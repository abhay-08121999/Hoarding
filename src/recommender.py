"""
The recommendation engine.

Content-based, with four independent similarity signals blended into one
score, optional filtering, MMR diversification, and per-recommendation
explanations.

Why content-based rather than collaborative filtering: the catalogue has
no user-interaction data at all (no ratings matrix, no watch history), so
item-item similarity over film metadata is the only approach that can
actually work here. The taste-profile feature below is a lightweight way
to get personalisation back without needing other users.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

from . import config
from .features import FeatureSpace
from .query_engine import SEARCH_FLOOR, QueryExpander, search_scores


@dataclass
class Recommendation:
    movie_id: str
    title: str
    score: float
    reasons: list
    breakdown: dict


class MovieRecommender:
    """Ranks films against a seed film, a set of seeds, or a free-text query."""

    def __init__(self, space: FeatureSpace, weights: dict | None = None):
        self.space = space
        self.df = space.df
        self.weights = dict(weights or config.SCORE_WEIGHTS)

        self._id_to_idx = {mid: i for i, mid in enumerate(self.df["movie_id"])}
        self._title_to_idx = {
            str(t).lower(): i for i, t in enumerate(self.df["title"])
        }

        # Row-normalise the indicator matrices once so their dot products
        # are cosine similarities (equivalently, set-overlap ratios).
        self._cat_norm = self._row_normalise(space.categorical)
        self._people_norm = self._row_normalise(space.people)

        self._quality = self._scaled_rating()
        self.query_expander = QueryExpander(space.vocabulary)

    # ------------------------------------------------------------- utils
    @staticmethod
    def _row_normalise(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
        norms = np.sqrt(matrix.multiply(matrix).sum(axis=1)).A.ravel()
        norms[norms == 0] = 1.0
        inv = sparse.diags(1.0 / norms)
        return (inv @ matrix).tocsr()

    def _scaled_rating(self) -> np.ndarray:
        r = self.df["rating"].astype(float).values
        if np.all(np.isnan(r)):
            return np.zeros(len(r))
        lo, hi = np.nanmin(r), np.nanmax(r)
        out = (r - lo) / (hi - lo) if hi > lo else np.zeros_like(r)
        return np.nan_to_num(out, nan=0.5)

    def index_of(self, ref: str) -> int:
        """Resolve a movie_id or a title (case-insensitive) to a row index."""
        if ref in self._id_to_idx:
            return self._id_to_idx[ref]
        key = str(ref).strip().lower()
        if key in self._title_to_idx:
            return self._title_to_idx[key]
        # Forgiving fallback: unique substring match on title.
        hits = [i for t, i in self._title_to_idx.items() if key and key in t]
        if len(hits) == 1:
            return hits[0]
        raise KeyError(f"Unknown film: {ref!r}")

    # -------------------------------------------------- similarity parts
    def _numeric_similarity(self, idx: int) -> np.ndarray:
        """
        Closeness in rating, release year and runtime.

        Each dimension contributes a decaying similarity rather than a hard
        match, so a 1994 film is 'near' a 1997 film but not a 2021 one.
        """
        num = self.space.numeric
        target = num[idx]
        diffs = np.abs(num - target)
        # Per-dimension tolerance: rating is tight, year looser, runtime loosest.
        tolerance = np.array([0.15, 0.30, 0.40])
        sims = np.exp(-diffs / tolerance)
        return sims.mean(axis=1)

    def _signal_matrix(self, idx: int) -> dict:
        text = cosine_similarity(
            self.space.tfidf[idx], self.space.tfidf
        ).ravel()
        cat = (self._cat_norm @ self._cat_norm[idx].T).toarray().ravel()
        people = (self._people_norm @ self._people_norm[idx].T).toarray().ravel()
        numeric = self._numeric_similarity(idx)
        return {
            "text": text,
            "categorical": cat,
            "people": people,
            "numeric": numeric,
        }

    def _blend(self, signals: dict) -> np.ndarray:
        total = np.zeros(len(self.df))
        for name, weight in self.weights.items():
            total += weight * signals[name]
        return total + config.POPULARITY_PRIOR * self._quality

    # ------------------------------------------------------- explanations
    def _explain(self, seed_idx: int, other_idx: int, limit: int = 4) -> list:
        """
        Say *why* two films were matched.

        Takes the element-wise minimum of the two TF-IDF vectors — the
        shared mass — and reports the heaviest shared terms, translated
        back from internal tokens into readable phrases.
        """
        a = self.space.tfidf[seed_idx].toarray().ravel()
        b = self.space.tfidf[other_idx].toarray().ravel()
        shared = np.minimum(a, b)
        if shared.sum() == 0:
            return []
        order = np.argsort(-shared)[: limit * 3]
        reasons = []
        seen = set()
        for j in order:
            if shared[j] <= 0:
                break
            label = self._pretty_term(self.space.vocabulary[j])
            if not label or label.lower() in seen:
                continue
            seen.add(label.lower())
            reasons.append(label)
            if len(reasons) >= limit:
                break
        return reasons

    @staticmethod
    def _pretty_term(token: str) -> str:
        """Turn an internal token back into something a person can read."""
        prefixes = {
            "person_": "", "kw_": "", "genre_": "", "sub_": "",
            "lang_": "", "country_": "", "era_": "", "mood_": "",
            "tag_": "",
        }
        for prefix, _ in prefixes.items():
            if token.startswith(prefix):
                body = token[len(prefix):].replace("_", " ")
                if prefix == "person_":
                    return body.title()
                if prefix == "tag_":
                    return None  # internal flag, not user-facing
                return body
        # Bare stemmed prose word — too mangled to show ("technologi").
        return None

    # ------------------------------------------------------------- public
    def similar_to(
        self,
        ref: str,
        top_n: int = 10,
        filters: dict | None = None,
        mmr_lambda: float | None = None,
    ) -> list:
        """Recommend films similar to a single seed film."""
        idx = self.index_of(ref)
        signals = self._signal_matrix(idx)
        scores = self._blend(signals)
        scores[idx] = -np.inf  # never recommend the seed back

        candidates = self._apply_filters(scores, filters)
        ranked = self._diversify(candidates, scores, top_n, mmr_lambda)

        return [
            Recommendation(
                movie_id=self.df.loc[i, "movie_id"],
                title=self.df.loc[i, "title"],
                score=round(float(scores[i]), 4),
                reasons=self._explain(idx, i),
                breakdown={k: round(float(v[i]), 3) for k, v in signals.items()},
            )
            for i in ranked
        ]

    def taste_profile(
        self,
        liked: list,
        disliked: list | None = None,
        alpha: float = 1.0,
        beta: float = 0.45,
    ) -> sparse.csr_matrix:
        """
        Build one vector representing a person's taste (Rocchio relevance
        feedback): the centroid of what they liked, pushed away from the
        centroid of what they didn't.

        Negative feedback is deliberately weighted lower than positive
        (beta < alpha) because a thumbs-down is a weaker, noisier signal
        than a thumbs-up.
        """
        liked_idx = [self.index_of(r) for r in liked]
        disliked_idx = [self.index_of(r) for r in (disliked or [])]

        if not liked_idx and not disliked_idx:
            raise ValueError("Need at least one liked or disliked film")

        dim = self.space.tfidf.shape[1]
        vec = sparse.csr_matrix((1, dim))
        if liked_idx:
            vec = vec + alpha * sparse.csr_matrix(
                self.space.tfidf[liked_idx].mean(axis=0)
            )
        if disliked_idx:
            vec = vec - beta * sparse.csr_matrix(
                self.space.tfidf[disliked_idx].mean(axis=0)
            )

        vec = vec.toarray().ravel()
        vec[vec < 0] = 0  # clipping keeps the profile a valid tf-idf point
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return sparse.csr_matrix(vec)

    def for_profile(
        self,
        liked: list,
        disliked: list | None = None,
        top_n: int = 12,
        filters: dict | None = None,
        mmr_lambda: float | None = None,
    ) -> list:
        """Recommend against a multi-film taste profile."""
        profile = self.taste_profile(liked, disliked)
        text = cosine_similarity(profile, self.space.tfidf).ravel()

        liked_idx = [self.index_of(r) for r in liked]
        disliked_idx = [self.index_of(r) for r in (disliked or [])]

        cat = np.zeros(len(self.df))
        people = np.zeros(len(self.df))
        if liked_idx:
            cat = np.asarray(
                (self._cat_norm @ self._cat_norm[liked_idx].T).mean(axis=1)
            ).ravel()
            people = np.asarray(
                (self._people_norm @ self._people_norm[liked_idx].T).mean(axis=1)
            ).ravel()

        numeric = (
            np.mean([self._numeric_similarity(i) for i in liked_idx], axis=0)
            if liked_idx
            else np.zeros(len(self.df))
        )

        signals = {
            "text": text, "categorical": cat,
            "people": people, "numeric": numeric,
        }
        scores = self._blend(signals)
        for i in liked_idx + disliked_idx:
            scores[i] = -np.inf  # don't recommend films already rated

        candidates = self._apply_filters(scores, filters)
        ranked = self._diversify(candidates, scores, top_n, mmr_lambda)

        nearest = liked_idx[0] if liked_idx else None
        return [
            Recommendation(
                movie_id=self.df.loc[i, "movie_id"],
                title=self.df.loc[i, "title"],
                score=round(float(scores[i]), 4),
                reasons=(
                    self._explain(self._closest_seed(i, liked_idx), i)
                    if liked_idx else []
                ),
                breakdown={k: round(float(v[i]), 3) for k, v in signals.items()},
            )
            for i in ranked
        ]

    def _closest_seed(self, idx: int, seeds: list) -> int:
        """Which liked film best explains this recommendation."""
        if not seeds:
            return idx
        sims = cosine_similarity(
            self.space.tfidf[idx], self.space.tfidf[seeds]
        ).ravel()
        return seeds[int(np.argmax(sims))]

    def search(
        self,
        query: str,
        top_n: int = 12,
        filters: dict | None = None,
    ) -> list:
        """
        Free-text search that ranks by meaning, not just title matching.

        The query is expanded onto the catalogue's atomic tokens first
        (see `query_engine.QueryExpander`), so "mind bending space movies"
        resolves to `kw_mind_bending` + `kw_space` and ranks films whose
        *content* matches, rather than any film whose synopsis happens to
        contain the word "space".
        """
        if not str(query).strip():
            return []

        relevance, matched = search_scores(self, query)
        # The popularity prior only breaks ties *after* a film has earned a
        # place. Adding it first let every well-rated film clear the floor.
        scores = relevance + config.POPULARITY_PRIOR * self._quality

        candidates = [i for i in self._apply_filters(scores, filters)
                      if relevance[i] >= SEARCH_FLOOR]
        ranked = sorted(candidates, key=lambda i: -scores[i])[:top_n]

        return [
            Recommendation(
                movie_id=self.df.loc[i, "movie_id"],
                title=self.df.loc[i, "title"],
                score=round(float(scores[i]), 4),
                reasons=matched[:4],
                breakdown={"text": round(float(scores[i]), 3)},
            )
            for i in ranked
        ]

    # ------------------------------------------------------ filter + MMR
    def _apply_filters(self, scores: np.ndarray, filters: dict | None) -> list:
        """Return the row indices still eligible after applying filters."""
        mask = np.isfinite(scores)
        if filters:
            df = self.df
            if filters.get("genres"):
                want = {g.lower() for g in filters["genres"]}
                mask &= df["genres"].map(
                    lambda gs: bool(want & {g.lower() for g in gs})
                ).values
            if filters.get("languages"):
                want = {l.lower() for l in filters["languages"]}
                mask &= df["language"].str.lower().isin(want).values
            if filters.get("moods"):
                want = set(filters["moods"])
                mask &= df["moods"].map(lambda ms: bool(want & set(ms))).values
            if filters.get("min_rating") is not None:
                mask &= (df["rating"].fillna(0) >= filters["min_rating"]).values
            if filters.get("year_range"):
                lo, hi = filters["year_range"]
                yr = df["year"].astype(float).fillna(-1)
                mask &= ((yr >= lo) & (yr <= hi)).values
            if filters.get("max_runtime") is not None:
                mask &= (
                    df["runtime"].fillna(9999) <= filters["max_runtime"]
                ).values
            if filters.get("age_bands"):
                mask &= df["age_band"].isin(filters["age_bands"]).values
        return [i for i in np.where(mask)[0]]

    def _diversify(
        self,
        candidates: list,
        scores: np.ndarray,
        top_n: int,
        mmr_lambda: float | None,
    ) -> list:
        """
        Maximal Marginal Relevance.

        Greedily picks the next film that is relevant to the seed but
        *unlike* what's already been picked. Without this, asking for
        films like the first Godfather returns Part II, Part III, and
        four more mafia dramas — technically correct and useless.
        """
        lam = config.DEFAULT_MMR_LAMBDA if mmr_lambda is None else mmr_lambda
        pool = sorted(candidates, key=lambda i: -scores[i])[: max(top_n * 5, 40)]
        if lam >= 0.999 or not pool:
            return pool[:top_n]

        selected: list = []
        pool = list(pool)
        while pool and len(selected) < top_n:
            if not selected:
                best = pool[0]
            else:
                sims = cosine_similarity(
                    self.space.tfidf[pool], self.space.tfidf[selected]
                ).max(axis=1)
                mmr = lam * scores[pool] - (1 - lam) * sims
                best = pool[int(np.argmax(mmr))]
            selected.append(best)
            pool.remove(best)
        return selected

    # ------------------------------------------------------------ extras
    def neighbours_table(self, top_n: int | None = None) -> dict:
        """Precompute top-N neighbours for every film (for the web export)."""
        top_n = top_n or config.TOP_N_NEIGHBOURS
        out = {}
        for idx, mid in enumerate(self.df["movie_id"]):
            signals = self._signal_matrix(idx)
            scores = self._blend(signals)
            scores[idx] = -np.inf
            order = np.argsort(-scores)[:top_n]
            out[mid] = [
                {
                    "id": self.df.loc[j, "movie_id"],
                    "score": round(float(scores[j]), 4),
                    "why": self._explain(idx, j, limit=3),
                }
                for j in order
                if np.isfinite(scores[j])
            ]
        return out