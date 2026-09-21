"""
Evaluation.

There is no user-interaction data in this catalogue, so there are no
held-out clicks or ratings to test against. Instead we use *proxy
relevance*: signals that a human would agree indicate a good
content-based recommendation, computed over the whole catalogue.

Metrics reported
----------------
genre_precision@k   fraction of recommendations sharing >=1 genre with the seed
theme_overlap@k     mean Jaccard overlap of curated keyword sets
director_recall     for films by a director with >1 title in the catalogue,
                    how often a sibling film appears in the top-k
intra_list_div      1 - mean pairwise cosine similarity inside one result
                    list (higher = less repetitive)
catalogue_coverage  share of films that appear in *someone's* top-k
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def _genre_set(row) -> set:
    return {g.strip().lower() for g in row["genres"] if g.strip()}


def _keyword_set(row) -> set:
    return {k.strip().lower() for k in row["keyword_list"] if k.strip()}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def evaluate(recommender, k: int = 10, mmr_lambda: float = 1.0) -> dict:
    """
    Run the full evaluation sweep.

    `mmr_lambda=1.0` measures pure relevance ranking. Pass a lower value
    to see the relevance/diversity trade-off MMR buys.
    """
    df = recommender.df
    n = len(df)

    genre_hits, theme_scores, div_scores = [], [], []
    seen_in_topk = set()

    # Directors with more than one film — used for the recall check.
    director_films = defaultdict(list)
    for idx, row in df.iterrows():
        for d in row["directors"]:
            director_films[d.strip().lower()].append(idx)
    multi_director = {
        d: idxs for d, idxs in director_films.items() if len(idxs) > 1
    }

    director_hits, director_total = 0, 0

    for idx in range(n):
        recs = recommender.similar_to(
            df.loc[idx, "movie_id"], top_n=k, mmr_lambda=mmr_lambda
        )
        if not recs:
            continue
        rec_idx = [recommender.index_of(r.movie_id) for r in recs]
        seen_in_topk.update(rec_idx)

        seed_genres = _genre_set(df.loc[idx])
        seed_keywords = _keyword_set(df.loc[idx])

        genre_hits.append(
            np.mean([
                1.0 if seed_genres & _genre_set(df.loc[j]) else 0.0
                for j in rec_idx
            ])
        )
        theme_scores.append(
            np.mean([_jaccard(seed_keywords, _keyword_set(df.loc[j]))
                     for j in rec_idx])
        )

        if len(rec_idx) > 1:
            sims = cosine_similarity(recommender.space.tfidf[rec_idx])
            upper = sims[np.triu_indices(len(rec_idx), k=1)]
            div_scores.append(1.0 - float(np.mean(upper)))

        # Director recall: does a sibling film by the same director show up?
        for d in df.loc[idx, "directors"]:
            key = d.strip().lower()
            if key in multi_director:
                siblings = set(multi_director[key]) - {idx}
                if siblings:
                    director_total += 1
                    if siblings & set(rec_idx):
                        director_hits += 1
                break

    return {
        "k": k,
        "mmr_lambda": mmr_lambda,
        "n_movies": n,
        "genre_precision@k": round(float(np.mean(genre_hits)), 4),
        "theme_overlap@k": round(float(np.mean(theme_scores)), 4),
        "director_recall@k": round(
            director_hits / director_total if director_total else 0.0, 4
        ),
        "intra_list_diversity": round(float(np.mean(div_scores)), 4),
        "catalogue_coverage": round(len(seen_in_topk) / n, 4),
    }


def ablation(recommender, k: int = 10) -> list:
    """
    Turn each similarity signal off in turn and re-measure.

    This is the honest way to show the hybrid is earning its keep: if
    dropping a signal doesn't move the metrics, that signal is decoration.
    """
    import copy

    base_weights = dict(recommender.weights)
    results = []

    configs = [("full hybrid", base_weights)]
    for name in base_weights:
        w = dict(base_weights)
        w[name] = 0.0
        total = sum(w.values())
        if total > 0:
            w = {kk: vv / total for kk, vv in w.items()}
        configs.append((f"without {name}", w))
    configs.append(
        ("text only", {"text": 1.0, "categorical": 0.0,
                       "people": 0.0, "numeric": 0.0})
    )

    for label, weights in configs:
        recommender.weights = weights
        metrics = evaluate(recommender, k=k, mmr_lambda=1.0)
        metrics["config"] = label
        results.append(metrics)

    recommender.weights = base_weights
    return results


def diversity_sweep(recommender, k: int = 10) -> list:
    """Show how MMR lambda trades relevance against variety."""
    out = []
    for lam in [1.0, 0.85, 0.7, 0.55, 0.4]:
        m = evaluate(recommender, k=k, mmr_lambda=lam)
        out.append(m)
    return out


def format_report(metrics_list, title: str) -> str:
    """Render a list of metric dicts as a fixed-width table."""
    if not metrics_list:
        return ""
    cols = [
        ("config", "configuration", 22),
        ("genre_precision@k", "genre P@k", 11),
        ("theme_overlap@k", "theme ovl", 11),
        ("director_recall@k", "dir recall", 11),
        ("intra_list_diversity", "diversity", 11),
        ("catalogue_coverage", "coverage", 10),
    ]
    if "config" not in metrics_list[0]:
        cols[0] = ("mmr_lambda", "mmr lambda", 22)

    lines = [title, "-" * 82]
    lines.append("".join(h.ljust(w) for _, h, w in cols))
    for m in metrics_list:
        lines.append(
            "".join(str(m.get(key, "")).ljust(w) for key, _, w in cols)
        )
    return "\n".join(lines)
