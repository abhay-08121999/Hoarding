"""
Export the trained model for the browser.

The front end is not a thin view over a Python server — it runs the same
ranking maths client-side so it works from a plain `file://` open with no
backend at all. To make that possible we ship:

  * the cleaned catalogue
  * every film's sparse L2-normalised TF-IDF vector
  * the vocabulary and its IDF weights
  * the phrase map + stem map so JS can vectorise a typed query
    *identically* to Python, without reimplementing the Porter stemmer
  * precomputed nearest neighbours, so opening a film is instant
  * aggregate stats for the charts

Output is a `.js` file assigning to `window.MOVIE_DB` rather than a
`.json` file, specifically so it loads over `file://`, where `fetch()` of
a local JSON is blocked by CORS.
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pandas as pd

from . import config
from .query_engine import SYNONYMS
from .text_nlp import STOPWORDS, normalise, stem

# Fields sent to the browser for display.
DISPLAY_FIELDS = [
    "movie_id", "title", "year", "runtime", "rating", "language", "country",
    "director", "cast", "description", "age_rating", "age_band", "awards",
    "era", "decade", "runtime_band", "primary_genre", "sub_genre",
    "has_awards", "source_file",
]
LIST_FIELDS = ["genres", "sub_genres", "cast_list", "directors",
               "keyword_list", "moods"]


def _jsonable(value):
    """Coerce numpy / pandas scalars into plain JSON types."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd._libs.missing.NAType.__class__):
        return None
    return value


def build_movie_records(space, neighbours: dict) -> list:
    """One JSON record per film, including its sparse TF-IDF vector."""
    df = space.df
    tfidf = space.tfidf
    records = []

    for i, row in df.iterrows():
        rec = {}
        for field in DISPLAY_FIELDS:
            val = row.get(field)
            if pd.isna(val) if not isinstance(val, (list, dict)) else False:
                val = None
            rec[field] = _jsonable(val)
        for field in LIST_FIELDS:
            rec[field] = list(row.get(field) or [])

        # Sparse vector as [termIndex, weight] pairs, weights rounded to
        # 4dp — plenty for ranking and it roughly halves the payload.
        start, end = tfidf.indptr[i], tfidf.indptr[i + 1]
        pairs = [
            [int(tfidf.indices[p]), round(float(tfidf.data[p]), 4)]
            for p in range(start, end)
            if tfidf.data[p] > 0.005
        ]
        pairs.sort(key=lambda x: -x[1])
        rec["vec"] = pairs
        rec["nn"] = [
            [n["id"], n["score"]] for n in neighbours.get(rec["movie_id"], [])
        ]
        records.append(rec)

    return records


def build_lexicon(space) -> dict:
    """
    Everything the browser needs to turn typed text into a query vector.

    `phrases` maps a human phrase onto the atomic token indices it should
    activate. `stems` maps raw corpus words straight onto vocabulary
    indices, which means the JS never has to run a stemmer to handle
    words it has already seen; a small suffix-stripper in JS covers the
    long tail of unseen variants.
    """
    vocab = space.vocabulary
    term_index = {t: i for i, t in enumerate(vocab)}

    phrases: dict[str, list] = {}
    prefixes = ("kw_", "genre_", "sub_", "person_", "mood_", "lang_",
                "country_", "era_")
    for token, idx in term_index.items():
        for prefix in prefixes:
            if token.startswith(prefix):
                phrase = token[len(prefix):].replace("_", " ").strip()
                if phrase:
                    phrases.setdefault(phrase, [])
                    if idx not in phrases[phrase]:
                        phrases[phrase].append(idx)
                break

    # Raw word -> vocabulary index, harvested from every text field.
    stems: dict[str, int] = {}
    df = space.df
    text_blob = " ".join(
        df["description"].astype(str).tolist()
        + df["title"].astype(str).tolist()
        + df["keywords"].astype(str).tolist()
    )
    for raw in set(normalise(text_blob).split()):
        if raw in STOPWORDS or len(raw) < 2:
            continue
        s = stem(raw)
        if s in term_index:
            stems[raw] = term_index[s]

    # Also index the bare stems themselves for JS-side fallback lookups.
    stem_index = {
        t: i for t, i in term_index.items() if not t.startswith(prefixes)
    }

    idf = space.vectorizer.idf_.round(4).tolist()

    return {
        "vocab": vocab,
        "idf": idf,
        "phrases": phrases,
        "stems": stems,
        "stemIndex": stem_index,
        "synonyms": SYNONYMS,
    }


def build_stats(df: pd.DataFrame) -> dict:
    """Aggregates powering the Explore tab's charts and facet counts."""
    genre_counts = Counter(g for gs in df["genres"] for g in gs)
    lang_counts = Counter(df["language"])
    country_counts = Counter(df["country"])
    mood_counts = Counter(m for ms in df["moods"] for m in ms)
    decade_counts = Counter(df["decade"])
    age_counts = Counter(df["age_rating"])

    director_counts = Counter(
        d for ds in df["directors"] for d in ds if d.strip()
    )
    actor_counts = Counter(
        a for cs in df["cast_list"] for a in cs if a.strip()
    )

    rating_hist = Counter(
        f"{float(r):.1f}"[:3] for r in df["rating"].dropna()
    )

    # Mean rating per genre — only where the sample is big enough to mean
    # anything, otherwise a single 9.3 film makes a genre look dominant.
    genre_rating = {}
    for genre in genre_counts:
        subset = df[df["genres"].map(lambda gs: genre in gs)]
        if len(subset) >= 3:
            genre_rating[genre] = round(float(subset["rating"].mean()), 2)

    decade_rating = {}
    for dec in sorted(decade_counts):
        subset = df[df["decade"] == dec]
        if len(subset) >= 2:
            decade_rating[dec] = round(float(subset["rating"].mean()), 2)

    return {
        "totals": {
            "movies": int(len(df)),
            "genres": len(genre_counts),
            "languages": len(lang_counts),
            "countries": len(country_counts),
            "directors": len(director_counts),
            "actors": len(actor_counts),
            "yearMin": int(df["year"].min()),
            "yearMax": int(df["year"].max()),
            "ratingMin": round(float(df["rating"].min()), 1),
            "ratingMax": round(float(df["rating"].max()), 1),
            "runtimeMin": int(df["runtime"].min()),
            "runtimeMax": int(df["runtime"].max()),
            "avgRating": round(float(df["rating"].mean()), 2),
            "avgRuntime": int(df["runtime"].mean()),
        },
        "genres": genre_counts.most_common(),
        "languages": lang_counts.most_common(),
        "countries": country_counts.most_common(),
        "moods": mood_counts.most_common(),
        "decades": sorted(decade_counts.items()),
        "ageRatings": age_counts.most_common(),
        "topDirectors": director_counts.most_common(20),
        "topActors": actor_counts.most_common(20),
        "ratingHistogram": sorted(rating_hist.items()),
        "genreRating": genre_rating,
        "decadeRating": decade_rating,
    }


def export(space, recommender, path=None, metrics=None) -> dict:
    """Write `web/js/data.js` and return a small summary of what was written."""
    path = path or config.WEB_DATA_JS
    path.parent.mkdir(parents=True, exist_ok=True)

    neighbours = recommender.neighbours_table()
    payload = {
        "meta": {
            "generated": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            "sources": sorted(
                {s for v in space.df["source_file"] for s in str(v).split(" + ")}
            ),
            "vocabSize": len(space.vocabulary),
            "scoreWeights": recommender.weights,
            "moodLexicon": {
                m: spec["cues"] for m, spec in config.MOOD_LEXICON.items()
            },
            "metrics": metrics or {},
        },
        "movies": build_movie_records(space, neighbours),
        "lexicon": build_lexicon(space),
        "stats": build_stats(space.df),
    }

    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    path.write_text(
        "/* Generated by build.py - do not edit by hand. */\n"
        "window.MOVIE_DB = " + body + ";\n",
        encoding="utf-8",
    )

    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "movies": len(payload["movies"]),
        "vocab": len(space.vocabulary),
    }
