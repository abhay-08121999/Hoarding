"""
Feature engineering.

Turns the cleaned catalogue into the numeric representations the
recommender ranks with:

  1. `soup`      - a weighted bag of atomic tokens per film
  2. `tfidf`     - L2-normalised TF-IDF matrix over that soup
  3. `categorical` - binary indicator matrix for genre/subgenre/language/mood
  4. `people`    - binary indicator matrix for director + cast
  5. `numeric`   - scaled rating / year / runtime columns

Keeping these separate (rather than dumping everything into one matrix)
is what makes the hybrid score explainable: we can report how much of a
recommendation came from theme vs. genre vs. cast vs. numbers.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

from . import config
from .text_nlp import analyzer, entity_token, phrase_token, tokenize


@dataclass
class FeatureSpace:
    """Everything the recommender needs to score a catalogue."""

    df: pd.DataFrame
    tfidf: sparse.csr_matrix
    vectorizer: TfidfVectorizer
    categorical: sparse.csr_matrix
    people: sparse.csr_matrix
    numeric: np.ndarray
    vocabulary: list = field(default_factory=list)

    @property
    def n_movies(self) -> int:
        return self.tfidf.shape[0]


def build_soup(row) -> str:
    """
    Compose one film's weighted token bag.

    Field weighting is done by repetition (see `config.FIELD_WEIGHTS`) and
    tempered afterwards by the vectoriser's sublinear TF, so a x3 field is
    meaningfully louder without completely drowning the synopsis.

    Multi-word values become atomic tokens: `kw_father_daughter`,
    `person_christopher_nolan`. This prevents spurious matches on shared
    common words ("father", "christopher") between unrelated films.
    """
    w = config.FIELD_WEIGHTS
    parts: list[str] = []

    def add(tokens, weight):
        tokens = [t for t in tokens if t]
        parts.extend(tokens * weight)

    add([f"genre_{g.lower().replace(' ', '_').replace('-', '_')}"
         for g in row["genres"]], w["genre"])
    add([f"sub_{s.lower().replace(' ', '_').replace('-', '_')}"
         for s in row["sub_genres"]], w["sub_genre"])
    add([phrase_token(k) for k in row["keyword_list"]], w["keywords"])
    add([entity_token(d, "person") for d in row["directors"]], w["director"])
    add([entity_token(a, "person") for a in row["cast_list"]], w["cast"])
    add([f"lang_{str(row['language']).lower().replace(' ', '_')}"], w["language"])
    add([f"country_{str(row['country']).lower().replace(' ', '_')}"], w["country"])
    add([f"era_{str(row['era']).lower().replace(' ', '_')}"], w["era"])
    add([f"mood_{m.replace('-', '_')}" for m in row["moods"]], w["mood"])

    # Free prose: stemmed content words from the synopsis.
    add(tokenize(row["description"]), w["description"])

    # Award text carries a weak quality/prestige signal.
    if row["has_awards"]:
        add(["tag_awarded"], w["awards"])

    return " ".join(parts)


def _scale(values: np.ndarray) -> np.ndarray:
    """Min-max scale to [0, 1], tolerating NaNs and constant columns."""
    v = np.asarray(values, dtype=float)
    if np.all(np.isnan(v)):
        return np.zeros_like(v)
    lo, hi = np.nanmin(v), np.nanmax(v)
    if hi - lo < 1e-9:
        return np.zeros_like(v)
    out = (v - lo) / (hi - lo)
    return np.nan_to_num(out, nan=float(np.nanmedian(out)))


def build_feature_space(df: pd.DataFrame) -> FeatureSpace:
    """Run the whole feature build and return every matrix bundled."""
    df = df.copy().reset_index(drop=True)
    df["soup"] = df.apply(build_soup, axis=1)

    # ---- 1. TF-IDF over the soup ---------------------------------------
    vectorizer = TfidfVectorizer(
        analyzer=analyzer,
        min_df=config.TFIDF["min_df"],
        max_df=config.TFIDF["max_df"],
        sublinear_tf=config.TFIDF["sublinear_tf"],
        norm=config.TFIDF["norm"],
    )
    tfidf = vectorizer.fit_transform(df["soup"])

    # ---- 2. Categorical indicators -------------------------------------
    def cat_tags(row):
        tags = []
        tags += [f"g:{g.lower()}" for g in row["genres"]]
        tags += [f"s:{s.lower()}" for s in row["sub_genres"]]
        tags.append(f"l:{str(row['language']).lower()}")
        tags.append(f"c:{str(row['country']).lower()}")
        tags += [f"m:{m}" for m in row["moods"]]
        tags.append(f"e:{str(row['era']).lower()}")
        return tags

    mlb_cat = MultiLabelBinarizer(sparse_output=True)
    categorical = mlb_cat.fit_transform(df.apply(cat_tags, axis=1)).tocsr()
    categorical = categorical.astype(np.float64)

    # ---- 3. People indicators ------------------------------------------
    def people_tags(row):
        return (
            [f"d:{d.lower()}" for d in row["directors"]]
            + [f"a:{a.lower()}" for a in row["cast_list"]]
        )

    mlb_people = MultiLabelBinarizer(sparse_output=True)
    people = mlb_people.fit_transform(df.apply(people_tags, axis=1)).tocsr()
    people = people.astype(np.float64)

    # ---- 4. Numeric block ----------------------------------------------
    numeric = np.column_stack(
        [
            _scale(df["rating"].astype(float).values),
            _scale(df["year"].astype(float).values),
            _scale(df["runtime"].astype(float).values),
        ]
    )

    return FeatureSpace(
        df=df,
        tfidf=tfidf.tocsr(),
        vectorizer=vectorizer,
        categorical=categorical,
        people=people,
        numeric=numeric,
        vocabulary=list(vectorizer.get_feature_names_out()),
    )
