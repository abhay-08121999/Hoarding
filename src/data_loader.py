"""
Load, clean, merge and enrich the four source workbooks.

Pipeline for each file:
    read -> map headers to canonical schema -> coerce types -> clean text
and then across files:
    concat -> dedupe on (title, year) with field-level coalescing
           -> derive era / runtime / mood / decade facets.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .schema import CANONICAL_FIELDS, map_columns, split_multi
from .text_nlp import normalise

# Certificates differ by country; map everything onto one ordered ladder
# so "suitable for kids?" is answerable across Indian and US ratings.
AGE_ORDER = {
    "U": 0, "G": 0,
    "U/A": 1, "PG": 1,
    "PG-13": 2, "UA": 1, "UA 13+": 2, "UA 16+": 3,
    "A": 4, "R": 4, "NC-17": 5,
}
AGE_BANDS = {
    0: "All ages",
    1: "Guidance",
    2: "Teens",
    3: "Teens",
    4: "Adults",
    5: "Adults",
}


def _clean_text(value) -> str:
    """Trim, collapse whitespace, and turn spreadsheet null markers into ''."""
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "n/a", "na", "-", "null", ""}:
        return ""
    return re.sub(r"\s+", " ", text)


def _to_number(value, cast=float):
    """Pull the first number out of a cell; return None if there isn't one."""
    text = _clean_text(value)
    if not text:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if not match:
        return None
    try:
        return cast(float(match.group()))
    except (TypeError, ValueError):
        return None


def _bucket(value, buckets, default="Unknown") -> str:
    if value is None:
        return default
    for low, high, label in buckets:
        if low <= value <= high:
            return label
    return default


def load_workbook(path: Path) -> pd.DataFrame:
    """Read one workbook and rename its columns to the canonical schema."""
    df = pd.read_excel(path, sheet_name=0)
    df = df.rename(columns=map_columns(df.columns))

    for field in CANONICAL_FIELDS:
        if field not in df.columns:
            df[field] = None

    df = df[CANONICAL_FIELDS].copy()
    df["source_file"] = path.name
    return df


def load_all(raw_dir: Path | None = None) -> pd.DataFrame:
    """Read every workbook in the raw directory into one frame."""
    raw_dir = Path(raw_dir or config.RAW_DIR)
    paths = sorted(raw_dir.glob("*.xlsx"))
    if not paths:
        raise FileNotFoundError(f"No .xlsx files found in {raw_dir}")
    frames = [load_workbook(p) for p in paths]
    return pd.concat(frames, ignore_index=True)


def _cue_hits(cue: str, haystack: str) -> int:
    """
    Count whole-word occurrences of a cue phrase.

    Word boundaries are essential: a naive substring test makes the cue
    "war" fire on every film whose awards column says "Academy Award",
    and "art" fire on "heart". Both were mislabelling large parts of the
    catalogue before this was tightened.
    """
    pattern = r"\b" + re.escape(normalise(cue)) + r"\b"
    return len(re.findall(pattern, haystack))


def _assign_moods(row) -> list[str]:
    """
    Tag a film with up to three moods, scored across weighted fields.

    Curated keywords and the subgenre are far more reliable mood signals
    than free prose, so they count for more. A mood needs a score of 2 to
    stick, which stops a single incidental word in a synopsis from
    labelling a horror film "feel-good".
    """
    fields = {
        "keywords": (str(row.get("keywords", "")), 2.0),
        "sub_genre": (str(row.get("sub_genre", "")), 2.0),
        "genre": (str(row.get("genre", "")), 1.5),
        "description": (str(row.get("description", "")), 0.75),
    }
    normalised = {k: (normalise(v), w) for k, (v, w) in fields.items()}

    film_genres = {g.strip().lower() for g in split_multi(row.get("genre", ""))}

    scored = []
    for mood, spec in config.MOOD_LEXICON.items():
        blockers = {b.lower() for b in spec.get("blocks", [])}
        if film_genres & blockers:
            continue
        score = 0.0
        for cue in spec["cues"]:
            for text, weight in normalised.values():
                score += _cue_hits(cue, text) * weight
        if score >= 2.0:
            scored.append((mood, score))

    scored.sort(key=lambda x: (-x[1], x[0]))
    return [m for m, _ in scored[:3]]


def _merge_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse rows describing the same film across workbooks.

    Key is (normalised title, year) rather than title alone: the catalogue
    contains two distinct films called *Drishyam* (Malayalam 2013 and its
    Hindi remake 2015) which must stay separate, while *3 Idiots*,
    *Dangal*, *Gladiator* and *The Dark Knight* genuinely appear in two
    files each and must merge.

    Merging is field-level: for each column we take the first non-empty
    value across the duplicate rows, so a row missing its synopsis inherits
    one from its twin instead of losing it.
    """
    df = df.copy()
    df["_key"] = (
        df["title"].map(normalise) + "__" + df["year"].astype("Int64").astype(str)
    )

    merged_rows = []
    for _, group in df.groupby("_key", sort=False):
        if len(group) == 1:
            merged_rows.append(group.iloc[0].to_dict())
            continue
        base = {}
        for col in group.columns:
            values = [v for v in group[col].tolist() if _clean_text(v) != ""]
            base[col] = values[0] if values else ""
        # Record the merge so provenance isn't silently lost.
        base["source_file"] = " + ".join(sorted(set(group["source_file"])))
        base["merged_from"] = len(group)
        merged_rows.append(base)

    out = pd.DataFrame(merged_rows)
    if "merged_from" not in out.columns:
        out["merged_from"] = 1
    out["merged_from"] = out["merged_from"].fillna(1).astype(int)
    return out.drop(columns=["_key"])


def build_clean_frame(raw_dir: Path | None = None) -> pd.DataFrame:
    """Full clean + enrich. This is the single entry point for the pipeline."""
    df = load_all(raw_dir)

    # ---- type coercion -------------------------------------------------
    for col in ["title", "genre", "sub_genre", "language", "director",
                "cast", "country", "description", "keywords", "age_rating",
                "awards", "source_id"]:
        df[col] = df[col].map(_clean_text)

    df["year"] = df["year"].map(lambda v: _to_number(v, int))
    df["runtime"] = df["runtime"].map(lambda v: _to_number(v, int))
    df["rating"] = df["rating"].map(lambda v: _to_number(v, float))

    # ---- drop unusable rows --------------------------------------------
    before = len(df)
    df = df[df["title"].astype(bool)].copy()
    df["year"] = df["year"].astype("Int64")
    dropped = before - len(df)

    # ---- merge cross-file duplicates -----------------------------------
    df = _merge_duplicates(df)

    # ---- multi-value fields as lists -----------------------------------
    df["genres"] = df["genre"].map(split_multi)
    df["sub_genres"] = df["sub_genre"].map(split_multi)
    df["cast_list"] = df["cast"].map(split_multi)
    df["directors"] = df["director"].map(split_multi)
    df["keyword_list"] = df["keywords"].map(split_multi)

    # ---- sensible fallbacks for missing values -------------------------
    df["description"] = df.apply(
        lambda r: r["description"]
        or f"{r['title']} is a {r['sub_genre'] or r['genre']} film "
           f"in {r['language'] or 'an unlisted language'}.",
        axis=1,
    )
    df["age_rating"] = df["age_rating"].replace("", "Not rated")
    df["awards"] = df["awards"].replace("", "No major awards listed")
    df["language"] = df["language"].replace("", "Unknown")
    df["country"] = df["country"].replace("", "Unknown")

    # ---- derived facets -------------------------------------------------
    df["era"] = df["year"].map(
        lambda y: _bucket(int(y) if pd.notna(y) else None, config.ERA_BUCKETS)
    )
    df["runtime_band"] = df["runtime"].map(
        lambda r: _bucket(r, config.RUNTIME_BUCKETS)
    )
    df["decade"] = df["year"].map(
        lambda y: f"{int(y) // 10 * 10}s" if pd.notna(y) else "Unknown"
    )
    df["age_level"] = df["age_rating"].map(lambda a: AGE_ORDER.get(a, 2))
    df["age_band"] = df["age_level"].map(lambda lv: AGE_BANDS.get(lv, "Teens"))
    df["has_awards"] = df["awards"].map(
        lambda a: not a.lower().startswith("no major")
    )
    df["moods"] = df.apply(_assign_moods, axis=1)
    df["primary_genre"] = df["genres"].map(lambda g: g[0] if g else "Unknown")
    df["is_indian"] = df["country"].map(lambda c: c.strip().lower() == "india")

    # ---- stable public id ----------------------------------------------
    df = df.sort_values(["title", "year"], kind="stable").reset_index(drop=True)
    df["movie_id"] = [f"m{idx:04d}" for idx in range(len(df))]

    df.attrs["rows_read"] = before
    df.attrs["rows_dropped"] = dropped
    df.attrs["duplicates_merged"] = int((df["merged_from"] > 1).sum())

    return df


def save_clean(df: pd.DataFrame) -> None:
    """Persist the cleaned catalogue as both CSV and JSON."""
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    flat = df.copy()
    for col in ["genres", "sub_genres", "cast_list", "directors",
                "keyword_list", "moods"]:
        flat[col] = flat[col].map(lambda xs: "|".join(xs))
    flat.to_csv(config.CLEAN_CSV, index=False)
    df.to_json(config.CLEAN_JSON, orient="records", indent=2)
