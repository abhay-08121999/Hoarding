"""
Schema reconciliation.

The four source workbooks carry the same 15 fields but name them
differently (`Movie Name` / `Title`, `Genre` / `Genres` / `Category`, ...)
and use different multi-value separators (`,` vs `|` vs `/`).

This module maps any of those headers onto one canonical schema so the
rest of the pipeline never has to care which file a row came from.
"""
from __future__ import annotations

import re

# Canonical field order used everywhere downstream.
CANONICAL_FIELDS = [
    "source_id",
    "title",
    "genre",
    "sub_genre",
    "language",
    "year",
    "runtime",
    "rating",
    "director",
    "cast",
    "country",
    "description",
    "keywords",
    "age_rating",
    "awards",
]

# Every header spelling observed across the four workbooks, lowercased and
# stripped of punctuation, mapped to its canonical name.
HEADER_ALIASES = {
    # identifier
    "movie id": "source_id", "id": "source_id", "movie_id": "source_id",
    "movieid": "source_id",
    # title
    "movie name": "title", "title": "title", "name": "title",
    "film": "title", "film name": "title",
    # genre
    "genre": "genre", "genres": "genre", "category": "genre",
    "categories": "genre",
    # sub-genre
    "sub-genre": "sub_genre", "subgenre": "sub_genre",
    "sub genre": "sub_genre", "sub_genre": "sub_genre",
    # language
    "language": "language", "original language": "language",
    "lang": "language",
    # year
    "release year": "year", "year": "year", "released": "year",
    "release_year": "year", "release date": "year",
    # runtime
    "runtime (min)": "runtime", "duration (mins)": "runtime",
    "runtime": "runtime", "length (min)": "runtime",
    "duration": "runtime", "length": "runtime",
    # rating
    "imdb rating": "rating", "rating": "rating", "imdb score": "rating",
    "imdb_rating": "rating", "score": "rating",
    # director
    "director": "director", "directed by": "director",
    "filmmaker": "director",
    # cast
    "cast": "cast", "actors": "cast", "starring": "cast",
    "lead actors": "cast",
    # country
    "country": "country", "country of origin": "country", "origin": "country",
    # description
    "description": "description", "plot": "description",
    "synopsis": "description", "summary": "description",
    "overview": "description",
    # keywords
    "keywords": "keywords", "tags": "keywords", "themes": "keywords",
    # age rating
    "age rating": "age_rating", "certificate": "age_rating",
    "rated": "age_rating", "age_rating": "age_rating",
    "certification": "age_rating",
    # awards
    "awards": "awards", "awards won": "awards", "accolades": "awards",
}

# Separators seen in multi-value cells across the four files.
MULTI_VALUE_SPLIT = re.compile(r"\s*[|/,;·]\s*|\s+&\s+|\s+and\s+")


def normalise_header(header: str) -> str:
    """Lowercase, collapse whitespace, drop stray punctuation from a header."""
    h = str(header).strip().lower()
    h = h.replace("\n", " ")
    h = re.sub(r"\s+", " ", h)
    return h


def map_columns(headers) -> dict:
    """
    Build {original_header: canonical_field} for one workbook.

    Falls back to positional mapping for any header we don't recognise,
    which keeps the loader working if a fifth file shows up with novel
    naming but the same 15-column layout.
    """
    mapping: dict[str, str] = {}
    unmatched_positions: list[int] = []

    for idx, raw in enumerate(headers):
        key = normalise_header(raw)
        if key in HEADER_ALIASES:
            mapping[raw] = HEADER_ALIASES[key]
        else:
            unmatched_positions.append(idx)

    # Positional rescue: fill canonical fields not yet claimed, in order.
    claimed = set(mapping.values())
    spare = [f for f in CANONICAL_FIELDS if f not in claimed]
    for pos, field in zip(unmatched_positions, spare):
        mapping[list(headers)[pos]] = field

    return mapping


def split_multi(value) -> list[str]:
    """Split a multi-value cell on any of the separators used in the data."""
    if value is None:
        return []
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "n/a", "-"}:
        return []
    parts = [p.strip() for p in MULTI_VALUE_SPLIT.split(text)]
    return [p for p in parts if p]
