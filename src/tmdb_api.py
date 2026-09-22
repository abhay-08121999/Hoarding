#!/usr/bin/env python3
"""
Grow the catalogue from TMDB.

    python tools/import_tmdb.py                     # medium: ~3-5 thousand films
    python tools/import_tmdb.py --size small        # ~1.5 thousand, quick trial
    python tools/import_tmdb.py --size large        # ~8 thousand
    python tools/import_tmdb.py --only korean,anime # just some collections
    python build.py                                 # then rebuild the site data

What it does
------------
1. Runs TMDB "discover" queries for each collection below (Indian cinema,
   regional-language, Korean, European, anime, documentaries, independent),
   most-voted first, so each collection gets its best-known films.
2. Fetches one detail record per film (cast, director, keywords, age
   certificate) and maps it onto the workbook schema the pipeline reads.
3. Writes data/raw/tmdb_import.xlsx. build.py picks it up automatically next
   to the hand-built workbooks; films already in those are merged, not
   duplicated, and keep their ids.

Results are cached in data/cache/tmdb_rows.json, so an interrupted run resumes
and a re-run only fetches what is new.

Needs a TMDB key (TMDB_API_KEY in the environment or .env). Ratings are TMDB
user ratings (0-10), which run a little lower than IMDb's; awards are not
available from TMDB and are left blank.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import tmdb_api  # noqa: E402
from src.tmdb_api import TMDBError  # noqa: E402

RAW_OUT = ROOT / "data" / "raw" / "tmdb_import.xlsx"
CACHE = ROOT / "data" / "cache" / "tmdb_rows.json"

# ------------------------------------------------------------------ mappings
LANGUAGES = {
    "en": "English", "hi": "Hindi", "ta": "Tamil", "te": "Telugu", "ml": "Malayalam",
    "kn": "Kannada", "bn": "Bengali", "mr": "Marathi", "pa": "Punjabi", "gu": "Gujarati",
    "ur": "Urdu", "or": "Odia", "as": "Assamese", "bh": "Bhojpuri", "ko": "Korean",
    "ja": "Japanese", "zh": "Mandarin", "cn": "Cantonese", "fr": "French", "de": "German",
    "it": "Italian", "es": "Spanish", "pt": "Portuguese", "sv": "Swedish", "da": "Danish",
    "no": "Norwegian", "nb": "Norwegian", "fi": "Finnish", "pl": "Polish", "ru": "Russian",
    "nl": "Dutch", "cs": "Czech", "hu": "Hungarian", "el": "Greek", "ro": "Romanian",
    "is": "Icelandic", "tr": "Turkish", "uk": "Ukrainian", "sr": "Serbian", "hr": "Croatian",
    "bg": "Bulgarian", "sk": "Slovak", "th": "Thai", "vi": "Vietnamese", "id": "Indonesian",
    "fa": "Persian", "ar": "Arabic", "he": "Hebrew", "sl": "Slovenian", "et": "Estonian",
    "lv": "Latvian", "lt": "Lithuanian", "ca": "Catalan",
}
COUNTRY_NAMES = {  # TMDB name -> the short form the hand-built workbooks use
    "United States of America": "USA", "United Kingdom": "UK",
    "Czechia": "Czech Republic", "Russian Federation": "Russia",
    "Korea": "South Korea", "South Korea": "South Korea",
}
GENRE_NAMES = {"Science Fiction": "Sci-Fi"}
DROP_GENRES = {"TV Movie"}
IN_CERTS = {"U": "U", "UA": "U/A", "U/A": "U/A", "A": "A", "S": "A", "UA 7+": "U/A",
            "UA 13+": "UA 13+", "UA 16+": "UA 16+"}
US_CERTS = {"G": "G", "PG": "PG", "PG-13": "PG-13", "R": "R", "NC-17": "NC-17"}

# ---------------------------------------------------------------- collections
# name -> list of discover queries. `weight` scales the per-query film target.
EUROPE = ["FR", "DE", "IT", "ES", "SE", "DK", "NO", "FI", "PL", "RU", "NL", "CZ",
          "HU", "GR", "RO", "PT", "AT", "BE", "IE", "IS"]
COLLECTIONS = {
    "hindi": [dict(with_original_language="hi", weight=4, min_votes=100)],
    "regional": [dict(with_original_language=l, weight=w, min_votes=20)
                 for l, w in (("ta", 2), ("te", 2), ("ml", 2), ("kn", 1), ("bn", 1),
                              ("mr", 1), ("pa", 1), ("gu", 0.5))],
    "korean": [dict(with_original_language="ko", weight=3, min_votes=100)],
    "european": [dict(with_origin_country=c, weight=1, min_votes=150) for c in EUROPE]
                + [dict(with_origin_country="GB", weight=2, min_votes=400)],
    "anime": [dict(with_original_language="ja", with_genres="16", weight=4, min_votes=100)],
    "documentaries": [dict(with_genres="99", weight=4, min_votes=60)],
    "independent": [dict(keyword="independent film", weight=4, min_votes=100),
                    dict(keyword="sundance", weight=1, min_votes=60)],
}
INDEPENDENT_TAG = {"independent"}
SIZES = {"small": 25, "medium": 60, "large": 120}  # films per weight-1 query


# ------------------------------------------------------------ pure functions
def year_of(detail: dict) -> int | None:
    try:
        return int(str(detail.get("release_date") or "")[:4])
    except ValueError:
        return None


def pick_certificate(detail: dict, country: str) -> str:
    """Age certificate: India's board for Indian films, otherwise the US one."""
    results = (detail.get("release_dates") or {}).get("results") or []
    by_country = {}
    for entry in results:
        for rel in entry.get("release_dates", []):
            cert = (rel.get("certification") or "").strip()
            if cert:
                by_country.setdefault(entry.get("iso_3166_1"), cert)
    if country == "India" and by_country.get("IN") in IN_CERTS:
        return IN_CERTS[by_country["IN"]]
    return US_CERTS.get(by_country.get("US", ""), IN_CERTS.get(by_country.get("IN", ""), "Not rated"))


def to_row(detail: dict, tags=(), min_votes: int = 0) -> dict | None:
    """Map one TMDB movie record onto the workbook schema. None = skip it."""
    year = year_of(detail)
    overview = (detail.get("overview") or "").strip()
    votes = detail.get("vote_count") or 0
    rating = round(float(detail.get("vote_average") or 0), 1)
    title = (detail.get("title") or "").strip()
    if detail.get("adult") or not title or not year or len(overview) < 25:
        return None
    if votes < min_votes or rating <= 0:
        return None
    genres = [GENRE_NAMES.get(g["name"], g["name"]) for g in detail.get("genres", [])
              if g.get("name") not in DROP_GENRES]
    runtime = detail.get("runtime") or 0
    if runtime and runtime < 40 and "Documentary" not in genres:
        return None  # shorts

    lang_code = detail.get("original_language") or ""
    language = LANGUAGES.get(lang_code)
    if not language:
        spoken = {s.get("iso_639_1"): s.get("english_name") for s in detail.get("spoken_languages", [])}
        language = spoken.get(lang_code) or lang_code.upper() or "Unknown"

    countries = [COUNTRY_NAMES.get(c["name"], c["name"]) for c in detail.get("production_countries", [])]
    country = countries[0] if countries else ""

    credits = detail.get("credits") or {}
    directors = [c["name"] for c in credits.get("crew", []) if c.get("job") == "Director"][:2]
    cast = [c["name"] for c in sorted(credits.get("cast", []), key=lambda c: c.get("order", 999))][:6]
    keywords = [k["name"] for k in (detail.get("keywords") or {}).get("keywords", [])][:10]

    return {
        "Movie ID": f"tmdb-{detail['id']}",
        "Title": title,
        "Genres": "|".join(genres),
        "Sub-Genre": "",
        "Language": language,
        "Release Year": year,
        "Runtime (min)": runtime or None,
        "Rating": rating,
        "Director": "|".join(directors),
        "Cast": "|".join(cast),
        "Country": country,
        "Description": overview,
        "Keywords": "|".join(keywords),
        "Age Rating": pick_certificate(detail, country),
        "Awards": "",
        "TMDB ID": detail["id"],
        "Poster Path": detail.get("poster_path") or "",
        "Collections": "|".join(sorted(tags)),
    }


# --------------------------------------------------------------- network work
def discover_ids(query: dict, target: int, today: str) -> list:
    """Film ids for one discover query, most-voted first, up to `target`."""
    params = dict(sort_by="vote_count.desc", include_adult="false", language="en-US",
                  **{"vote_count.gte": query["min_votes"], "primary_release_date.lte": today,
                     "with_runtime.gte": 40})
    for key in ("with_original_language", "with_origin_country", "with_genres"):
        if key in query:
            params[key] = query[key]
    if "keyword" in query:
        found = tmdb_api.get("/search/keyword", query=query["keyword"]).get("results", [])
        exact = [k for k in found if k["name"].lower() == query["keyword"].lower()]
        if not exact:
            print(f"     ! keyword '{query['keyword']}' not found on TMDB, skipping")
            return []
        params["with_keywords"] = exact[0]["id"]

    ids, page = [], 1
    while len(ids) < target:
        data = tmdb_api.get("/discover/movie", page=page, **params)
        ids += [m["id"] for m in data.get("results", [])]
        if page >= min(data.get("total_pages", 1), 500) or not data.get("results"):
            break
        page += 1
    return ids[:target]


def fetch_row(film_id: int, tags: set, min_votes: int) -> dict | None:
    detail = tmdb_api.get(f"/movie/{film_id}", append_to_response="credits,keywords,release_dates",
                          language="en-US")
    return to_row(detail, tags, min_votes)


def load_cache() -> dict:
    try:
        return json.loads(CACHE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_cache(cache: dict) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


def write_workbook(rows: list) -> None:
    import pandas as pd
    RAW_OUT.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0].keys())
    pd.DataFrame(rows, columns=columns).to_excel(RAW_OUT, index=False)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Import films from TMDB")
    ap.add_argument("--size", choices=SIZES, default="medium")
    ap.add_argument("--only", help="comma list of: " + ", ".join(COLLECTIONS))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-films", type=int, help="stop after this many (for trials)")
    args = ap.parse_args(argv)

    tmdb_api.load_dotenv()
    if not tmdb_api.api_key():
        print("No TMDB key found. Set TMDB_API_KEY in the environment or .env "
              "(free key: https://www.themoviedb.org/settings/api).")
        return 2

    names = [n.strip() for n in args.only.split(",")] if args.only else list(COLLECTIONS)
    unknown = [n for n in names if n not in COLLECTIONS]
    if unknown:
        print("Unknown collection(s):", ", ".join(unknown))
        return 2

    base = SIZES[args.size]
    today = date.today().isoformat()
    wanted: dict = {}  # film id -> (tags, min_votes)
    try:
        print(f"1/3  Discovering films ({args.size}: ~{base} per query unit)")
        for name in names:
            found = 0
            for q in COLLECTIONS[name]:
                try:
                    ids = discover_ids(q, max(5, int(base * q["weight"])), today)
                except TMDBError as exc:
                    if "rejected the API key" in str(exc):
                        raise
                    print(f"     ! {name} query {q.get('with_original_language') or q.get('with_origin_country') or q.get('keyword') or ''}: {exc} (skipped)")
                    continue
                for fid in ids:
                    tags, mv = wanted.setdefault(fid, (set(), q["min_votes"]))
                    if name in INDEPENDENT_TAG:
                        tags.add("Independent")
                    wanted[fid] = (tags, min(mv, q["min_votes"]))
                found += len(ids)
            print(f"     {name:14s} {found:5d} candidates")
    except TMDBError as exc:
        print("Stopped:", exc)
        return 1

    order = list(wanted)[: args.max_films] if args.max_films else list(wanted)
    cache = load_cache()
    todo = [f for f in order if str(f) not in cache]
    print(f"2/3  Fetching details: {len(order)} films, {len(order) - len(todo)} already cached")

    lock, done, failed = threading.Lock(), 0, 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_row, f, wanted[f][0], wanted[f][1]): f for f in todo}
        for fut in as_completed(futures):
            fid = futures[fut]
            try:
                row = fut.result()
            except TMDBError as exc:
                failed += 1
                if failed <= 3:
                    print(f"     ! {fid}: {exc}")
                if "rejected the API key" in str(exc):
                    print("Stopped: the API key was rejected.")
                    return 1
                continue
            with lock:
                cache[str(fid)] = row  # None is cached too: it means "skip"
                done += 1
                if done % 200 == 0:
                    save_cache(cache)
                    print(f"     {done}/{len(todo)}")
    save_cache(cache)

    rows = [cache[str(f)] for f in order if cache.get(str(f))]
    if not rows:
        print("Nothing to write.")
        return 1
    print(f"3/3  Writing {len(rows)} films to {RAW_OUT.relative_to(ROOT)} "
          f"({len(order) - len(rows)} skipped: no synopsis, too few votes, shorts...)")
    write_workbook(rows)
    print("Done. Now run:  python build.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
