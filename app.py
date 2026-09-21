#!/usr/bin/env python3
"""
Optional Flask server.

The web app already runs entirely in the browser, so this is not required
to use the project — open web/index.html and everything works. Run this
when you want the recommender as a JSON API for another program, or to
serve the page over http:// instead of file://.

    python app.py            -> http://127.0.0.1:5000

Endpoints
---------
GET  /api/movies?q=&genre=&language=&limit=
GET  /api/movies/<movie_id>
GET  /api/similar/<movie_id>?n=&lambda=
GET  /api/search?q=&n=
POST /api/profile          {"liked": [ids], "disliked": [ids], "n": 12}
GET  /api/compare/<a>/<b>
GET  /api/stats
GET  /api/metadata/<movie_id>[?lite=1]   TMDB poster / backdrop / trailer
GET  /api/health
"""
from __future__ import annotations

import json
import os
import re
import secrets
import threading
import time
import unicodedata
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from flask import Flask, jsonify, request, send_from_directory

from src import config
from src.data_loader import build_clean_frame
from src.export_web import build_stats
from src.features import build_feature_space
from src.recommender import MovieRecommender
from src.account_store import authenticate, create_user, get_preferences, save_preferences, user_by_id



def _load_dotenv() -> None:
    """Tiny .env reader (no extra dependency). Real environment variables win."""
    path = config.ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()

app = Flask(__name__, static_folder=None)
_sessions: dict[str, int] = {}

print("Building model...")
_df = build_clean_frame()
_space = build_feature_space(_df)
_rec = MovieRecommender(_space)
_stats = build_stats(_df)
print(f"Ready: {len(_df)} films, {len(_space.vocabulary)} terms")


def _movie_json(idx: int) -> dict:
    row = _space.df.loc[idx]
    return {
        "movie_id": row["movie_id"],
        "title": row["title"],
        "year": int(row["year"]) if row["year"] == row["year"] else None,
        "runtime": int(row["runtime"]) if row["runtime"] == row["runtime"] else None,
        "rating": float(row["rating"]) if row["rating"] == row["rating"] else None,
        "language": row["language"],
        "country": row["country"],
        "genres": list(row["genres"]),
        "sub_genre": row["sub_genre"],
        "director": row["director"],
        "cast": list(row["cast_list"]),
        "keywords": list(row["keyword_list"]),
        "moods": list(row["moods"]),
        "description": row["description"],
        "age_rating": row["age_rating"],
        "awards": row["awards"],
        "era": row["era"],
    }


def _rec_json(r) -> dict:
    idx = _rec.index_of(r.movie_id)
    payload = _movie_json(idx)
    payload["score"] = r.score
    payload["why"] = r.reasons
    payload["signals"] = r.breakdown
    return payload


def _filters_from_request() -> dict:
    f = {}
    if request.args.get("genre"):
        f["genres"] = request.args.getlist("genre")
    if request.args.get("language"):
        f["languages"] = request.args.getlist("language")
    if request.args.get("mood"):
        f["moods"] = request.args.getlist("mood")
    if request.args.get("min_rating"):
        f["min_rating"] = float(request.args["min_rating"])
    if request.args.get("max_runtime"):
        f["max_runtime"] = int(request.args["max_runtime"])
    return f


def _current_user():
    token = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    user_id = _sessions.get(token)
    return user_by_id(user_id) if user_id else None


# ----------------------------------------------------------------- static
@app.route("/")
def index():
    return send_from_directory(config.WEB_DIR, "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(config.WEB_DIR, filename)


# -------------------------------------------------------------------- api
@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "movies": len(_df), "tmdb_configured": bool(_tmdb_key())})


@app.route("/api/auth/register", methods=["POST"])
def register():
    body = request.get_json(silent=True) or {}
    try:
        user = create_user(str(body.get("username", "")), str(body.get("password", "")))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    token = secrets.token_urlsafe(32)
    _sessions[token] = user["id"]
    return jsonify({"user": user, "token": token, "preferences": get_preferences(user["id"])})


@app.route("/api/auth/login", methods=["POST"])
def login():
    body = request.get_json(silent=True) or {}
    user = authenticate(str(body.get("username", "")), str(body.get("password", "")))
    if not user:
        return jsonify({"error": "invalid username or password"}), 401
    token = secrets.token_urlsafe(32)
    _sessions[token] = user["id"]
    return jsonify({"user": user, "token": token, "preferences": get_preferences(user["id"])})


@app.route("/api/me")
def me():
    user = _current_user()
    if not user:
        return jsonify({"error": "not authenticated"}), 401
    return jsonify({"user": user, "preferences": get_preferences(user["id"])})


@app.route("/api/preferences", methods=["GET", "PUT"])
def preferences():
    user = _current_user()
    if not user:
        return jsonify({"error": "not authenticated"}), 401
    if request.method == "PUT":
        return jsonify(save_preferences(user["id"], request.get_json(silent=True) or {}))
    return jsonify(get_preferences(user["id"]))


# ------------------------------------------------------------------- TMDB
TMDB_BASE = os.getenv("TMDB_API_BASE", "https://api.themoviedb.org/3").rstrip("/")
TMDB_CACHE_PATH = config.ROOT / "data" / "tmdb_cache.json"
_TMDB_HIT_TTL = 30 * 24 * 3600
_TMDB_MISS_TTL = 6 * 3600
_tmdb_lock = threading.Lock()


class TMDBError(Exception):
    """A failure with a message that is safe and useful to show the person."""


def _tmdb_key() -> str:
    return os.getenv("TMDB_API_KEY", "").strip().strip('"').strip("'")


def _load_tmdb_cache() -> dict:
    try:
        return json.loads(TMDB_CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


_tmdb_cache: dict = _load_tmdb_cache()


def _save_tmdb_cache() -> None:
    try:
        TMDB_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        TMDB_CACHE_PATH.write_text(json.dumps(_tmdb_cache), encoding="utf-8")
    except OSError:
        pass  # read-only or ephemeral disk: the in-memory cache still works


def _tmdb_get(path: str, **params) -> dict:
    """GET one TMDB endpoint. Accepts a v3 API key or a v4 read-access token."""
    key = _tmdb_key()
    headers = {"Accept": "application/json", "User-Agent": "movie-recommender/1.1"}
    if key.startswith("eyJ") or len(key) > 40:
        headers["Authorization"] = "Bearer " + key
    else:
        params["api_key"] = key
    params = {k: v for k, v in params.items() if v not in (None, "")}
    url = f"{TMDB_BASE}{path}?{urlencode(params)}"
    try:
        with urlopen(Request(url, headers=headers), timeout=8) as response:
            return json.load(response)
    except HTTPError as exc:
        if exc.code == 401:
            raise TMDBError("TMDB rejected the API key (401) - check TMDB_API_KEY") from exc
        if exc.code == 429:
            raise TMDBError("TMDB rate limit reached (429) - try again shortly") from exc
        raise TMDBError(f"TMDB returned HTTP {exc.code}") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise TMDBError("Could not reach TMDB from the server (network/firewall)") from exc


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text or ""))
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _pick_best(results: list, title: str, year: int | None) -> dict | None:
    """Choose the right TMDB hit; year carries most weight (two 'Drishyam's exist)."""
    want = _norm(title)
    want_tok = set(want.split())
    best, best_score = None, 0.0
    for r in (results or [])[:10]:
        titles = [_norm(r.get("title")), _norm(r.get("original_title"))]
        if want in titles:
            t = 3.0
        elif any(x and (want in x or x in want) for x in titles):
            t = 1.5
        else:
            tt = set(titles[0].split())
            overlap = len(want_tok & tt) / max(len(want_tok), len(tt), 1)
            t = 1.0 if overlap >= 0.6 else 0.0
        if not t:
            continue
        y = 0.0
        try:
            ry = int(str(r.get("release_date") or "")[:4])
        except ValueError:
            ry = 0
        if ry and year:
            d = abs(ry - year)
            y = 3.0 if d == 0 else 1.5 if d == 1 else -0.5 if d <= 3 else -2.0
        votes = min(len(str(int(r.get("vote_count") or 0))), 4) * 0.3
        score = t + y + votes + (0.3 if r.get("poster_path") else 0.0)
        if score > best_score:
            best, best_score = r, score
    return best if best_score >= 2.0 else None


def _pick_trailer(videos: dict | None) -> dict | None:
    items = [v for v in (videos or {}).get("results", []) if v.get("site") == "YouTube" and v.get("key")]
    if not items:
        return None

    def rank(v):
        r = {"Trailer": 0, "Teaser": 2, "Clip": 3}.get(v.get("type"), 4)
        if v.get("type") == "Trailer" and v.get("official"):
            r -= 0.5
        if v.get("iso_639_1") == "en":
            r -= 0.2
        return r

    items.sort(key=lambda v: str(v.get("published_at", "")), reverse=True)  # newest first
    items.sort(key=rank)  # stable: best type first, newest within a type
    v = items[0]
    return {
        "key": v["key"],
        "name": v.get("name") or "Trailer",
        "type": v.get("type") or "",
        "url": "https://www.youtube.com/watch?v=" + v["key"],
    }


def _lookup_tmdb(title: str, year: int | None, full: bool) -> dict:
    """Search TMDB (retrying without the year), optionally fetch the trailer."""
    data = _tmdb_get("/search/movie", query=title, year=year, include_adult="false", language="en-US")
    hit = _pick_best(data.get("results"), title, year)
    if hit is None and year:
        data = _tmdb_get("/search/movie", query=title, include_adult="false", language="en-US")
        hit = _pick_best(data.get("results"), title, year)
    if hit is None:
        return {"matched": False}

    out = {
        "matched": True,
        "tmdb_id": hit["id"],
        "title": hit.get("title"),
        "poster": hit.get("poster_path"),
        "backdrop": hit.get("backdrop_path"),
        "overview": hit.get("overview") or "",
        "release_date": hit.get("release_date") or "",
        "vote_average": hit.get("vote_average"),
        "vote_count": hit.get("vote_count") or 0,
    }
    if full:
        langs = ["en", "null"]
        orig = hit.get("original_language")
        if orig and orig not in langs:
            langs.append(orig)
        detail = _tmdb_get(
            f"/movie/{hit['id']}", append_to_response="videos",
            language="en-US", include_video_language=",".join(langs),
        )
        out["tagline"] = detail.get("tagline") or ""
        out["overview"] = detail.get("overview") or out["overview"]
        out["runtime"] = detail.get("runtime")
        out["poster"] = detail.get("poster_path") or out["poster"]
        out["backdrop"] = detail.get("backdrop_path") or out["backdrop"]
        out["trailer"] = _pick_trailer(detail.get("videos"))
        out["trailer_url"] = out["trailer"]["url"] if out["trailer"] else None
    return out


@app.route("/api/metadata/<movie_id>")
def metadata(movie_id):
    """
    TMDB poster / backdrop / trailer for one film.

    ?lite=1 skips the second request (trailer) - cards use it, the detail
    modal doesn't. The response always says *why* nothing was found:
      configured:false  -> no TMDB_API_KEY on the server
      error: "..."      -> TMDB rejected the key / was unreachable
      matched:false     -> TMDB has no film that matches title + year
    """
    try:
        idx = _rec.index_of(movie_id)
    except KeyError:
        return jsonify({"error": "not found"}), 404
    row = _space.df.loc[idx]
    year = int(row["year"]) if row["year"] == row["year"] else None
    lite = request.args.get("lite") in ("1", "true", "yes")
    base = {"source": "catalogue", "configured": bool(_tmdb_key()), "imdb_rating": row["rating"],
            "release_date": str(year) if year else None, "poster": None, "backdrop": None}

    if not _tmdb_key():
        base["configured"] = False
        return jsonify(base)

    now = time.time()
    with _tmdb_lock:
        cached = _tmdb_cache.get(movie_id)
    if cached and now - cached["ts"] < (_TMDB_HIT_TTL if cached["data"].get("matched") else _TMDB_MISS_TTL):
        data = cached["data"]
        # a lite entry can't answer a full request (no trailer yet)
        if lite or "trailer" in data or not data.get("matched"):
            return jsonify({**base, **data, "source": "tmdb", "cached": True})

    try:
        data = _lookup_tmdb(str(row["title"]), year, full=not lite)
    except TMDBError as exc:
        return jsonify({**base, "error": str(exc)})
    with _tmdb_lock:
        _tmdb_cache[movie_id] = {"ts": now, "data": data}
        _save_tmdb_cache()
    return jsonify({**base, **data, "source": "tmdb"})


@app.route("/api/movies")
def movies():
    limit = int(request.args.get("limit", 50))
    query = request.args.get("q", "").strip()
    filters = _filters_from_request()

    if query:
        results = _rec.search(query, top_n=limit, filters=filters)
        return jsonify([_rec_json(r) for r in results])

    idxs = _rec._apply_filters(
        __import__("numpy").zeros(len(_df)), filters
    )
    idxs = sorted(idxs, key=lambda i: -(_space.df.loc[i, "rating"] or 0))
    return jsonify([_movie_json(i) for i in idxs[:limit]])


@app.route("/api/movies/<movie_id>")
def movie_detail(movie_id):
    try:
        return jsonify(_movie_json(_rec.index_of(movie_id)))
    except KeyError:
        return jsonify({"error": "not found"}), 404


@app.route("/api/similar/<movie_id>")
def similar(movie_id):
    n = int(request.args.get("n", 10))
    lam = float(request.args.get("lambda", config.DEFAULT_MMR_LAMBDA))
    try:
        results = _rec.similar_to(
            movie_id, top_n=n, filters=_filters_from_request(), mmr_lambda=lam
        )
    except KeyError:
        return jsonify({"error": "not found"}), 404
    return jsonify([_rec_json(r) for r in results])


@app.route("/api/search")
def search():
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"error": "q parameter required"}), 400
    n = int(request.args.get("n", 12))
    results = _rec.search(query, top_n=n, filters=_filters_from_request())
    return jsonify([_rec_json(r) for r in results])


@app.route("/api/profile", methods=["POST"])
def profile():
    body = request.get_json(silent=True) or {}
    liked = body.get("liked", [])
    disliked = body.get("disliked", [])
    if not liked:
        return jsonify({"error": "at least one liked film required"}), 400
    try:
        results = _rec.for_profile(
            liked, disliked, top_n=int(body.get("n", 12))
        )
    except KeyError as exc:
        return jsonify({"error": str(exc)}), 404
    return jsonify([_rec_json(r) for r in results])


@app.route("/api/compare/<a>/<b>")
def compare(a, b):
    try:
        ia, ib = _rec.index_of(a), _rec.index_of(b)
    except KeyError:
        return jsonify({"error": "not found"}), 404
    signals = _rec._signal_matrix(ia)
    return jsonify({
        "a": _movie_json(ia),
        "b": _movie_json(ib),
        "signals": {k: round(float(v[ib]), 4) for k, v in signals.items()},
        "shared": _rec._explain(ia, ib, limit=8),
    })


@app.route("/api/stats")
def stats():
    return jsonify(_stats)


if __name__ == "__main__":
    # Render and similar hosting platforms provide PORT dynamically.
    # The local default remains 5000 for Windows and local development.
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)