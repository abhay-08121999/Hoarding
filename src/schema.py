"""
Small TMDB HTTP client shared by app.py (live lookups) and
tools/import_tmdb.py (bulk catalogue import).

Accepts either a v3 "API Key" or a v4 "API Read Access Token", retries on
rate limits (429) and transient 5xx, and raises TMDBError with a message that
is safe and useful to show a person.
"""
from __future__ import annotations

import json
import os
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from . import config


class TMDBError(Exception):
    """A failure with a message that is safe and useful to show the person."""


def load_dotenv() -> None:
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


def api_key() -> str:
    return os.getenv("TMDB_API_KEY", "").strip().strip('"').strip("'")


def base_url() -> str:
    return os.getenv("TMDB_API_BASE", "https://api.themoviedb.org/3").rstrip("/")


def get(path: str, retries: int = 3, timeout: int = 10, **params) -> dict:
    """GET one TMDB endpoint and return the decoded JSON."""
    key = api_key()
    headers = {"Accept": "application/json", "User-Agent": "movie-recommender/1.2"}
    if key.startswith("eyJ") or len(key) > 40:
        headers["Authorization"] = "Bearer " + key
    else:
        params["api_key"] = key
    params = {k: v for k, v in params.items() if v not in (None, "")}
    url = f"{base_url()}{path}?{urlencode(params)}"

    for attempt in range(retries + 1):
        try:
            with urlopen(Request(url, headers=headers), timeout=timeout) as response:
                return json.load(response)
        except HTTPError as exc:
            if exc.code == 401:
                raise TMDBError("TMDB rejected the API key (401) - check TMDB_API_KEY") from exc
            if exc.code == 404:
                raise TMDBError("TMDB has no such record (404)") from exc
            if exc.code in (429, 500, 502, 503, 504) and attempt < retries:
                wait = float(exc.headers.get("Retry-After", 0) or 0) or 1.5 * (attempt + 1)
                time.sleep(min(wait, 10))
                continue
            if exc.code == 429:
                raise TMDBError("TMDB rate limit reached (429) - try again shortly") from exc
            raise TMDBError(f"TMDB returned HTTP {exc.code}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            if attempt < retries:
                time.sleep(1.0 * (attempt + 1))
                continue
            raise TMDBError("Could not reach TMDB (network/firewall)") from exc
    raise TMDBError("TMDB request failed")
