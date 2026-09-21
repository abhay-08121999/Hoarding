"""Small SQLite-backed account and preference store for the demo app."""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import sqlite3
from pathlib import Path

DB_PATH = Path(os.getenv("MOVIE_RECOMMENDER_DB", Path(__file__).resolve().parent.parent / "data" / "accounts.sqlite3"))


def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript("""
      CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL UNIQUE,
        password_hash TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
      );
      CREATE TABLE IF NOT EXISTS preferences (
        user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
        watchlist TEXT NOT NULL DEFAULT '[]',
        liked TEXT NOT NULL DEFAULT '[]',
        disliked TEXT NOT NULL DEFAULT '[]',
        recent TEXT NOT NULL DEFAULT '[]',
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
      );
    """)
    return conn


def _hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 180_000)
    return salt.hex() + ":" + digest.hex()


def _check_password(password: str, encoded: str) -> bool:
    try:
        salt_hex, digest_hex = encoded.split(":", 1)
        actual = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex), 180_000)
        return hmac.compare_digest(actual.hex(), digest_hex)
    except (ValueError, TypeError):
        return False


def create_user(username: str, password: str) -> dict:
    username = username.strip().lower()
    if len(username) < 3 or len(username) > 40:
        raise ValueError("username must be 3–40 characters")
    if len(password) < 6:
        raise ValueError("password must be at least 6 characters")
    with _connect() as conn:
        try:
            cur = conn.execute("INSERT INTO users(username, password_hash) VALUES (?, ?)", (username, _hash_password(password)))
        except sqlite3.IntegrityError as exc:
            raise ValueError("username already exists") from exc
        user_id = cur.lastrowid
        conn.execute("INSERT INTO preferences(user_id) VALUES (?)", (user_id,))
        return {"id": user_id, "username": username}


def authenticate(username: str, password: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute("SELECT id, username, password_hash FROM users WHERE username = ?", (username.strip().lower(),)).fetchone()
    if not row or not _check_password(password, row["password_hash"]):
        return None
    return {"id": row["id"], "username": row["username"]}


def get_preferences(user_id: int) -> dict:
    with _connect() as conn:
        row = conn.execute("SELECT watchlist, liked, disliked, recent FROM preferences WHERE user_id = ?", (user_id,)).fetchone()
    if not row:
        return {"watchlist": [], "liked": [], "disliked": [], "recent": []}
    return {key: json.loads(row[key] or "[]") for key in ("watchlist", "liked", "disliked", "recent")}


def save_preferences(user_id: int, payload: dict) -> dict:
    clean = {}
    for key in ("watchlist", "liked", "disliked", "recent"):
        values = payload.get(key, [])
        clean[key] = list(dict.fromkeys(str(x) for x in values if x))[:500]
    with _connect() as conn:
        conn.execute("""INSERT INTO preferences(user_id, watchlist, liked, disliked, recent)
          VALUES (?, ?, ?, ?, ?)
          ON CONFLICT(user_id) DO UPDATE SET watchlist=excluded.watchlist,
          liked=excluded.liked, disliked=excluded.disliked, recent=excluded.recent,
          updated_at=CURRENT_TIMESTAMP""", (user_id, *(json.dumps(clean[k]) for k in ("watchlist", "liked", "disliked", "recent"))))
    return clean


def user_by_id(user_id: int) -> dict | None:
    with _connect() as conn:
        row = conn.execute("SELECT id, username FROM users WHERE id = ?", (user_id,)).fetchone()
    return dict(row) if row else None


# Initialize lazily on import so the app always has its schema.
_connect().close()

__all__ = ["create_user", "authenticate", "get_preferences", "save_preferences", "user_by_id"]
