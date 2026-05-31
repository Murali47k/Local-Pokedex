import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "pokedex_cache.db")


def _get_conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS pokemon_cache "
        "(name TEXT PRIMARY KEY, data TEXT)"
    )
    conn.commit()
    return conn


def get_cached(name: str) -> str | None:
    conn = _get_conn()
    row = conn.execute(
        "SELECT data FROM pokemon_cache WHERE name = ?", (name,)
    ).fetchone()
    conn.close()
    return row[0] if row else None


def set_cached(name: str, data: str):
    conn = _get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO pokemon_cache (name, data) VALUES (?, ?)",
        (name, data),
    )
    conn.commit()
    conn.close()