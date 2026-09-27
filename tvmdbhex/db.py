"""SQLite storage shared by the ingester (writer) and the API (reader)."""
from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone

MEDIA_TYPES = ("movie", "tv")

# status values
PENDING = "pending"  # known title, not processed yet (or re-queued by a change)
DONE = "done"  # colours extracted
NO_POSTER = "no_poster"  # TMDB has no poster for this title
NOT_FOUND = "not_found"  # title removed from TMDB
ERROR = "error"  # transient failure, retried on the next run

SCHEMA = """
CREATE TABLE IF NOT EXISTS titles (
    media_type      TEXT    NOT NULL CHECK (media_type IN ('movie', 'tv')),
    tmdb_id         INTEGER NOT NULL,
    title           TEXT,
    adult           INTEGER NOT NULL DEFAULT 0,
    popularity      REAL,
    poster_path     TEXT,
    primary_hex     TEXT,
    primary_ratio   REAL,
    secondary_hex   TEXT,
    secondary_ratio REAL,
    tertiary_hex    TEXT,
    tertiary_ratio  REAL,
    status          TEXT    NOT NULL DEFAULT 'pending',
    error           TEXT,
    attempts        INTEGER NOT NULL DEFAULT 0,
    updated_at      TEXT    NOT NULL,
    PRIMARY KEY (media_type, tmdb_id)
);
CREATE INDEX IF NOT EXISTS idx_titles_status ON titles (status, media_type);
CREATE INDEX IF NOT EXISTS idx_titles_poster ON titles (poster_path);
CREATE INDEX IF NOT EXISTS idx_titles_popularity ON titles (status, popularity DESC);

CREATE TABLE IF NOT EXISTS sync_state (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

TITLE_COLUMNS = (
    "media_type, tmdb_id, title, poster_path, primary_hex, primary_ratio, "
    "secondary_hex, secondary_ratio, tertiary_hex, tertiary_ratio, status, updated_at"
)


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def connect(path: str, readonly: bool = False) -> sqlite3.Connection:
    if path != ":memory:" and not readonly:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if readonly:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
    else:
        conn = sqlite3.connect(path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.executescript(SCHEMA)
    conn.execute("PRAGMA busy_timeout=10000")
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    try:
        yield conn
        conn.commit()
    except BaseException:
        conn.rollback()
        raise


def upsert_seed(conn: sqlite3.Connection, media_type: str, rows: Iterable[dict]) -> int:
    """Insert titles from a TMDB export. Existing rows keep their colours."""
    ts = now_iso()
    count = 0
    with transaction(conn):
        for row in rows:
            conn.execute(
                """
                INSERT INTO titles (media_type, tmdb_id, title, adult, popularity, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT (media_type, tmdb_id) DO UPDATE SET
                    title = excluded.title,
                    adult = excluded.adult,
                    popularity = excluded.popularity
                """,
                (
                    media_type,
                    row["id"],
                    row.get("title"),
                    int(bool(row.get("adult"))),
                    row.get("popularity"),
                    ts,
                ),
            )
            count += 1
    return count


def requeue(conn: sqlite3.Connection, media_type: str, ids: Iterable[int]) -> int:
    """Mark titles as pending (inserting unknown ones) so they get re-processed."""
    ts = now_iso()
    count = 0
    with transaction(conn):
        for tmdb_id in ids:
            conn.execute(
                """
                INSERT INTO titles (media_type, tmdb_id, status, updated_at)
                VALUES (?, ?, 'pending', ?)
                ON CONFLICT (media_type, tmdb_id) DO UPDATE SET status = 'pending'
                """,
                (media_type, int(tmdb_id), ts),
            )
            count += 1
    return count


def pending(
    conn: sqlite3.Connection,
    media_type: str | None,
    include_adult: bool,
    retry_errors: bool,
    max_attempts: int,
    limit: int | None,
) -> list[tuple[str, int]]:
    statuses = [PENDING] + ([ERROR] if retry_errors else [])
    sql = f"SELECT media_type, tmdb_id FROM titles WHERE status IN ({','.join('?' * len(statuses))})"
    params: list = list(statuses)
    sql += " AND attempts < ?"
    params.append(max_attempts)
    if media_type:
        sql += " AND media_type = ?"
        params.append(media_type)
    if not include_adult:
        sql += " AND adult = 0"
    sql += " ORDER BY popularity IS NULL, popularity DESC, tmdb_id"
    if limit:
        sql += " LIMIT ?"
        params.append(limit)
    return [(r[0], r[1]) for r in conn.execute(sql, params)]


def find_by_poster(conn: sqlite3.Connection, poster_path: str) -> sqlite3.Row | None:
    """Reuse colours when another title already has the exact same poster file."""
    return conn.execute(
        "SELECT * FROM titles WHERE poster_path = ? AND status = 'done' LIMIT 1",
        (poster_path,),
    ).fetchone()


def save_result(
    conn: sqlite3.Connection,
    media_type: str,
    tmdb_id: int,
    *,
    status: str,
    title: str | None = None,
    adult: bool | None = None,
    poster_path: str | None = None,
    palette: list[tuple[str, float]] | None = None,
    error: str | None = None,
) -> None:
    colours: list = [None] * 6
    if palette:
        colours = [v for swatch in palette for v in swatch]
    with transaction(conn):
        conn.execute(
            """
            INSERT INTO titles (media_type, tmdb_id, updated_at) VALUES (?, ?, ?)
            ON CONFLICT (media_type, tmdb_id) DO NOTHING
            """,
            (media_type, tmdb_id, now_iso()),
        )
        conn.execute(
            """
            UPDATE titles SET
                title = COALESCE(?, title),
                adult = COALESCE(?, adult),
                poster_path = ?,
                primary_hex = ?, primary_ratio = ?,
                secondary_hex = ?, secondary_ratio = ?,
                tertiary_hex = ?, tertiary_ratio = ?,
                status = ?,
                error = ?,
                attempts = CASE WHEN ? = 'error' THEN attempts + 1 ELSE 0 END,
                updated_at = ?
            WHERE media_type = ? AND tmdb_id = ?
            """,
            (
                title,
                None if adult is None else int(adult),
                poster_path,
                *colours,
                status,
                error,
                status,
                now_iso(),
                media_type,
                tmdb_id,
            ),
        )


def search(
    conn: sqlite3.Connection,
    query: str = "",
    media_type: str | None = None,
    status: str | None = None,
    limit: int = 24,
    offset: int = 0,
) -> list[sqlite3.Row]:
    """Titles matching `query` (title substring or TMDB ID), most popular first."""
    where, params = [], []
    order = "popularity IS NULL, popularity DESC, tmdb_id"
    query = query.strip()
    if query:
        pattern = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        clause = "title LIKE ? ESCAPE '\\'"
        params.append(f"%{pattern}%")
        if query.isdigit():
            clause = f"({clause} OR tmdb_id = ?)"
            params.append(int(query))
        where.append(clause)
        # Exact matches first, then titles starting with the query, then the rest.
        order = "(title = ? COLLATE NOCASE) DESC, (title LIKE ? ESCAPE '\\') DESC, " + order
    if media_type:
        where.append("media_type = ?")
        params.append(media_type)
    if status:
        where.append("status = ?")
        params.append(status)
    sql = f"SELECT {TITLE_COLUMNS} FROM titles"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += f" ORDER BY {order} LIMIT ? OFFSET ?"
    if query:
        params += [query, f"{pattern}%"]
    params += [limit, offset]
    return conn.execute(sql, params).fetchall()


def get_state(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM sync_state WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None


def set_state(conn: sqlite3.Connection, key: str, value: str) -> None:
    with transaction(conn):
        conn.execute(
            "INSERT INTO sync_state (key, value) VALUES (?, ?) "
            "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
            (key, value),
        )


def stats(conn: sqlite3.Connection) -> dict:
    out: dict = {m: {} for m in MEDIA_TYPES}
    for row in conn.execute(
        "SELECT media_type, status, COUNT(*) FROM titles GROUP BY media_type, status"
    ):
        out[row[0]][row[1]] = row[2]
    return out
