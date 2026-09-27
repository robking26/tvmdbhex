"""Storage shared by the ingester (writer) and the API (reader).

Two backends with one dialect of SQL (`?` placeholders):
- SQLite for local use and tests (`TVMDBHEX_DB_PATH`)
- Postgres for hosted deployments such as Vercel (`DATABASE_URL`)
"""
from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from itertools import islice
from typing import Any

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


def is_postgres_url(target: str) -> bool:
    return target.startswith(("postgres://", "postgresql://"))


class SQLiteConnection(sqlite3.Connection):
    dialect = "sqlite"


class Row(tuple):
    """Result row addressable by column name or position (like sqlite3.Row)."""

    _index: dict[str, int]

    def __new__(cls, values: Iterable[Any], index: dict[str, int]) -> "Row":
        row = super().__new__(cls, values)
        row._index = index
        return row

    def __getitem__(self, key):  # type: ignore[override]
        if isinstance(key, str):
            key = self._index[key]
        return tuple.__getitem__(self, key)

    def keys(self) -> list[str]:
        return list(self._index)


class PostgresConnection:
    """Thin psycopg wrapper exposing the sqlite3 connection methods we use."""

    dialect = "postgres"

    def __init__(self, url: str, autocommit: bool = False):
        import psycopg  # only needed for hosted deployments

        def row_factory(cursor):
            index = {d.name: i for i, d in enumerate(cursor.description or [])}
            return lambda values: Row(values, index)

        # prepare_threshold=None: no server-side prepared statements, which can break
        # behind transaction-mode poolers such as Neon's pooled DATABASE_URL.
        self._conn = psycopg.connect(
            url, autocommit=autocommit, row_factory=row_factory, prepare_threshold=None
        )

    @staticmethod
    def _sql(sql: str) -> str:
        return sql.replace("?", "%s")

    def execute(self, sql: str, params: Iterable[Any] = ()):
        return self._conn.execute(self._sql(sql), tuple(params))

    def executemany(self, sql: str, seq: Iterable[Iterable[Any]]) -> None:
        with self._conn.cursor() as cur:
            cur.executemany(self._sql(sql), [tuple(p) for p in seq])

    def executescript(self, script: str) -> None:
        for statement in script.split(";"):
            if statement.strip():
                self._conn.execute(statement)
        if not self._conn.autocommit:
            self._conn.commit()

    def commit(self) -> None:
        self._conn.commit()

    def rollback(self) -> None:
        self._conn.rollback()

    def close(self) -> None:
        self._conn.close()

    @property
    def closed(self) -> bool:
        return self._conn.closed or self._conn.broken


Connection = Any  # SQLiteConnection | PostgresConnection

PG_SCHEMA = SCHEMA.replace(" REAL", " DOUBLE PRECISION")


def connect(target: str, readonly: bool = False) -> Connection:
    """Open the database at `target` (a SQLite path or a postgres:// URL).

    `readonly` connections (used by the API) don't create the schema.
    """
    if is_postgres_url(target):
        conn = PostgresConnection(target, autocommit=readonly)
        if not readonly:
            conn.executescript(PG_SCHEMA)
        return conn

    if target != ":memory:" and not readonly:
        os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
    if readonly:
        conn = sqlite3.connect(
            f"file:{target}?mode=ro", uri=True, check_same_thread=False, factory=SQLiteConnection
        )
    else:
        conn = sqlite3.connect(target, check_same_thread=False, factory=SQLiteConnection)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.executescript(SCHEMA)
    conn.execute("PRAGMA busy_timeout=10000")
    conn.row_factory = sqlite3.Row
    return conn


def ensure_schema(conn: Connection) -> None:
    conn.executescript(PG_SCHEMA if conn.dialect == "postgres" else SCHEMA)


def _batches(items: Iterable, size: int = 5000) -> Iterator[list]:
    it = iter(items)
    while batch := list(islice(it, size)):
        yield batch


@contextmanager
def transaction(conn: Connection) -> Iterator[Connection]:
    try:
        yield conn
        conn.commit()
    except BaseException:
        conn.rollback()
        raise


def upsert_seed(conn: Connection, media_type: str, rows: Iterable[dict]) -> int:
    """Insert titles from a TMDB export. Existing rows keep their colours."""
    ts = now_iso()
    count = 0
    for batch in _batches(rows):
        with transaction(conn):
            conn.executemany(
                """
                INSERT INTO titles (media_type, tmdb_id, title, adult, popularity, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT (media_type, tmdb_id) DO UPDATE SET
                    title = COALESCE(titles.title, excluded.title),
                    adult = excluded.adult,
                    popularity = excluded.popularity
                """,
                [
                    (
                        media_type,
                        row["id"],
                        row.get("title"),
                        int(bool(row.get("adult"))),
                        row.get("popularity"),
                        ts,
                    )
                    for row in batch
                ],
            )
        count += len(batch)
    return count


def requeue(conn: Connection, media_type: str, ids: Iterable[int]) -> int:
    """Mark titles as pending (inserting unknown ones) so they get re-processed."""
    ts = now_iso()
    count = 0
    for batch in _batches(ids):
        with transaction(conn):
            conn.executemany(
                """
                INSERT INTO titles (media_type, tmdb_id, status, updated_at)
                VALUES (?, ?, 'pending', ?)
                ON CONFLICT (media_type, tmdb_id) DO UPDATE SET status = 'pending', attempts = 0
                """,
                [(media_type, int(tmdb_id), ts) for tmdb_id in batch],
            )
        count += len(batch)
    return count


def pending(
    conn: Connection,
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


def find_by_poster(conn: Connection, poster_path: str) -> Row | None:
    """Reuse colours when another title already has the exact same poster file."""
    return conn.execute(
        "SELECT * FROM titles WHERE poster_path = ? AND status = 'done' LIMIT 1",
        (poster_path,),
    ).fetchone()


def save_result(
    conn: Connection,
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
                attempts = CASE WHEN ? = 1 THEN attempts + 1 ELSE 0 END,
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
                int(status == ERROR),
                now_iso(),
                media_type,
                tmdb_id,
            ),
        )


def search(
    conn: Connection,
    query: str = "",
    media_type: str | None = None,
    status: str | None = None,
    limit: int = 24,
    offset: int = 0,
) -> list[Row]:
    """Titles matching `query` (title substring or TMDB ID), most popular first."""
    where, params = [], []
    order = "popularity IS NULL, popularity DESC, tmdb_id"
    query = query.strip()
    like = "ILIKE" if conn.dialect == "postgres" else "LIKE"  # case-insensitive in both
    if query:
        pattern = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        clause = f"title {like} ? ESCAPE '\\'"
        params.append(f"%{pattern}%")
        if query.isdigit():
            clause = f"({clause} OR tmdb_id = ?)"
            params.append(int(query))
        where.append(clause)
        # Exact matches first, then titles starting with the query, then the rest.
        order = f"(lower(title) = lower(?)) DESC, (title {like} ? ESCAPE '\\') DESC, " + order
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


def get_state(conn: Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM sync_state WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None


def set_state(conn: Connection, key: str, value: str) -> None:
    with transaction(conn):
        conn.execute(
            "INSERT INTO sync_state (key, value) VALUES (?, ?) "
            "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
            (key, value),
        )


def stats(conn: Connection) -> dict:
    out: dict = {m: {} for m in MEDIA_TYPES}
    for row in conn.execute(
        "SELECT media_type, status, COUNT(*) FROM titles GROUP BY media_type, status"
    ):
        out[row[0]][row[1]] = row[2]
    return out
