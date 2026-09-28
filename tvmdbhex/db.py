"""Storage shared by the ingester (writer) and the API (reader).

Two backends with one dialect of SQL (`?` placeholders):
- SQLite for local use and tests (`TVMDBHEX_DB_PATH`)
- Postgres for hosted deployments such as Vercel (`DATABASE_URL`)
"""
from __future__ import annotations

import logging
import os
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from itertools import islice
from typing import Any

log = logging.getLogger("tvmdbhex.db")

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
    palette_version INTEGER,
    base_hex        TEXT,
    identity1_hex   TEXT,
    identity2_hex   TEXT,
    highlight1_hex  TEXT,
    highlight2_hex  TEXT,
    accent_hex      TEXT,
    logo_path       TEXT,
    updated_at      TEXT    NOT NULL,
    PRIMARY KEY (media_type, tmdb_id)
);
CREATE INDEX IF NOT EXISTS idx_titles_popularity ON titles (status, popularity DESC);
CREATE INDEX IF NOT EXISTS idx_titles_type_popularity ON titles (status, media_type, popularity DESC);

CREATE TABLE IF NOT EXISTS sync_state (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

SEMANTIC_ROLES = ("base", "identity1", "identity2", "highlight1", "highlight2", "accent")
ADDED_COLUMNS = ("palette_version INTEGER",) + tuple(f"{r}_hex TEXT" for r in SEMANTIC_ROLES) + ("logo_path TEXT",)

TITLE_COLUMNS = (
    "media_type, tmdb_id, title, poster_path, primary_hex, primary_ratio, "
    "secondary_hex, secondary_ratio, tertiary_hex, tertiary_ratio, status, updated_at, "
    "palette_version, logo_path, " + ", ".join(f"{r}_hex" for r in SEMANTIC_ROLES)
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
            _migrate(conn)
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
        _migrate(conn)
    conn.execute("PRAGMA busy_timeout=10000")
    conn.row_factory = sqlite3.Row
    return conn


def ensure_schema(conn: Connection) -> None:
    conn.executescript(PG_SCHEMA if conn.dialect == "postgres" else SCHEMA)
    _migrate(conn)


def _migrate(conn: Connection) -> None:
    """Add columns introduced after a database was created (metadata-only, cheap)."""
    if conn.dialect == "postgres":
        # Only ALTER/DROP when something is actually missing: both take an
        # exclusive lock even when there's nothing to do, which would block
        # (or be blocked by) the API and ingester sharing the table.
        existing = {
            r[0]
            for r in conn.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name = 'titles'"
            )
        }
        missing = [col for col in ADDED_COLUMNS if col.split()[0] not in existing]
        for col in missing:
            conn.execute(f"ALTER TABLE titles ADD COLUMN IF NOT EXISTS {col}")
        stale = conn.execute("SELECT 1 FROM pg_indexes WHERE indexname = 'idx_titles_poster'").fetchone()
        if stale:
            conn.execute("DROP INDEX IF EXISTS idx_titles_poster")  # no longer used
        conn.commit()
        return
    existing = {r[1] for r in conn.execute("PRAGMA table_info(titles)")}
    missing = [col for col in ADDED_COLUMNS if col.split()[0] not in existing]
    for col in missing:
        conn.execute(f"ALTER TABLE titles ADD COLUMN {col}")
    if conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'idx_titles_poster'").fetchone():
        conn.execute("DROP INDEX idx_titles_poster")
    if missing:
        conn.commit()


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
    """Insert titles from a TMDB export. Existing rows keep their colours, and
    unchanged rows aren't rewritten (every rewrite leaves a dead row behind
    until vacuum, which counts against hosted database size limits)."""
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
                WHERE titles.popularity IS DISTINCT FROM excluded.popularity
                   OR titles.adult IS DISTINCT FROM excluded.adult
                   OR titles.title IS NULL
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
    """Flag changed titles for re-processing (existing rows only).

    Coloured titles keep serving their current colours but are marked as an
    outdated palette version, so the next run re-colours them; titles without
    colours go back to pending. Unknown IDs are ignored: new titles arrive via
    the daily seed, which respects the catalogue cap.
    """
    count = 0
    for batch in _batches(ids):
        with transaction(conn):
            conn.executemany(
                """
                UPDATE titles SET
                    palette_version = CASE WHEN status = 'done' THEN NULL ELSE palette_version END,
                    status = CASE WHEN status = 'done' THEN 'done' ELSE 'pending' END,
                    attempts = 0
                WHERE media_type = ? AND tmdb_id = ?
                """,
                [(media_type, int(tmdb_id)) for tmdb_id in batch],
            )
        count += len(batch)
    return count


def rank_titles(rows: Iterable, include_adult: bool = False) -> list:
    """Order titles the way the whole service ranks them: movies and TV
    alternating, each by popularity (their scales differ). `rows` are tuples
    starting (media_type, tmdb_id, popularity, adult, ...). Adult titles are
    left out unless include_adult."""
    by_type: dict[str, list] = {m: [] for m in MEDIA_TYPES}
    for r in rows:
        if include_adult or not r[3]:
            by_type[r[0]].append(r)
    for items in by_type.values():
        items.sort(key=lambda r: (r[2] is None, -(r[2] or 0.0), r[1]))
    ranked: list = []
    movies, shows = by_type["movie"], by_type["tv"]
    for i in range(max(len(movies), len(shows))):
        ranked.extend(x[i] for x in (movies, shows) if i < len(x))
    return ranked


def pending(
    conn: Connection,
    media_type: str | None,
    include_adult: bool,
    retry_errors: bool,
    max_attempts: int,
    limit: int | None,
    top: int | None = None,
    palette_version: int | None = None,
) -> list[tuple[str, int]]:
    """Titles still to process, most popular first.

    `top` caps the catalogue to the N most popular titles overall (whatever their
    status): titles ranked below it are never returned. `limit` caps this batch.
    With `palette_version`, titles coloured by an older algorithm version are
    returned too, so they get re-coloured.
    """
    statuses = {PENDING} | ({ERROR} if retry_errors else set())
    sql = "SELECT media_type, tmdb_id, popularity, adult, status, attempts, palette_version FROM titles"
    params: list = []
    if media_type:
        sql += " WHERE media_type = ?"
        params.append(media_type)
    ranked = rank_titles(conn.execute(sql, params), include_adult)
    if top:
        ranked = ranked[:top]

    def todo(status: str, attempts: int, version: int | None) -> bool:
        if status in statuses and attempts < max_attempts:
            return True
        return palette_version is not None and status == DONE and (version or 1) < palette_version

    out = [(r[0], r[1]) for r in ranked if todo(r[4], r[5], r[6])]
    return out[:limit] if limit else out


def prune(conn: Connection, top: int, include_adult: bool = False) -> int:
    """Delete titles ranked below `top` that have no colours, then vacuum.

    Coloured titles are always kept (even if their rank has slipped), so
    nothing Hoozat already uses disappears. Returns the number deleted.
    """
    rows = list(conn.execute("SELECT media_type, tmdb_id, popularity, adult, status FROM titles"))
    keep = {(r[0], r[1]) for r in rank_titles(rows, include_adult)[:top]}
    doomed = [(r[0], r[1]) for r in rows if r[4] != DONE and (r[0], r[1]) not in keep]
    for batch in _batches(doomed):
        with transaction(conn):
            conn.executemany("DELETE FROM titles WHERE media_type = ? AND tmdb_id = ?", batch)
    if doomed:
        try:
            if len(doomed) > 0.2 * len(rows):
                compact(conn)
            else:
                vacuum(conn)
        except Exception as exc:  # autovacuum will get there; deleted space is reusable anyway
            log.warning("vacuum failed: %s", exc)
    return len(doomed)


SECONDARY_INDEXES = ("idx_titles_status", "idx_titles_popularity", "idx_titles_type_popularity")


def compact(conn: Connection) -> None:
    """Shrink the table on disk after a large delete.

    Plain VACUUM only makes space reusable; the files keep their size, and
    hosted Postgres (Neon) counts file size against its limit. VACUUM FULL
    rewrites the table but needs room for the copy, which a full database
    doesn't have, so drop the secondary indexes first (freeing their files),
    rewrite, then recreate the indexes, now sized for the smaller table.
    """
    if conn.dialect != "postgres":
        vacuum(conn)
        return
    raw = conn._conn
    raw.commit()
    previous, raw.autocommit = raw.autocommit, True
    try:
        for name in SECONDARY_INDEXES:
            raw.execute(f"DROP INDEX IF EXISTS {name}")
        try:
            raw.execute("VACUUM (FULL, ANALYZE) titles")
        finally:
            raw.autocommit = previous
            ensure_schema(conn)  # recreates the indexes
    finally:
        raw.autocommit = previous


def vacuum(conn: Connection) -> None:
    """Make space from deleted/updated rows reusable (must run outside a transaction)."""
    if conn.dialect == "postgres":
        raw = conn._conn
        raw.commit()
        previous, raw.autocommit = raw.autocommit, True
        try:
            raw.execute("VACUUM (ANALYZE) titles")
        finally:
            raw.autocommit = previous
    else:
        conn.commit()
        conn.execute("VACUUM")


def save_result(conn: Connection, media_type: str, tmdb_id: int, **fields: Any) -> None:
    save_results(conn, [{"media_type": media_type, "tmdb_id": tmdb_id, **fields}])


def save_results(conn: Connection, results: list[dict]) -> None:
    """Store processing outcomes in one transaction (two round trips per batch).

    Each result: media_type, tmdb_id, status and optionally title, adult,
    poster_path, logo_path, palette [(hex, ratio) x3], semantic {role: hex},
    palette_version, error.
    """
    if not results:
        return
    ts = now_iso()
    updates = []
    for r in results:
        palette = r.get("palette")
        colours = [v for swatch in palette for v in swatch] if palette else [None] * 6
        semantic = r.get("semantic") or {}
        adult = r.get("adult")
        updates.append(
            (
                r.get("title"),
                None if adult is None else int(adult),
                r.get("poster_path"),
                *colours,
                *(semantic.get(role) for role in SEMANTIC_ROLES),
                r.get("logo_path"),
                r["status"],
                r.get("error"),
                int(r["status"] == ERROR),
                r.get("palette_version") if palette else None,
                ts,
                r["media_type"],
                r["tmdb_id"],
            )
        )
    with transaction(conn):
        conn.executemany(
            """
            INSERT INTO titles (media_type, tmdb_id, updated_at) VALUES (?, ?, ?)
            ON CONFLICT (media_type, tmdb_id) DO NOTHING
            """,
            [(r["media_type"], r["tmdb_id"], ts) for r in results],
        )
        conn.executemany(
            """
            UPDATE titles SET
                title = COALESCE(?, title),
                adult = COALESCE(?, adult),
                poster_path = ?,
                primary_hex = ?, primary_ratio = ?,
                secondary_hex = ?, secondary_ratio = ?,
                tertiary_hex = ?, tertiary_ratio = ?,
                base_hex = ?, identity1_hex = ?, identity2_hex = ?,
                highlight1_hex = ?, highlight2_hex = ?, accent_hex = ?,
                logo_path = ?,
                status = ?,
                error = ?,
                attempts = CASE WHEN ? = 1 THEN attempts + 1 ELSE 0 END,
                palette_version = ?,
                updated_at = ?
            WHERE media_type = ? AND tmdb_id = ?
            """,
            updates,
        )


def poster_palettes(conn: Connection, palette_version: int) -> dict[str, dict]:
    """"poster_path|logo_path" -> stored colours for every title coloured by this
    algorithm version, to reuse without re-downloading (TMDB image files are
    immutable, so the same pair always gives the same colours)."""
    rows = conn.execute(
        "SELECT poster_path, logo_path, primary_hex, primary_ratio, secondary_hex, secondary_ratio, "
        "tertiary_hex, tertiary_ratio, " + ", ".join(f"{r}_hex" for r in SEMANTIC_ROLES) + " FROM titles "
        "WHERE status = 'done' AND poster_path IS NOT NULL AND palette_version = ?",
        (palette_version,),
    )
    out = {}
    for r in rows:
        out[f"{r[0]}|{r[1] or ''}"] = {
            "palette": [(r[2], r[3]), (r[4], r[5]), (r[6], r[7])],
            "semantic": dict(zip(SEMANTIC_ROLES, r[8:14])),
        }
    return out


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


def top_titles(
    conn: Connection, media_type: str | None = None, limit: int = 60, offset: int = 0
) -> list[tuple[int, Row]]:
    """Titles with colours as a ranked list: [(rank, row), ...].

    Ranked like the ingester (movie and TV alternating, each by popularity).
    Titles are coloured in rank order, so this matches the catalogue ranking.
    """
    def fetch(mt: str, n: int) -> list[Row]:
        return conn.execute(
            f"SELECT {TITLE_COLUMNS} FROM titles WHERE status = 'done' AND media_type = ? "
            "ORDER BY popularity IS NULL, popularity DESC, tmdb_id LIMIT ?",
            (mt, n),
        ).fetchall()

    if media_type:
        rows = fetch(media_type, offset + limit)
    else:
        movies, shows = fetch("movie", offset + limit), fetch("tv", offset + limit)
        rows = []
        for i in range(max(len(movies), len(shows))):
            rows.extend(x[i] for x in (movies, shows) if i < len(x))
    return list(enumerate(rows, start=1))[offset : offset + limit]


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
