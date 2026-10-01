"""TheTVDB pipeline: TheTVDB titles -> poster (+ logo) -> six colours -> Hoozat.

Replaces the TMDB pipeline for Hoozat. Titles and artwork come from TheTVDB, so the
colours are keyed by TheTVDB ID, the same ID as Hoozat's credits: no ID mapping.

State is one local SQLite file (kept between CI runs in the Actions cache). Nothing
is served from here: finished colours are pushed to the Hoozat API, which stores
them in Cloudflare D1 and reads them on each scan.

    seed     page through every TheTVDB series and movie (id, name, score, poster)
    process  colour titles not done yet, best-scored first, within a time limit
    export   push new or changed colours to Hoozat, within a daily write budget
"""
from __future__ import annotations

import asyncio
import logging
import sqlite3
import time
from datetime import datetime, timezone

import httpx

from . import semantic
from .semantic import SEMANTIC_VERSION as PALETTE_VERSION
from .tmdb import NotFound
from .tvdb import KINDS, TVDBClient

log = logging.getLogger("tvmdbhex.tvdb")

ROLES = ("base", "identity1", "identity2", "highlight1", "highlight2", "accent")
MAX_ATTEMPTS = 5
EXPORT_BATCH = 500
# Cloudflare D1's free plan allows 100,000 rows written a day; leave room for the API.
DEFAULT_MAX_WRITES = 80_000

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS tvdb_titles (
    kind            TEXT    NOT NULL CHECK (kind IN ('series', 'movie')),
    tvdb_id         INTEGER NOT NULL,
    title           TEXT,
    score           REAL,
    image           TEXT,
    logo            TEXT,
    status          TEXT    NOT NULL DEFAULT 'pending',
    error           TEXT,
    attempts        INTEGER NOT NULL DEFAULT 0,
    palette_version INTEGER,
    {", ".join(f"{r} TEXT" for r in ROLES)},
    dirty           INTEGER NOT NULL DEFAULT 0,  -- 1: Hoozat hasn't got the latest result
    seen            INTEGER,                     -- the last seed (generation) that listed it
    updated_at      TEXT    NOT NULL,
    PRIMARY KEY (kind, tvdb_id)
);
CREATE INDEX IF NOT EXISTS idx_tvdb_status ON tvdb_titles (status, kind, score DESC);
CREATE INDEX IF NOT EXISTS idx_tvdb_dirty ON tvdb_titles (dirty, score DESC);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(SCHEMA)
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------- seed

async def seed(conn: sqlite3.Connection, client: TVDBClient, kinds: tuple[str, ...] = KINDS) -> dict:
    """Store every title TheTVDB lists. A new poster re-queues the title for colouring.

    Titles that have disappeared from a complete listing are marked not_found and,
    if they had colours, queued for deletion from Hoozat.
    """
    counts = {}
    for kind in kinds:
        gen = conn.execute("SELECT COALESCE(MAX(seen), 0) + 1 FROM tvdb_titles WHERE kind = ?", (kind,)).fetchone()[0]
        ts = now_iso()
        batch, total = [], 0
        async for t in client.titles(kind):
            batch.append((kind, t["id"], t["title"], t["score"], t["image"], gen, ts))
            if len(batch) >= 5000:
                _upsert(conn, batch)
                total += len(batch)
                batch = []
        _upsert(conn, batch)
        total += len(batch)
        # The listing finished, so anything it didn't include is gone from TheTVDB.
        if total:
            with conn:
                gone = conn.execute(
                    "UPDATE tvdb_titles SET status = 'not_found', dirty = CASE WHEN status = 'done' THEN 1 ELSE dirty END, "
                    "updated_at = ? WHERE kind = ? AND (seen IS NULL OR seen < ?) AND status != 'not_found'",
                    (now_iso(), kind, gen),
                ).rowcount
        else:
            gone = 0
        counts[kind] = {"listed": total, "gone": gone}
        log.info("seeded %d %s titles (%d gone)", total, kind, gone)
    return counts


def _upsert(conn: sqlite3.Connection, rows: list[tuple]) -> None:
    if not rows:
        return
    with conn:
        conn.executemany(
            """
            INSERT INTO tvdb_titles (kind, tvdb_id, title, score, image, seen, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (kind, tvdb_id) DO UPDATE SET
                title = excluded.title,
                score = excluded.score,
                seen = excluded.seen,
                -- A changed poster means new colours: done titles keep serving the old
                -- ones until re-coloured; anything else goes back to pending.
                palette_version = CASE WHEN image IS NOT excluded.image THEN NULL ELSE palette_version END,
                status = CASE
                    WHEN image IS NOT excluded.image AND status != 'done' THEN 'pending'
                    WHEN status = 'not_found' THEN 'pending'
                    ELSE status END,
                attempts = CASE WHEN image IS NOT excluded.image THEN 0 ELSE attempts END,
                image = excluded.image
            """,
            rows,
        )


# ---------------------------------------------------------------- process

def pending(conn: sqlite3.Connection, limit: int | None = None) -> list[sqlite3.Row]:
    """Titles to colour, best-scored first, series and movies alternating
    (their scores are on different scales)."""
    by_kind = {}
    for kind in KINDS:
        by_kind[kind] = conn.execute(
            """
            SELECT kind, tvdb_id, image FROM tvdb_titles
            WHERE kind = ? AND (
                (status IN ('pending', 'error') AND attempts < ?)
                OR (status = 'done' AND (palette_version IS NULL OR palette_version < ?)))
            ORDER BY score IS NULL, score DESC, tvdb_id
            """ + (" LIMIT ?" if limit else ""),
            (kind, MAX_ATTEMPTS, PALETTE_VERSION, *([limit] if limit else [])),
        ).fetchall()
    out = []
    series, movies = by_kind["series"], by_kind["movie"]
    for i in range(max(len(series), len(movies))):
        out.extend(x[i] for x in (series, movies) if i < len(x))
    return out[:limit] if limit else out


async def _colours(client: TVDBClient, poster_url: str, logo_url: str | None) -> dict[str, str]:
    poster = await client.image(poster_url)
    logo = None
    if logo_url:
        try:
            logo = await client.image(logo_url, prefer_thumbnail=False)  # logos are small already
        except Exception as exc:  # a missing logo shouldn't fail the title
            log.info("logo %s unavailable: %s", logo_url, exc)
    result = await asyncio.to_thread(semantic.analyse_bytes, poster, logo)
    return result.hexes()


async def process_one(client: TVDBClient, kind: str, tvdb_id: int, image: str | None,
                      cache: dict[str, dict], inflight: dict[str, asyncio.Task]) -> dict:
    key = {"kind": kind, "tvdb_id": tvdb_id}
    if not image:
        return {**key, "status": "no_poster"}
    try:
        logo = await client.logo(kind, tvdb_id)
    except NotFound:
        return {**key, "status": "not_found"}
    except Exception as exc:  # colour from the poster alone rather than fail the title
        log.info("%s %s: no logo (%s)", kind, tvdb_id, exc)
        logo = None
    pair = f"{image}|{logo or ''}"  # artwork files don't change, so a pair always gives the same colours
    colours = cache.get(pair)
    if colours is None:
        task = inflight.get(pair)
        if task is None:
            task = inflight[pair] = asyncio.ensure_future(_colours(client, image, logo))
            task.add_done_callback(lambda _: inflight.pop(pair, None))
        try:
            colours = await asyncio.shield(task)
        except NotFound:
            return {**key, "status": "no_poster"}  # listed, but the file is gone
        cache[pair] = colours
    return {**key, "status": "done", "logo": logo, "colours": colours}


def save(conn: sqlite3.Connection, results: list[dict]) -> None:
    ts = now_iso()
    rows = []
    for r in results:
        c = r.get("colours") or {}
        done = r["status"] == "done"
        rows.append((
            r["status"], r.get("error"), int(r["status"] == "error"),
            PALETTE_VERSION if done else None, r.get("logo"),
            *(c.get(role) for role in ROLES),
            # Hoozat needs the result: new colours, or a deletion of colours it may have.
            int(r["status"] in ("done", "no_poster", "not_found")),
            ts, r["kind"], r["tvdb_id"],
        ))
    with conn:
        conn.executemany(
            f"""
            UPDATE tvdb_titles SET
                status = ?, error = ?,
                attempts = CASE WHEN ? = 1 THEN attempts + 1 ELSE 0 END,
                palette_version = ?, logo = ?,
                {", ".join(f"{role} = ?" for role in ROLES)},
                dirty = CASE WHEN ? = 1 THEN 1 ELSE dirty END,
                updated_at = ?
            WHERE kind = ? AND tvdb_id = ?
            """,
            rows,
        )


async def process(conn: sqlite3.Connection, client: TVDBClient, concurrency: int = 16,
                  limit: int | None = None, max_minutes: float | None = None) -> dict:
    todo = pending(conn, limit)
    cache: dict[str, dict] = {}
    inflight: dict[str, asyncio.Task] = {}
    queue: asyncio.Queue = asyncio.Queue()
    for row in todo:
        queue.put_nowait((row["kind"], row["tvdb_id"], row["image"]))
    log.info("colouring %d titles with %d workers", len(todo), concurrency)
    counts: dict[str, int] = {}
    buffer: list[dict] = []
    started = time.monotonic()
    deadline = started + max_minutes * 60 if max_minutes else None

    def flush() -> None:
        nonlocal buffer
        batch, buffer = buffer, []
        save(conn, batch)

    async def worker() -> None:
        while not (deadline and time.monotonic() > deadline):
            try:
                kind, tvdb_id, image = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                result = await process_one(client, kind, tvdb_id, image, cache, inflight)
            except Exception as exc:  # retried on a later run
                log.warning("%s %s failed: %s", kind, tvdb_id, exc)
                result = {"kind": kind, "tvdb_id": tvdb_id, "status": "error", "error": str(exc)[:500]}
            buffer.append(result)
            counts[result["status"]] = counts.get(result["status"], 0) + 1
            if len(buffer) >= 200:
                flush()
            done = sum(counts.values())
            if done % 1000 == 0:
                log.info("%d/%d (%.1f/s) %s", done, len(todo), done / max(time.monotonic() - started, 1e-6), counts)

    try:
        await asyncio.gather(*(worker() for _ in range(max(1, concurrency))))
    finally:
        flush()
    log.info("finished: %s", counts)
    return counts


# ---------------------------------------------------------------- export

async def export(conn: sqlite3.Connection, url: str, write_key: str,
                 max_writes: int = DEFAULT_MAX_WRITES, http: httpx.AsyncClient | None = None) -> dict:
    """Push results Hoozat hasn't got yet, best-scored first: colours for done titles,
    deletions for titles that lost theirs. Stops at max_writes (D1's daily allowance)."""
    rows = conn.execute(
        f"SELECT kind, tvdb_id, status, {', '.join(ROLES)} FROM tvdb_titles WHERE dirty = 1 "
        "ORDER BY score IS NULL, score DESC LIMIT ?",
        (max_writes,),
    ).fetchall()
    sent = 0
    own = http is None
    http = http or httpx.AsyncClient(timeout=60.0)
    try:
        for i in range(0, len(rows), EXPORT_BATCH):
            batch = rows[i : i + EXPORT_BATCH]
            items = [
                {"key": f"{r['kind']}:{r['tvdb_id']}",
                 "colours": {role: r[role] for role in ROLES} if r["status"] == "done" else None}
                for r in batch
            ]
            resp = await http.put(url, json={"items": items}, headers={"X-Colours-Write-Key": write_key})
            if resp.status_code >= 400:
                raise RuntimeError(f"Hoozat refused the colours: HTTP {resp.status_code} {resp.text[:200]}")
            with conn:
                conn.executemany(
                    "UPDATE tvdb_titles SET dirty = 0 WHERE kind = ? AND tvdb_id = ?",
                    [(r["kind"], r["tvdb_id"]) for r in batch],
                )
            sent += len(batch)
    finally:
        if own:
            await http.aclose()
    remaining = conn.execute("SELECT COUNT(*) FROM tvdb_titles WHERE dirty = 1").fetchone()[0]
    log.info("exported %d to Hoozat; %d still to send", sent, remaining)
    return {"sent": sent, "remaining": remaining}


def stats(conn: sqlite3.Connection) -> dict:
    out: dict = {k: {} for k in KINDS}
    for kind, status, n in conn.execute("SELECT kind, status, COUNT(*) FROM tvdb_titles GROUP BY kind, status"):
        out[kind][status] = n
    out["unsent"] = conn.execute("SELECT COUNT(*) FROM tvdb_titles WHERE dirty = 1").fetchone()[0]
    return out
