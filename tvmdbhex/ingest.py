"""Ingestion pipeline: TMDB -> poster -> dominant colours -> SQLite."""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import date, timedelta

from . import db
from . import semantic
from .semantic import SEMANTIC_VERSION as PALETTE_VERSION
from .config import Settings
from .tmdb import NotFound, TMDBClient, parse_export, pick_logo

log = logging.getLogger("tvmdbhex.ingest")

MAX_ATTEMPTS = 5
CHANGES_MAX_WINDOW = 14  # days per TMDB /changes call


async def seed(
    conn: db.Connection,
    client: TMDBClient,
    media_types: list[str],
    top: int | None = None,
    include_adult: bool = False,
) -> dict:
    """Load title IDs from TMDB's daily exports into the database.

    With `top`, only the N most popular titles (movies and TV alternating, as
    ranked everywhere else) are stored; the rest of the catalogue would only
    ever be "pending", and storing ~1.5M such rows is what fills a small
    hosted database.
    """
    exports: dict[str, tuple] = {}
    for media_type in media_types:
        day, payload = await client.daily_export(media_type)
        if db.get_state(conn, f"seed:{media_type}") == day.isoformat():
            log.info("%s export for %s already loaded", media_type, day)
            continue
        exports[media_type] = (day, list(parse_export(media_type, payload)))
    if not exports:
        return {m: 0 for m in media_types}

    keep: set[tuple[str, int]] | None = None
    if top:
        rows = [(m, r["id"], r.get("popularity"), r.get("adult")) for m, (_, items) in exports.items() for r in items]
        keep = {(r[0], r[1]) for r in db.rank_titles(rows, include_adult)[:top]}

    counts = {m: 0 for m in media_types}
    for media_type, (day, items) in exports.items():
        if keep is not None:
            items = [r for r in items if (media_type, r["id"]) in keep]
        counts[media_type] = db.upsert_seed(conn, media_type, items)
        db.set_state(conn, f"seed:{media_type}", day.isoformat())
        # The export is a snapshot; changes since then are picked up by sync_changes.
        if db.get_state(conn, f"changes:{media_type}") is None:
            db.set_state(conn, f"changes:{media_type}", day.isoformat())
        log.info("seeded %d %s titles from %s export", counts[media_type], media_type, day)
    return counts


def prune(conn: db.Connection, top: int, include_adult: bool = False) -> int:
    deleted = db.prune(conn, top, include_adult)
    log.info("pruned %d uncoloured titles ranked below the top %d", deleted, top)
    return deleted


# Legacy `colors` (primary/secondary/tertiary) are derived from the semantic
# roles so existing API consumers keep working: identity -> primary, the base
# environment -> secondary, accent -> tertiary. Ratios are the roles' design weights.
LEGACY_FROM_SEMANTIC = (("identity1", 0.275), ("base", 0.45), ("accent", 0.055))


async def _colours_for(client: TMDBClient, settings: Settings, poster_path: str, logo_path: str | None) -> dict:
    poster = await client.poster(poster_path, settings.poster_size)
    logo = None
    if logo_path:
        try:
            logo = await client.image(logo_path, "w300")
        except Exception as exc:  # a missing logo shouldn't fail the title
            log.info("logo %s unavailable: %s", logo_path, exc)
    result = await asyncio.to_thread(semantic.analyse_bytes, poster, logo)
    roles = result.hexes()
    return {
        "semantic": roles,
        "palette": [(roles[role], weight) for role, weight in LEGACY_FROM_SEMANTIC],
    }


async def process_one(
    client: TMDBClient,
    settings: Settings,
    media_type: str,
    tmdb_id: int,
    inflight: dict[str, asyncio.Task],
    posters: dict[str, dict],
) -> dict:
    """Fetch one title's colours. Returns the result to store (no DB access)."""
    key = {"media_type": media_type, "tmdb_id": tmdb_id}
    try:
        details = await client.details(media_type, tmdb_id, with_images=True)
    except NotFound:
        return {**key, "status": db.NOT_FOUND}

    title = details.get("title") or details.get("name")
    adult = bool(details.get("adult", False))
    poster_path = details.get("poster_path")
    if not poster_path:
        return {**key, "status": db.NO_POSTER, "title": title, "adult": adult}
    logo_path = pick_logo(details)

    # Image files are immutable, so colours computed once for a poster+logo
    # pair can be reused by any title using the same pair.
    cache_key = f"{poster_path}|{logo_path or ''}"
    colours = posters.get(cache_key)
    if colours is None:
        # Titles processed concurrently may share images: download them once.
        task = inflight.get(cache_key)
        if task is None:
            task = inflight[cache_key] = asyncio.ensure_future(_colours_for(client, settings, poster_path, logo_path))
            task.add_done_callback(lambda _: inflight.pop(cache_key, None))
        colours = await asyncio.shield(task)
        posters[cache_key] = colours

    return {
        **key,
        "status": db.DONE,
        "title": title,
        "adult": adult,
        "poster_path": poster_path,
        "logo_path": logo_path,
        "palette": colours["palette"],
        "semantic": colours["semantic"],
        "palette_version": PALETTE_VERSION,
    }


class _BatchWriter:
    """Buffers results and writes them in batches off the event loop.

    Per-title writes cost several round trips each; to a remote database
    (e.g. Neon from a CI runner) that, not TMDB, was the bottleneck.
    """

    def __init__(self, conn: db.Connection, batch_size: int = 200, max_delay: float = 5.0):
        self.conn = conn
        self.batch_size = batch_size
        self.max_delay = max_delay
        self.buffer: list[dict] = []
        self.lock = asyncio.Lock()
        self.last_flush = time.monotonic()

    async def add(self, result: dict) -> None:
        self.buffer.append(result)
        if len(self.buffer) >= self.batch_size or time.monotonic() - self.last_flush > self.max_delay:
            await self.flush()

    async def flush(self) -> None:
        async with self.lock:
            batch, self.buffer = self.buffer, []
            self.last_flush = time.monotonic()
            if batch:
                await asyncio.to_thread(db.save_results, self.conn, batch)


async def process(
    conn: db.Connection,
    client: TMDBClient,
    settings: Settings,
    media_type: str | None = None,
    limit: int | None = None,
    max_minutes: float | None = None,
    retry_errors: bool = True,
) -> dict:
    """Extract colours for every pending title (most popular first)."""
    todo = db.pending(
        conn, media_type, settings.include_adult, retry_errors, MAX_ATTEMPTS, limit, settings.max_titles,
        palette_version=PALETTE_VERSION,
    )
    posters = db.poster_palettes(conn, PALETTE_VERSION)
    log.info(
        "processing %d titles with %d workers (%d known posters)", len(todo), settings.concurrency, len(posters)
    )
    queue: asyncio.Queue[tuple[str, int]] = asyncio.Queue()
    for item in todo:
        queue.put_nowait(item)

    counts: dict[str, int] = {}
    inflight: dict[str, asyncio.Task] = {}
    writer = _BatchWriter(conn)
    started = time.monotonic()
    deadline = started + max_minutes * 60 if max_minutes else None

    async def worker() -> None:
        while True:
            if deadline and time.monotonic() > deadline:
                return  # out of time; the rest stays pending for the next run
            try:
                mt, tmdb_id = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                result = await process_one(client, settings, mt, tmdb_id, inflight, posters)
            except Exception as exc:  # keep going; retried on the next run
                log.warning("%s/%s failed: %s", mt, tmdb_id, exc)
                result = {"media_type": mt, "tmdb_id": tmdb_id, "status": db.ERROR, "error": str(exc)[:500]}
            await writer.add(result)
            status = result["status"]
            counts[status] = counts.get(status, 0) + 1
            done = sum(counts.values())
            if done % 1000 == 0:
                rate = done / max(time.monotonic() - started, 1e-6)
                log.info("%d/%d done (%.1f/s) %s", done, len(todo), rate, counts)

    try:
        await asyncio.gather(*(worker() for _ in range(max(1, settings.concurrency))))
    finally:
        await writer.flush()
    log.info("finished: %s", counts)
    return counts


async def sync_changes(
    conn: db.Connection, client: TMDBClient, media_types: list[str], today: date | None = None
) -> dict:
    """Re-queue titles TMDB reports as changed (e.g. new poster) since the last sync."""
    today = today or date.today()
    counts = {}
    for media_type in media_types:
        key = f"changes:{media_type}"
        last = db.get_state(conn, key)
        start = date.fromisoformat(last) if last else today - timedelta(days=1)
        ids: set[int] = set()
        while start < today:
            end = min(start + timedelta(days=CHANGES_MAX_WINDOW), today)
            ids |= await client.changes(media_type, start, end)
            start = end
        counts[media_type] = db.requeue(conn, media_type, ids)
        db.set_state(conn, key, today.isoformat())
        log.info("re-queued %d changed %s titles", counts[media_type], media_type)
    return counts
