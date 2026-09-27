"""Ingestion pipeline: TMDB -> poster -> dominant colours -> SQLite."""
from __future__ import annotations

import asyncio
import logging
import sqlite3
import time
from datetime import date, timedelta

from . import db
from .colors import extract_palette_from_bytes
from .config import Settings
from .tmdb import NotFound, TMDBClient, parse_export

log = logging.getLogger("tvmdbhex.ingest")

MAX_ATTEMPTS = 5
CHANGES_MAX_WINDOW = 14  # days per TMDB /changes call


async def seed(conn: sqlite3.Connection, client: TMDBClient, media_types: list[str]) -> dict:
    """Load every title ID from TMDB's daily exports into the database."""
    counts = {}
    for media_type in media_types:
        day, payload = await client.daily_export(media_type)
        counts[media_type] = db.upsert_seed(conn, media_type, parse_export(media_type, payload))
        db.set_state(conn, f"seed:{media_type}", day.isoformat())
        # The export is a snapshot; changes since then are picked up by sync_changes.
        if db.get_state(conn, f"changes:{media_type}") is None:
            db.set_state(conn, f"changes:{media_type}", day.isoformat())
        log.info("seeded %d %s titles from %s export", counts[media_type], media_type, day)
    return counts


async def _palette_for(client: TMDBClient, settings: Settings, poster_path: str) -> list[tuple[str, float]]:
    image = await client.poster(poster_path, settings.poster_size)
    result = await asyncio.to_thread(extract_palette_from_bytes, image)
    return [(s.hex, s.ratio) for s in result.as_list()]


async def process_one(
    conn: sqlite3.Connection,
    client: TMDBClient,
    settings: Settings,
    media_type: str,
    tmdb_id: int,
    inflight: dict[str, asyncio.Task] | None = None,
) -> str:
    try:
        details = await client.details(media_type, tmdb_id)
    except NotFound:
        db.save_result(conn, media_type, tmdb_id, status=db.NOT_FOUND)
        return db.NOT_FOUND

    title = details.get("title") or details.get("name")
    adult = bool(details.get("adult", False))
    poster_path = details.get("poster_path")
    if not poster_path:
        db.save_result(conn, media_type, tmdb_id, status=db.NO_POSTER, title=title, adult=adult)
        return db.NO_POSTER

    existing = db.find_by_poster(conn, poster_path)
    if existing is not None:
        palette = [
            (existing["primary_hex"], existing["primary_ratio"]),
            (existing["secondary_hex"], existing["secondary_ratio"]),
            (existing["tertiary_hex"], existing["tertiary_ratio"]),
        ]
    else:
        # Titles processed concurrently may share a poster file: download it once.
        inflight = {} if inflight is None else inflight
        task = inflight.get(poster_path)
        if task is None:
            task = inflight[poster_path] = asyncio.ensure_future(_palette_for(client, settings, poster_path))
            task.add_done_callback(lambda _: inflight.pop(poster_path, None))
        palette = await asyncio.shield(task)

    db.save_result(
        conn,
        media_type,
        tmdb_id,
        status=db.DONE,
        title=title,
        adult=adult,
        poster_path=poster_path,
        palette=palette,
    )
    return db.DONE


async def process(
    conn: sqlite3.Connection,
    client: TMDBClient,
    settings: Settings,
    media_type: str | None = None,
    limit: int | None = None,
    retry_errors: bool = True,
) -> dict:
    """Extract colours for every pending title (most popular first)."""
    todo = db.pending(conn, media_type, settings.include_adult, retry_errors, MAX_ATTEMPTS, limit)
    log.info("processing %d titles with %d workers", len(todo), settings.concurrency)
    queue: asyncio.Queue[tuple[str, int]] = asyncio.Queue()
    for item in todo:
        queue.put_nowait(item)

    counts: dict[str, int] = {}
    inflight: dict[str, asyncio.Task] = {}
    started = time.monotonic()

    async def worker() -> None:
        while True:
            try:
                mt, tmdb_id = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                status = await process_one(conn, client, settings, mt, tmdb_id, inflight)
            except Exception as exc:  # keep going; retried on the next run
                log.warning("%s/%s failed: %s", mt, tmdb_id, exc)
                db.save_result(conn, mt, tmdb_id, status=db.ERROR, error=str(exc)[:500])
                status = db.ERROR
            counts[status] = counts.get(status, 0) + 1
            done = sum(counts.values())
            if done % 1000 == 0:
                rate = done / max(time.monotonic() - started, 1e-6)
                log.info("%d/%d done (%.1f/s) %s", done, len(todo), rate, counts)

    await asyncio.gather(*(worker() for _ in range(max(1, settings.concurrency))))
    log.info("finished: %s", counts)
    return counts


async def sync_changes(
    conn: sqlite3.Connection, client: TMDBClient, media_types: list[str], today: date | None = None
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
