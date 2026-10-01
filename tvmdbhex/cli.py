"""Command line entry point: `tvmdbhex <command>`."""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging

import os

from . import db, ingest, tvdb_ingest
from .config import Settings
from .tmdb import TMDBClient
from .tvdb import TVDBClient


def _media_types(value: str) -> list[str]:
    return list(db.MEDIA_TYPES) if value == "all" else [value]


def _describe_db(settings: Settings) -> str:
    if settings.database_url:
        host = settings.database_url.split("@", 1)[-1].split("/", 1)[0]  # no credentials
        return f"postgres at {host}"
    return f"sqlite at {settings.db_path}"


async def _run(args: argparse.Namespace, settings: Settings) -> dict:
    logging.getLogger("tvmdbhex").info("database: %s", _describe_db(settings))
    conn = db.connect(settings.db_target)
    try:
        async with TMDBClient(
            settings.tmdb_api_key, settings.tmdb_read_token, settings.requests_per_second
        ) as client:
            media_types = _media_types(args.media)
            size = settings.catalogue_size  # seed/prune: catalogue size only, never the run's --top
            if args.command == "seed":
                return await ingest.seed(conn, client, media_types, size, settings.include_adult)
            if args.command == "sync":
                return await ingest.sync_changes(conn, client, media_types)
            if args.command == "process":
                media = None if args.media == "all" else args.media
                return await ingest.process(conn, client, settings, media, args.limit, args.max_minutes)
            if args.command == "run":  # full refresh: seed + changes + process
                result = {}
                if size:  # prune first: frees space before the seed writes anything
                    result["pruned"] = ingest.prune(conn, size, settings.include_adult)
                result["seed"] = await ingest.seed(conn, client, media_types, size, settings.include_adult)
                result["sync"] = await ingest.sync_changes(conn, client, media_types)
                media = None if args.media == "all" else args.media
                result["process"] = await ingest.process(
                    conn, client, settings, media, args.limit, args.max_minutes
                )
                return result
    finally:
        conn.close()
    raise ValueError(args.command)


async def _run_tvdb(args: argparse.Namespace, settings: Settings) -> dict:
    """`tvmdbhex tvdb <action>`: the TheTVDB pipeline that feeds Hoozat."""
    os.makedirs(os.path.dirname(os.path.abspath(settings.tvdb_db_path)), exist_ok=True)
    conn = tvdb_ingest.connect(settings.tvdb_db_path)
    try:
        if args.action == "stats":
            return tvdb_ingest.stats(conn)
        result: dict = {}
        if args.action in ("export", "run") and not settings.colours_write_key:
            raise SystemExit("Set COLOURS_WRITE_KEY (the same value as the Hoozat API's secret) to export")
        if args.action in ("seed", "process", "run"):
            async with TVDBClient(settings.tvdb_api_key, settings.tvdb_pin, settings.tvdb_requests_per_second) as client:
                if args.action in ("seed", "run"):
                    result["seed"] = await tvdb_ingest.seed(conn, client)
                if args.action in ("process", "run"):
                    result["process"] = await tvdb_ingest.process(
                        conn, client, settings.concurrency, args.limit, args.max_minutes
                    )
        if args.action in ("export", "run"):
            result["export"] = await tvdb_ingest.export(
                conn, settings.colours_url, settings.colours_write_key, args.max_writes
            )
        result["stats"] = tvdb_ingest.stats(conn)
        return result
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="tvmdbhex")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in [
        ("seed", "Load every movie/TV ID from TMDB's daily export"),
        ("sync", "Re-queue titles changed on TMDB since the last sync"),
        ("process", "Fetch posters and extract colours for pending titles"),
        ("run", "prune + seed (catalogue size) + sync + process (use this on a daily schedule)"),
    ]:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--media", choices=["movie", "tv", "all"], default="all")
        p.add_argument(
            "--top", type=int, default=None,
            help="Only process titles ranked in the top N this run (default: TVMDBHEX_MAX_TITLES); "
            "the catalogue size (TVMDBHEX_CATALOGUE_SIZE) is separate",
        )
        if name in ("process", "run"):
            p.add_argument("--limit", type=int, default=None, help="Max titles to process")
            p.add_argument(
                "--max-minutes", type=float, default=None,
                help="Stop cleanly after this long (e.g. to fit a CI job limit); resumes next run",
            )
    prune = sub.add_parser("prune", help="Delete uncoloured titles ranked below the catalogue size and vacuum")
    prune.add_argument("--size", type=int, default=None, help="Catalogue size (default: TVMDBHEX_CATALOGUE_SIZE)")
    sub.add_parser("stats", help="Show counts by status")
    tv = sub.add_parser("tvdb", help="TheTVDB pipeline for Hoozat: seed, process, export, run or stats")
    tv.add_argument("action", choices=["seed", "process", "export", "run", "stats"])
    tv.add_argument("--limit", type=int, default=None, help="Max titles to colour this run")
    tv.add_argument("--max-minutes", type=float, default=None, help="Stop colouring after this long")
    tv.add_argument(
        "--max-writes", type=int, default=tvdb_ingest.DEFAULT_MAX_WRITES,
        help="Max rows to send to Hoozat this run (Cloudflare D1 free plan: 100,000 writes a day)",
    )
    serve = sub.add_parser("serve", help="Run the HTTP API")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per request is too noisy
    settings = Settings.from_env()
    if getattr(args, "top", None) and args.command != "prune":
        settings = dataclasses.replace(settings, max_titles=args.top)

    if args.command == "prune":
        size = args.size or settings.catalogue_size
        if not size:
            parser.error("prune needs --size or TVMDBHEX_CATALOGUE_SIZE")
        conn = db.connect(settings.db_target)
        print(json.dumps({"pruned": ingest.prune(conn, size, settings.include_adult)}))
        return
    if args.command == "stats":
        conn = db.connect(settings.db_target)
        print(json.dumps(db.stats(conn), indent=2))
        return
    if args.command == "tvdb":
        print(json.dumps(asyncio.run(_run_tvdb(args, settings)), indent=2))
        return
    if args.command == "serve":
        import uvicorn

        from .api import create_app

        uvicorn.run(create_app(settings), host=args.host, port=args.port)
        return
    print(json.dumps(asyncio.run(_run(args, settings)), indent=2))


if __name__ == "__main__":
    main()
