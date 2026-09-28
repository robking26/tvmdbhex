"""Command line entry point: `tvmdbhex <command>`."""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging

from . import db, ingest
from .config import Settings
from .tmdb import TMDBClient


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
            top = settings.max_titles
            if args.command == "seed":
                return await ingest.seed(conn, client, media_types, top, settings.include_adult)
            if args.command == "sync":
                return await ingest.sync_changes(conn, client, media_types)
            if args.command == "process":
                media = None if args.media == "all" else args.media
                return await ingest.process(conn, client, settings, media, args.limit, args.max_minutes)
            if args.command == "run":  # full refresh: seed + changes + process
                result = {}
                if top:  # prune first: frees space before the seed writes anything
                    result["pruned"] = ingest.prune(conn, top, settings.include_adult)
                result["seed"] = await ingest.seed(conn, client, media_types, top, settings.include_adult)
                result["sync"] = await ingest.sync_changes(conn, client, media_types)
                media = None if args.media == "all" else args.media
                result["process"] = await ingest.process(
                    conn, client, settings, media, args.limit, args.max_minutes
                )
                return result
    finally:
        conn.close()
    raise ValueError(args.command)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="tvmdbhex")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in [
        ("seed", "Load every movie/TV ID from TMDB's daily export"),
        ("sync", "Re-queue titles changed on TMDB since the last sync"),
        ("process", "Fetch posters and extract colours for pending titles"),
        ("run", "prune + seed + sync + process (use this on a daily schedule)"),
    ]:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--media", choices=["movie", "tv", "all"], default="all")
        p.add_argument(
            "--top", type=int, default=None,
            help="Only store/process the N most popular titles overall (default: TVMDBHEX_MAX_TITLES)",
        )
        if name in ("process", "run"):
            p.add_argument("--limit", type=int, default=None, help="Max titles to process")
            p.add_argument(
                "--max-minutes", type=float, default=None,
                help="Stop cleanly after this long (e.g. to fit a CI job limit); resumes next run",
            )
    prune = sub.add_parser("prune", help="Delete uncoloured titles ranked below --top and vacuum")
    prune.add_argument("--top", type=int, default=None, help="Default: TVMDBHEX_MAX_TITLES")
    sub.add_parser("stats", help="Show counts by status")
    serve = sub.add_parser("serve", help="Run the HTTP API")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per request is too noisy
    settings = Settings.from_env()
    if getattr(args, "top", None):
        settings = dataclasses.replace(settings, max_titles=args.top)

    if args.command == "prune":
        top = args.top or settings.max_titles
        if not top:
            parser.error("prune needs --top or TVMDBHEX_MAX_TITLES")
        conn = db.connect(settings.db_target)
        print(json.dumps({"pruned": ingest.prune(conn, top, settings.include_adult)}))
        return
    if args.command == "stats":
        conn = db.connect(settings.db_target)
        print(json.dumps(db.stats(conn), indent=2))
        return
    if args.command == "serve":
        import uvicorn

        from .api import create_app

        uvicorn.run(create_app(settings), host=args.host, port=args.port)
        return
    print(json.dumps(asyncio.run(_run(args, settings)), indent=2))


if __name__ == "__main__":
    main()
