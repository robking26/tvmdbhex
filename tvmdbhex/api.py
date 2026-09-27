"""Read-only HTTP API over the colour database. Never calls TMDB."""
from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from importlib import resources
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Path, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from . import __version__, db
from .config import Settings

MediaType = Literal["movie", "tv"]
MAX_BATCH = 500


class Swatch(BaseModel):
    hex: str = Field(examples=["#1B2A3C"])
    ratio: float = Field(description="Share of the poster's pixels in this colour (0-1)")


class Colors(BaseModel):
    primary: Swatch
    secondary: Swatch
    tertiary: Swatch


class TitleColors(BaseModel):
    media_type: MediaType
    tmdb_id: int
    title: str | None
    poster_path: str | None
    status: str = Field(description="done | pending | no_poster | not_found | error")
    colors: Colors | None
    updated_at: str


class LookupItem(BaseModel):
    media_type: MediaType
    tmdb_id: int


class LookupRequest(BaseModel):
    items: list[LookupItem] = Field(max_length=MAX_BATCH)


class LookupResponse(BaseModel):
    results: list[TitleColors]
    missing: list[LookupItem]


def _row_to_model(row: db.Row) -> TitleColors:
    colors = None
    if row["status"] == db.DONE and row["primary_hex"]:
        colors = Colors(
            primary=Swatch(hex=row["primary_hex"], ratio=row["primary_ratio"]),
            secondary=Swatch(hex=row["secondary_hex"], ratio=row["secondary_ratio"]),
            tertiary=Swatch(hex=row["tertiary_hex"], ratio=row["tertiary_ratio"]),
        )
    return TitleColors(
        media_type=row["media_type"],
        tmdb_id=row["tmdb_id"],
        title=row["title"],
        poster_path=row["poster_path"],
        status=row["status"],
        colors=colors,
        updated_at=row["updated_at"],
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    target = settings.db_target
    postgres = db.is_postgres_url(target)
    if not postgres:
        db.connect(target).close()  # create the file/schema if missing

    app = FastAPI(
        title="tvmdbhex",
        version=__version__,
        description="Primary, secondary and tertiary poster colours (hex) for TMDB movies and TV series.",
    )

    # Postgres: one connection per instance, reused across (serverless) requests.
    shared: dict = {}
    lock = threading.Lock()

    def postgres_conn() -> db.Connection:
        with lock:
            conn = shared.get("conn")
            if conn is None or conn.closed:
                conn = db.connect(target, readonly=True)
                try:
                    db.ensure_schema(conn)  # first deploy, before the ingester has run
                except Exception as exc:  # e.g. a concurrent cold start created it
                    logging.getLogger("tvmdbhex.api").info("schema check skipped: %s", exc)
                shared["conn"] = conn
            return conn

    def get_conn() -> Iterator[db.Connection]:
        if postgres:
            yield postgres_conn()
            return
        conn = db.connect(target, readonly=True)
        try:
            yield conn
        finally:
            conn.close()

    def require_key(x_api_key: str | None = Header(default=None)) -> None:
        if settings.api_keys and x_api_key not in settings.api_keys:
            raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")

    index_html = resources.files("tvmdbhex").joinpath("static/index.html").read_text("utf-8")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> str:
        # The page shell is public; its data calls still need X-API-Key.
        return index_html

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/v1/stats", dependencies=[Depends(require_key)])
    def stats(conn: db.Connection = Depends(get_conn)) -> dict:
        return db.stats(conn)

    @app.get(
        "/v1/search",
        response_model=list[TitleColors],
        dependencies=[Depends(require_key)],
        summary="Search titles by name or TMDB ID, most popular first",
    )
    def search(
        q: str = Query("", max_length=200, description="Title substring or TMDB ID; empty = browse"),
        media_type: MediaType | None = None,
        status: str | None = Query(None, description="e.g. `done` to only return titles with colours"),
        limit: int = Query(24, ge=1, le=100),
        offset: int = Query(0, ge=0, le=10_000),
        conn: db.Connection = Depends(get_conn),
    ) -> list[TitleColors]:
        rows = db.search(conn, q, media_type, status, limit, offset)
        return [_row_to_model(r) for r in rows]

    @app.get(
        "/v1/{media_type}/{tmdb_id}",
        response_model=TitleColors,
        dependencies=[Depends(require_key)],
    )
    def get_title(
        media_type: MediaType,
        tmdb_id: int = Path(ge=1),
        conn: db.Connection = Depends(get_conn),
    ) -> TitleColors:
        row = conn.execute(
            f"SELECT {db.TITLE_COLUMNS} FROM titles WHERE media_type = ? AND tmdb_id = ?",
            (media_type, tmdb_id),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Unknown title")
        return _row_to_model(row)

    @app.post("/v1/lookup", response_model=LookupResponse, dependencies=[Depends(require_key)])
    def lookup(req: LookupRequest, conn: db.Connection = Depends(get_conn)) -> LookupResponse:
        found: dict[tuple[str, int], TitleColors] = {}
        for media_type in db.MEDIA_TYPES:
            ids = sorted({i.tmdb_id for i in req.items if i.media_type == media_type})
            if not ids:
                continue
            rows = conn.execute(
                f"SELECT {db.TITLE_COLUMNS} FROM titles WHERE media_type = ? "
                f"AND tmdb_id IN ({','.join('?' * len(ids))})",
                (media_type, *ids),
            )
            for row in rows:
                found[(row["media_type"], row["tmdb_id"])] = _row_to_model(row)
        results, missing, seen = [], [], set()
        for item in req.items:
            key = (item.media_type, item.tmdb_id)
            if key in seen:
                continue
            seen.add(key)
            if key in found:
                results.append(found[key])
            else:
                missing.append(item)
        return LookupResponse(results=results, missing=missing)

    @app.get(
        "/v1/{media_type}",
        response_model=list[TitleColors],
        dependencies=[Depends(require_key)],
        summary="Page through all titles with extracted colours (for bulk sync)",
    )
    def list_titles(
        media_type: MediaType,
        after_id: int = Query(0, ge=0, description="Return titles with tmdb_id greater than this"),
        updated_since: str | None = Query(None, description="ISO-8601 timestamp filter"),
        limit: int = Query(100, ge=1, le=1000),
        conn: db.Connection = Depends(get_conn),
    ) -> list[TitleColors]:
        sql = (
            f"SELECT {db.TITLE_COLUMNS} FROM titles "
            "WHERE media_type = ? AND status = 'done' AND tmdb_id > ?"
        )
        params: list = [media_type, after_id]
        if updated_since:
            sql += " AND updated_at >= ?"
            params.append(updated_since)
        sql += " ORDER BY tmdb_id LIMIT ?"
        params.append(limit)
        return [_row_to_model(r) for r in conn.execute(sql, params)]

    return app
