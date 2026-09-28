"""Read-only HTTP API over the colour database. Never calls TMDB."""
from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from importlib import resources
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Path, Query, Request
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


class SemanticPalette(BaseModel):
    """Six-role palette. Approximate prominence in generated artwork: base
    40-50%, identity1 25-30%, identity2 10-20%, highlight1 10-20%,
    highlight2 10-20%, accent 3-8%."""

    base: str = Field(examples=["#0D83C2"], description="Upper-corner environment of the poster")
    identity1: str = Field(examples=["#E91919"], description="Main title/logo colour, else strongest artwork colour")
    identity2: str = Field(examples=["#FFE65A"], description="Second logo colour, else next strongest artwork colour")
    highlight1: str = Field(examples=["#937279"], description="Strongest artwork colour not used above")
    highlight2: str = Field(examples=["#CC9B6B"], description="Next strongest unused artwork colour")
    accent: str = Field(examples=["#653178"], description="Next unused artwork colour, leaning vivid")


class TitleColors(BaseModel):
    rank: int | None = Field(None, description="Popularity rank (only in /v1/top)")
    media_type: MediaType
    tmdb_id: int
    title: str | None
    poster_path: str | None
    logo_path: str | None = Field(None, description="TMDB title logo used for identity colours, if any")
    status: str = Field(description="done | pending | no_poster | not_found | error")
    palette: SemanticPalette | None = Field(
        None, description="Six-role semantic palette; null until the title is coloured by algorithm v3"
    )
    colors: Colors | None = Field(None, description="Legacy 3-colour palette (derived from `palette` for v3 titles)")
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
    # All six palette columns are nullable and `status` is not a promise about them: the
    # schema lets a row be `done` with any subset present. Testing `primary_hex` alone was
    # enough while writes came from `save_results`, which stores three swatches or none —
    # but a row with two, or with a hex and no ratio, made `Swatch` raise, and an
    # exception here is a 500 for whatever asked. `/v1/lookup` asks about hundreds at a
    # time, so one such row took the whole batch with it.
    #
    # `colors: null` is what this returns instead, which is a documented answer rather
    # than a repair: it is the same thing a pending title or one with no poster gets.
    swatches = [
        (row["primary_hex"], row["primary_ratio"]),
        (row["secondary_hex"], row["secondary_ratio"]),
        (row["tertiary_hex"], row["tertiary_ratio"]),
    ]
    colors = None
    if row["status"] == db.DONE and all(hex_ and ratio is not None for hex_, ratio in swatches):
        primary, secondary, tertiary = (Swatch(hex=h, ratio=r) for h, r in swatches)
        colors = Colors(primary=primary, secondary=secondary, tertiary=tertiary)
    # Same rule for the six-role palette: all six stored, or null.
    roles = {role: row[f"{role}_hex"] for role in db.SEMANTIC_ROLES}
    palette = SemanticPalette(**roles) if row["status"] == db.DONE and all(roles.values()) else None
    return TitleColors(
        media_type=row["media_type"],
        tmdb_id=row["tmdb_id"],
        title=row["title"],
        poster_path=row["poster_path"],
        logo_path=row["logo_path"],
        status=row["status"],
        palette=palette,
        colors=colors,
        updated_at=row["updated_at"],
    )


log = logging.getLogger("tvmdbhex.api")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    target = settings.db_target
    postgres = db.is_postgres_url(target)
    if not postgres:
        db.connect(target).close()  # create the file/schema if missing

    app = FastAPI(
        title="tvmdbhex",
        version=__version__,
        description="Six-role semantic palettes (base, identity1, identity2, highlight1, highlight2, accent) "
        "for TMDB movies and TV series, from each poster's title treatment and artwork.",
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
        "/v1/top",
        response_model=list[TitleColors],
        dependencies=[Depends(require_key)],
        summary="Most popular titles with colours, ranked (movies and TV alternating)",
    )
    def top(
        media_type: MediaType | None = None,
        limit: int = Query(60, ge=1, le=200),
        offset: int = Query(0, ge=0, le=10_000),
        conn: db.Connection = Depends(get_conn),
    ) -> list[TitleColors]:
        out = []
        for rank, row in db.top_titles(conn, media_type, limit, offset):
            model = _row_to_model(row)
            model.rank = rank
            out.append(model)
        return out

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
                key = (row["media_type"], row["tmdb_id"])
                try:
                    found[key] = _row_to_model(row)
                except Exception as exc:
                    # **One row must never cost the batch**, which is the fault that
                    # brought this endpoint down: a caller asking about a whole
                    # filmography got a 500 and nothing, where the one title at fault
                    # should simply have come back as missing. `_row_to_model` above no
                    # longer raises on the case that did it; this is the guarantee rather
                    # than the repair, and it names the row instead of swallowing it.
                    log.warning("lookup: skipping %s %s — %s", *key, exc)
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

    if settings.debug:
        _add_debug_routes(app, require_key)
    return app


MAX_DEBUG_IMAGE_BYTES = 10 * 1024 * 1024


def _add_debug_routes(app: FastAPI, require_key) -> None:
    """Palette tuning tools, only mounted when TVMDBHEX_DEBUG is on.

    Image libraries are imported lazily so the production API stays lean.
    """
    debug_html = resources.files("tvmdbhex").joinpath("static/debug.html").read_text("utf-8")

    @app.get("/debug", response_class=HTMLResponse, include_in_schema=False)
    def debug_page() -> str:
        return debug_html

    image_path = r"^/[A-Za-z0-9_.-]+\.(jpg|jpeg|png|webp)$"

    def fetch(path: str, size: str) -> bytes:
        import httpx

        try:
            resp = httpx.get(f"https://image.tmdb.org/t/p/{size}{path}", timeout=20, follow_redirects=True)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"Could not fetch {path}: {exc}") from exc
        return resp.content

    @app.post("/v1/debug/palette", dependencies=[Depends(require_key)], tags=["debug"])
    async def debug_palette_upload(
        request: Request,
        logo_path: str | None = Query(None, pattern=image_path, description="TMDB logo to use for the title treatment"),
    ) -> dict:
        """Analyse an uploaded poster (raw image bytes as the request body):
        six-role palette with logo location, logo/artwork queues and the
        dominant ladder, plus the previous algorithms for comparison."""
        from .palette_debug import compare_bytes

        data = await request.body()
        if not data or len(data) > MAX_DEBUG_IMAGE_BYTES:
            raise HTTPException(status_code=400, detail="Send an image (max 10 MB) as the request body")
        logo = fetch(logo_path, "w300") if logo_path else None
        try:
            return compare_bytes(data, logo)
        except Exception as exc:  # unreadable image
            raise HTTPException(status_code=400, detail=f"Could not analyse image: {exc}") from exc

    @app.get("/v1/debug/palette", dependencies=[Depends(require_key)], tags=["debug"])
    def debug_palette_tmdb(
        poster_path: str = Query(pattern=image_path),
        logo_path: str | None = Query(None, pattern=image_path),
    ) -> dict:
        """Analyse a TMDB poster (and optionally its logo) fetched from TMDB's image CDN."""
        from .palette_debug import compare_bytes

        return compare_bytes(fetch(poster_path, "w342"), fetch(logo_path, "w300") if logo_path else None)
