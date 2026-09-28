"""Minimal async TMDB client (used only by the ingester)."""
from __future__ import annotations

import asyncio
import gzip
import json
import time
from collections.abc import Iterator
from datetime import date, timedelta

import httpx

API_BASE = "https://api.themoviedb.org/3"
IMAGE_BASE = "https://image.tmdb.org/t/p"
EXPORT_BASE = "https://files.tmdb.org/p/exports"
EXPORT_NAMES = {"movie": "movie_ids", "tv": "tv_series_ids"}

RETRY_STATUSES = {429, 500, 502, 503, 504}
MAX_RETRIES = 5


class NotFound(Exception):
    pass


class RateLimiter:
    """Evenly spaces requests to stay under TMDB's per-IP limit (~50 req/s)."""

    def __init__(self, per_second: float):
        self.interval = 1.0 / per_second if per_second > 0 else 0.0
        self._next = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        if not self.interval:
            return
        async with self._lock:
            now = time.monotonic()
            delay = self._next - now
            self._next = max(now, self._next) + self.interval
        if delay > 0:
            await asyncio.sleep(delay)


class TMDBClient:
    def __init__(
        self,
        api_key: str = "",
        read_token: str = "",
        requests_per_second: float = 40.0,
        client: httpx.AsyncClient | None = None,
    ):
        if not api_key and not read_token:
            raise ValueError("Set TMDB_READ_TOKEN or TMDB_API_KEY")
        self.api_key = api_key
        self.read_token = read_token
        self.limiter = RateLimiter(requests_per_second)
        self.http = client or httpx.AsyncClient(timeout=30.0, follow_redirects=True)

    async def aclose(self) -> None:
        await self.http.aclose()

    async def __aenter__(self) -> "TMDBClient":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    async def _request(self, url: str, params: dict | None = None, api: bool = True) -> httpx.Response:
        headers = {}
        params = dict(params or {})
        if api:
            if self.read_token:
                headers["Authorization"] = f"Bearer {self.read_token}"
            else:
                params["api_key"] = self.api_key
        for attempt in range(MAX_RETRIES + 1):
            if api:
                await self.limiter.wait()
            try:
                resp = await self.http.get(url, params=params, headers=headers)
            except httpx.TransportError:
                if attempt == MAX_RETRIES:
                    raise
                await asyncio.sleep(2**attempt)
                continue
            if resp.status_code == 404:
                raise NotFound(url)
            if resp.status_code in RETRY_STATUSES and attempt < MAX_RETRIES:
                retry_after = resp.headers.get("Retry-After")
                try:
                    wait = float(retry_after) if retry_after else 2**attempt
                except ValueError:
                    wait = 2**attempt
                await asyncio.sleep(min(wait, 60))
                continue
            resp.raise_for_status()
            return resp
        raise RuntimeError("unreachable")

    async def details(self, media_type: str, tmdb_id: int, with_images: bool = False) -> dict:
        """Title details. with_images also returns the title's logos (official
        title treatments as transparent PNGs) in the same request."""
        params = {"append_to_response": "images", "include_image_language": "en,null"} if with_images else None
        resp = await self._request(f"{API_BASE}/{media_type}/{tmdb_id}", params)
        return resp.json()

    async def poster(self, poster_path: str, size: str = "w185") -> bytes:
        resp = await self._request(f"{IMAGE_BASE}/{size}{poster_path}", api=False)
        return resp.content

    async def image(self, path: str, size: str) -> bytes:
        resp = await self._request(f"{IMAGE_BASE}/{size}{path}", api=False)
        return resp.content

    async def changes(self, media_type: str, start: date, end: date) -> set[int]:
        """IDs changed between start and end (TMDB allows at most 14 days per call)."""
        ids: set[int] = set()
        page, total_pages = 1, 1
        while page <= total_pages:
            resp = await self._request(
                f"{API_BASE}/{media_type}/changes",
                {"start_date": start.isoformat(), "end_date": end.isoformat(), "page": page},
            )
            body = resp.json()
            ids.update(item["id"] for item in body.get("results", []) if item.get("id"))
            total_pages = body.get("total_pages") or 1
            page += 1
        return ids

    async def daily_export(self, media_type: str, max_days_back: int = 3) -> tuple[date, bytes]:
        """Download the newest daily ID export (published ~08:00 UTC each day)."""
        today = date.today()
        last_error: Exception | None = None
        for back in range(max_days_back + 1):
            day = today - timedelta(days=back)
            url = f"{EXPORT_BASE}/{EXPORT_NAMES[media_type]}_{day:%m_%d_%Y}.json.gz"
            try:
                resp = await self._request(url, api=False)
                return day, resp.content
            except (NotFound, httpx.HTTPStatusError) as exc:  # not published yet
                last_error = exc
        raise RuntimeError(f"No {media_type} export found in the last {max_days_back} days") from last_error


def parse_export(media_type: str, gz_bytes: bytes) -> Iterator[dict]:
    """Yield {id, title, adult, popularity} from a gzip'd JSON-lines export."""
    title_key = "original_title" if media_type == "movie" else "original_name"
    for line in gzip.decompress(gz_bytes).splitlines():
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        yield {
            "id": item["id"],
            "title": item.get(title_key),
            "adult": item.get("adult", False),
            "popularity": item.get("popularity"),
        }


def pick_logo(details: dict) -> str | None:
    """Best title-treatment logo from a details response fetched with_images:
    English first, then language-neutral; PNG only (TMDB also has SVG logos,
    which Pillow can't read); highest-voted, then widest."""
    logos = (details.get("images") or {}).get("logos") or []
    usable = [lg for lg in logos if str(lg.get("file_path", "")).lower().endswith(".png")]
    if not usable:
        return None
    lang_rank = {"en": 0, None: 1}
    usable.sort(
        key=lambda lg: (
            lang_rank.get(lg.get("iso_639_1"), 2),
            -(lg.get("vote_average") or 0),
            -(lg.get("width") or 0),
        )
    )
    return usable[0]["file_path"]
