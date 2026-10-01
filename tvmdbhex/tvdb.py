"""Minimal async TheTVDB v4 client (used only by the TheTVDB ingester).

TheTVDB is the source of titles and artwork for the colours Hoozat uses: its
credits are keyed by TheTVDB IDs, so colours keyed the same way need no ID
mapping. Licence: free under $50k a year in revenue, with attribution.
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import httpx

from .tmdb import MAX_RETRIES, RETRY_STATUSES, NotFound, RateLimiter

API_BASE = "https://api4.thetvdb.com/v4"
ARTWORK_BASE = "https://artworks.thetvdb.com"
KINDS = ("series", "movie")  # TheTVDB record types; the list endpoints are /series and /movies
LIST_PATH = {"series": "series", "movie": "movies"}


def absolute(url: str | None) -> str | None:
    """Artwork URLs are usually absolute, but older records give a path."""
    if not url:
        return None
    if url.startswith("http"):
        return url
    return f"{ARTWORK_BASE}{url if url.startswith('/') else '/' + url}"


def thumbnail(url: str) -> str:
    """TheTVDB keeps a smaller copy of each artwork beside it, named <name>_t.<ext>."""
    stem, dot, ext = url.rpartition(".")
    return f"{stem}_t.{ext}" if dot and "/" not in ext else url


class TVDBClient:
    def __init__(
        self,
        api_key: str,
        pin: str = "",
        requests_per_second: float = 20.0,
        client: httpx.AsyncClient | None = None,
    ):
        if not api_key:
            raise ValueError("Set TVDB_API_KEY")
        self.api_key = api_key
        self.pin = pin
        self.limiter = RateLimiter(requests_per_second)
        self.http = client or httpx.AsyncClient(timeout=30.0, follow_redirects=True)
        self._token: str | None = None
        self._login_lock = asyncio.Lock()
        self._logo_types: dict[str, set[int]] | None = None

    async def aclose(self) -> None:
        await self.http.aclose()

    async def __aenter__(self) -> "TVDBClient":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    async def _login(self) -> str:
        async with self._login_lock:
            if self._token:
                return self._token
            body = {"apikey": self.api_key, **({"pin": self.pin} if self.pin else {})}
            resp = await self.http.post(f"{API_BASE}/login", json=body)
            resp.raise_for_status()
            self._token = resp.json()["data"]["token"]
            return self._token

    async def _get(self, url: str, params: dict | None = None, api: bool = True) -> httpx.Response:
        for attempt in range(MAX_RETRIES + 1):
            headers = {}
            if api:
                await self.limiter.wait()
                headers["Authorization"] = f"Bearer {self._token or await self._login()}"
            try:
                resp = await self.http.get(url, params=params, headers=headers)
            except httpx.TransportError:
                if attempt == MAX_RETRIES:
                    raise
                await asyncio.sleep(2**attempt)
                continue
            if resp.status_code == 401 and api and attempt == 0:
                self._token = None  # expired: log in again once
                continue
            if resp.status_code == 404:
                raise NotFound(url)
            if resp.status_code in RETRY_STATUSES and attempt < MAX_RETRIES:
                await asyncio.sleep(min(2**attempt, 60))
                continue
            resp.raise_for_status()
            return resp
        raise RuntimeError("unreachable")

    async def titles(self, kind: str) -> AsyncIterator[dict]:
        """Every title of a kind: {id, title, score, image}, a page (500) at a time."""
        page = 0
        while True:
            body = (await self._get(f"{API_BASE}/{LIST_PATH[kind]}", {"page": page})).json()
            for item in body.get("data") or []:
                if item.get("id"):
                    yield {
                        "id": int(item["id"]),
                        "title": item.get("name"),
                        "score": item.get("score"),
                        "image": absolute(item.get("image")),
                    }
            if not (body.get("links") or {}).get("next") or not body.get("data"):
                return
            page += 1

    async def logo_type_ids(self) -> dict[str, set[int]]:
        """Artwork type IDs that are title treatments (ClearLogo), per record type."""
        if self._logo_types is None:
            types = (await self._get(f"{API_BASE}/artwork/types")).json().get("data") or []
            out: dict[str, set[int]] = {k: set() for k in KINDS}
            for t in types:
                name = f"{t.get('name', '')} {t.get('slug', '')}".lower().replace(" ", "")
                kind = str(t.get("recordType", "")).lower()
                if "clearlogo" in name and kind in out:
                    out[kind].add(int(t["id"]))
            self._logo_types = out
        return self._logo_types

    async def logo(self, kind: str, tvdb_id: int) -> str | None:
        """Best title-treatment logo URL: English first, then language-neutral; highest score."""
        types = (await self.logo_type_ids()).get(kind) or set()
        if not types:
            return None
        if kind == "series":
            url, params = f"{API_BASE}/series/{tvdb_id}/artworks", {"lang": "eng"}
        else:  # movies have no artworks endpoint; the extended record carries them
            url, params = f"{API_BASE}/movies/{tvdb_id}/extended", None
        try:
            data = (await self._get(url, params)).json().get("data") or {}
        except NotFound:
            return None
        logos = [a for a in data.get("artworks") or [] if a.get("type") in types and a.get("image")]
        if not logos:
            return None
        lang_rank = {"eng": 0, None: 1, "": 1}
        logos.sort(key=lambda a: (lang_rank.get(a.get("language"), 2), -(a.get("score") or 0)))
        return absolute(logos[0]["image"])

    async def image(self, url: str, prefer_thumbnail: bool = True) -> bytes:
        """Artwork bytes, from the small copy when there is one."""
        if prefer_thumbnail:
            try:
                return (await self._get(thumbnail(url), api=False)).content
            except (NotFound, httpx.HTTPStatusError):
                pass
        return (await self._get(url, api=False)).content
