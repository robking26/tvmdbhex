from __future__ import annotations

import os
from dataclasses import dataclass, field


def _bool(value: str | None, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    db_path: str = "data/tvmdbhex.db"
    tmdb_api_key: str = ""
    tmdb_read_token: str = ""
    api_keys: frozenset[str] = field(default_factory=frozenset)
    concurrency: int = 16
    requests_per_second: float = 40.0
    poster_size: str = "w185"
    include_adult: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        keys = os.environ.get("TVMDBHEX_API_KEYS", "")
        return cls(
            db_path=os.environ.get("TVMDBHEX_DB_PATH", cls.db_path),
            tmdb_api_key=os.environ.get("TMDB_API_KEY", ""),
            tmdb_read_token=os.environ.get("TMDB_READ_TOKEN", ""),
            api_keys=frozenset(k.strip() for k in keys.split(",") if k.strip()),
            concurrency=int(os.environ.get("TVMDBHEX_CONCURRENCY", cls.concurrency)),
            requests_per_second=float(
                os.environ.get("TVMDBHEX_REQUESTS_PER_SECOND", cls.requests_per_second)
            ),
            poster_size=os.environ.get("TVMDBHEX_POSTER_SIZE", cls.poster_size),
            include_adult=_bool(os.environ.get("TVMDBHEX_INCLUDE_ADULT"), cls.include_adult),
        )
