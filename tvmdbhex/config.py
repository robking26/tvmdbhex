from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

_PG_URL = re.compile(r"postgres(?:ql)?://\S+")


def parse_database_url(raw: str | None) -> str:
    """Normalise a DATABASE_URL value; '' if unset.

    Tolerates values pasted from a dashboard: surrounding whitespace/quotes, a
    `DATABASE_URL=` prefix, or a whole .env snippet (the DATABASE_URL line wins).
    Anything else raises, rather than silently falling back to a local SQLite file.
    """
    value = (raw or "").strip()
    if not value:
        return ""
    candidates = [value]
    for line in value.splitlines():
        key, sep, rest = line.strip().partition("=")
        if sep and key.strip() in ("DATABASE_URL", "POSTGRES_URL"):
            candidates.insert(0, rest)
    for candidate in candidates:
        match = _PG_URL.search(candidate.strip().strip("'\""))
        if match:
            return match.group(0).rstrip("'\"")
    raise ValueError(
        "DATABASE_URL is set but doesn't contain a postgres:// or postgresql:// URL. "
        "Set it to just the connection string, e.g. postgresql://user:pass@host/db?sslmode=require"
    )


def _bool(value: str | None, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    db_path: str = "data/tvmdbhex.db"
    database_url: str = ""
    tmdb_api_key: str = ""
    tmdb_read_token: str = ""
    api_keys: frozenset[str] = field(default_factory=frozenset)
    concurrency: int = 16
    requests_per_second: float = 40.0
    poster_size: str = "w185"
    include_adult: bool = False
    max_titles: int | None = None  # only ever process the N most popular titles

    @property
    def db_target(self) -> str:
        """Postgres URL when configured, otherwise the local SQLite path."""
        return self.database_url or self.db_path

    @classmethod
    def from_env(cls) -> "Settings":
        keys = os.environ.get("TVMDBHEX_API_KEYS", "")
        return cls(
            db_path=os.environ.get("TVMDBHEX_DB_PATH", cls.db_path),
            # Vercel's Neon/Postgres integrations set DATABASE_URL / POSTGRES_URL.
            database_url=parse_database_url(
                os.environ.get("DATABASE_URL") or os.environ.get("POSTGRES_URL")
            ),
            tmdb_api_key=os.environ.get("TMDB_API_KEY", ""),
            tmdb_read_token=os.environ.get("TMDB_READ_TOKEN", ""),
            api_keys=frozenset(k.strip() for k in keys.split(",") if k.strip()),
            concurrency=int(os.environ.get("TVMDBHEX_CONCURRENCY", cls.concurrency)),
            requests_per_second=float(
                os.environ.get("TVMDBHEX_REQUESTS_PER_SECOND", cls.requests_per_second)
            ),
            poster_size=os.environ.get("TVMDBHEX_POSTER_SIZE", cls.poster_size),
            include_adult=_bool(os.environ.get("TVMDBHEX_INCLUDE_ADULT"), cls.include_adult),
            max_titles=int(os.environ["TVMDBHEX_MAX_TITLES"]) if os.environ.get("TVMDBHEX_MAX_TITLES") else None,
        )
