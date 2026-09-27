"""Vercel entry point: serves the website and API as one serverless function.

vercel.json routes every path to this file via an explicit @vercel/python
build, so it works regardless of the project's framework preset. The database is Postgres
(DATABASE_URL / POSTGRES_URL, set by Vercel's Neon integration); the
ingester runs separately as a GitHub Actions workflow.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

from tvmdbhex.api import create_app  # noqa: E402
from tvmdbhex.config import Settings  # noqa: E402


def _not_configured() -> FastAPI:
    # Vercel's filesystem is read-only, so there's no SQLite fallback here.
    fallback = FastAPI()

    @fallback.api_route("/{path:path}", methods=["GET", "POST"])
    def not_configured(path: str) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={
                "error": "Database not configured",
                "fix": "In Vercel: Storage -> Create Database -> Neon (Postgres), connect it to "
                "this project (sets DATABASE_URL), then redeploy.",
            },
        )

    return fallback


settings = Settings.from_env()

# Vercel finds the entrypoint by looking for a top-level `app` assignment.
app = create_app(settings) if settings.database_url else _not_configured()
