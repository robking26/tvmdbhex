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

settings = Settings.from_env()

if settings.database_url:
    app = create_app(settings)
else:
    # Vercel's filesystem is read-only, so there's no SQLite fallback here.
    app = FastAPI()

    @app.api_route("/{path:path}", methods=["GET", "POST"])
    def not_configured(path: str) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={
                "error": "Database not configured",
                "fix": "In Vercel: Storage -> Create Database -> Neon (Postgres), connect it to "
                "this project (sets DATABASE_URL), then redeploy.",
            },
        )
