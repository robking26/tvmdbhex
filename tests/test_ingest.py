import asyncio
import gzip
import json
from datetime import date

import httpx
import pytest
import respx
from conftest import make_poster, to_jpeg

from tvmdbhex import db, ingest
from tvmdbhex.config import Settings
from tvmdbhex.tmdb import TMDBClient, parse_export


def export(lines):
    return gzip.compress("\n".join(json.dumps(l) for l in lines).encode())


MOVIE_EXPORT = export(
    [
        {"adult": False, "id": 550, "original_title": "Fight Club", "popularity": 60.1, "video": False},
        {"adult": False, "id": 551, "original_title": "No Poster", "popularity": 1.0, "video": False},
        {"adult": False, "id": 552, "original_title": "Gone", "popularity": 0.5, "video": False},
        {"adult": False, "id": 553, "original_title": "Same Poster", "popularity": 0.4, "video": False},
        {"adult": True, "id": 554, "original_title": "Adult", "popularity": 9.0, "video": False},
    ]
)
TV_EXPORT = export([{"id": 1399, "original_name": "Game of Thrones", "popularity": 300.0}])
POSTER = to_jpeg(make_poster([((20, 30, 120), 0.6), ((230, 40, 40), 0.25), ((240, 220, 60), 0.15)]))


@pytest.fixture
def settings(db_settings):
    return Settings(**db_settings, tmdb_read_token="t", requests_per_second=0)


@pytest.fixture
def mock_tmdb():
    with respx.mock(assert_all_called=False) as m:
        m.get(url__regex=r".*/exports/movie_ids_.*").respond(content=MOVIE_EXPORT)
        m.get(url__regex=r".*/exports/tv_series_ids_.*").respond(content=TV_EXPORT)
        m.get("https://api.themoviedb.org/3/movie/550").respond(
            json={"id": 550, "title": "Fight Club", "adult": False, "poster_path": "/fc.jpg"}
        )
        m.get("https://api.themoviedb.org/3/movie/551").respond(
            json={"id": 551, "title": "No Poster", "poster_path": None}
        )
        m.get("https://api.themoviedb.org/3/movie/552").respond(status_code=404)
        m.get("https://api.themoviedb.org/3/movie/553").respond(
            json={"id": 553, "title": "Same Poster", "poster_path": "/fc.jpg"}
        )
        m.get("https://api.themoviedb.org/3/tv/1399").respond(
            json={"id": 1399, "name": "Game of Thrones", "poster_path": "/got.jpg"}
        )
        m.get(url__regex=r"https://image\.tmdb\.org/t/p/w185/.*").respond(content=POSTER)
        yield m


def run(coro):
    return asyncio.run(coro)


async def _full(settings):
    conn = db.connect(settings.db_target)
    async with TMDBClient(read_token="t", requests_per_second=0) as client:
        seeded = await ingest.seed(conn, client, ["movie", "tv"])
        counts = await ingest.process(conn, client, settings)
    return conn, seeded, counts


def test_parse_export_keeps_every_title():
    rows = list(parse_export("movie", MOVIE_EXPORT))
    assert [r["id"] for r in rows] == [550, 551, 552, 553, 554]
    assert list(parse_export("tv", TV_EXPORT))[0]["title"] == "Game of Thrones"


def test_seed_and_process(settings, mock_tmdb):
    conn, seeded, counts = run(_full(settings))
    assert seeded == {"movie": 5, "tv": 1}
    assert counts == {"done": 3, "no_poster": 1, "not_found": 1}

    row = conn.execute("SELECT * FROM titles WHERE media_type='movie' AND tmdb_id=550").fetchone()
    assert row["status"] == "done"
    assert row["primary_hex"].startswith("#") and len(row["primary_hex"]) == 7
    assert row["primary_ratio"] > row["secondary_ratio"] > row["tertiary_ratio"]

    # The second title with an identical poster file reuses colours without a download.
    twin = conn.execute("SELECT * FROM titles WHERE tmdb_id=553").fetchone()
    assert twin["primary_hex"] == row["primary_hex"]
    assert mock_tmdb.routes[-1].call_count == 2  # fc.jpg once + got.jpg once

    # Adult titles are skipped unless TVMDBHEX_INCLUDE_ADULT is set.
    assert conn.execute("SELECT status FROM titles WHERE tmdb_id=554").fetchone()[0] == "pending"
    assert db.stats(conn)["tv"] == {"done": 1}


def test_errors_are_recorded_and_retried(settings, mock_tmdb):
    mock_tmdb.get("https://api.themoviedb.org/3/movie/550").respond(status_code=401)

    async def go():
        conn = db.connect(settings.db_target)
        db.upsert_seed(conn, "movie", [{"id": 550}])
        async with TMDBClient(read_token="t", requests_per_second=0) as client:
            first = await ingest.process(conn, client, settings)
            mock_tmdb.get("https://api.themoviedb.org/3/movie/550").respond(
                json={"id": 550, "title": "Fight Club", "poster_path": "/fc.jpg"}
            )
            second = await ingest.process(conn, client, settings)
        return conn, first, second

    conn, first, second = run(go())
    assert first == {"error": 1} and second == {"done": 1}
    assert conn.execute("SELECT attempts FROM titles").fetchone()[0] == 0


def test_rate_limit_is_retried(settings, mock_tmdb, monkeypatch):
    async def no_sleep(_):
        return None

    monkeypatch.setattr("tvmdbhex.tmdb.asyncio.sleep", no_sleep)
    mock_tmdb.get("https://api.themoviedb.org/3/tv/1399").mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "1"}),
            httpx.Response(200, json={"id": 1399, "name": "GoT", "poster_path": "/got.jpg"}),
        ]
    )

    async def go():
        async with TMDBClient(read_token="t", requests_per_second=0) as client:
            return await client.details("tv", 1399)

    assert run(go())["name"] == "GoT"


def test_sync_changes_requeues(settings, mock_tmdb):
    route = mock_tmdb.get("https://api.themoviedb.org/3/movie/changes").respond(
        json={"results": [{"id": 550}, {"id": 9999}], "page": 1, "total_pages": 1}
    )

    async def go():
        conn, _, _ = await _full(settings)
        db.set_state(conn, "changes:movie", "2026-09-01")
        async with TMDBClient(read_token="t", requests_per_second=0) as client:
            counts = await ingest.sync_changes(conn, client, ["movie"], today=date(2026, 9, 27))
        return conn, counts

    conn, counts = run(go())
    assert counts == {"movie": 2}
    assert route.call_count == 2  # 26 days split into 14-day windows
    assert conn.execute("SELECT status FROM titles WHERE tmdb_id=550").fetchone()[0] == "pending"
    assert conn.execute("SELECT status FROM titles WHERE tmdb_id=9999").fetchone()[0] == "pending"
    assert db.get_state(conn, "changes:movie") == "2026-09-27"


def test_pending_alternates_movie_and_tv(settings):
    conn = db.connect(settings.db_target)
    db.upsert_seed(conn, "movie", [{"id": i, "popularity": 10 - i} for i in range(1, 4)])
    db.upsert_seed(conn, "tv", [{"id": 100 + i, "popularity": 1000 - i} for i in range(1, 3)])
    assert db.pending(conn, None, False, True, 5, None) == [
        ("movie", 1), ("tv", 101), ("movie", 2), ("tv", 102), ("movie", 3)
    ]
    assert db.pending(conn, None, False, True, 5, 2) == [("movie", 1), ("tv", 101)]
    assert db.pending(conn, "tv", False, True, 5, None) == [("tv", 101), ("tv", 102)]
