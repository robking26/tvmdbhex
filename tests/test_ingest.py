import asyncio
import gzip
import json
from datetime import date

import httpx
import pytest
import respx
from conftest import make_logo_png, make_poster, to_jpeg

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
LOGO = make_logo_png()
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
            json={
                "id": 550, "title": "Fight Club", "adult": False, "poster_path": "/fc.jpg",
                "images": {"logos": [
                    {"file_path": "/fc_logo.svg", "iso_639_1": "en", "vote_average": 9},
                    {"file_path": "/fc_logo.png", "iso_639_1": "en", "vote_average": 5},
                ]},
            }
        )
        m.get(url__regex=r"https://image\.tmdb\.org/t/p/w300/fc_logo\.png").respond(content=LOGO)
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
        m.get(url__regex=r"https://image\.tmdb\.org/t/p/w\d+/(fc|got)\.jpg").respond(content=POSTER)
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
    # Six-role palette stored; the TMDB logo (PNG preferred over SVG) was used.
    assert all(row[f"{r}_hex"] and len(row[f"{r}_hex"]) == 7 for r in db.SEMANTIC_ROLES)
    assert row["logo_path"] == "/fc_logo.png"
    assert row["palette_version"] == 3
    # Legacy colours are derived: primary = identity1, secondary = base, tertiary = accent.
    assert (row["primary_hex"], row["secondary_hex"], row["tertiary_hex"]) == (
        row["identity1_hex"], row["base_hex"], row["accent_hex"]
    )

    # Same poster file, but 553 has no logo: its colours are computed separately
    # (the logo feeds identity), so fc.jpg is downloaded once per poster+logo pair.
    twin = conn.execute("SELECT * FROM titles WHERE tmdb_id=553").fetchone()
    assert twin["logo_path"] is None and twin["base_hex"] == row["base_hex"]
    posters = next(r for r in mock_tmdb.routes if "fc|got" in str(r.pattern))
    assert posters.call_count == 3  # fc.jpg with logo, fc.jpg without, got.jpg

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
        json={"results": [{"id": 550}, {"id": 551}, {"id": 9999}], "page": 1, "total_pages": 1}
    )

    async def go():
        conn, _, _ = await _full(settings)
        db.set_state(conn, "changes:movie", "2026-09-01")
        async with TMDBClient(read_token="t", requests_per_second=0) as client:
            counts = await ingest.sync_changes(conn, client, ["movie"], today=date(2026, 9, 27))
        return conn, counts

    conn, counts = run(go())
    assert route.call_count == 2  # 26 days split into 14-day windows
    row = lambda i: conn.execute("SELECT status, palette_version FROM titles WHERE tmdb_id=?", (i,)).fetchone()  # noqa: E731
    # Coloured title keeps serving colours but is flagged for re-colouring.
    assert tuple(row(550)) == ("done", None)
    assert db.pending(conn, None, False, True, 5, None, palette_version=2)[:1] == [("movie", 550)]
    assert row(551)[0] == "pending"  # no_poster -> retried
    assert row(9999) is None  # unknown IDs aren't inserted (seed handles new titles)
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


def test_pending_top_caps_by_rank_not_batch(settings):
    conn = db.connect(settings.db_target)
    db.upsert_seed(conn, "movie", [{"id": i, "popularity": 10 - i} for i in range(1, 4)])
    db.upsert_seed(conn, "tv", [{"id": 100 + i, "popularity": 1000 - i} for i in range(1, 3)])
    db.save_result(conn, "movie", 1, status="done", palette=[("#000000", 1.0)] * 3)
    # Ranking: movie 1 (done), tv 101, movie 2, tv 102, movie 3. Top 3 leaves 2 to do.
    assert db.pending(conn, None, False, True, 5, None, top=3) == [("tv", 101), ("movie", 2)]
    assert db.pending(conn, None, False, True, 5, 1, top=3) == [("tv", 101)]


def test_older_palettes_are_recoloured(settings):
    conn = db.connect(settings.db_target)
    db.upsert_seed(conn, "movie", [{"id": 1, "popularity": 9}, {"id": 2, "popularity": 8}])
    pal = [("#000000", 1.0)] * 3
    db.save_result(conn, "movie", 1, status="done", poster_path="/a.jpg", palette=pal)  # legacy: no version
    db.save_result(conn, "movie", 2, status="done", poster_path="/b.jpg", palette=pal, palette_version=2)
    assert db.pending(conn, None, False, True, 5, None) == []
    assert db.pending(conn, None, False, True, 5, None, palette_version=2) == [("movie", 1)]
    assert set(db.poster_palettes(conn, 2)) == {"/b.jpg|"}


def test_migration_adds_palette_version(tmp_path):
    import sqlite3

    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.executescript(db.SCHEMA.replace("    palette_version INTEGER,\n", ""))
    old.close()
    conn = db.connect(str(path))
    assert "palette_version" in {r[1] for r in conn.execute("PRAGMA table_info(titles)")}



def test_seed_with_top_stores_only_top_titles(settings, mock_tmdb):
    async def go():
        conn = db.connect(settings.db_target)
        async with TMDBClient(read_token="t", requests_per_second=0) as client:
            counts = await ingest.seed(conn, client, ["movie", "tv"], top=3)
        return conn, counts

    conn, counts = run(go())
    # Ranking: movie 550 (60.1), tv 1399, movie 551 (1.0); adult 554 is excluded.
    assert counts == {"movie": 2, "tv": 1}
    assert {(r[0], r[1]) for r in conn.execute("SELECT media_type, tmdb_id FROM titles")} == {
        ("movie", 550), ("tv", 1399), ("movie", 551)
    }


def test_prune_deletes_uncoloured_titles_below_cap(settings):
    conn = db.connect(settings.db_target)
    db.upsert_seed(conn, "movie", [{"id": i, "popularity": 100 - i} for i in range(1, 7)])
    db.upsert_seed(conn, "movie", [{"id": 50, "popularity": 1, "adult": True}])
    # A coloured title whose rank has slipped below the cap is kept.
    db.save_result(conn, "movie", 6, status="done", palette=[("#000000", 1.0)] * 3, palette_version=2)
    assert db.prune(conn, top=3) == 3  # movies 4, 5 and adult 50
    left = sorted(r[0] for r in conn.execute("SELECT tmdb_id FROM titles"))
    assert left == [1, 2, 3, 6]
    assert db.prune(conn, top=3) == 0


def test_run_top_never_shrinks_the_catalogue(tmp_path, mock_tmdb, monkeypatch, capsys):
    """Regression: `run --top 1` once pruned every uncoloured title below rank 1."""
    import json as _json

    from tvmdbhex import cli

    path = str(tmp_path / "run.db")
    monkeypatch.setenv("TVMDBHEX_DB_PATH", path)
    monkeypatch.setenv("TMDB_READ_TOKEN", "t")
    monkeypatch.setenv("TVMDBHEX_REQUESTS_PER_SECOND", "0")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("TVMDBHEX_CATALOGUE_SIZE", "10")
    mock_tmdb.get(url__regex=r".*/(movie|tv)/changes.*").respond(json={"results": [], "total_pages": 1})

    cli.main(["run", "--top", "1"])
    out = _json.loads(capsys.readouterr().out)
    assert out["pruned"] == 0
    conn = db.connect(path)
    # All 5 non-adult titles stay stored; only the top-ranked one was processed.
    assert conn.execute("SELECT COUNT(*) FROM titles").fetchone()[0] == 5
    assert out["process"] == {"done": 1}
