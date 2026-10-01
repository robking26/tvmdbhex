"""The TheTVDB pipeline: seed -> colour -> export to Hoozat, with every service mocked."""
import asyncio
import json

import httpx
import pytest
import respx
from conftest import make_logo_png, make_poster, to_jpeg

from tvmdbhex import tvdb_ingest
from tvmdbhex.tvdb import TVDBClient, absolute, thumbnail

API = "https://api4.thetvdb.com/v4"
ART = "https://artworks.thetvdb.com/banners/v4"
HOOZAT = "https://hoozat.example/v1/colours"
POSTER = to_jpeg(make_poster([((20, 30, 120), 0.6), ((230, 40, 40), 0.25), ((240, 220, 60), 0.15)]))
LOGO = make_logo_png()


def run(coro):
    return asyncio.run(coro)


def listing(kind_path, pages):
    """Mock /series or /movies paging: pages is a list of lists of records."""
    def handler(request):
        page = int(request.url.params.get("page", 0))
        data = pages[page] if page < len(pages) else []
        nxt = f"{API}/{kind_path}?page={page + 1}" if page + 1 < len(pages) else None
        return httpx.Response(200, json={"status": "success", "data": data, "links": {"next": nxt}})
    return handler


@pytest.fixture
def tvdb():
    """Two series (one paged), two movies; one series has no poster, one movie's file is gone."""
    state = {"series_pages": [
        [{"id": 81189, "name": "Breaking Bad", "score": 900, "image": f"{ART}/series/81189/posters/bb.jpg"}],
        [{"id": 1, "name": "No Poster", "score": 5, "image": None}],
    ], "movie_pages": [[
        {"id": 13, "name": "Forrest Gump", "score": 800, "image": "/banners/v4/movie/13/posters/fg.jpg"},
        {"id": 99, "name": "Missing File", "score": 1, "image": f"{ART}/movie/99/posters/gone.jpg"},
    ]]}
    with respx.mock(assert_all_called=False) as m:
        m.post(f"{API}/login").respond(json={"data": {"token": "tok"}})
        m.get(f"{API}/series").mock(side_effect=lambda r: listing("series", state["series_pages"])(r))
        m.get(f"{API}/movies").mock(side_effect=lambda r: listing("movies", state["movie_pages"])(r))
        m.get(f"{API}/artwork/types").respond(json={"data": [
            {"id": 2, "name": "Poster", "recordType": "series"},
            {"id": 23, "name": "ClearLogo", "recordType": "series"},
            {"id": 25, "name": "ClearLogo", "recordType": "movie"},
        ]})
        m.get(f"{API}/series/81189/artworks").respond(json={"data": {"artworks": [
            {"type": 23, "language": "spa", "score": 99, "image": f"{ART}/series/81189/clearlogo/es.png"},
            {"type": 23, "language": "eng", "score": 10, "image": f"{ART}/series/81189/clearlogo/en.png"},
        ]}})
        m.get(f"{API}/movies/13/extended").respond(json={"data": {"artworks": []}})
        m.get(f"{API}/movies/99/extended").respond(json={"data": {"artworks": []}})
        m.get(f"{ART}/series/81189/posters/bb_t.jpg").respond(content=POSTER)
        m.get(f"{ART}/series/81189/clearlogo/en.png").respond(content=LOGO)
        m.get(f"{ART}/movie/13/posters/fg_t.jpg").respond(status_code=404)  # no small copy: use the full one
        m.get(f"{ART}/movie/13/posters/fg.jpg").respond(content=POSTER)
        m.get(url__regex=r".*/gone(_t)?\.jpg").respond(status_code=404)
        state["mock"] = m
        yield state


def test_urls():
    assert absolute("/banners/x.jpg") == "https://artworks.thetvdb.com/banners/x.jpg"
    assert absolute(None) is None
    assert thumbnail("https://a/b/c.jpg") == "https://a/b/c_t.jpg"


def test_seed_colour_and_export(tvdb, tmp_path):
    conn = tvdb_ingest.connect(str(tmp_path / "t.db"))
    sent = []

    def hoozat(request):
        assert request.headers["X-Colours-Write-Key"] == "wk"
        sent.extend(json.loads(request.content)["items"])
        return httpx.Response(200, json={"written": 1})

    tvdb["mock"].put(HOOZAT).mock(side_effect=hoozat)

    async def go():
        async with TVDBClient("k", requests_per_second=0) as client:
            seeded = await tvdb_ingest.seed(conn, client)
            counts = await tvdb_ingest.process(conn, client, concurrency=4)
        exported = await tvdb_ingest.export(conn, HOOZAT, "wk")
        return seeded, counts, exported

    seeded, counts, exported = run(go())
    assert seeded["series"]["listed"] == 2 and seeded["movie"]["listed"] == 2
    assert counts == {"done": 2, "no_poster": 2}

    row = conn.execute("SELECT * FROM tvdb_titles WHERE kind = 'series' AND tvdb_id = 81189").fetchone()
    assert row["status"] == "done" and row["logo"].endswith("/en.png"), "English logo wins over a higher-scored Spanish one"
    assert all(row[r].startswith("#") and len(row[r]) == 7 for r in tvdb_ingest.ROLES)

    keys = {i["key"]: i["colours"] for i in sent}
    assert set(keys) == {"series:81189", "movie:13", "series:1", "movie:99"}
    assert keys["series:81189"]["base"].startswith("#")
    assert keys["series:1"] is None and keys["movie:99"] is None, "no colours: Hoozat deletes any it had"
    assert exported == {"sent": 4, "remaining": 0}

    # Nothing changed: a second run colours and sends nothing.
    sent.clear()
    seeded, counts, exported = run(go())
    assert counts == {} and exported["sent"] == 0


def test_new_poster_recolours_and_removed_titles_are_deleted(tvdb, tmp_path):
    conn = tvdb_ingest.connect(str(tmp_path / "t.db"))
    tvdb["mock"].put(HOOZAT).respond(json={})

    async def cycle():
        async with TVDBClient("k", requests_per_second=0) as client:
            await tvdb_ingest.seed(conn, client)
            counts = await tvdb_ingest.process(conn, client, concurrency=2)
        await tvdb_ingest.export(conn, HOOZAT, "wk")
        return counts

    run(cycle())
    # Breaking Bad gets a new poster; Forrest Gump disappears from TheTVDB.
    tvdb["series_pages"][0][0]["image"] = f"{ART}/series/81189/posters/bb2.jpg"
    tvdb["mock"].get(f"{ART}/series/81189/posters/bb2_t.jpg").respond(content=POSTER)
    tvdb["movie_pages"][0] = [tvdb["movie_pages"][0][1]]

    async def seed_only():
        async with TVDBClient("k", requests_per_second=0) as client:
            return await tvdb_ingest.seed(conn, client)

    seeded = run(seed_only())
    assert seeded["movie"]["gone"] == 1
    gump = conn.execute("SELECT status, dirty FROM tvdb_titles WHERE kind = 'movie' AND tvdb_id = 13").fetchone()
    assert tuple(gump) == ("not_found", 1), "a title with colours that disappears is deleted from Hoozat"
    bb = conn.execute("SELECT status, palette_version FROM tvdb_titles WHERE tvdb_id = 81189").fetchone()
    assert bb["status"] == "done" and bb["palette_version"] is None, "keeps serving old colours until recoloured"
    assert [r["tvdb_id"] for r in tvdb_ingest.pending(conn)] == [81189]


def test_export_respects_the_write_budget_and_stops_on_refusal(tmp_path):
    conn = tvdb_ingest.connect(str(tmp_path / "t.db"))
    with conn:
        conn.executemany(
            "INSERT INTO tvdb_titles (kind, tvdb_id, score, status, dirty, updated_at) VALUES ('movie', ?, ?, 'no_poster', 1, 'x')",
            [(i, i) for i in range(1, 1201)],
        )
    with respx.mock() as m:
        m.put(HOOZAT).respond(json={})
        out = run(tvdb_ingest.export(conn, HOOZAT, "wk", max_writes=700))
        assert out == {"sent": 700, "remaining": 500}
        assert m.calls.call_count == 2, "batches of 500"
        first = json.loads(m.calls[0].request.content)["items"][0]["key"]
        assert first == "movie:1200", "best-scored first"
    with respx.mock() as m:
        m.put(HOOZAT).respond(status_code=401, text="wrong key")
        with pytest.raises(RuntimeError, match="401"):
            run(tvdb_ingest.export(conn, HOOZAT, "bad"))
    assert tvdb_ingest.stats(conn)["unsent"] == 500, "nothing marked sent when Hoozat refuses"


def test_pending_alternates_series_and_movies(tmp_path):
    conn = tvdb_ingest.connect(str(tmp_path / "t.db"))
    with conn:
        conn.executemany(
            "INSERT INTO tvdb_titles (kind, tvdb_id, score, updated_at) VALUES (?, ?, ?, 'x')",
            [("series", 1, 10), ("series", 2, 90), ("movie", 3, 5000), ("movie", 4, None)],
        )
    assert [(r["kind"], r["tvdb_id"]) for r in tvdb_ingest.pending(conn)] == [
        ("series", 2), ("movie", 3), ("series", 1), ("movie", 4)]
