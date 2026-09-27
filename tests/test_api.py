import pytest
from fastapi.testclient import TestClient

from tvmdbhex import db
from tvmdbhex.api import create_app
from tvmdbhex.config import Settings

PALETTE = [("#14207A", 0.6), ("#E62828", 0.25), ("#F0DC3C", 0.15)]


@pytest.fixture
def client(tmp_path):
    settings = Settings(db_path=str(tmp_path / "api.db"), api_keys=frozenset({"secret"}))
    conn = db.connect(settings.db_path)
    db.save_result(conn, "movie", 550, status="done", title="Fight Club", poster_path="/fc.jpg", palette=PALETTE)
    db.save_result(conn, "movie", 551, status="no_poster", title="No Poster")
    db.save_result(conn, "tv", 1399, status="done", title="GoT", poster_path="/got.jpg", palette=PALETTE)
    conn.close()
    return TestClient(create_app(settings), headers={"X-API-Key": "secret"})


def test_auth_required(client):
    assert client.get("/v1/movie/550", headers={"X-API-Key": "nope"}).status_code == 401
    assert client.get("/health", headers={"X-API-Key": ""}).status_code == 200


def test_get_title(client):
    body = client.get("/v1/movie/550").json()
    assert body["status"] == "done"
    assert body["colors"]["primary"] == {"hex": "#14207A", "ratio": 0.6}
    assert body["colors"]["secondary"]["hex"] == "#E62828"
    assert body["colors"]["tertiary"]["hex"] == "#F0DC3C"

    assert client.get("/v1/movie/551").json()["colors"] is None
    assert client.get("/v1/movie/12345").status_code == 404
    assert client.get("/v1/book/1").status_code == 422


def test_lookup(client):
    body = client.post(
        "/v1/lookup",
        json={"items": [
            {"media_type": "tv", "tmdb_id": 1399},
            {"media_type": "movie", "tmdb_id": 550},
            {"media_type": "movie", "tmdb_id": 999},
            {"media_type": "movie", "tmdb_id": 550},
        ]},
    ).json()
    assert [(r["media_type"], r["tmdb_id"]) for r in body["results"]] == [("tv", 1399), ("movie", 550)]
    assert body["missing"] == [{"media_type": "movie", "tmdb_id": 999}]


def test_list_and_stats(client):
    assert [r["tmdb_id"] for r in client.get("/v1/movie").json()] == [550]
    assert client.get("/v1/movie?after_id=550").json() == []
    assert client.get("/v1/stats").json()["movie"] == {"done": 1, "no_poster": 1}


def test_search(tmp_path):
    settings = Settings(db_path=str(tmp_path / "s.db"))
    conn = db.connect(settings.db_path)
    db.upsert_seed(conn, "movie", [
        {"id": 1, "title": "The Matrix", "popularity": 50},
        {"id": 2, "title": "Matrix", "popularity": 5},
        {"id": 3, "title": "100% Wolf", "popularity": 9},
        {"id": 4, "title": "Some Film", "popularity": 99},
    ])
    db.save_result(conn, "movie", 4, status="done", title="Some Film", poster_path="/a.jpg", palette=PALETTE)
    conn.close()
    c = TestClient(create_app(settings))

    ids = lambda r: [t["tmdb_id"] for t in r.json()]
    assert ids(c.get("/v1/search?q=matrix")) == [2, 1]  # exact match first, then popularity
    assert ids(c.get("/v1/search?q=%25")) == [3]  # LIKE wildcards are escaped
    assert ids(c.get("/v1/search?q=4")) == [4]  # numeric query matches TMDB ID
    assert ids(c.get("/v1/search")) == [4, 1, 3, 2]  # browse = popularity order
    assert ids(c.get("/v1/search?status=done")) == [4]
    assert ids(c.get("/v1/search?media_type=tv")) == []
    assert ids(c.get("/v1/search?limit=2&offset=1")) == [1, 3]


def test_index_page_is_public(client):
    r = client.get("/", headers={"X-API-Key": ""})
    assert r.status_code == 200 and "tvmdbhex" in r.text
