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
