import io
import os

import pytest
from PIL import Image

from tvmdbhex import db

# Set to a disposable Postgres database to also run the storage tests on Postgres,
# e.g. postgresql://postgres:pg@localhost/tvmdbhex_test
PG_URL = os.environ.get("TVMDBHEX_TEST_DATABASE_URL")


@pytest.fixture(params=["sqlite", "postgres"])
def db_settings(request, tmp_path):
    """Settings kwargs pointing at an empty database on each backend."""
    if request.param == "postgres":
        if not PG_URL:
            pytest.skip("TVMDBHEX_TEST_DATABASE_URL not set")
        conn = db.connect(PG_URL)
        conn.execute("TRUNCATE titles, sync_state")
        conn.commit()
        conn.close()
        return {"database_url": PG_URL}
    return {"db_path": str(tmp_path / "test.db")}


def make_poster(bands: list[tuple[tuple[int, int, int], float]], size=(185, 278)) -> Image.Image:
    """Poster made of horizontal colour bands; each band covers `share` of the height."""
    img = Image.new("RGB", size)
    y = 0
    for i, (colour, share) in enumerate(bands):
        h = size[1] - y if i == len(bands) - 1 else round(size[1] * share)
        img.paste(colour, (0, y, size[0], y + h))
        y += h
    return img


def to_jpeg(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=95)
    return buf.getvalue()


def make_logo_png(colour=(230, 30, 60), size=(300, 90)) -> bytes:
    """A transparent PNG with blocky 'letters', like a TMDB title logo."""
    from PIL import ImageDraw

    img = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for i in range(5):
        x = 10 + i * 58
        d.rectangle([x, 10, x + 40, size[1] - 10], fill=colour + (255,))
        d.rectangle([x + 12, 25, x + 28, size[1] - 25], fill=(0, 0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()
