"""Six-role semantic palette: queue logic, logo handling, base and duplicates."""
import io

import numpy as np
from PIL import Image, ImageDraw

from tvmdbhex import semantic as S
from tvmdbhex.colors import srgb_to_oklab

ROLES = S.ROLES


def hex_rgb(h):
    return tuple(int(h[i : i + 2], 16) for i in (1, 3, 5))


def close(h, rgb, tol=0.06):
    a, b = srgb_to_oklab(np.array([hex_rgb(h), rgb]))
    return float(np.linalg.norm(a - b)) < tol


def logo_png(colours, size=(300, 80)):
    """Transparent logo: blocky letters, each letter one of `colours` in turn."""
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    n = 6
    w = size[0] // n
    for i in range(n):
        c = colours[i % len(colours)]
        x = i * w + 4
        d.rectangle([x, 6, x + w - 10, size[1] - 6], fill=c + (255,))
        d.rectangle([x + 8, 22, x + w - 18, size[1] - 22], fill=(0, 0, 0, 0))
    return img


def poster(bands, logo=None, logo_at=(0.15, 0.05, 0.85), size=(342, 513)):
    """Horizontal colour bands (top to bottom) with an optional logo pasted on."""
    img = Image.new("RGB", size)
    y = 0
    for i, (colour, share) in enumerate(bands):
        h = size[1] - y if i == len(bands) - 1 else round(size[1] * share)
        img.paste(colour, (0, y, size[0], y + h))
        y += h
    if logo is not None:
        x0, y0, x1 = logo_at
        w = int(size[0] * (x1 - x0))
        lg = logo.resize((w, int(w * logo.height / logo.width)))
        img.paste(lg, (int(size[0] * x0), int(size[1] * y0)), lg)
    return img


SKY, NAVY, ORANGE, CREAM, PURPLE, RED = (40, 120, 200), (20, 30, 70), (230, 120, 30), (240, 230, 200), (110, 50, 150), (200, 30, 40)
ART = [(SKY, 0.35), (NAVY, 0.25), (ORANGE, 0.15), (CREAM, 0.12), (PURPLE, 0.08), (RED, 0.05)]
PINK, GOLD = (240, 30, 160), (210, 170, 60)


def test_case_a_two_logo_colours():
    logo = logo_png([PINK, GOLD])
    r = S.analyse(poster(ART, logo), logo)
    assert r.logo_source == "tmdb-located"
    assert close(r.base.hex, SKY)
    assert close(r.identity1.hex, PINK) or close(r.identity1.hex, GOLD)
    assert {r.identity1.source, r.identity2.source} == {"logo"}
    # Highlights start at the top of the artwork ladder (base colour excluded).
    assert r.highlight1.source == r.highlight2.source == r.accent.source == "artwork"
    assert close(r.highlight1.hex, NAVY)


def test_case_b_one_logo_colour_then_artwork_queue():
    logo = logo_png([GOLD])
    r = S.analyse(poster(ART, logo), logo)
    assert close(r.identity1.hex, GOLD) and r.identity1.source == "logo"
    assert r.identity2.source == "artwork"
    # identity2 consumed the strongest unused dominant, highlights shift down.
    assert close(r.identity2.hex, NAVY)
    assert not close(r.highlight1.hex, NAVY)


def test_black_and_white_logo_falls_back_to_artwork():
    logo = logo_png([(255, 255, 255), (0, 0, 0)])
    r = S.analyse(poster(ART, logo), logo)
    assert r.logo_colours == []
    assert r.identity1.source == r.identity2.source == "artwork"
    for role in ("identity1", "identity2"):
        assert getattr(r, role).C >= S.DEFAULT_CONFIG.neutral_chroma  # never white/black/grey


def test_near_identical_logo_colours_merge():
    logo = logo_png([(0xD8, 0x3D, 0x48), (0xE1, 0x44, 0x4C)])
    r = S.analyse(poster(ART, logo), logo)
    assert len(r.logo_colours) == 1
    assert r.identity1.source == "logo" and r.identity2.source == "artwork"


def test_no_duplicate_roles():
    logo = logo_png([PINK])
    r = S.analyse(poster(ART, logo), logo)
    cols = [getattr(r, role) for role in ROLES]
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            assert S.dist(a, b) >= S.DEFAULT_CONFIG.min_distance * S.DEFAULT_CONFIG.relax_steps[-1]


def test_base_is_upper_corner_environment():
    # Top band dark green across both corners; big orange area below.
    art = [((20, 70, 40), 0.2), (ORANGE, 0.6), (NAVY, 0.2)]
    r = S.analyse(poster(art), None)
    assert close(r.base.hex, (20, 70, 40))
    assert r.base.hex not in {r.identity1.hex, r.identity2.hex}


def test_logo_is_masked_out_of_artwork():
    # A big pink logo on a sky/navy poster: pink must not become an artwork dominant.
    logo = logo_png([PINK], size=(300, 120))
    r = S.analyse(poster([(SKY, 0.5), (NAVY, 0.5)], logo, logo_at=(0.05, 0.08, 0.95)), logo)
    assert r.logo_source == "tmdb-located"
    assert not any(close(c.hex, PINK, 0.08) for c in r.ladder)


def test_unlocated_logo_colours_must_exist_on_poster():
    # Logo not on the poster at all: its pink isn't in the artwork either -> unused.
    logo = logo_png([PINK])
    r = S.analyse(poster(ART), logo)
    assert r.logo_source == "tmdb"
    assert r.logo_colours == []


def test_title_detection_without_tmdb_logo():
    img = poster([(NAVY, 1.0)])
    d = ImageDraw.Draw(img)
    for i in range(7):  # a wide block of bright "letters" near the top
        x = 30 + i * 42
        d.rectangle([x, 50, x + 28, 110], fill=GOLD)
        d.rectangle([x + 9, 66, x + 19, 94], fill=NAVY)
    r = S.analyse(img, None)
    assert r.logo_source == "detected"
    assert close(r.identity1.hex, GOLD, 0.08)


def test_monochrome_poster_stays_monochrome():
    art = [((10, 10, 10), 0.5), ((90, 88, 85), 0.3), ((220, 218, 210), 0.2)]
    r = S.analyse(poster(art), None)
    assert all(getattr(r, role).C < 0.05 for role in ROLES)


def test_deterministic_and_bytes_entry_point():
    logo = logo_png([PINK, GOLD])
    img = poster(ART, logo)
    buf, lbuf = io.BytesIO(), io.BytesIO()
    img.save(buf, "JPEG", quality=95)
    logo.save(lbuf, "PNG")
    a = S.analyse_bytes(buf.getvalue(), lbuf.getvalue()).to_dict()
    b = S.analyse_bytes(buf.getvalue(), lbuf.getvalue()).to_dict()
    assert a == b
    assert set(a["palette"]) == set(ROLES)
    assert all(v.startswith("#") and len(v) == 7 for v in a["palette"].values())
