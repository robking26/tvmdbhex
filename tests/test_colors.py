import numpy as np
from conftest import make_poster, to_jpeg
from PIL import Image

from tvmdbhex.colors import _srgb_to_lab, extract_palette, extract_palette_from_bytes, rgb_to_hex


def hex_to_rgb(h):
    return tuple(int(h[i : i + 2], 16) for i in (1, 3, 5))


def close(h, rgb, tol=6):
    return all(abs(a - b) <= tol for a, b in zip(hex_to_rgb(h), rgb))


def test_rgb_to_hex():
    assert rgb_to_hex((255, 0, 16)) == "#FF0010"
    assert rgb_to_hex(np.array([12.4, 12.6, 300])) == "#0C0DFF"


def test_lab_reference_values():
    lab = _srgb_to_lab(np.array([[255, 255, 255], [0, 0, 0], [255, 0, 0]]))
    np.testing.assert_allclose(lab[0], [100, 0, 0], atol=0.1)
    np.testing.assert_allclose(lab[1], [0, 0, 0], atol=0.1)
    np.testing.assert_allclose(lab[2], [53.24, 80.09, 67.20], atol=0.1)


def test_ranked_by_coverage():
    img = make_poster([((20, 30, 120), 0.5), ((230, 40, 40), 0.3), ((240, 220, 60), 0.2)])
    p = extract_palette(img)
    assert close(p.primary.hex, (20, 30, 120))
    assert close(p.secondary.hex, (230, 40, 40))
    assert close(p.tertiary.hex, (240, 220, 60))
    assert p.primary.ratio > p.secondary.ratio > p.tertiary.ratio
    assert abs(p.primary.ratio - 0.5) < 0.03


def test_near_duplicate_shades_are_skipped():
    # Two near-identical blacks dominate; secondary/tertiary should be distinct colours.
    img = make_poster(
        [((10, 10, 10), 0.35), ((16, 14, 12), 0.3), ((200, 30, 30), 0.2), ((30, 160, 60), 0.15)]
    )
    p = extract_palette(img)
    assert close(p.primary.hex, (13, 12, 11))
    assert close(p.secondary.hex, (200, 30, 30))
    assert close(p.tertiary.hex, (30, 160, 60))


def test_single_colour_poster_fills_all_slots():
    p = extract_palette(Image.new("RGB", (100, 150), (40, 80, 120)))
    assert p.primary.hex == p.secondary.hex == p.tertiary.hex == "#285078"


def test_two_colour_poster():
    p = extract_palette(make_poster([((0, 0, 0), 0.6), ((255, 255, 255), 0.4)]))
    assert p.primary.hex == "#000000"
    assert p.secondary.hex == "#FFFFFF"
    assert p.tertiary.hex in {"#000000", "#FFFFFF"}


def test_from_jpeg_bytes_and_transparency():
    data = to_jpeg(make_poster([((250, 120, 0), 0.7), ((0, 90, 200), 0.3)]))
    p = extract_palette_from_bytes(data)
    assert close(p.primary.hex, (250, 120, 0), tol=10)
    rgba = Image.new("RGBA", (50, 75), (0, 0, 0, 0))
    assert extract_palette(rgba).primary.hex == "#FFFFFF"


def test_deterministic():
    rng = np.random.default_rng(1)
    img = Image.fromarray(rng.integers(0, 256, (150, 100, 3), dtype=np.uint8))
    assert extract_palette(img) == extract_palette(img)
