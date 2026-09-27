import numpy as np
import pytest
from conftest import make_poster, to_jpeg
from PIL import Image, ImageDraw

from tvmdbhex import colors_legacy
from tvmdbhex.colors import (
    DEFAULT_CONFIG,
    PaletteConfig,
    analyse,
    extract_palette,
    extract_palette_from_bytes,
    oklab_to_srgb,
    rgb_to_hex,
    srgb_to_oklab,
)


def hex_to_rgb(h):
    return tuple(int(h[i : i + 2], 16) for i in (1, 3, 5))


def close(h, rgb, tol=12):
    return all(abs(a - b) <= tol for a, b in zip(hex_to_rgb(h), rgb))


def distance(h1, h2):
    a, b = srgb_to_oklab(np.array([hex_to_rgb(h1), hex_to_rgb(h2)]))
    return float(np.linalg.norm(a - b))


def test_rgb_to_hex():
    assert rgb_to_hex((255, 0, 16)) == "#FF0010"
    assert rgb_to_hex(np.array([12.4, 12.6, 300])) == "#0C0DFF"


def test_oklab_reference_values_and_round_trip():
    lab = srgb_to_oklab(np.array([[255, 255, 255], [0, 0, 0], [255, 0, 0]]))
    np.testing.assert_allclose(lab[0], [1.0, 0, 0], atol=1e-3)
    np.testing.assert_allclose(lab[1], [0, 0, 0], atol=1e-3)
    np.testing.assert_allclose(lab[2], [0.6280, 0.2249, 0.1258], atol=1e-3)  # Ottosson's reference
    rgb = np.array([[12, 200, 99], [250, 10, 180], [128, 128, 128]])
    np.testing.assert_allclose(oklab_to_srgb(srgb_to_oklab(rgb)), rgb, atol=0.5)


def test_barbie_identity_colour_beats_bigger_sky():
    # 55% sky blue, 25% hot pink, 12% skin, 8% white.
    img = make_poster([((120, 190, 235), 0.55), ((236, 64, 152), 0.25), ((230, 190, 160), 0.12), ((250, 250, 250), 0.08)])
    p = extract_palette(img)
    assert close(p.primary.hex, (236, 64, 152))  # pink is the identity colour
    assert close(p.secondary.hex, (120, 190, 235))  # blue is the environment
    # The legacy algorithm picked by area, so blue came first.
    assert close(colors_legacy.extract_palette(img).primary.hex, (120, 190, 235))


def test_dark_poster_with_small_gold_keeps_black_primary():
    # Kingsman-like: mostly black, small gold title, some dark brown, cream.
    img = make_poster([((12, 10, 9), 0.72), ((200, 160, 60), 0.08), ((45, 32, 25), 0.10), ((235, 225, 200), 0.10)])
    p = extract_palette(img)
    assert close(p.primary.hex, (12, 10, 9))
    assert close(p.secondary.hex, (200, 160, 60)) or close(p.tertiary.hex, (200, 160, 60))


def test_monochrome_poster_stays_neutral():
    # The Witch-like: black, muted warm grey, pale cream. No colour to invent.
    img = make_poster([((15, 14, 13), 0.6), ((95, 85, 75), 0.25), ((215, 205, 185), 0.15)])
    a = analyse(img)
    assert a.colour_strength < 0.1
    assert close(a.palette.primary.hex, (15, 14, 13))
    for s in a.palette.as_list():
        r, g, b = hex_to_rgb(s.hex)
        assert max(r, g, b) - min(r, g, b) < 40  # nothing saturated was forced in


def test_black_and_white_with_small_coloured_title_keeps_bw_primary():
    # Roma-like: B&W photo with a small yellow title -> title becomes the accent.
    img = make_poster([((240, 240, 240), 0.35), ((64, 64, 63), 0.35), ((20, 20, 20), 0.26), ((221, 183, 55), 0.04)])
    p = extract_palette(img)
    assert not close(p.primary.hex, (221, 183, 55))
    assert any(close(s.hex, (221, 183, 55)) for s in p.as_list())


def test_picks_are_perceptually_distinct():
    # Near-identical dark browns shouldn't take all three slots when better options exist.
    img = make_poster(
        [((25, 21, 18), 0.3), ((36, 28, 24), 0.25), ((53, 42, 34), 0.25), ((200, 30, 30), 0.1), ((230, 220, 200), 0.1)]
    )
    p = extract_palette(img)
    hexes = [s.hex for s in p.as_list()]
    assert min(distance(a, b) for i, a in enumerate(hexes) for b in hexes[i + 1 :]) >= DEFAULT_CONFIG.min_distance


def test_secondary_avoids_repeating_primary_hue():
    # Kill Bill-like: yellow with a mustard variant and black text.
    img = make_poster([((254, 227, 32), 0.5), ((232, 188, 72), 0.2), ((5, 3, 3), 0.2), ((238, 95, 38), 0.1)])
    p = extract_palette(img)
    yellows = [(254, 227, 32), (232, 188, 72)]
    # Whichever yellow leads, the secondary must not be the other yellow.
    assert any(close(p.primary.hex, y) for y in yellows)
    assert not any(close(p.secondary.hex, y) for y in yellows)


def test_small_vivid_accent_is_found():
    # Tiny yellow detail on a purple poster (La La Land dress) becomes a candidate.
    img = Image.new("RGB", (200, 300), (40, 20, 110))
    ImageDraw.Draw(img).rectangle([90, 200, 104, 225], fill=(250, 210, 40))  # ~0.6% of the poster
    a = analyse(img)
    assert any(close(c.hex, (250, 210, 40), tol=30) for c in a.candidates)


def test_single_colour_poster_fills_all_slots():
    p = extract_palette(Image.new("RGB", (100, 150), (40, 80, 120)))
    assert p.primary.hex == p.secondary.hex == p.tertiary.hex
    assert close(p.primary.hex, (40, 80, 120), tol=2)


def test_from_jpeg_bytes_and_transparency():
    data = to_jpeg(make_poster([((250, 120, 0), 0.7), ((0, 90, 200), 0.3)]))
    p = extract_palette_from_bytes(data)
    assert {p.primary.hex, p.secondary.hex} and any(close(s.hex, (250, 120, 0), tol=16) for s in p.as_list())
    assert extract_palette(Image.new("RGBA", (50, 75), (0, 0, 0, 0))).primary.hex == "#FFFFFF"


def test_deterministic():
    rng = np.random.default_rng(1)
    img = Image.fromarray(rng.integers(0, 256, (150, 100, 3), dtype=np.uint8))
    assert analyse(img).to_dict() == analyse(img).to_dict()


def test_config_is_the_single_source_of_tuning():
    img = make_poster([((120, 190, 235), 0.55), ((236, 64, 152), 0.25), ((250, 250, 250), 0.2)])
    # With vividness weights removed, primary falls back to presence (the sky).
    from dataclasses import replace

    from tvmdbhex.colors import RoleWeights

    flat = replace(
        DEFAULT_CONFIG, primary=RoleWeights(coverage=1.0), colourfulness_ramp=(0.0, 1e-9)
    )
    assert isinstance(flat, PaletteConfig)
    assert close(extract_palette(img, flat).primary.hex, (120, 190, 235))


def test_analysis_exposes_features_and_scores():
    a = analyse(make_poster([((20, 30, 120), 0.6), ((230, 40, 40), 0.3), ((240, 220, 60), 0.1)]))
    d = a.to_dict()
    assert set(d["palette"]) == {"primary", "secondary", "tertiary"}
    cand = d["candidates"][0]
    for key in ("coverage", "chroma", "saturation", "lightness", "distinctiveness", "contrast", "neutrality", "scores"):
        assert key in cand
    assert sorted(c["role"] for c in d["candidates"] if c["role"]) == ["primary", "secondary", "tertiary"]


@pytest.mark.parametrize("size", [(64, 96), (500, 750), (185, 278)])
def test_any_input_size(size):
    img = make_poster([((20, 30, 120), 0.6), ((230, 40, 40), 0.4)], size=size)
    assert extract_palette(img).primary.hex.startswith("#")
