"""Palette debugging: run the legacy and current algorithms side by side and
flag suspicious results. Used by the /debug web view and tools/palette_report.py.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image

from . import colors, colors_legacy

# Thresholds for the automatic review flags (not used by the algorithm itself).
FLAG_SIMILAR = 0.08  # min OKLab distance between any two picks
FLAG_ACCENT_CHROMA = 0.12  # a "distinctive accent" candidate...
FLAG_ACCENT_COVERAGE = 0.05  # ...with at least this much presence
FLAG_SPECK_COVERAGE = 0.05  # vivid primary covering less than this = over-selected accent
FLAG_LIGHTNESS_DRIFT = 0.22  # palette vs poster average lightness


def _oklab(hex_: str) -> np.ndarray:
    rgb = [int(hex_[i : i + 2], 16) for i in (1, 3, 5)]
    return colors.srgb_to_oklab(np.array([rgb]))[0]


def flags(analysis: colors.Analysis, image: Image.Image) -> list[str]:
    picks = analysis.palette.as_list()
    labs = [_oklab(s.hex) for s in picks]
    out = []

    min_d = min(np.linalg.norm(labs[i] - labs[j]) for i in range(3) for j in range(i + 1, 3))
    if min_d < FLAG_SIMILAR:
        out.append(f"too similar (min distance {min_d:.3f})")

    primary = next(c for c in analysis.candidates if c.role == "primary")
    accents = [
        c for c in analysis.candidates
        if c.chroma >= FLAG_ACCENT_CHROMA and c.coverage >= FLAG_ACCENT_COVERAGE and c is not primary
    ]
    if primary.neutrality > 0.6 and accents:
        best = max(accents, key=lambda c: c.chroma)
        out.append(f"neutral primary beat accent {best.hex} ({best.coverage:.0%})")
    if primary.chroma >= FLAG_ACCENT_CHROMA and primary.coverage < FLAG_SPECK_COVERAGE:
        out.append(f"vivid primary covers only {primary.coverage:.1%}")

    # Drift: a dark poster should still read dark, a light one light.
    small = np.asarray(image.convert("RGB").resize((40, 60), Image.Resampling.BOX)).reshape(-1, 3)
    poster_l = float(colors.srgb_to_oklab(small)[:, 0].mean())
    weights = np.array([max(s.ratio, 1e-6) for s in picks])
    palette_l = float(np.dot(weights, [lab[0] for lab in labs]) / weights.sum())
    if abs(poster_l - palette_l) > FLAG_LIGHTNESS_DRIFT:
        out.append(f"lightness drift (poster {poster_l:.2f} vs palette {palette_l:.2f})")
    return out


def compare(image: Image.Image) -> dict:
    """Legacy vs current palette, candidate details and review flags."""
    analysis = colors.analyse(image)
    legacy = colors_legacy.extract_palette(image)
    return {
        "version": colors.PALETTE_VERSION,
        "current": analysis.to_dict(),
        "legacy": {
            role: {"hex": s.hex, "ratio": s.ratio}
            for role, s in zip(("primary", "secondary", "tertiary"), legacy.as_list())
        },
        "flags": flags(analysis, image),
    }


def compare_bytes(data: bytes) -> dict:
    with Image.open(io.BytesIO(data)) as img:
        img.load()
        return compare(img)
