"""Dominant colour extraction.

Posters are downscaled, converted to CIELAB (a perceptually uniform colour
space) and clustered with k-means. Clusters are ranked by how many pixels they
cover; primary/secondary/tertiary are picked greedily in that order, skipping
clusters that are perceptually too close to one already picked so the three
colours are visually distinct. If a poster doesn't contain three distinct
colours, the remaining slots fall back to the next-largest clusters, and
finally repeat the last colour (e.g. a flat single-colour poster).
"""
from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image

SAMPLE_SIZE = (64, 96)  # poster aspect ratio (2:3), ~6k pixels
N_CLUSTERS = 8
MAX_ITERATIONS = 25
# CIE76 delta-E below which two colours are treated as "the same" colour.
# ~2.3 is a just-noticeable difference; 15 is clearly different at a glance.
DISTINCT_DELTA_E = 15.0
# Clusters smaller than this share of the poster (e.g. anti-aliased edges,
# small text) are never considered dominant.
MIN_SHARE = 0.02


@dataclass(frozen=True)
class Swatch:
    hex: str
    ratio: float  # share of the poster's pixels in this colour's cluster (0-1)


@dataclass(frozen=True)
class Palette:
    primary: Swatch
    secondary: Swatch
    tertiary: Swatch

    def as_list(self) -> list[Swatch]:
        return [self.primary, self.secondary, self.tertiary]


def rgb_to_hex(rgb: np.ndarray | tuple[int, int, int]) -> str:
    r, g, b = (int(round(float(c))) for c in rgb)
    return f"#{max(0, min(255, r)):02X}{max(0, min(255, g)):02X}{max(0, min(255, b)):02X}"


def _srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """(N, 3) uint8/float sRGB -> (N, 3) CIELAB (D65)."""
    c = rgb.astype(np.float64) / 255.0
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    m = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
    xyz = c @ m.T
    xyz /= np.array([0.95047, 1.0, 1.08883])
    eps, kappa = 216 / 24389, 24389 / 27
    f = np.where(xyz > eps, np.cbrt(xyz), (kappa * xyz + 16) / 116)
    lab = np.empty_like(f)
    lab[:, 0] = 116 * f[:, 1] - 16
    lab[:, 1] = 500 * (f[:, 0] - f[:, 1])
    lab[:, 2] = 200 * (f[:, 1] - f[:, 2])
    return lab


def _kmeans(points: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Return a cluster label per point. k-means++ seeding, Lloyd iterations."""
    n = len(points)
    centers = np.empty((k, points.shape[1]))
    centers[0] = points[rng.integers(n)]
    d2 = ((points - centers[0]) ** 2).sum(axis=1)
    for i in range(1, k):
        total = d2.sum()
        idx = rng.integers(n) if total == 0 else rng.choice(n, p=d2 / total)
        centers[i] = points[idx]
        d2 = np.minimum(d2, ((points - centers[i]) ** 2).sum(axis=1))

    labels = np.zeros(n, dtype=np.int64)
    for it in range(MAX_ITERATIONS):
        dist = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = dist.argmin(axis=1)
        if it > 0 and np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for i in range(k):
            members = points[labels == i]
            if len(members):
                centers[i] = members.mean(axis=0)
    return labels


def extract_palette(image: Image.Image) -> Palette:
    img = image
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img)
    img = img.convert("RGB")
    img = img.resize(SAMPLE_SIZE, Image.Resampling.BILINEAR)

    rgb = np.asarray(img, dtype=np.uint8).reshape(-1, 3)
    lab = _srgb_to_lab(rgb)

    k = min(N_CLUSTERS, len(np.unique(rgb, axis=0)))
    labels = _kmeans(lab, k, np.random.default_rng(0))

    total = len(rgb)
    clusters = []
    for i in range(k):
        mask = labels == i
        count = int(mask.sum())
        if count == 0:
            continue
        clusters.append((count, rgb[mask].mean(axis=0), lab[mask].mean(axis=0)))
    clusters.sort(key=lambda c: c[0], reverse=True)
    clusters = clusters[:1] + [c for c in clusters[1:] if c[0] / total >= MIN_SHARE]

    picked: list[tuple[int, np.ndarray, np.ndarray]] = []
    for cluster in clusters:
        if all(np.linalg.norm(cluster[2] - p[2]) >= DISTINCT_DELTA_E for p in picked):
            picked.append(cluster)
        if len(picked) == 3:
            break
    for cluster in clusters:  # not enough distinct colours: fall back by size
        if len(picked) == 3:
            break
        if not any(cluster is p for p in picked):
            picked.append(cluster)
    while len(picked) < 3:  # e.g. a single flat colour
        picked.append(picked[-1])

    swatches = [Swatch(rgb_to_hex(c[1]), round(c[0] / total, 4)) for c in picked]
    return Palette(*swatches)


def extract_palette_from_bytes(data: bytes) -> Palette:
    with Image.open(io.BytesIO(data)) as img:
        img.load()
        return extract_palette(img)
