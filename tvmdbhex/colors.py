"""Designer-style poster palette: primary / secondary / tertiary.

The old algorithm (tvmdbhex.colors_legacy) ranked colours purely by pixel
count, so a big blue sky beat Barbie pink. This one asks "which three colours
would a designer pick to represent this poster?":

1. Extract candidates: downscale, convert to OKLab (perceptually uniform), run
   deterministic k-means (~12 clusters), merge near-duplicates (one sky, not
   three; all near-blacks are one black), then an *accent pass* pulls small
   vivid details that k-means averaged away (a yellow dress on a purple
   poster) into their own candidates.
2. Measure each candidate: coverage (mildly centre-weighted), chroma,
   saturation, lightness, distinctiveness from the rest of the poster, local
   contrast against neighbouring regions, and neutrality (grey, near-black,
   near-white, beige/brown).
3. Measure the poster's *colourfulness* (average chroma). This scales how much
   vividness matters: a black-and-white poster is ranked mostly on presence
   and contrast, a colourful one on identity. Neutral penalties scale the same
   way, so monochrome posters stay monochrome.
4. Pick by role, each with its own score:
   - primary   = identity colour: vivid, distinctive, contrasting, with real
                 presence (soft ramp, so a title can qualify but a speck can't).
                 A dominant neutral in a mostly-neutral poster keeps primary.
   - secondary = strongest supporting/environmental colour: mostly coverage,
                 plus separation from primary.
   - tertiary  = accent: far from both, reasonably vivid, genuinely present.
   Later picks must be perceptually distinct from earlier ones (OKLab
   distance) and are penalised for repeating the primary's hue family; the
   distance threshold relaxes only when the poster has nothing better.

Deterministic (fixed seeding, stable tie-breaks), numpy-only, tens of
milliseconds per poster. Every tunable number lives in PaletteConfig; use
analyse() to see each candidate's features and scores.
"""
from __future__ import annotations

import io
from dataclasses import asdict, dataclass, field, replace

import numpy as np
from PIL import Image

# Bump when the algorithm changes in a way that should re-colour stored titles.
PALETTE_VERSION = 2


@dataclass(frozen=True)
class RoleWeights:
    coverage: float = 0.0
    saturation: float = 0.0
    chroma: float = 0.0
    contrast: float = 0.0  # local contrast with neighbouring regions
    distinctiveness: float = 0.0  # difference from the poster's overall palette
    dist_primary: float = 0.0  # separation from the chosen primary
    dist_secondary: float = 0.0  # separation from the chosen secondary


@dataclass(frozen=True)
class PaletteConfig:
    # --- extraction -------------------------------------------------------
    # 60x90 keeps the 2:3 poster shape; box-downsampling averages away film
    # grain and anti-aliased text so they don't become candidates.
    sample_size: tuple[int, int] = (60, 90)
    n_clusters: int = 12
    kmeans_iterations: int = 30
    seed: int = 0
    # Clusters closer than this (OKLab distance) are merged: one sky, not three.
    merge_distance: float = 0.065
    # All near-black clusters merge into one black (same for near-white): a
    # poster that is 60% black shouldn't look like four minor dark colours.
    near_black: tuple[float, float] = (0.22, 0.04)  # (max L, max chroma)
    near_white: tuple[float, float] = (0.93, 0.03)  # (min L, max chroma)
    # Candidates below this share of the poster are ignored (specks, small text),
    # except vivid ones, which only need accent_min_coverage.
    min_coverage: float = 0.015
    # Accent pass: vivid pixels far from their cluster's colour were averaged
    # away by k-means; if there are enough of them they become accent candidates.
    accent_outlier_distance: float = 0.12
    accent_min_chroma: float = 0.09
    accent_min_coverage: float = 0.004
    accent_clusters: int = 3
    # Mild centre weighting for "salient coverage": edge pixels count this much
    # less than centre pixels. Kept small on purpose; posters put identity
    # colours (titles, logos) near edges too.
    centre_weight: float = 0.3

    # --- feature normalisation (OKLab units) ------------------------------
    chroma_full: float = 0.20  # OKLCH chroma treated as "fully vivid" (sRGB max ~0.32)
    contrast_full: float = 0.40  # neighbour difference treated as maximal contrast
    distinct_full: float = 0.40  # difference from the rest of the poster treated as maximal
    separation_full: float = 0.50  # pick-to-pick distance treated as maximal

    # --- poster colourfulness ------------------------------------------------
    # Average pixel chroma. Measured on the tuning set: monochrome posters sit
    # below ~0.012 (The Witch 0.003, Roma 0.009), muted ones 0.015-0.035,
    # colourful ones above 0.05. Colour "strength" ramps 0 -> 1 across this
    # range and scales (a) neutral penalties and (b) how much chroma/saturation
    # count for primary. Relative, never absolute: a B&W poster stays B&W.
    colourfulness_ramp: tuple[float, float] = (0.010, 0.040)

    # --- neutral detection (smooth ramps, OKLab/OKLCH units) ---------------
    grey_chroma: tuple[float, float] = (0.025, 0.06)  # fully grey below, not grey above
    dark_lightness: tuple[float, float] = (0.20, 0.34)  # near-black below
    light_lightness: tuple[float, float] = (0.88, 0.96)  # near-white above
    # Beige/brown: warm hue with modest chroma, e.g. skin, wood, sepia.
    warm_hue_deg: tuple[float, float] = (30.0, 100.0)
    brown_chroma: tuple[float, float] = (0.05, 0.10)

    # --- neutral penalties (fraction of score removed at full strength) ---
    grey_penalty: float = 0.70
    dark_penalty: float = 0.50
    light_penalty: float = 0.50
    brown_penalty: float = 0.30
    # Secondary is "environmental", so neutrals are penalised less there.
    secondary_neutral_scale: float = 0.5

    # --- dominant-neutral identity (The Witch, Kingsman) -------------------
    # A neutral covering most of a muted poster *is* its identity (black and
    # gold, not gold and black). Bonus = weight * neutrality * ramp(coverage)
    # * (1 - ramp(colourfulness)).
    neutral_identity_bonus: float = 0.45
    neutral_identity_coverage: tuple[float, float] = (0.30, 0.65)
    neutral_identity_colourfulness: tuple[float, float] = (0.020, 0.050)

    # --- role weights ------------------------------------------------------
    # Primary's saturation+chroma weight is scaled by colour strength and the
    # remainder moves to coverage, so on a monochrome poster presence wins.
    primary: RoleWeights = field(
        default_factory=lambda: RoleWeights(
            coverage=0.25, saturation=0.25, chroma=0.20, contrast=0.15, distinctiveness=0.15
        )
    )
    secondary: RoleWeights = field(
        default_factory=lambda: RoleWeights(coverage=0.50, dist_primary=0.20, saturation=0.15, chroma=0.15)
    )
    tertiary: RoleWeights = field(
        default_factory=lambda: RoleWeights(
            dist_primary=0.30, dist_secondary=0.30, saturation=0.20, coverage=0.20
        )
    )
    # Primary needs real presence: its score is multiplied by a ramp over this
    # coverage range, so a title (2-3%) can qualify but a speck can't.
    primary_presence: tuple[float, float] = (0.008, 0.035)

    # --- separation between picks -------------------------------------------
    # Minimum OKLab distance between chosen colours (~0.02 is a just-noticeable
    # difference; 0.12 is clearly a different colour). If nothing qualifies the
    # threshold relaxes step by step, so monochrome posters still get 3 colours.
    min_distance: float = 0.12
    relax_steps: tuple[float, ...] = (1.0, 0.6, 0.3, 0.0)
    # Two chromatic colours whose hues are within this range (degrees) are
    # "the same hue family" (yellow vs mustard, red vs dark red); a later pick
    # in the same family as an earlier one loses this fraction of its score.
    same_hue_deg: tuple[float, float] = (15.0, 45.0)
    same_hue_min_chroma: float = 0.05
    same_hue_penalty: float = 0.40


DEFAULT_CONFIG = PaletteConfig()


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


@dataclass
class Candidate:
    """A candidate colour with the features and scores used to rank it."""

    hex: str
    oklab: tuple[float, float, float]
    coverage: float
    salient_coverage: float
    lightness: float
    chroma: float
    hue: float
    saturation: float
    distinctiveness: float
    contrast: float
    neutrality: float = 0.0
    neutral_factor: float = 1.0  # multiplier applied to primary score (1 = no penalty)
    accent: bool = False  # found by the accent pass
    scores: dict[str, float] = field(default_factory=dict)
    role: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["oklab"] = [round(v, 4) for v in self.oklab]
        d["scores"] = {k: round(v, 4) for k, v in self.scores.items()}
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}


@dataclass
class Analysis:
    palette: Palette
    candidates: list[Candidate]
    colourfulness: float
    colour_strength: float

    def to_dict(self) -> dict:
        return {
            "palette": {
                role: asdict(s) for role, s in zip(("primary", "secondary", "tertiary"), self.palette.as_list())
            },
            "colourfulness": round(self.colourfulness, 4),
            "colour_strength": round(self.colour_strength, 4),
            "candidates": [c.to_dict() for c in self.candidates],
        }


# --------------------------------------------------------------------------
# Colour science: sRGB <-> OKLab (Björn Ottosson's reference matrices)
# --------------------------------------------------------------------------

_M1 = np.array(
    [
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ]
)
_M2 = np.array(
    [
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ]
)
_M1_INV = np.linalg.inv(_M1)
_M2_INV = np.linalg.inv(_M2)


def srgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """(N, 3) sRGB 0-255 -> (N, 3) OKLab (L 0-1)."""
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return np.cbrt(lin @ _M1.T) @ _M2.T


def oklab_to_srgb(lab: np.ndarray) -> np.ndarray:
    """(N, 3) OKLab -> (N, 3) sRGB 0-255 floats (clipped to gamut)."""
    lms = (np.asarray(lab, dtype=np.float64) @ _M2_INV.T) ** 3
    lin = np.clip(lms @ _M1_INV.T, 0.0, 1.0)
    c = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * np.power(lin, 1 / 2.4) - 0.055)
    return np.clip(c * 255.0, 0, 255)


def rgb_to_hex(rgb: np.ndarray | tuple[int, int, int]) -> str:
    r, g, b = (int(round(float(c))) for c in rgb)
    return f"#{max(0, min(255, r)):02X}{max(0, min(255, g)):02X}{max(0, min(255, b)):02X}"


def _ramp(x, lo: float, hi: float) -> float:
    """Smoothstep from 0 (x <= lo) to 1 (x >= hi)."""
    t = min(max((float(x) - lo) / (hi - lo), 0.0), 1.0)
    return t * t * (3 - 2 * t)


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------


def _prepare(image: Image.Image, size: tuple[int, int]) -> np.ndarray:
    img = image
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img)
    img = img.convert("RGB").resize(size, Image.Resampling.BOX)
    return np.asarray(img, dtype=np.uint8)  # (H, W, 3)


def _kmeans(points: np.ndarray, k: int, iterations: int, seed: int) -> np.ndarray:
    """Deterministic k-means++ seeding + Lloyd iterations. Returns labels.

    Distances use |x|^2 - 2x.c + |c|^2 (one matrix multiply) and centres are
    updated with bincount, which keeps a poster well under 10 ms here.
    """
    rng = np.random.default_rng(seed)
    n = len(points)
    centres = np.empty((k, 3))
    centres[0] = points[rng.integers(n)]
    d2 = ((points - centres[0]) ** 2).sum(axis=1)
    for i in range(1, k):
        total = d2.sum()
        idx = int(d2.argmax()) if total == 0 else rng.choice(n, p=d2 / total)
        centres[i] = points[idx]
        d2 = np.minimum(d2, ((points - centres[i]) ** 2).sum(axis=1))
    sq = (points**2).sum(axis=1)[:, None]
    labels = np.full(n, -1)
    for _ in range(iterations):
        dist = sq - 2 * points @ centres.T + (centres**2).sum(axis=1)[None, :]
        new = dist.argmin(axis=1)
        if np.array_equal(new, labels):
            break
        labels = new
        counts = np.bincount(labels, minlength=k)
        for axis in range(3):
            sums = np.bincount(labels, weights=points[:, axis], minlength=k)
            np.divide(sums, counts, out=centres[:, axis], where=counts > 0)
    return labels


def _centres(labels: np.ndarray, lab: np.ndarray) -> dict[int, np.ndarray]:
    return {int(i): lab[labels == i].mean(axis=0) for i in np.unique(labels)}


def _merge(labels: np.ndarray, lab: np.ndarray, cfg: PaletteConfig) -> np.ndarray:
    labels = labels.copy()
    # 1. All near-black clusters become one black; all near-white one white.
    centres = _centres(labels, lab)
    for is_group in (
        lambda c: c[0] <= cfg.near_black[0] and np.hypot(c[1], c[2]) <= cfg.near_black[1],
        lambda c: c[0] >= cfg.near_white[0] and np.hypot(c[1], c[2]) <= cfg.near_white[1],
    ):
        group = [i for i, c in centres.items() if is_group(c)]
        for i in group[1:]:
            labels[labels == i] = group[0]
    # 2. Merge the closest pair of clusters until none is closer than merge_distance.
    while True:
        centres = _centres(labels, lab)
        ids = list(centres)
        if len(ids) < 2:
            return labels
        table = np.array([centres[i] for i in ids])
        d = np.linalg.norm(table[:, None] - table[None], axis=2)
        np.fill_diagonal(d, np.inf)
        a, b = np.unravel_index(d.argmin(), d.shape)
        if d[a, b] >= cfg.merge_distance:
            return labels
        labels[labels == ids[b]] = ids[a]


def _accent_pass(labels: np.ndarray, lab: np.ndarray, cfg: PaletteConfig) -> tuple[np.ndarray, set[int]]:
    """Split vivid, badly-represented pixels out into their own accent clusters."""
    centres = _centres(labels, lab)
    lookup = np.zeros((int(labels.max()) + 1, 3))
    for i, c in centres.items():
        lookup[i] = c
    table = lookup[labels]
    chroma = np.hypot(lab[:, 1], lab[:, 2])
    outlier = (np.linalg.norm(lab - table, axis=1) > cfg.accent_outlier_distance) & (
        chroma > cfg.accent_min_chroma
    )
    n = len(lab)
    if outlier.sum() < cfg.accent_min_coverage * n:
        return labels, set()
    pts = lab[outlier]
    k = min(cfg.accent_clusters, len(np.unique(np.round(pts, 3), axis=0)))
    sub = _kmeans(pts, k, cfg.kmeans_iterations, cfg.seed) if k > 1 else np.zeros(len(pts), dtype=int)
    labels = labels.copy()
    base = int(labels.max()) + 1
    labels[outlier] = base + sub
    accents = {base + int(s) for s in np.unique(sub) if (sub == s).sum() >= cfg.accent_min_coverage * n}
    return labels, accents


def _local_contrast(label_map: np.ndarray, centres: dict[int, np.ndarray]) -> dict[int, float]:
    """Average colour difference across each cluster's region boundaries:
    how strongly a colour stands out against whatever it touches."""
    ids = sorted(centres)
    index = {i: n for n, i in enumerate(ids)}
    lookup = np.zeros(max(ids) + 1, dtype=int)
    for i, n in index.items():
        lookup[i] = n
    dense = lookup[label_map]
    table = np.array([centres[i] for i in ids])
    total = np.zeros(len(ids))
    edges = np.zeros(len(ids))
    for a, b in ((dense[:, :-1], dense[:, 1:]), (dense[:-1, :], dense[1:, :])):
        diff = a != b
        x, y = a[diff], b[diff]
        dist = np.linalg.norm(table[x] - table[y], axis=1)
        for side in (x, y):
            np.add.at(total, side, dist)
            np.add.at(edges, side, 1)
    return {i: float(total[n] / edges[n]) if edges[n] else 0.0 for i, n in index.items()}


def _saturation(rgb: np.ndarray, lightness: float) -> float:
    """HSV-style saturation (how "pure" a colour looks, even when dark), faded
    out for near-black where tiny channel differences read as huge saturation."""
    hi, lo = float(rgb.max()), float(rgb.min())
    s = (hi - lo) / hi if hi > 0 else 0.0
    return s * _ramp(lightness, 0.12, 0.30)


def _neutral_parts(c: Candidate, cfg: PaletteConfig) -> tuple[float, float, float, float]:
    grey = 1 - _ramp(c.chroma, *cfg.grey_chroma)
    dark = 1 - _ramp(c.lightness, *cfg.dark_lightness)
    light = _ramp(c.lightness, *cfg.light_lightness)
    lo, hi = cfg.warm_hue_deg
    warm = 1.0 if lo <= c.hue <= hi else 0.0
    brown = warm * (1 - _ramp(c.chroma, *cfg.brown_chroma)) * (1 - grey)
    return grey, dark, light, brown


def _candidates(image: Image.Image, cfg: PaletteConfig) -> tuple[list[Candidate], float, float]:
    rgb_img = _prepare(image, cfg.sample_size)
    h, w, _ = rgb_img.shape
    rgb = rgb_img.reshape(-1, 3)
    lab = srgb_to_oklab(rgb)
    n = len(rgb)

    colourfulness = float(np.hypot(lab[:, 1], lab[:, 2]).mean())
    strength = _ramp(colourfulness, *cfg.colourfulness_ramp)

    # Mild centre weighting for salient coverage.
    yy, xx = np.mgrid[0:h, 0:w]
    r2 = ((xx - (w - 1) / 2) / (w / 2)) ** 2 + ((yy - (h - 1) / 2) / (h / 2)) ** 2
    weights = (1.0 - cfg.centre_weight * np.clip(r2 / 2, 0, 1)).reshape(-1)

    k = min(cfg.n_clusters, len(np.unique(rgb, axis=0)))
    labels = _kmeans(lab, k, cfg.kmeans_iterations, cfg.seed)
    labels = _merge(labels, lab, cfg)
    labels, accents = _accent_pass(labels, lab, cfg)

    centres = _centres(labels, lab)
    ids = sorted(centres)
    coverage = {i: float((labels == i).sum()) / n for i in ids}
    salient = {i: float(weights[labels == i].sum() / weights.sum()) for i in ids}
    contrast = _local_contrast(labels.reshape(h, w), centres)

    cands: list[Candidate] = []
    for i in ids:
        members = lab[labels == i]
        centre = centres[i]
        # Averaging colours of slightly different hue dulls them; keep the
        # cluster's *typical* chroma (mean of member chromas) so the swatch
        # looks like the colour on the poster rather than a muddier average.
        mean_chroma = float(np.hypot(members[:, 1], members[:, 2]).mean())
        centre_chroma = float(np.hypot(centre[1], centre[2]))
        rep = centre.copy()
        if centre_chroma > 1e-6:
            rep[1:] *= mean_chroma / centre_chroma
        L, C = float(rep[0]), float(np.hypot(rep[1], rep[2]))
        rep_rgb = oklab_to_srgb(rep[None])[0]
        # Distinctiveness: coverage-weighted distance to every other colour.
        others = [j for j in ids if j != i]
        other_cov = sum(coverage[j] for j in others)
        distinct = (
            sum(coverage[j] * float(np.linalg.norm(rep - centres[j])) for j in others) / other_cov
            if other_cov
            else 0.0
        )
        cands.append(
            Candidate(
                hex=rgb_to_hex(rep_rgb),
                oklab=(L, float(rep[1]), float(rep[2])),
                coverage=coverage[i],
                salient_coverage=salient[i],
                lightness=L,
                chroma=C,
                hue=float(np.degrees(np.arctan2(rep[2], rep[1])) % 360),
                saturation=_saturation(rep_rgb, L),
                distinctiveness=min(distinct / cfg.distinct_full, 1.0),
                contrast=min(contrast[i] / cfg.contrast_full, 1.0),
                accent=i in accents,
            )
        )

    cands.sort(key=lambda c: (-c.coverage, c.hex))
    # Vivid clusters (and accent-pass clusters) may be much smaller than the
    # general floor: a yellow raincoat split by shading into two 1% clusters
    # is still a legitimate accent.
    cands = cands[:1] + [
        c for c in cands[1:]
        if c.coverage >= cfg.min_coverage
        or ((c.accent or c.chroma >= cfg.accent_min_chroma) and c.coverage >= cfg.accent_min_coverage)
    ]

    for c in cands:
        grey, dark, light, brown = _neutral_parts(c, cfg)
        c.neutrality = max(grey, dark, light, brown)
        raw = (
            (1 - cfg.grey_penalty * grey)
            * (1 - cfg.dark_penalty * dark)
            * (1 - cfg.light_penalty * light)
            * (1 - cfg.brown_penalty * brown)
        )
        c.neutral_factor = 1 - strength * (1 - raw)
    return cands, colourfulness, strength


# --------------------------------------------------------------------------
# Role scoring and selection
# --------------------------------------------------------------------------


def _dist(a: Candidate, b: Candidate) -> float:
    return float(np.linalg.norm(np.array(a.oklab) - np.array(b.oklab)))


def _same_hue(a: Candidate, b: Candidate, cfg: PaletteConfig) -> float:
    """1 when both are chromatic and share a hue family, fading to 0."""
    if min(a.chroma, b.chroma) < cfg.same_hue_min_chroma:
        return 0.0
    diff = abs(a.hue - b.hue) % 360
    return 1 - _ramp(min(diff, 360 - diff), *cfg.same_hue_deg)


def _score(c: Candidate, w: RoleWeights, cfg: PaletteConfig, max_cov: float, picks: list[Candidate]) -> float:
    # Coverage is relative to the largest candidate and square-rooted: presence
    # matters, but a colour with a quarter of the leader's area keeps half its
    # coverage credit, so identity colours can compete with backgrounds.
    cov = (c.salient_coverage / max_cov) ** 0.5 if max_cov > 0 else 0.0
    sep = lambda other: min(_dist(c, other) / cfg.separation_full, 1.0)  # noqa: E731
    return (
        w.coverage * cov
        + w.saturation * c.saturation
        + w.chroma * min(c.chroma / cfg.chroma_full, 1.0)
        + w.contrast * c.contrast
        + w.distinctiveness * c.distinctiveness
        + (w.dist_primary * sep(picks[0]) if picks else 0.0)
        + (w.dist_secondary * sep(picks[1]) if len(picks) > 1 else 0.0)
    )


def _pick(cands: list[Candidate], picks: list[Candidate], key, cfg: PaletteConfig) -> Candidate | None:
    """Best-scoring candidate that is far enough from earlier picks, relaxing
    the distance threshold only if nothing qualifies."""
    pool = [c for c in cands if all(c is not p for p in picks)]
    for step in cfg.relax_steps:
        ok = [c for c in pool if all(_dist(c, p) >= cfg.min_distance * step for p in picks)]
        if ok:
            return max(ok, key=lambda c: (key(c), c.coverage, c.hex))
    return None


def analyse(image: Image.Image, cfg: PaletteConfig = DEFAULT_CONFIG) -> Analysis:
    cands, colourfulness, strength = _candidates(image, cfg)
    max_cov = max(c.salient_coverage for c in cands)

    # PRIMARY: identity colour. Vividness only counts as far as the poster is
    # colourful; the weight it loses goes to coverage (presence).
    p = cfg.primary
    moved = (p.saturation + p.chroma) * (1 - strength)
    primary_w = replace(
        p, saturation=p.saturation * strength, chroma=p.chroma * strength, coverage=p.coverage + moved
    )
    identity_scale = 1 - _ramp(colourfulness, *cfg.neutral_identity_colourfulness)
    for c in cands:
        base = _score(c, primary_w, cfg, max_cov, []) * c.neutral_factor
        bonus = (
            cfg.neutral_identity_bonus
            * c.neutrality
            * _ramp(c.coverage, *cfg.neutral_identity_coverage)
            * identity_scale
        )
        c.scores["primary"] = (base + bonus) * _ramp(c.coverage, *cfg.primary_presence)
    primary = max(cands, key=lambda c: (c.scores["primary"], c.coverage, c.hex))
    picks = [primary]

    def hue_factor(c: Candidate) -> float:
        return 1 - cfg.same_hue_penalty * max(_same_hue(c, q, cfg) for q in picks)

    # SECONDARY: supporting / environmental colour. Neutrals penalised less.
    def secondary_key(c: Candidate) -> float:
        neutral = 1 - cfg.secondary_neutral_scale * (1 - c.neutral_factor)
        return _score(c, cfg.secondary, cfg, max_cov, picks[:1]) * neutral * hue_factor(c)

    for c in cands:
        if c is not primary:
            c.scores["secondary"] = secondary_key(c)
    secondary = _pick(cands, picks, secondary_key, cfg) or primary
    picks.append(secondary)

    # TERTIARY: accent, far from both earlier picks.
    def tertiary_key(c: Candidate) -> float:
        return _score(c, cfg.tertiary, cfg, max_cov, picks[:2]) * hue_factor(c)

    for c in cands:
        if c is not primary and c is not secondary:
            c.scores["tertiary"] = tertiary_key(c)
    tertiary = _pick(cands, picks, tertiary_key, cfg) or secondary

    for role, c in (("tertiary", tertiary), ("secondary", secondary), ("primary", primary)):
        c.role = role  # a candidate reused for several slots keeps its best role
    palette = Palette(*(Swatch(c.hex, round(c.coverage, 4)) for c in (primary, secondary, tertiary)))
    return Analysis(palette, cands, colourfulness, strength)


def extract_palette(image: Image.Image, cfg: PaletteConfig = DEFAULT_CONFIG) -> Palette:
    return analyse(image, cfg).palette


def analyse_bytes(data: bytes, cfg: PaletteConfig = DEFAULT_CONFIG) -> Analysis:
    with Image.open(io.BytesIO(data)) as img:
        img.load()
        return analyse(img, cfg)


def extract_palette_from_bytes(data: bytes) -> Palette:
    return analyse_bytes(data).palette
