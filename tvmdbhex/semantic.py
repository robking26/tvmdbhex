"""Six-role semantic palette: base / identity1 / identity2 / highlight1 / highlight2 / accent.

Colours come from two sources, kept separate:

1. The TITLE TREATMENT (logo). Preferably TMDB's official logo PNG for the
   title (exact letterforms and colours, transparent background). It is then
   *located* on the poster by multi-scale edge matching so it can be masked
   out of the artwork. Without a TMDB logo, a text-region detector looks for
   the main title on the poster itself (large, wide, high-contrast text block,
   not the small billing block or taglines).
2. The ARTWORK, with the logo masked out: perceptual (OKLab) clusters, merged
   near-duplicates, ranked into a coverage-driven "dominant ladder".

Roles are then filled from two queues, in this order:

    base       <- upper-corner environment (top ~18% band, both corners)
    identity1  <- logo1, else strongest unused (non-neutral) artwork dominant
    identity2  <- logo2, else strongest unused (non-neutral) artwork dominant
    highlight1 <- strongest unused artwork dominant
    highlight2 <- next strongest unused artwork dominant
    accent     <- next unused artwork dominant, choosing between the next two
                  with a little extra weight on chroma and distinctiveness

Every assignment is checked against all earlier ones (OKLab distance), so no
two roles are effectively the same colour; the threshold relaxes only when
the poster genuinely has nothing else (monochrome art). Everything stays
traceable to real poster/logo colours: the only invented colour is a last-
resort tint of base for a completely flat image.

Deterministic, numpy/scipy only (no OCR or ML models), ~50 ms per poster.
All tunables live in SemanticConfig.
"""
from __future__ import annotations

import io
from dataclasses import asdict, dataclass, field

import numpy as np
from PIL import Image
from scipy import fft as sfft
from scipy import ndimage

from .colors import _kmeans, _ramp, oklab_to_srgb, rgb_to_hex, srgb_to_oklab

SEMANTIC_VERSION = 3  # stored as palette_version; bump to re-colour stored titles
ROLES = ("base", "identity1", "identity2", "highlight1", "highlight2", "accent")
# Approximate prominence of each role in generated artwork (for consumers; the
# algorithm doesn't use these). Ranges from the design brief, midpoints here.
ROLE_WEIGHTS = {
    "base": (0.40, 0.50),
    "identity1": (0.25, 0.30),
    "identity2": (0.10, 0.20),
    "highlight1": (0.10, 0.20),
    "highlight2": (0.10, 0.20),
    "accent": (0.03, 0.08),
}


@dataclass(frozen=True)
class SemanticConfig:
    # --- working resolutions -------------------------------------------------
    detect_size: tuple[int, int] = (150, 225)  # logo location / title detection
    cluster_size: tuple[int, int] = (80, 120)  # artwork clustering
    border_trim: float = 0.03  # ignore this fraction at each edge (frames, borders)

    # --- logo location (TMDB logo PNG -> position on poster) -----------------
    # Logo widths tried, as a fraction of poster width.
    logo_scales: tuple[float, ...] = tuple(np.round(np.linspace(0.28, 0.98, 11), 3))
    logo_max_height: float = 0.55  # a title treatment taller than this isn't plausible
    # Location acceptance: NCC must beat logo_match_min + logo_small_penalty
    # (scaled down to 0 at full width) by logo_match_margin. Tuned on the
    # 44-poster set: 34 correct locations, no false ones.
    logo_match_min: float = 0.42
    logo_small_penalty: float = 0.33
    logo_match_margin: float = 0.05
    logo_mask_dilate: int = 2  # px (detect resolution) around the letterforms

    # --- title detection fallback (no TMDB logo) ------------------------------
    # Tuned on the 35 posters whose logo location is known: 31 found, 0 wrong.
    text_gradient_min: float = 0.26  # OKLab L step that counts as a letter edge
    text_close: tuple[int, int] = (3, 9)  # (rows, cols) closing to merge letters into words
    text_min_width: float = 0.30  # of poster width
    text_height: tuple[float, float] = (0.035, 0.40)  # of poster height
    text_min_density: float = 0.12  # edge pixels inside the box
    text_min_score: float = 0.05
    # Letter colours: at least this share of the block, and this much more
    # common inside the block than in the ring around it.
    text_fg_min_share: float = 0.08
    text_fg_enrichment: float = 0.06

    # --- logo colours -------------------------------------------------------
    logo_clusters: int = 6
    logo_max_samples: int = 4000
    logo_merge_distance: float = 0.08  # e.g. #D83D48 vs #E1444C -> one colour
    logo_min_share: float = 0.06  # of the logo's pixels
    # "Meaningful chromatic": OKLCH chroma at/above this. Keeps cream, gold,
    # beige, deep navy, dark red; drops white, black and neutral greys.
    logo_min_chroma: float = 0.035
    logo_dark_lightness: float = 0.30  # below this lightness...
    logo_dark_chroma_slope: float = 0.25  # ...required chroma rises (L 0.12 -> 0.08)
    # A TMDB logo colour that can't be located on the poster is only used if the
    # poster really contains it (>= this share of pixels within trace distance).
    logo_trace_distance: float = 0.10
    logo_trace_share: float = 0.004
    # Located logos: a TMDB logo colour must cover this share of the poster's
    # letter pixels to count; poster samples this close to the surrounding
    # background are discarded.
    logo_located_trace_share: float = 0.05
    logo_background_distance: float = 0.08
    logo_sampled_min_chroma: float = 0.045
    # Metallic (silver) treatments: low chroma but a strong lightness gradient.
    logo_metallic_lightness_std: float = 0.12
    logo_metallic_lightness: tuple[float, float] = (0.45, 0.92)

    # --- artwork clusters & dominant ladder ---------------------------------
    n_clusters: int = 16
    kmeans_iterations: int = 30
    seed: int = 0
    merge_distance: float = 0.065
    near_black: tuple[float, float] = (0.22, 0.04)  # (max L, max chroma) -> one black
    near_white: tuple[float, float] = (0.93, 0.03)  # (min L, max chroma) -> one white
    min_coverage: float = 0.01
    grid: tuple[int, int] = (4, 6)  # cells for "spatial coverage"
    grid_presence: float = 0.08  # a cluster "appears" in a cell above this share
    ladder_length: int = 8
    # Weight of dominance (vs vividness) per ladder slot: slots 1-2 are almost
    # pure coverage/spatial significance, later slots allow a little vividness.
    ladder_dominance_weights: tuple[float, ...] = (0.92, 0.9, 0.8, 0.72, 0.62, 0.6, 0.6, 0.6)
    chroma_full: float = 0.20
    contrast_full: float = 0.40

    # --- base (upper-corner environment) --------------------------------------
    base_band: float = 0.18  # top fraction of the poster
    base_corner_width: float = 0.35  # each corner window, fraction of width
    base_related_distance: float = 0.12  # corner colours this close are "related"

    # --- role separation -------------------------------------------------------
    min_distance: float = 0.10  # OKLab; ~0.02 is a just-noticeable difference
    relax_steps: tuple[float, ...] = (1.0, 0.6, 0.35)
    same_hue_deg: float = 20.0  # same hue family: within this hue angle...
    same_hue_max_distance: float = 0.18  # ...and closer than this overall
    same_hue_min_chroma: float = 0.05
    neutral_chroma: float = 0.035  # below: grey/black/white for identity purposes
    accent_lookahead: int = 2  # accent chooses among the next N unused dominants
    accent_vividness: float = 0.35  # extra weight on chroma/distinctiveness for accent


DEFAULT_CONFIG = SemanticConfig()


@dataclass
class Colour:
    hex: str
    oklab: tuple[float, float, float]
    share: float = 0.0  # coverage within its source (logo pixels or unmasked artwork)
    source: str = ""  # "logo" | "artwork" | "base" | "derived"
    info: dict = field(default_factory=dict)

    @property
    def L(self) -> float:
        return self.oklab[0]

    @property
    def C(self) -> float:
        return float(np.hypot(self.oklab[1], self.oklab[2]))

    @property
    def hue(self) -> float:
        return float(np.degrees(np.arctan2(self.oklab[2], self.oklab[1])) % 360)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["oklab"] = [round(v, 4) for v in self.oklab]
        d["share"] = round(self.share, 4)
        d["chroma"] = round(self.C, 4)
        d["info"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.info.items()}
        return d


@dataclass
class SemanticPalette:
    base: Colour
    identity1: Colour
    identity2: Colour
    highlight1: Colour
    highlight2: Colour
    accent: Colour
    # "tmdb-located": TMDB logo found on the poster (colours sampled there);
    # "tmdb": TMDB logo not found on the poster (its colours, if traceable);
    # "detected": no TMDB logo, title found by text detection; "none".
    logo_source: str
    logo_match: float | None  # edge NCC of the TMDB logo on the poster, or detection score
    logo_box: tuple[int, int, int, int] | None  # (x0, y0, x1, y1) as poster fractions*1000
    logo_colours: list[Colour]
    ladder: list[Colour]

    def hexes(self) -> dict[str, str]:
        return {r: getattr(self, r).hex for r in ROLES}

    def to_dict(self) -> dict:
        return {
            "palette": self.hexes(),
            "sources": {r: getattr(self, r).source for r in ROLES},
            "roles": {r: getattr(self, r).to_dict() for r in ROLES},
            "logo": {
                "source": self.logo_source,
                "match": None if self.logo_match is None else round(self.logo_match, 3),
                "box": self.logo_box,
                "colours": [c.to_dict() for c in self.logo_colours],
            },
            "ladder": [c.to_dict() for c in self.ladder],
        }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _rgb(image: Image.Image, size: tuple[int, int] | None = None) -> np.ndarray:
    img = image
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    img = img.convert("RGB")
    if size:
        img = img.resize(size, Image.Resampling.BOX)
    return np.asarray(img, dtype=np.uint8)


def _colour(lab_pixels: np.ndarray, **kw) -> Colour:
    """Representative colour of a pixel group: OKLab mean, rescaled to the
    group's typical chroma (averaging several hues otherwise dulls it)."""
    centre = lab_pixels.mean(axis=0)
    mean_c = float(np.hypot(lab_pixels[:, 1], lab_pixels[:, 2]).mean())
    cc = float(np.hypot(centre[1], centre[2]))
    if cc > 1e-6:
        centre = centre.copy()
        centre[1:] *= mean_c / cc
    return Colour(rgb_to_hex(oklab_to_srgb(centre[None])[0]), tuple(float(v) for v in centre), **kw)


def dist(a: Colour, b: Colour) -> float:
    return float(np.linalg.norm(np.subtract(a.oklab, b.oklab)))


def _too_close(c: Colour, others: list[Colour], cfg: SemanticConfig, scale: float, hue_rule: bool = True) -> bool:
    for o in others:
        d = dist(c, o)
        if d < cfg.min_distance * scale:
            return True
        # Same hue family and not far apart overall: slightly different shades of
        # one blue shouldn't fill several roles. (Not applied to logo colours:
        # a red title on a red-sky base is still the title's identity colour.)
        if hue_rule and min(c.C, o.C) >= cfg.same_hue_min_chroma and d < cfg.same_hue_max_distance * scale:
            dh = abs(c.hue - o.hue) % 360
            if min(dh, 360 - dh) < cfg.same_hue_deg:
                return True
    return False


def _is_neutral(c: Colour, cfg: SemanticConfig) -> bool:
    return c.C < cfg.neutral_chroma


def _cluster(lab: np.ndarray, k: int, cfg: SemanticConfig, merge: float) -> np.ndarray:
    """k-means + merging of near-duplicates (all near-blacks one black, all
    near-whites one white, then closest pairs under `merge`)."""
    if len(lab) < 4 * k:  # tiny inputs: don't ask for more clusters than colours
        k = max(1, min(k, len(np.unique(np.round(lab, 3), axis=0))))
    labels = _kmeans(lab, k, cfg.kmeans_iterations, cfg.seed)

    def centres():
        return {int(i): lab[labels == i].mean(axis=0) for i in np.unique(labels)}

    cs = centres()
    for is_group in (
        lambda c: c[0] <= cfg.near_black[0] and np.hypot(c[1], c[2]) <= cfg.near_black[1],
        lambda c: c[0] >= cfg.near_white[0] and np.hypot(c[1], c[2]) <= cfg.near_white[1],
    ):
        group = [i for i, c in cs.items() if is_group(c)]
        for i in group[1:]:
            labels[labels == i] = group[0]
    while True:
        cs = centres()
        ids = list(cs)
        if len(ids) < 2:
            return labels
        table = np.array([cs[i] for i in ids])
        d = np.linalg.norm(table[:, None] - table[None], axis=2)
        np.fill_diagonal(d, np.inf)
        a, b = np.unravel_index(d.argmin(), d.shape)
        if d[a, b] >= merge:
            return labels
        labels[labels == ids[b]] = ids[a]


# ---------------------------------------------------------------------------
# 1. title / logo
# ---------------------------------------------------------------------------


def _edges(values: np.ndarray) -> np.ndarray:
    g = np.hypot(ndimage.sobel(values, axis=0), ndimage.sobel(values, axis=1))
    return ndimage.gaussian_filter(g, 1.0)


def _box_sums(a: np.ndarray, th: int, tw: int) -> np.ndarray:
    """Sum of every th x tw window (valid positions), via an integral image."""
    c = np.pad(a, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    return c[th:, tw:] - c[:-th, tw:] - c[th:, :-tw] + c[:-th, :-tw]


def locate_logo(poster_L: np.ndarray, logo_rgba: np.ndarray, cfg: SemanticConfig):
    """Find the TMDB logo on the poster.

    Template = edges of the logo's own lightness composited on mid-grey (so
    letter interiors, outlines and inner strokes all count, not just the
    silhouette). Score = normalised cross-correlation with the poster's edges,
    minus a size-dependent threshold: small templates match random texture
    easily, so they must correlate much better to be believed.
    Returns (margin, ncc, mask, box, colour_mask); mask/box are None when the
    best match isn't convincing.
    """
    H, W = poster_L.shape
    alpha = logo_rgba[:, :, 3].astype(np.float64) / 255.0
    ys, xs = np.nonzero(alpha > 0.5)
    if len(ys) < 20:
        return -1.0, 0.0, None, None, None
    sl = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    alpha = alpha[sl]
    logo_L = srgb_to_oklab(logo_rgba[sl][:, :, :3].reshape(-1, 3)).reshape(alpha.shape + (3,))[:, :, 0]
    comp = np.clip(logo_L * alpha + 0.5 * (1 - alpha), 0, 1)
    aspect = alpha.shape[0] / alpha.shape[1]
    E = _edges(poster_L)
    E_sq = E**2
    fshape = (sfft.next_fast_len(2 * H, real=True), sfft.next_fast_len(2 * W, real=True))
    E_f = sfft.rfft2(E, fshape, workers=1)
    lo = min(cfg.logo_scales)
    best = (-9.0, 0.0, None)
    for s in cfg.logo_scales:
        tw = max(8, int(round(W * s)))
        th = max(4, int(round(tw * aspect)))
        if th > H * cfg.logo_max_height or tw > W or th > H:
            continue
        t_img = np.asarray(Image.fromarray((comp * 255).astype(np.uint8)).resize((tw, th), Image.Resampling.BOX))
        T = _edges(t_img.astype(np.float64) / 255.0)
        T = T - T.mean()
        t_norm = np.sqrt((T**2).sum())
        if t_norm < 1e-9:
            continue
        full = sfft.irfft2(E_f * sfft.rfft2(T[::-1, ::-1], fshape), fshape)
        num = full[th - 1 : H, tw - 1 : W]
        s1, s2 = _box_sums(E, th, tw), _box_sums(E_sq, th, tw)
        ncc = num / (np.sqrt(np.maximum(s2 - s1**2 / T.size, 1e-9)) * t_norm)
        y, x = np.unravel_index(int(ncc.argmax()), ncc.shape)
        need = cfg.logo_match_min + cfg.logo_small_penalty * (1 - s) / max(1 - lo, 1e-6)
        margin = float(ncc[y, x]) - need
        if margin > best[0]:
            best = (margin, float(ncc[y, x]), (x, y, tw, th))
    margin, ncc_val, where = best
    if where is None or margin < cfg.logo_match_margin:
        return margin, ncc_val, None, None, None
    x, y, tw, th = where
    a = np.asarray(Image.fromarray((alpha * 255).astype(np.uint8)).resize((tw, th), Image.Resampling.BOX))
    solid = np.zeros((H, W), dtype=bool)
    solid[y : y + th, x : x + tw] = a > 200  # letter interiors, for sampling colours
    mask = np.zeros((H, W), dtype=bool)
    mask[y : y + th, x : x + tw] = a > 60
    mask = ndimage.binary_dilation(mask, iterations=cfg.logo_mask_dilate)  # for masking the artwork
    box = (int(1000 * x / W), int(1000 * y / H), int(1000 * (x + tw) / W), int(1000 * (y + th) / H))
    return margin, ncc_val, mask, box, solid


def detect_title(lab_img: np.ndarray, cfg: SemanticConfig):
    """Fallback title detector (no TMDB logo): the most title-like text block.

    Letter edges (strong lightness steps) are closed horizontally into word
    blocks; each block is scored on width, height, edge density and position.
    Tall-enough, wide blocks win; the billing block (tiny text at the bottom),
    taglines and credits are too short or too sparse to qualify.
    Returns (score, mask, box, foreground_lab_pixels).
    """
    H, W, _ = lab_img.shape
    L = lab_img[:, :, 0]
    grad = np.hypot(ndimage.sobel(L, axis=0), ndimage.sobel(L, axis=1)) / 4.0
    strokes = grad > cfg.text_gradient_min
    blocks = ndimage.binary_closing(strokes, structure=np.ones(cfg.text_close))
    blocks = ndimage.binary_opening(blocks, structure=np.ones((2, 2)))
    labels, n = ndimage.label(blocks)
    best = (0.0, None)
    for i, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is None:
            continue
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        wf, hf = w / W, h / H
        if wf < cfg.text_min_width or not (cfg.text_height[0] <= hf <= cfg.text_height[1]):
            continue
        density = float(strokes[sl].mean())
        if density < cfg.text_min_density:
            continue
        cy = (sl[0].start + sl[0].stop) / 2 / H
        cx = (sl[1].start + sl[1].stop) / 2 / W
        position = 1.0 if (cy < 0.35 or cy > 0.6) else 0.6  # titles sit top or bottom
        centred = 1.0 - min(abs(cx - 0.5) * 1.5, 0.6)
        score = wf * hf**0.5 * density * position * centred
        if score > best[0]:
            best = (score, sl)
    score, sl = best
    if sl is None or score < cfg.text_min_score:
        return 0.0, None, None, None
    # Letters = colours concentrated inside the title block compared with a
    # ring just outside it (a title's colour is over-represented in its box;
    # the background continues outside).
    pad_y, pad_x = max(2, int(0.25 * (sl[0].stop - sl[0].start))), max(2, int(0.08 * (sl[1].stop - sl[1].start)))
    outer = (slice(max(0, sl[0].start - pad_y), min(H, sl[0].stop + pad_y)),
             slice(max(0, sl[1].start - pad_x), min(W, sl[1].stop + pad_x)))
    inside = np.zeros((H, W), dtype=bool)
    inside[sl] = True
    ring = np.zeros((H, W), dtype=bool)
    ring[outer] = True
    ring &= ~inside
    region = lab_img[sl]
    px_in, px_ring = lab_img[inside], lab_img[ring]
    labels = _cluster(np.concatenate([px_in, px_ring]), 6, cfg, cfg.logo_merge_distance)
    lab_in, lab_ring = labels[: len(px_in)], labels[len(px_in) :]
    fg_ids = []
    for i in np.unique(labels):
        share_in = float((lab_in == i).mean())
        share_ring = float((lab_ring == i).mean()) if len(lab_ring) else 0.0
        if share_in >= cfg.text_fg_min_share and share_in - share_ring >= cfg.text_fg_enrichment:
            fg_ids.append(i)
    if not fg_ids:
        return 0.0, None, None, None
    fg_flat = np.isin(lab_in, fg_ids)
    fg = fg_flat.reshape(region.shape[:2])
    mask = np.zeros((H, W), dtype=bool)
    mask[sl] = fg
    mask = ndimage.binary_dilation(mask, iterations=1)
    box = (int(1000 * sl[1].start / W), int(1000 * sl[0].start / H), int(1000 * sl[1].stop / W), int(1000 * sl[0].stop / H))
    return float(score), mask, box, region[fg]


def logo_colours(lab: np.ndarray, cfg: SemanticConfig) -> list[Colour]:
    """Usable identity colours of a title treatment, best first (max 2).

    Clusters the logo's pixels, merges near-duplicates, then keeps colours with
    real presence and genuine chroma (or a metallic gradient). White, black
    and neutral greys are dropped.
    """
    if lab is None or len(lab) < 10:
        return []
    if len(lab) > cfg.logo_max_samples:  # deterministic, evenly spaced subsample
        lab = lab[:: int(np.ceil(len(lab) / cfg.logo_max_samples))]
    labels = _cluster(lab, cfg.logo_clusters, cfg, cfg.logo_merge_distance)
    out: list[Colour] = []
    for i in np.unique(labels):
        px = lab[labels == i]
        share = len(px) / len(lab)
        if share < cfg.logo_min_share:
            continue
        c = _colour(px, share=share, source="logo")
        l_std = float(px[:, 0].std())
        lo, hi = cfg.logo_metallic_lightness
        metallic = l_std >= cfg.logo_metallic_lightness_std and lo <= c.L <= hi and share >= 0.25
        # Darker colours need more chroma to count: very dark red/navy with
        # genuine colour qualifies, a near-black shadow or outline doesn't.
        need = cfg.logo_min_chroma + max(0.0, cfg.logo_dark_lightness - c.L) * cfg.logo_dark_chroma_slope
        chromatic = c.C >= need
        c.info = {"lightness_std": l_std, "chromatic": chromatic, "metallic": metallic}
        if chromatic or metallic:
            # Rank: presence first, then vividness.
            c.info["score"] = 0.6 * share**0.5 + 0.4 * min(c.C / cfg.chroma_full, 1.0)
            out.append(c)
    out.sort(key=lambda c: (-c.info["score"], c.hex))
    picked: list[Colour] = []
    for c in out:
        if not _too_close(c, picked, cfg, 1.0):
            picked.append(c)
        if len(picked) == 2:
            break
    return picked


# ---------------------------------------------------------------------------
# 2. artwork dominant ladder
# ---------------------------------------------------------------------------


def dominant_ladder(lab_img: np.ndarray, keep: np.ndarray, cfg: SemanticConfig):
    """Ranked, perceptually distinct artwork colours (logo pixels excluded).

    Returns (ladder, cluster_labels_map, cluster_colours)."""
    H, W, _ = lab_img.shape
    lab = lab_img[keep]
    labels = _cluster(lab, cfg.n_clusters, cfg, cfg.merge_distance)
    label_map = np.full((H, W), -1)
    label_map[keep] = labels
    n = len(lab)
    ids = [int(i) for i in np.unique(labels)]
    colours: dict[int, Colour] = {}
    gy, gx = cfg.grid[1], cfg.grid[0]
    for i in ids:
        px = lab[labels == i]
        share = len(px) / n
        m = label_map == i
        # Spatial coverage: share of grid cells where this colour really appears.
        cells = 0
        for r in range(gy):
            for q in range(gx):
                cell = m[r * H // gy : (r + 1) * H // gy, q * W // gx : (q + 1) * W // gx]
                cells += cell.mean() >= cfg.grid_presence
        spatial = cells / (gx * gy)
        # Coherence: how often a pixel's right/down neighbour is the same colour.
        same = (m[:, :-1] & m[:, 1:]).sum() + (m[:-1, :] & m[1:, :]).sum()
        coherence = float(same / max(2 * m.sum(), 1))
        comp, nc = ndimage.label(m)
        region = float(np.bincount(comp.ravel())[1:].max() / m.sum()) if nc else 0.0
        c = _colour(px, share=share, source="artwork")
        c.info = {"spatial": spatial, "coherence": coherence, "region": region}
        colours[i] = c
    # Local contrast: mean colour difference across each colour's boundaries.
    idx = {i: n_ for n_, i in enumerate(ids)}
    dense = np.full(label_map.shape, -1)
    for i, n_ in idx.items():
        dense[label_map == i] = n_
    table = np.array([colours[i].oklab for i in ids])
    total, edges = np.zeros(len(ids)), np.zeros(len(ids))
    for a, b in ((dense[:, :-1], dense[:, 1:]), (dense[:-1, :], dense[1:, :])):
        diff = (a != b) & (a >= 0) & (b >= 0)
        x, y = a[diff], b[diff]
        d = np.linalg.norm(table[x] - table[y], axis=1)
        for side in (x, y):
            np.add.at(total, side, d)
            np.add.at(edges, side, 1)
    max_share = max(c.share for c in colours.values())
    for i, c in colours.items():
        n_ = idx[i]
        contrast = min((total[n_] / edges[n_]) / cfg.contrast_full, 1.0) if edges[n_] else 0.0
        others = [o for j, o in colours.items() if j != i]
        wsum = sum(o.share for o in others)
        distinct = sum(o.share * dist(c, o) for o in others) / wsum if wsum else 0.0
        c.info.update(
            contrast=contrast,
            distinct=min(distinct / 0.4, 1.0),
            dominance=0.55 * (c.share / max_share) ** 0.5
            + 0.25 * c.info["spatial"]
            + 0.10 * c.info["coherence"]
            + 0.10 * c.info["region"],
            vividness=0.6 * min(c.C / cfg.chroma_full, 1.0) + 0.4 * contrast,
        )
    pool = [c for c in colours.values() if c.share >= cfg.min_coverage] or list(colours.values())
    ladder: list[Colour] = []
    for slot in range(cfg.ladder_length):
        w = cfg.ladder_dominance_weights[min(slot, len(cfg.ladder_dominance_weights) - 1)]
        remaining = [c for c in pool if all(c is not p for p in ladder)]
        if not remaining:
            break
        for scale in cfg.relax_steps:
            ok = [c for c in remaining if not _too_close(c, ladder, cfg, scale)]
            if ok:
                break
        else:
            ok = remaining if len(ladder) < 5 else []
        if not ok:
            break
        pick = max(ok, key=lambda c: (w * c.info["dominance"] + (1 - w) * c.info["vividness"], c.share, c.hex))
        pick.info["rank"] = slot + 1
        ladder.append(pick)
    return ladder, label_map, colours


# ---------------------------------------------------------------------------
# 3. base
# ---------------------------------------------------------------------------


def base_colour(lab_img: np.ndarray, label_map: np.ndarray, colours: dict[int, Colour], cfg: SemanticConfig) -> Colour:
    """Environment colour meeting the two upper corners (top band, both corners)."""
    H, W, _ = lab_img.shape
    band = slice(0, max(2, int(round(H * cfg.base_band))))
    cw = max(2, int(round(W * cfg.base_corner_width)))
    left, right = label_map[band, :cw], label_map[band, W - cw :]

    def shares(region):
        vals = region[region >= 0]
        if not len(vals):
            return {}
        ids, counts = np.unique(vals, return_counts=True)
        return {int(i): c / len(vals) for i, c in zip(ids, counts)}

    sl, sr = shares(left), shares(right)
    top_band = label_map[band]
    weights = np.ones(top_band.shape)
    weights[:, :cw] = weights[:, W - cw :] = 2.0  # corners count double
    band_share: dict[int, float] = {}
    for i in np.unique(top_band[top_band >= 0]):
        band_share[int(i)] = float(weights[top_band == i].sum() / weights[top_band >= 0].sum())
    if not band_share:  # the whole top band is logo: fall back to the whole image
        band_share = {i: c.share for i, c in colours.items()}
        chosen = [max(band_share, key=band_share.get)]
    else:
        l_top = max(sl, key=sl.get) if sl else None
        r_top = max(sr, key=sr.get) if sr else None
        if l_top is not None and r_top is not None and (
            l_top == r_top or dist(colours[l_top], colours[r_top]) < cfg.base_related_distance
        ):
            chosen = sorted({l_top, r_top})  # related corners: blend them
        else:
            chosen = [max(band_share, key=band_share.get)]  # differ: best overall upper environment
    sel = np.isin(top_band, chosen)
    px = lab_img[band][sel] if sel.any() else np.concatenate([lab_img[label_map == i] for i in chosen])
    c = _colour(px, share=float(sum(band_share.get(i, 0) for i in chosen)), source="base")
    c.info = {"clusters": [colours[i].hex for i in chosen]}
    return c


# ---------------------------------------------------------------------------
# 4. role assignment
# ---------------------------------------------------------------------------


def assign(base: Colour, logo: list[Colour], ladder: list[Colour], extra: list[Colour], cfg: SemanticConfig):
    """Fill identity/highlight/accent from the logo queue and artwork queue."""
    assigned: list[Colour] = [base]
    used: set[int] = set()  # indices into ladder consumed so far

    def take_artwork(prefer_colourful: bool, scale: float = 1.0) -> Colour | None:
        order = [i for i in range(len(ladder)) if i not in used]
        if prefer_colourful and any(not _is_neutral(ladder[i], cfg) for i in order):
            order = [i for i in order if not _is_neutral(ladder[i], cfg)] + [
                i for i in order if _is_neutral(ladder[i], cfg)
            ]
        for i in order:
            if not _too_close(ladder[i], assigned, cfg, scale):
                used.add(i)
                return ladder[i]
        return None

    def fallback(prefer_colourful: bool) -> Colour:
        for scale in cfg.relax_steps[1:]:
            c = take_artwork(prefer_colourful, scale)
            if c:
                return c
        for c in extra:  # any real artwork colour not already used
            if all(c is not a for a in assigned) and not _too_close(c, assigned, cfg, cfg.relax_steps[-1]):
                return c
        # Last resort (flat image): a tint/shade of base, still on its hue.
        L, a, b = base.oklab
        k = len(assigned)
        L2 = L + (0.18 + 0.06 * k) * (1 if L < 0.5 else -1)
        lab = np.array([[min(max(L2, 0.02), 0.98), a, b]])
        return Colour(rgb_to_hex(oklab_to_srgb(lab)[0]), tuple(float(v) for v in lab[0]), source="derived")

    roles: dict[str, Colour] = {}
    # Identity: logo queue first (duplicate-protected), else artwork.
    logo_q = list(logo)
    for role in ("identity1", "identity2"):
        pick = None
        while logo_q and pick is None:
            cand = logo_q.pop(0)
            if not _too_close(cand, assigned, cfg, 1.0, hue_rule=False):
                pick = cand
        if pick is None:
            pick = take_artwork(prefer_colourful=True) or fallback(prefer_colourful=True)
        roles[role] = pick
        assigned.append(pick)
    for role in ("highlight1", "highlight2"):
        pick = take_artwork(prefer_colourful=False) or fallback(prefer_colourful=False)
        roles[role] = pick
        assigned.append(pick)
    # Accent: the next unused dominants, leaning a little towards vividness.
    options = [i for i in range(len(ladder)) if i not in used and not _too_close(ladder[i], assigned, cfg, 1.0)]
    options = options[: cfg.accent_lookahead]
    if options:
        v = cfg.accent_vividness

        def accent_score(i):
            c = ladder[i]
            return (1 - v) * c.info["dominance"] + v * (0.6 * min(c.C / cfg.chroma_full, 1) + 0.4 * c.info["distinct"])

        i = max(options, key=lambda i: (accent_score(i), -i))
        used.add(i)
        roles["accent"] = ladder[i]
    else:
        roles["accent"] = fallback(prefer_colourful=True)
    return roles


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------


def _background_ring(lab_d: np.ndarray, mask: np.ndarray, cfg: SemanticConfig) -> np.ndarray:
    """Main colours immediately around the located letters (what shows through
    between thin strokes); poster samples matching these aren't logo colours."""
    ring = ndimage.binary_dilation(mask, iterations=3) & ~mask
    px = lab_d[ring]
    if len(px) < 10:
        return np.zeros((0, 3))
    labels = _cluster(px, 3, cfg, cfg.logo_merge_distance)
    return np.array([px[labels == i].mean(axis=0) for i in np.unique(labels) if (labels == i).mean() >= 0.2])


def _located_logo_colours(poster, tmdb_lab, solid, bg, cfg: SemanticConfig) -> list[Colour]:
    """Identity colours of a logo located on the poster.

    1. TMDB logo colours (clean, exact) that the poster really shows under the
       letters: preferred, they aren't blurred by outlines or backgrounds.
    2. The poster's own samples under the letters, minus anything matching the
       surrounding background: tops up when the poster recolours the logo.
    """
    sampled = _sample_full_res(poster, solid)
    out: list[Colour] = []
    if tmdb_lab is not None and sampled is not None:
        for c in logo_colours(tmdb_lab, cfg):
            near = (np.linalg.norm(sampled - np.array(c.oklab), axis=1) < cfg.logo_trace_distance * 1.2).mean()
            if near >= cfg.logo_located_trace_share:
                c.info["traced"] = float(near)
                out.append(c)
    if sampled is not None and len(out) < 2:
        if len(bg):
            d = np.linalg.norm(sampled[:, None, :] - bg[None], axis=2).min(axis=1)
            sampled = sampled[d >= cfg.logo_background_distance]
        for c in logo_colours(sampled, cfg):
            # Thin white letters over a coloured background sample as faintly
            # tinted greys, so poster samples need a little more chroma.
            if c.info.get("chromatic") and c.C < cfg.logo_sampled_min_chroma:
                continue
            if len(out) < 2 and not _too_close(c, out, cfg, 1.0, hue_rule=False):
                c.info["sampled"] = True
                out.append(c)
    return out


def _sample_full_res(poster: Image.Image, solid: np.ndarray) -> np.ndarray | None:
    """Poster pixels inside the located letterforms, at the poster's own
    resolution and eroded by a pixel, so thin strokes aren't averaged with
    the background behind them."""
    full = _rgb(poster)
    H, W = full.shape[:2]
    m = np.asarray(Image.fromarray(solid.astype(np.uint8) * 255).resize((W, H), Image.Resampling.NEAREST)) > 0
    inner = ndimage.binary_erosion(m, iterations=1)
    if inner.sum() >= 30:
        m = inner
    if m.sum() < 20:
        return None
    return srgb_to_oklab(full[m])


def analyse(poster: Image.Image, logo: Image.Image | None = None, cfg: SemanticConfig = DEFAULT_CONFIG) -> SemanticPalette:
    rgb_d = _rgb(poster, cfg.detect_size)
    lab_d = srgb_to_oklab(rgb_d.reshape(-1, 3)).reshape(rgb_d.shape)
    Hd, Wd = lab_d.shape[:2]

    # 1. Title treatment. TMDB logo located on the poster -> sample the
    # poster's own pixels under the letterforms (true to this poster). Not
    # located -> the TMDB logo's colours, kept only where the poster contains
    # them, and the text detector still masks the title if it finds one.
    # No TMDB logo -> text detector alone.
    logo_source, match, box, logo_mask, logo_lab = "none", None, None, None, None
    trace_only = False
    located = None
    tmdb_lab = None
    if logo is not None:
        rgba = np.asarray(logo.convert("RGBA"))
        if max(rgba.shape[:2]) > 300:
            f = 300 / max(rgba.shape[:2])
            rgba = np.asarray(logo.convert("RGBA").resize((max(1, int(rgba.shape[1] * f)), max(1, int(rgba.shape[0] * f))), Image.Resampling.BOX))
        solid_px = rgba[:, :, 3] > 200
        if solid_px.sum() >= 20:
            tmdb_lab = srgb_to_oklab(rgba[:, :, :3][solid_px])
            margin, match, mask, box, solid = locate_logo(lab_d[:, :, 0], rgba, cfg)
            if mask is not None:
                logo_source, logo_mask = "tmdb-located", mask
                located = (solid, _background_ring(lab_d, mask, cfg))
            else:
                logo_source, logo_lab, trace_only = "tmdb", srgb_to_oklab(rgba[:, :, :3][solid_px]), True
    if logo_mask is None:
        score, mask, dbox, fg = detect_title(lab_d, cfg)
        if mask is not None:
            logo_mask = mask
            if logo_source == "none":
                logo_source, match, box, logo_lab = "detected", score, dbox, fg
            else:
                box = dbox  # masked by detection; colours still from the TMDB logo
    if located is not None:
        logo_cols = _located_logo_colours(poster, tmdb_lab, *located, cfg)
    else:
        logo_cols = logo_colours(logo_lab, cfg)
    if trace_only and logo_cols:
        poster_lab = lab_d.reshape(-1, 3)
        logo_cols = [
            c for c in logo_cols
            if (np.linalg.norm(poster_lab - np.array(c.oklab), axis=1) < cfg.logo_trace_distance).mean()
            >= cfg.logo_trace_share
        ]

    # 2. Artwork (logo masked out, borders trimmed).
    size = cfg.cluster_size
    rgb_c = _rgb(poster, size)
    lab_c = srgb_to_oklab(rgb_c.reshape(-1, 3)).reshape(rgb_c.shape)
    Hc, Wc = lab_c.shape[:2]
    keep = np.ones((Hc, Wc), dtype=bool)
    t = int(round(cfg.border_trim * min(Hc, Wc)))
    if t:
        keep[:t], keep[-t:], keep[:, :t], keep[:, -t:] = False, False, False, False
    if logo_mask is not None:
        small = np.asarray(Image.fromarray(logo_mask.astype(np.uint8) * 255).resize((Wc, Hc), Image.Resampling.BOX)) > 60
        if (keep & ~small).sum() > 0.2 * keep.sum():  # never mask away most of the poster
            keep &= ~small
    ladder, label_map, colours = dominant_ladder(lab_c, keep, cfg)

    # 3. Base, 4. roles.
    base = base_colour(lab_c, label_map, colours, cfg)
    extra = sorted(colours.values(), key=lambda c: -c.share)
    roles = assign(base, logo_cols, ladder, extra, cfg)
    return SemanticPalette(
        base=base,
        **roles,
        logo_source=logo_source,
        logo_match=match,
        logo_box=box,
        logo_colours=logo_cols,
        ladder=ladder,
    )


def analyse_bytes(poster: bytes, logo: bytes | None = None, cfg: SemanticConfig = DEFAULT_CONFIG) -> SemanticPalette:
    with Image.open(io.BytesIO(poster)) as p:
        p.load()
        if logo:
            with Image.open(io.BytesIO(logo)) as lg:
                lg.load()
                return analyse(p, lg, cfg)
        return analyse(p, None, cfg)
