"""
Build a pixel-mask-based zone classifier from src/lol_cv/features/zone_mask.png
and verify it by overlaying the classified zones on a real minimap frame.

Workflow:
    1. Load the mask image.
    2. For each pixel in the mask, find the nearest palette colour and assign
       the corresponding zone label. Pre-compute this as a (H, W) zone grid so
       classification is O(1) per query.
    3. Apply the river split rule: pixels labelled ``river`` get split into
       ``river_top`` vs ``river_bot`` based on the anti-diagonal `x + y < 1.0`.
    4. Render THREE verification views saved to charts/:
         - zone_mask_classified.png — the mask itself recoloured by classified
           zone (so you can see how anti-aliased border pixels get resolved)
         - zone_mask_overlay.png — a real minimap frame with the classified
           zones overlaid as a translucent colour wash
         - zone_mask_comparison.png — side-by-side of the original rectangle
           overlay and the new mask overlay, for the "before/after" story
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
MASK_PATH = REPO_ROOT / "src" / "lol_cv" / "features" / "zone_mask.png"
MINIMAP_PATH = REPO_ROOT / "charts" / "slide1_minimap_hero.png"
OUT_CLASSIFIED = REPO_ROOT / "charts" / "zone_mask_classified.png"
OUT_OVERLAY = REPO_ROOT / "charts" / "zone_mask_overlay.png"
OUT_COMPARISON = REPO_ROOT / "charts" / "zone_mask_comparison.png"

# ── Mask scaling ──
#
# The painted mask covers the full [0, 1] × [0, 1] space of the minimap crop.
# In practice the minimap *content* (the playable map) is slightly smaller
# than the crop — there's a thin broadcast border around the map frame. We
# shrink the mask by this factor so the painted zones sit on the actual
# playable area. A factor of 1.0 disables shrinking.
#
# Shrinking means: the mask is "placed" inside the central SHRINK_FACTOR
# portion of the minimap crop. Coordinates outside that inner region are
# classified as "unknown". Iterate this value if your minimap crop has
# different border proportions.
SHRINK_FACTOR = 0.95

# ── Mask translation ──
#
# After shrinking, the mask can be nudged around inside the minimap crop.
# Units are normalised [0, 1]. A ~200 px minimap crop has ~0.005 per pixel.
#   SHIFT_X < 0 → mask moves LEFT
#   SHIFT_X > 0 → mask moves RIGHT
#   SHIFT_Y < 0 → mask moves UP
#   SHIFT_Y > 0 → mask moves DOWN  (image y-axis grows downward)
SHIFT_X = -2 / 200   # 1 pixel left (on a ~200 px wide minimap)
SHIFT_Y = +1 / 200   # 1 pixel down

# ── Palette: dominant RGB → zone name ───────────────────────────────────────
#
# These anchors are the most-common quantised colours from the mask (see
# scripts/extract_mask_colors.py). At classification time we snap every pixel
# to its nearest anchor via L2 distance, which resolves anti-aliased edges
# cleanly and is robust to minor colour drift.
COLOR_TO_ZONE: dict[tuple[int, int, int], str] = {
    (16, 16, 16):       "unknown",          # black background outside the map
    (112, 16, 16):      "red_base",         # dark red, top-right corner
    (16, 48, 112):      "blue_base",        # dark navy, bottom-left corner
    (48, 80, 16):       "top_jungle_blue",  # dark green, left side
    (112, 80, 48):      "bot_jungle_red",   # brown, right side
    (240, 240, 80):     "bot_lane",         # yellow, bottom-right L-shape
    (176, 48, 112):     "top_lane",         # pink/magenta, top-left L-shape
    (208, 112, 48):     "top_jungle_red",   # orange, top centre
    (80, 48, 144):      "bot_jungle_blue",  # purple, bottom centre
    (208, 208, 176):    "mid_lane",         # light grey/beige diagonal bar
    (112, 208, 208):    "river",            # cyan — will be split by (x+y)
    (208, 176, 48):     "baron_pit",        # mustard, near baron area
    (176, 240, 80):     "dragon_pit",       # lime, near dragon area
}

# Display colours for the overlay — one RGB per zone. These are the
# visualisation-only colours; the classifier's output is the string zone name.
ZONE_DISPLAY_COLORS: dict[str, tuple[float, float, float]] = {
    "blue_base":        (0.12, 0.24, 0.55),   # navy
    "red_base":         (0.55, 0.12, 0.12),   # dark red
    "top_lane":         (0.85, 0.25, 0.55),   # pink
    "bot_lane":         (0.95, 0.85, 0.25),   # yellow
    "top_jungle_blue":  (0.20, 0.50, 0.20),   # green
    "top_jungle_red":   (1.00, 0.55, 0.15),   # orange
    "bot_jungle_red":   (0.55, 0.35, 0.20),   # brown
    "bot_jungle_blue":  (0.45, 0.20, 0.70),   # purple
    "mid_lane":         (0.85, 0.85, 0.75),   # light beige
    "river_top":        (0.30, 0.85, 0.95),   # bright cyan (baron side)
    "river_bot":        (0.45, 0.85, 0.90),   # cooler cyan (dragon side)
    "baron_pit":        (0.85, 0.75, 0.15),   # mustard
    "dragon_pit":       (0.65, 0.95, 0.30),   # lime
    "unknown":          (0.00, 0.00, 0.00),   # black — reserved
}


# ── ZoneMask classifier ─────────────────────────────────────────────────────


class ZoneMask:
    """Pixel-mask zone classifier with post-processing river split.

    Usage:
        mask = ZoneMask(path, COLOR_TO_ZONE)
        zone = mask.classify(x_norm, y_norm)
    """

    def __init__(
        self,
        mask_path: Path,
        palette: dict[tuple[int, int, int], str],
        shrink_factor: float = 1.0,
        shift_x: float = 0.0,
        shift_y: float = 0.0,
    ):
        img = np.asarray(Image.open(mask_path).convert("RGB"))
        self.h, self.w = img.shape[:2]
        self.shrink_factor = float(shrink_factor)
        self.shift_x = float(shift_x)
        self.shift_y = float(shift_y)
        # Inset applied on each side so the mask is centred at 1.0 - shrink
        # total coverage. e.g. shrink=0.9 → 0.05 border on each side.
        self._offset = (1.0 - self.shrink_factor) / 2.0

        # Pre-compute a (H, W) integer grid of zone indices by nearest-colour
        # lookup. This is faster than classifying each point lazily and also
        # lets us visualise the resolved classification directly.
        anchors = np.array(list(palette.keys()), dtype=np.int32)  # (K, 3)
        self._labels = np.array(list(palette.values()))           # (K,)

        pixels = img.astype(np.int32).reshape(-1, 3)  # (H*W, 3)
        # L2 distance to each anchor; take argmin
        dists = np.sum(
            (pixels[:, None, :] - anchors[None, :, :]) ** 2, axis=-1
        )  # (H*W, K)
        nearest = np.argmin(dists, axis=-1)  # (H*W,)
        self.zone_idx_grid = nearest.reshape(self.h, self.w).astype(np.int32)

    # ── coordinate remapping ──
    def _minimap_to_mask(self, x: float, y: float) -> tuple[float, float]:
        """Map a normalised minimap coordinate into the mask's coordinate
        space, accounting for shrink and translation.

        The shift is applied BEFORE the shrink: the caller's (x, y) is first
        moved by (-shift_x, -shift_y) so that a mask shifted right by Δx
        (positive shift_x) is "found" Δx to the left of where we'd normally
        look. Then the shrink-inset is applied.
        """
        xs = x - self.shift_x
        ys = y - self.shift_y
        mx = (xs - self._offset) / self.shrink_factor
        my = (ys - self._offset) / self.shrink_factor
        return mx, my

    # ── single-point classification ────────────────────────────────────
    def classify(self, x: float, y: float) -> str:
        """Return the zone name at normalised coordinates (x, y) in [0, 1].

        Points outside the inset (i.e. in the broadcast border that the
        painted mask doesn't cover) return ``unknown``.
        """
        mx, my = self._minimap_to_mask(x, y)
        if mx < 0.0 or mx >= 1.0 or my < 0.0 or my >= 1.0:
            return "unknown"
        px = int(np.clip(mx * self.w, 0, self.w - 1))
        py = int(np.clip(my * self.h, 0, self.h - 1))
        zone = str(self._labels[self.zone_idx_grid[py, px]])
        if zone == "river":
            return "river_top" if (x + y) < 1.0 else "river_bot"
        return zone

    # ── vectorised classification for pre-rendering ────────────────────
    def classify_grid(self, h: int, w: int) -> np.ndarray:
        """Return a (h, w) array of zone name strings at a chosen resolution.

        Coordinates outside the inset (determined by shrink_factor) are
        labelled ``unknown``. Applies the river split rule after lookup.
        """
        # Output-grid normalised coordinates (centre of each pixel)
        y_norm = (np.arange(h) + 0.5) / h  # (h,)
        x_norm = (np.arange(w) + 0.5) / w  # (w,)

        # Remap into mask space: shift first, then shrink
        my_norm = ((y_norm - self.shift_y) - self._offset) / self.shrink_factor  # (h,)
        mx_norm = ((x_norm - self.shift_x) - self._offset) / self.shrink_factor  # (w,)

        valid_y = (my_norm >= 0.0) & (my_norm < 1.0)
        valid_x = (mx_norm >= 0.0) & (mx_norm < 1.0)

        my_pix = np.clip((my_norm * self.h).astype(np.int32), 0, self.h - 1)
        mx_pix = np.clip((mx_norm * self.w).astype(np.int32), 0, self.w - 1)

        src = self.zone_idx_grid[my_pix[:, None], mx_pix[None, :]]  # (h, w)
        labels = self._labels[src].astype(object)  # (h, w) of strings

        # Points outside the inset are unknown
        valid_grid = valid_y[:, None] & valid_x[None, :]
        labels[~valid_grid] = "unknown"

        # Replace "river" with "river_top" / "river_bot" using the ORIGINAL
        # (un-shrunk) coordinates — the split diagonal is anchored to the
        # minimap, not the mask.
        diag = x_norm[None, :] + y_norm[:, None]  # (h, w)
        river_mask = labels == "river"
        labels[river_mask & (diag < 1.0)] = "river_top"
        labels[river_mask & (diag >= 1.0)] = "river_bot"

        return labels


# ── Rendering ──────────────────────────────────────────────────────────────


def zone_grid_to_rgb(
    zone_grid: np.ndarray, display_colors: dict
) -> np.ndarray:
    """Map a (h, w) string array to a (h, w, 3) float RGB image."""
    h, w = zone_grid.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    for zone_name, color in display_colors.items():
        mask = zone_grid == zone_name
        rgb[mask] = color
    return rgb


def compute_zone_centroids(
    zone_grid: np.ndarray, zones_to_label: list[str]
) -> dict[str, tuple[float, float]]:
    """Return a dict of zone → (cx, cy) in normalised [0, 1] coords.

    Used to place on-chart text labels at the centre of each zone region.
    Skips zones with no pixels in the grid.
    """
    h, w = zone_grid.shape
    out: dict[str, tuple[float, float]] = {}
    for zone in zones_to_label:
        mask = zone_grid == zone
        if not mask.any():
            continue
        ys, xs = np.where(mask)
        cx = float(xs.mean() / w)
        cy = float(ys.mean() / h)
        out[zone] = (cx, cy)
    return out


def draw_zone_labels(
    ax, centroids: dict[str, tuple[float, float]], fontsize: int = 8
) -> None:
    """Place a label at each zone centroid with a white outline for
    readability over arbitrary background colours.
    """
    import matplotlib.patheffects as pe
    for zone, (cx, cy) in centroids.items():
        ax.text(
            cx, cy, zone,
            ha="center", va="center",
            fontsize=fontsize, fontweight="bold",
            color="black",
            path_effects=[
                pe.withStroke(linewidth=2.2, foreground="white"),
            ],
        )


def build_classified_view(zone_mask: ZoneMask) -> None:
    """Render the mask with every pixel snapped to its nearest zone colour.

    This shows how anti-aliased border pixels in the original mask get
    resolved — useful for spotting any mis-assignments.
    """
    # Render at 1/4 resolution for speed
    h_out = zone_mask.h // 4
    w_out = zone_mask.w // 4
    grid = zone_mask.classify_grid(h_out, w_out)
    rgb = zone_grid_to_rgb(grid, ZONE_DISPLAY_COLORS)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(rgb, extent=[0, 1, 1, 0])
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_title(
        "Classified zone mask — every pixel snapped to nearest zone",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.set_xlabel("x (normalised)")
    ax.set_ylabel("y (normalised)")

    # Zone name labels placed at each zone's centroid
    label_zones = [
        "blue_base", "red_base", "top_lane", "bot_lane",
        "top_jungle_blue", "top_jungle_red",
        "bot_jungle_blue", "bot_jungle_red",
        "mid_lane", "river_top", "river_bot",
        "baron_pit", "dragon_pit",
    ]
    centroids = compute_zone_centroids(grid, label_zones)
    draw_zone_labels(ax, centroids, fontsize=10)

    fig.text(
        0.5, 0.015,
        "River is split into top/bot by the anti-diagonal x + y < 1.0",
        ha="center", fontsize=9, color="#555555",
    )
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(OUT_CLASSIFIED, dpi=180, facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUT_CLASSIFIED}")


def build_minimap_overlay(zone_mask: ZoneMask) -> None:
    """Overlay the classified zones on the real minimap hero frame.

    Uses translucent colour fills so the underlying minimap texture is still
    visible through the zone layer.
    """
    if not MINIMAP_PATH.exists():
        raise SystemExit(f"Missing {MINIMAP_PATH}")
    minimap = np.asarray(Image.open(MINIMAP_PATH).convert("RGB"))
    mh, mw = minimap.shape[:2]

    # Classify at the minimap's resolution
    grid = zone_mask.classify_grid(mh, mw)
    zone_rgb = zone_grid_to_rgb(grid, ZONE_DISPLAY_COLORS)

    # Blend: 55% zone wash + 45% minimap texture
    minimap_f = minimap.astype(np.float32) / 255.0
    alpha = 0.55
    blended = alpha * zone_rgb + (1 - alpha) * minimap_f
    blended = np.clip(blended, 0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8.5))

    # Left: raw minimap
    axes[0].imshow(minimap_f, extent=[0, 1, 1, 0])
    axes[0].set_title("Original minimap frame", fontsize=13, fontweight="bold")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(1, 0)

    # Right: blended overlay
    axes[1].imshow(blended, extent=[0, 1, 1, 0])
    axes[1].set_title(
        "Classified zones (mask-based)",
        fontsize=13, fontweight="bold",
    )
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(1, 0)

    # Zone name labels at centroids on the right panel
    label_zones = [
        "blue_base", "red_base", "top_lane", "bot_lane",
        "top_jungle_blue", "top_jungle_red",
        "bot_jungle_blue", "bot_jungle_red",
        "mid_lane", "river_top", "river_bot",
        "baron_pit", "dragon_pit",
    ]
    centroids = compute_zone_centroids(grid, label_zones)
    draw_zone_labels(axes[1], centroids, fontsize=9)

    for ax in axes:
        ax.set_xlabel("x (normalised)")
        ax.set_ylabel("y (normalised)")

    fig.suptitle(
        f"ZoneMask verification — real minimap overlay  "
        f"(shrink={SHRINK_FACTOR:.2f}, shift=({SHIFT_X:+.4f}, {SHIFT_Y:+.4f}))",
        fontsize=14, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(OUT_OVERLAY, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_OVERLAY}")


def build_before_after_comparison(zone_mask: ZoneMask) -> None:
    """Side-by-side: old rectangle overlay vs new mask overlay.

    Reuses the existing ``charts/zone_overlay_verification.png`` as the "old"
    panel if it exists; otherwise skips this view with a warning.
    """
    old_path = REPO_ROOT / "charts" / "zone_overlay_verification.png"
    if not old_path.exists():
        print(f"Skipping comparison — {old_path} not found")
        return

    old_img = np.asarray(Image.open(old_path).convert("RGB"))
    minimap = np.asarray(Image.open(MINIMAP_PATH).convert("RGB"))
    mh, mw = minimap.shape[:2]

    grid = zone_mask.classify_grid(mh, mw)
    zone_rgb = zone_grid_to_rgb(grid, ZONE_DISPLAY_COLORS)
    alpha = 0.55
    blended = alpha * zone_rgb + (1 - alpha) * (minimap / 255.0)
    blended = np.clip(blended, 0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    axes[0].imshow(old_img)
    axes[0].set_title(
        "Before: hardcoded rectangles", fontsize=14, fontweight="bold",
    )
    axes[0].axis("off")

    axes[1].imshow(blended, extent=[0, 1, 1, 0])
    axes[1].set_title(
        "After: pixel mask (with river split)", fontsize=14, fontweight="bold",
    )
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(1, 0)
    axes[1].set_xlabel("x (normalised)")
    axes[1].set_ylabel("y (normalised)")

    fig.suptitle(
        "Zone classification before/after — rectangles vs pixel mask",
        fontsize=16, fontweight="bold", y=0.995,
    )
    fig.tight_layout()
    fig.savefig(OUT_COMPARISON, dpi=160, facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUT_COMPARISON}")


# ── Smoke test ─────────────────────────────────────────────────────────────


def smoke_test(zone_mask: ZoneMask) -> None:
    """Probe a handful of known positions and print the classified zone.

    Catches obvious palette / coordinate-system bugs before the user eyeballs
    the rendered overlay.
    """
    probes = [
        ("blue base centre",      0.10, 0.90, "blue_base"),
        ("red base centre",       0.90, 0.10, "red_base"),
        ("baron pit centre",      0.38, 0.22, "baron_pit"),
        ("dragon pit centre",     0.62, 0.78, "dragon_pit"),
        ("mid lane centre",       0.50, 0.50, "mid_lane"),
        ("top lane middle",       0.10, 0.30, "top_lane"),
        ("bot lane middle",       0.90, 0.70, "bot_lane"),
        ("top blue jungle",       0.25, 0.50, "top_jungle_blue"),
        ("top red jungle",        0.55, 0.25, "top_jungle_red"),
        ("bot blue jungle",       0.45, 0.75, "bot_jungle_blue"),
        ("bot red jungle",        0.75, 0.45, "bot_jungle_red"),
        ("river top (near baron)", 0.40, 0.40, "river_top"),  # expected if the
                                                               # cyan X extends
                                                               # here
        ("river bot (near dragon)", 0.60, 0.60, "river_bot"),
    ]
    print("\n=== Smoke test ===")
    print(f"{'position':28s}  {'(x, y)':>14s}  {'classified':>18s}  "
          f"{'expected':>18s}  {'ok?':>4s}")
    print("-" * 100)
    hits = 0
    for name, x, y, expected in probes:
        got = zone_mask.classify(x, y)
        ok = got == expected
        hits += int(ok)
        mark = "YES" if ok else "no"
        print(f"{name:28s}  ({x:.2f}, {y:.2f})  {got:>18s}  "
              f"{expected:>18s}  {mark:>4s}")
    print(f"\nSmoke test: {hits}/{len(probes)} expected matches")
    print(
        "(Jungle / river positions are approximate — mismatches may just "
        "mean the probe point sits in a neighbouring zone in your painting.)"
    )


# ── Main ───────────────────────────────────────────────────────────────────


def main() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "savefig.bbox": "tight",
    })

    if not MASK_PATH.exists():
        sys.exit(f"Missing {MASK_PATH}")

    print(f"Loading mask from {MASK_PATH}")
    print(f"Shrink factor: {SHRINK_FACTOR:.2f} "
          f"(centred inset, {(1 - SHRINK_FACTOR) / 2 * 100:.1f}% border on each side)")
    print(f"Shift:   x={SHIFT_X:+.4f}  y={SHIFT_Y:+.4f}  "
          f"(~{SHIFT_X * 200:+.1f} px, ~{SHIFT_Y * 200:+.1f} px on a 200px minimap)")
    zone_mask = ZoneMask(
        MASK_PATH, COLOR_TO_ZONE,
        shrink_factor=SHRINK_FACTOR,
        shift_x=SHIFT_X, shift_y=SHIFT_Y,
    )
    print(f"Mask resolved to a {zone_mask.h}x{zone_mask.w} zone grid")

    # Report the distribution of zones in the classified mask
    unique, counts = np.unique(zone_mask.zone_idx_grid, return_counts=True)
    labels = zone_mask._labels[unique]
    total = counts.sum()
    print("\nZone distribution in the mask:")
    for label, count in sorted(
        zip(labels, counts), key=lambda kv: -kv[1]
    ):
        print(f"  {label:20s} {count / total * 100:6.2f}%")

    smoke_test(zone_mask)

    print("\nRendering outputs...")
    build_classified_view(zone_mask)
    build_minimap_overlay(zone_mask)
    build_before_after_comparison(zone_mask)

    print("\nDone.")


if __name__ == "__main__":
    main()
