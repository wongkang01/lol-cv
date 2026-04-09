"""
Extract the dominant colours from the zone mask and report where each one
lives on the map. Used once to identify the palette for the ZoneMask
classifier; subsequent classification uses the resulting COLOR_TO_ZONE
mapping directly (no re-extraction needed).

Prints for each dominant colour:
    - RGB tuple
    - pixel count / fraction of the image
    - centroid (x, y) in normalised [0, 1] coords
    - bounding box
    - a guess at which zone it is based on centroid position
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
MASK_PATH = REPO_ROOT / "src" / "lol_cv" / "features" / "zone_mask.png"

# Colour quantisation: round each channel to a multiple of 32 so that
# anti-aliased edges collapse onto a small palette of dominant colours.
QUANT_STEP = 32
MIN_PIXEL_FRACTION = 0.005  # ignore colours that cover <0.5% of the image


def quantise(rgb: np.ndarray, step: int = QUANT_STEP) -> np.ndarray:
    """Round each channel to a multiple of `step`, then clip to 0..255."""
    return np.clip((rgb // step) * step + step // 2, 0, 255).astype(np.uint8)


def guess_zone_by_centroid(cx: float, cy: float) -> str:
    """Rough positional heuristic — ONLY to help identify the palette.
    This is NOT the final classifier; it just gives the user a sanity hint
    next to each dominant colour so they can confirm the mapping.
    """
    if cy > 0.75 and cx < 0.25:
        return "blue_base (bottom-left corner)"
    if cy < 0.25 and cx > 0.75:
        return "red_base (top-right corner)"
    if cy < 0.25 and cx < 0.55:
        return "top area — top_lane border / top_jungle_red / baron_pit"
    if cy > 0.75 and cx > 0.45:
        return "bottom area — bot_lane border / bot_jungle_blue / dragon_pit"
    if cx < 0.25 and 0.25 < cy < 0.75:
        return "left side — top_lane border / top_jungle_blue"
    if cx > 0.75 and 0.25 < cy < 0.75:
        return "right side — bot_lane border / bot_jungle_red"
    return "centre — mid_lane / river"


def main() -> None:
    if not MASK_PATH.exists():
        sys.exit(f"Missing {MASK_PATH}")

    img = np.asarray(Image.open(MASK_PATH).convert("RGB"))
    h, w = img.shape[:2]
    total_pixels = h * w
    print(f"Mask: {w}x{h} = {total_pixels} pixels\n")

    # Quantise and count.
    flat = img.reshape(-1, 3)
    q = quantise(flat)
    counts = Counter(map(tuple, q))

    print(f"Unique quantised colours: {len(counts)}")
    print(f"Reporting colours covering >={MIN_PIXEL_FRACTION*100:.1f}% of the image\n")
    print("=" * 100)

    # Compute centroid + bbox per dominant colour.
    rows = []
    for colour, count in counts.most_common():
        frac = count / total_pixels
        if frac < MIN_PIXEL_FRACTION:
            continue
        mask = np.all(q.reshape(h, w, 3) == np.array(colour), axis=-1)
        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue
        cx = float(xs.mean() / w)
        cy = float(ys.mean() / h)
        xmin, xmax = xs.min() / w, xs.max() / w
        ymin, ymax = ys.min() / h, ys.max() / h
        rows.append({
            "colour": colour,
            "count": count,
            "frac": frac,
            "centroid": (cx, cy),
            "bbox": (xmin, ymin, xmax, ymax),
            "guess": guess_zone_by_centroid(cx, cy),
        })

    # Sort by frac descending for easy scanning
    rows.sort(key=lambda r: -r["frac"])

    print(f"{'RGB':>18s}  {'%':>6s}  {'centroid (x, y)':>20s}  {'bbox (x,y,x,y)':>28s}  guess")
    print("-" * 120)
    for r in rows:
        cx, cy = r["centroid"]
        bx1, by1, bx2, by2 = r["bbox"]
        rgb_str = f"({r['colour'][0]:>3d},{r['colour'][1]:>3d},{r['colour'][2]:>3d})"
        centroid_str = f"({cx:.3f}, {cy:.3f})"
        bbox_str = f"({bx1:.2f},{by1:.2f},{bx2:.2f},{by2:.2f})"
        print(
            f"{rgb_str:>18s}  {r['frac']*100:>5.2f}%  "
            f"{centroid_str:>20s}  {bbox_str:>28s}  {r['guess']}"
        )

    print("\nDominant colours above MIN_PIXEL_FRACTION:", len(rows))
    print(
        "\nReview this table, then fill in the COLOR_TO_ZONE dict "
        "in scripts/build_zone_mask_overlay.py"
    )


if __name__ == "__main__":
    main()
