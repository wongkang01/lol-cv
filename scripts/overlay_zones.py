"""
Overlay the hardcoded ZONES and OBJECTIVES from spatial.py on a real minimap
frame so the user can visually verify the zone rectangles are correct.

Loads charts/slide1_minimap_hero.png (the neutral mid-game frame picked for
the presentation), draws each zone rectangle with a distinct colour, labels
it at its centre, marks the canonical dragon/baron locations, and saves the
result to charts/zone_overlay_verification.png.

If any zone is off, just edit ZONES in src/lol_cv/features/spatial.py and
re-run this script to check the fix.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

# Keep this script self-contained — hardcode the same zones as spatial.py
# so the visualisation can't silently drift out of sync.
ZONES = {
    "top_lane": (0.0, 0.0, 0.2, 0.55),
    "mid_lane": (0.3, 0.3, 0.7, 0.7),
    "bot_lane": (0.45, 0.8, 1.0, 1.0),
    "top_jungle_blue": (0.1, 0.35, 0.35, 0.65),
    "top_jungle_red": (0.1, 0.05, 0.4, 0.35),
    "bot_jungle_blue": (0.6, 0.65, 0.9, 0.95),
    "bot_jungle_red": (0.65, 0.35, 0.9, 0.65),
    "dragon_pit": (0.55, 0.7, 0.7, 0.85),
    "baron_pit": (0.3, 0.15, 0.45, 0.3),
    "river_top": (0.2, 0.15, 0.45, 0.4),
    "river_bot": (0.55, 0.6, 0.8, 0.85),
    "blue_base": (0.0, 0.8, 0.2, 1.0),
    "red_base": (0.8, 0.0, 1.0, 0.2),
}

OBJECTIVES = {
    "dragon": (0.62, 0.78),
    "baron": (0.38, 0.22),
}

# Colour by category so overlapping zones are distinguishable at a glance.
# Using a qualitative palette inspired by tab20.
ZONE_COLORS = {
    "top_lane":        "#ff7f0e",  # orange
    "mid_lane":        "#2ca02c",  # green
    "bot_lane":        "#d62728",  # red
    "top_jungle_blue": "#1f77b4",  # blue
    "top_jungle_red":  "#9467bd",  # purple
    "bot_jungle_blue": "#17becf",  # cyan
    "bot_jungle_red":  "#e377c2",  # pink
    "dragon_pit":      "#bcbd22",  # olive
    "baron_pit":       "#8c564b",  # brown
    "river_top":       "#7f7f7f",  # grey
    "river_bot":       "#aec7e8",  # light blue
    "blue_base":       "#0d3b66",  # navy
    "red_base":        "#a52a2a",  # dark red
}

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_PNG = REPO_ROOT / "charts" / "slide1_minimap_hero.png"
OUTPUT_ALL = REPO_ROOT / "charts" / "zone_overlay_verification.png"
OUTPUT_PAIR = REPO_ROOT / "charts" / "zone_overlay_split.png"


def load_minimap(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img)


def draw_zones(ax, zones: dict, colors: dict, alpha_fill: float = 0.28):
    """Draw a labelled rectangle per zone on the given axes.

    Axes must be configured with extent=[0, 1, 1, 0] so that image
    coordinates (origin top-left, y growing downward) match the
    (x_min, y_min, x_max, y_max) convention used in ZONES.
    """
    for name, (x0, y0, x1, y1) in zones.items():
        color = colors[name]
        rect = patches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            linewidth=2.0,
            edgecolor=color,
            facecolor=color,
            alpha=alpha_fill,
        )
        ax.add_patch(rect)
        # Also draw a solid border on top so it stands out against the busy
        # minimap background.
        border = patches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            linewidth=2.0,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(border)
        # Label at the centre.
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        ax.text(
            cx, cy, name,
            ha="center", va="center",
            fontsize=8.5, fontweight="bold",
            color="white",
            bbox=dict(
                facecolor=color, edgecolor="white",
                boxstyle="round,pad=0.2", linewidth=0.8,
            ),
        )


def draw_objectives(ax, objectives: dict):
    for name, (x, y) in objectives.items():
        ax.plot(x, y, marker="*", markersize=22,
                markerfacecolor="yellow", markeredgecolor="black",
                markeredgewidth=1.5, zorder=5)
        ax.text(x, y - 0.03, name.upper(),
                ha="center", va="bottom",
                fontsize=9, fontweight="bold",
                color="yellow",
                path_effects=None,
                bbox=dict(facecolor="black", edgecolor="yellow",
                          boxstyle="round,pad=0.2", linewidth=0.8))


def build_single_overlay(img: np.ndarray) -> None:
    """All 13 zones + both objective stars on one figure."""
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img, extent=[0, 1, 1, 0])
    draw_zones(ax, ZONES, ZONE_COLORS)
    draw_objectives(ax, OBJECTIVES)

    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)  # flipped y so blue base (high y) is at the bottom
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("x (normalised)")
    ax.set_ylabel("y (normalised)")
    ax.set_title(
        "Hardcoded ZONES overlay — verify each rectangle lands on the right area",
        pad=14, fontsize=14, fontweight="bold",
    )
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.6)

    # Subtitle note
    fig.text(
        0.5, 0.02,
        "Source: src/lol_cv/features/spatial.py:29-43 | "
        "Blue base is bottom-left, red base is top-right | "
        "Yellow stars = dragon and baron pit centres",
        ha="center", fontsize=9, color="#555555",
    )

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(OUTPUT_ALL, dpi=180, facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUTPUT_ALL}")


def build_split_overlay(img: np.ndarray) -> None:
    """Two-panel version: bases/objectives on the left, lanes/jungle on the right.

    Makes the visual less crowded so you can spot small misalignments.
    """
    group_a = {
        k: ZONES[k]
        for k in ["blue_base", "red_base", "dragon_pit", "baron_pit",
                  "river_top", "river_bot"]
    }
    group_b = {
        k: ZONES[k]
        for k in ["top_lane", "mid_lane", "bot_lane",
                  "top_jungle_blue", "top_jungle_red",
                  "bot_jungle_blue", "bot_jungle_red"]
    }

    fig, axes = plt.subplots(1, 2, figsize=(17, 9))
    titles = [
        "Bases, objectives, and river",
        "Lanes and jungle quadrants",
    ]
    for ax, group, title in zip(axes, [group_a, group_b], titles):
        ax.imshow(img, extent=[0, 1, 1, 0])
        draw_zones(ax, group, ZONE_COLORS)
        if group is group_a:
            draw_objectives(ax, OBJECTIVES)
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.2, linestyle="--", linewidth=0.5)

    fig.suptitle(
        "ZONES verification — two-panel view for less visual overlap",
        fontsize=15, fontweight="bold", y=0.995,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_PAIR, dpi=180, facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUTPUT_PAIR}")


def main() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "savefig.bbox": "tight",
    })

    if not INPUT_PNG.exists():
        raise SystemExit(f"Missing {INPUT_PNG} — run scripts/extract_slide_frames.py first")

    img = load_minimap(INPUT_PNG)
    print(f"Loaded minimap: {img.shape}")

    build_single_overlay(img)
    build_split_overlay(img)


if __name__ == "__main__":
    main()
