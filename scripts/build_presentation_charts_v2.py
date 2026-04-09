"""
Build every chart referenced by docs/presentation_outline_v2.md.

All outputs land in charts/v2/ with transparent backgrounds, no figure titles
(the user puts those on the slides themselves), and the "Cinematic Nexus"
design-system palette from docs/design.md.

Run from repo root:

    python scripts/build_presentation_charts_v2.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.patheffects import Stroke, Normal

# ──────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = REPO_ROOT / "data" / "processed"
OUT_DIR = REPO_ROOT / "charts" / "v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────
# Cinematic Nexus palette (from docs/design.md)
# ──────────────────────────────────────────────────────────────────────────
SURFACE              = "#151313"
SURFACE_LOW          = "#1d1b1b"
SURFACE_HIGH         = "#2c2929"
SURFACE_HIGHEST      = "#373434"
SURFACE_BRIGHT       = "#3b3838"

PRIMARY              = "#b9f6ff"   # pale cyan — high-priority headings
PRIMARY_CONTAINER    = "#10e4f9"   # electric cyan — fills / CTAs
PRIMARY_DIM          = "#00dbef"   # deeper cyan — active indicators
SECONDARY_CONTAINER  = "#46456d"   # muted purple — chips

TERTIARY             = "#ffe6e2"   # light pink — live badges
TERTIARY_CONTAINER   = "#ff6a5c"   # red-orange — warnings / "worse"

ON_SURFACE           = "#e7e1e0"   # headings on dark
ON_SURFACE_VARIANT   = "#bac9cc"   # body text on dark
OUTLINE_VARIANT      = "#3b494b"   # "ghost border" fallback

# Cyan heatmap gradient: dark surface → electric primary-container.
# The bottom anchor is slightly lifted from SURFACE so low cells are still
# visually distinct on a transparent slide background.
HEAT_CMAP = LinearSegmentedColormap.from_list(
    "cinematic_cyan",
    [
        (0.00, "#1a1a1f"),
        (0.25, "#1f3a4a"),
        (0.50, "#146a83"),
        (0.75, "#10a9c3"),
        (1.00, "#10e4f9"),
    ],
)


# ──────────────────────────────────────────────────────────────────────────
# Style
# ──────────────────────────────────────────────────────────────────────────
def apply_style() -> None:
    """Apply shared matplotlib rcParams for the entire v2 chart set."""
    mpl.rcParams.update({
        # Transparent bg everywhere — slide substrate shows through.
        "figure.facecolor":   "none",
        "axes.facecolor":     "none",
        "savefig.facecolor":  "none",
        "savefig.edgecolor":  "none",
        "savefig.transparent": True,

        # Fonts: the design system calls for Space Grotesk (display) + Inter
        # (body) but neither ships on most machines. DejaVu Sans is a clean
        # fallback that already looks geometric enough for this aesthetic.
        "font.family": ["DejaVu Sans", "sans-serif"],
        "font.size": 11,
        "text.color":        ON_SURFACE,
        "axes.labelcolor":   ON_SURFACE_VARIANT,
        "xtick.color":       ON_SURFACE_VARIANT,
        "ytick.color":       ON_SURFACE_VARIANT,

        # "No-line" rule — kill all spines by default; components that need
        # a guide re-enable a single faint one explicitly.
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.spines.left":  False,
        "axes.spines.bottom": False,
        "axes.edgecolor":    OUTLINE_VARIANT,

        "axes.grid":         False,
        "grid.color":        OUTLINE_VARIANT,
        "grid.alpha":        0.25,
        "grid.linewidth":    0.6,

        "axes.titleweight":  "bold",
        "axes.titlecolor":   ON_SURFACE,
        "axes.titlepad":     10,

        "legend.frameon":    False,
        "legend.labelcolor": ON_SURFACE_VARIANT,

        "savefig.bbox":      "tight",
        "savefig.dpi":       200,
    })


def cyan_glow(text_obj, color: str = PRIMARY, blur: float = 3.0) -> None:
    """Apply a subtle cyan 'text outline' glow to a matplotlib text object."""
    text_obj.set_path_effects([
        Stroke(linewidth=blur, foreground=color, alpha=0.22),
        Normal(),
    ])


def glassy_panel(ax, x, y, w, h, tier="high", edge_alpha=0.35) -> None:
    """Draw a glassy tonal panel (no visible 1px border — uses a faint
    outline_variant stroke at low alpha as the 'ghost border' fallback).
    """
    tier_fill = {
        "low":     SURFACE_LOW,
        "high":    SURFACE_HIGH,
        "highest": SURFACE_HIGHEST,
        "bright":  SURFACE_BRIGHT,
    }[tier]
    rect = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0,rounding_size=0",
        facecolor=tier_fill,
        edgecolor=to_rgba(OUTLINE_VARIANT, edge_alpha),
        linewidth=1.1,
        joinstyle="miter",
    )
    ax.add_patch(rect)


# ──────────────────────────────────────────────────────────────────────────
# Chart 1 — pipeline_diagram.png
# ──────────────────────────────────────────────────────────────────────────
def build_pipeline_diagram() -> None:
    stages = [
        ("01", "VOD FRAME\nEXTRACTION",  "mp4 → 1 fps png"),
        ("02", "MINIMAP\nCROP",          "fixed crop box"),
        ("03", "YOLOv11m\nDETECTION",    "95% F1, 14.5 fps"),
        ("04", "TRACKING +\nZONE MASK",  "per-champion (x, y, zone)"),
        ("05", "FEATURE\nEXTRACTION",    "213 features / game"),
    ]

    fig, ax = plt.subplots(figsize=(18, 4.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 10)
    ax.set_axis_off()

    n = len(stages)
    box_w = 15.5
    gap_w = (100 - n * box_w) / (n - 1)
    y_bottom = 2.4
    box_h = 5.2

    for i, (num, title, sub) in enumerate(stages):
        x0 = i * (box_w + gap_w)

        # Main glass panel.
        glassy_panel(ax, x0, y_bottom, box_w, box_h, tier="high",
                     edge_alpha=0.45)

        # Cyan top accent bar (the "neon tube").
        accent_h = 0.34
        ax.add_patch(patches.Rectangle(
            (x0, y_bottom + box_h - accent_h),
            box_w, accent_h,
            facecolor=PRIMARY_CONTAINER, edgecolor="none",
        ))
        ax.add_patch(patches.Rectangle(
            (x0, y_bottom + box_h - accent_h - 0.25),
            box_w, 0.25,
            facecolor=PRIMARY, edgecolor="none", alpha=0.22,
        ))

        # Stage index (tonal authority — all caps, wide spaced).
        idx_txt = ax.text(
            x0 + 0.7, y_bottom + box_h - 1.25, num,
            fontsize=11, fontweight="bold",
            color=PRIMARY_DIM, ha="left", va="top",
        )
        idx_txt.set_fontfamily(["Space Grotesk", "DejaVu Sans"])

        # Stage title.
        title_txt = ax.text(
            x0 + box_w / 2, y_bottom + box_h * 0.54,
            title,
            fontsize=13.5, fontweight="bold",
            color=ON_SURFACE, ha="center", va="center",
            linespacing=1.05,
        )
        title_txt.set_fontfamily(["Space Grotesk", "DejaVu Sans"])
        cyan_glow(title_txt, color=PRIMARY, blur=2.4)

        # Sub-caption.
        ax.text(
            x0 + box_w / 2, y_bottom + 0.75,
            sub,
            fontsize=9.2, color=ON_SURFACE_VARIANT,
            ha="center", va="center",
        )

        # Arrow to next stage.
        if i < n - 1:
            arrow_y = y_bottom + box_h / 2
            arrow_start = x0 + box_w + 0.4
            arrow_end = x0 + box_w + gap_w - 0.4
            ax.annotate(
                "",
                xy=(arrow_end, arrow_y),
                xytext=(arrow_start, arrow_y),
                arrowprops=dict(
                    arrowstyle="-|>,head_width=0.28,head_length=0.42",
                    linewidth=1.6,
                    color=PRIMARY_DIM,
                    shrinkA=0, shrinkB=0,
                ),
            )

    fig.savefig(OUT_DIR / "pipeline_diagram.png", dpi=220)
    plt.close(fig)
    print("  wrote pipeline_diagram.png")


# ──────────────────────────────────────────────────────────────────────────
# Chart 2 — strategic_priorities_per_window.png
# ──────────────────────────────────────────────────────────────────────────
def build_strategic_priorities() -> None:
    tiles = [
        {
            "window": "0 – 5 min",
            "label":  "PICK\n& PATH",
            "category": "map_control_territory",
            "auc": 0.813,
            "takeaway": (
                "Control WHERE the five champions stand at\n"
                "t=180 and t=300. Centroid + enemy-half count\n"
                "beats everything else in the first-clear window."
            ),
        },
        {
            "window": "0 – 10 min",
            "label":  "FIRST\nDRAGON DANCE",
            "category": "objective_contestation",
            "auc": 0.907,
            "takeaway": (
                "Convergence speed to the dragon quadrant,\n"
                "grouping density around the pit, and first-\n"
                "objective timing are the decisive measurements."
            ),
        },
        {
            "window": "0 – 15 min",
            "label":  "MACRO\nSHAPE",
            "category": "map_control_territory",
            "auc": 0.975,
            "takeaway": (
                "At 12 – 15:00 laning-phase territory alone\n"
                "nearly saturates AUC. Team centroid in the\n"
                "enemy half is the single clearest signal."
            ),
        },
        {
            "window": "FULL GAME",
            "label":  "SUSTAINED\nTERRITORY",
            "category": "map_control_territory",
            "auc": 1.000,
            "takeaway": (
                "Territory → objectives → jungle pathing is the\n"
                "dominant ordering. Lane priority and kill tempo\n"
                "sit at a clear second tier (0.74 and 0.57)."
            ),
        },
    ]

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_axis_off()

    # 2×2 tiles.
    tile_w = 46
    tile_h = 42
    gap = 5
    origin_x = (100 - (2 * tile_w + gap)) / 2
    origin_y = (100 - (2 * tile_h + gap)) / 2

    coords = [
        (origin_x,                       origin_y + tile_h + gap),
        (origin_x + tile_w + gap,        origin_y + tile_h + gap),
        (origin_x,                       origin_y),
        (origin_x + tile_w + gap,        origin_y),
    ]

    for (x0, y0), tile in zip(coords, tiles):
        glassy_panel(ax, x0, y0, tile_w, tile_h, tier="high", edge_alpha=0.42)

        # Left cyan accent strip.
        ax.add_patch(patches.Rectangle(
            (x0, y0), 0.9, tile_h,
            facecolor=PRIMARY_CONTAINER, edgecolor="none",
        ))

        # Window (all caps, tonal authority).
        ax.text(
            x0 + 2.5, y0 + tile_h - 3.5,
            tile["window"].upper(),
            fontsize=12, fontweight="bold",
            color=PRIMARY_DIM, ha="left", va="top",
        )

        # Phase label (display type) — constrained to left ~60% of tile so
        # it can never overlap the AUC number in the bottom-right corner.
        lbl_txt = ax.text(
            x0 + 2.5, y0 + tile_h - 8,
            tile["label"],
            fontsize=17, fontweight="bold",
            color=ON_SURFACE, ha="left", va="top",
            linespacing=1.0,
        )
        cyan_glow(lbl_txt, color=PRIMARY, blur=3.0)

        # Category chip (below the 2-line label).
        ax.text(
            x0 + 2.5, y0 + tile_h - 22.8,
            tile["category"].replace("_", " "),
            fontsize=9.5, color=ON_SURFACE_VARIANT,
            ha="left", va="top",
        )

        # Takeaway body (below the category chip, full width).
        ax.text(
            x0 + 2.5, y0 + tile_h - 26.5,
            tile["takeaway"],
            fontsize=9.8, color=ON_SURFACE_VARIANT,
            ha="left", va="top",
            linespacing=1.45,
        )

        # Giant AUC number — bottom-right corner, out of the label's lane.
        auc_str = f"{tile['auc']:.3f}"
        auc_txt = ax.text(
            x0 + tile_w - 2.5, y0 + 2.8,
            auc_str,
            fontsize=30, fontweight="bold",
            color=PRIMARY_CONTAINER, ha="right", va="bottom",
        )
        cyan_glow(auc_txt, color=PRIMARY, blur=4.0)

        ax.text(
            x0 + tile_w - 2.5, y0 + 2.2,
            "BEST AUC",
            fontsize=8.5, color=ON_SURFACE_VARIANT,
            ha="right", va="top",
        )

    fig.savefig(OUT_DIR / "strategic_priorities_per_window.png", dpi=200)
    plt.close(fig)
    print("  wrote strategic_priorities_per_window.png")


# ──────────────────────────────────────────────────────────────────────────
# Chart 3 — category_heatmap_v2.png
# ──────────────────────────────────────────────────────────────────────────
def build_category_heatmap_v2() -> None:
    df = pd.read_csv(PROCESSED / "analysis_corrected_v2" / "best_auc_wide.csv")
    windows = ["0-5 min", "0-10 min", "0-15 min", "Full"]
    df = df.sort_values("avg", ascending=False).reset_index(drop=True)
    values = df[windows].to_numpy()
    categories = df["category"].tolist()

    fig, ax = plt.subplots(figsize=(11.5, 7.5))

    im = ax.imshow(values, cmap=HEAT_CMAP, vmin=0.35, vmax=1.0, aspect="auto")

    # Cell annotations.
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            # Text color flips based on cell brightness so it stays legible.
            txt_color = SURFACE if v > 0.78 else ON_SURFACE
            ax.text(
                j, i, f"{v:.3f}",
                ha="center", va="center",
                fontsize=11.5, fontweight="bold", color=txt_color,
            )

    ax.set_xticks(range(len(windows)))
    ax.set_xticklabels(windows, fontsize=11, color=ON_SURFACE_VARIANT)
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(
        [c.replace("_", " ") for c in categories],
        fontsize=11, color=ON_SURFACE,
    )
    ax.tick_params(axis="both", which="both", length=0)

    # Top-label strip (kill xlabel, use an "AUC WINDOW" caption).
    ax.set_xlabel("")
    ax.set_ylabel("")

    # Colorbar styled to the palette.
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(colors=ON_SURFACE_VARIANT, length=0, labelsize=9)
    cbar.ax.yaxis.set_tick_params(color=ON_SURFACE_VARIANT)

    # Tonal authority caption.
    ax.text(
        -0.02, 1.07,
        "BEST ROC-AUC · 5-FOLD CV · n=34",
        transform=ax.transAxes,
        fontsize=9.5, color=PRIMARY_DIM,
        ha="left", va="bottom", fontweight="bold",
    )

    fig.savefig(OUT_DIR / "category_heatmap_v2.png", dpi=200)
    plt.close(fig)
    print("  wrote category_heatmap_v2.png")


# ──────────────────────────────────────────────────────────────────────────
# Chart 4 — r2_emergence.png
# ──────────────────────────────────────────────────────────────────────────
def build_r2_emergence() -> None:
    windows = ["0-5 min", "0-10 min", "0-15 min", "Full game"]
    r2 = [0.194, 0.553, 0.599, 0.564]

    fig, ax = plt.subplots(figsize=(11, 6.2))

    xs = np.arange(len(windows))

    # Faint vertical guides as "tonal shift" markers.
    for xv in xs:
        ax.axvline(xv, color=OUTLINE_VARIANT, linewidth=0.8, alpha=0.25)

    # Horizontal zero guide.
    ax.axhline(0, color=OUTLINE_VARIANT, linewidth=0.8, alpha=0.4,
               linestyle="--")

    # Area fill under the line — cyan gradient via imshow trick.
    # Use a simple fill_between with low-alpha cyan for transparency safety.
    ax.fill_between(xs, r2, 0, color=PRIMARY_CONTAINER, alpha=0.14,
                    zorder=1)

    # Primary line (the neon tube).
    ax.plot(
        xs, r2,
        color=PRIMARY_CONTAINER, linewidth=3.2,
        solid_capstyle="round", zorder=3,
    )

    # Markers — filled cyan circle with a primary halo.
    for x, y in zip(xs, r2):
        ax.scatter(
            x, y,
            s=260, facecolor=PRIMARY, edgecolor=PRIMARY_CONTAINER,
            linewidth=2.0, zorder=5,
        )
        ax.scatter(
            x, y,
            s=900, facecolor=PRIMARY_CONTAINER, alpha=0.18, zorder=4,
            linewidth=0,
        )

    # Value labels above each point.
    for x, y in zip(xs, r2):
        t = ax.text(
            x, y + 0.045, f"{y:.3f}",
            ha="center", va="bottom",
            fontsize=14, fontweight="bold", color=ON_SURFACE,
        )
        t.set_fontfamily(["Space Grotesk", "DejaVu Sans"])
        cyan_glow(t, color=PRIMARY, blur=3.0)

    ax.set_xticks(xs)
    ax.set_xticklabels(windows, fontsize=11.5, color=ON_SURFACE_VARIANT)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6])
    ax.set_yticklabels(["0.00", "0.20", "0.40", "0.60"],
                       fontsize=10, color=ON_SURFACE_VARIANT)
    ax.set_ylim(-0.05, 0.75)
    ax.set_xlim(-0.4, len(windows) - 0.6)

    ax.set_ylabel("Best R² (5-fold CV)",
                  fontsize=11, color=ON_SURFACE_VARIANT, labelpad=10)

    # Baseline caption.
    ax.text(
        -0.02, 1.08,
        "TARGET · GOLD_DIFF_FINAL_API · BEST OF LINEAR / RIDGE / GBR",
        transform=ax.transAxes,
        fontsize=9, fontweight="bold", color=PRIMARY_DIM,
        ha="left", va="bottom",
    )

    ax.tick_params(axis="both", length=0)

    fig.savefig(OUT_DIR / "r2_emergence.png", dpi=200)
    plt.close(fig)
    print("  wrote r2_emergence.png")


# ──────────────────────────────────────────────────────────────────────────
# Data helpers for remaining charts
# ──────────────────────────────────────────────────────────────────────────
def load_features_with_meta() -> pd.DataFrame:
    """Join features_corrected + targets + features_meta on match_id.

    NOTE: `features_meta.csv` has `blue_team`/`red_team` columns that
    contain the *champion picks* (comma-separated), and `blue_team_code`/
    `red_team_code` that contain the actual team abbreviations (BLG, BFX,
    …). We rename the code columns to `blue_team`/`red_team` downstream so
    the rest of the script can treat the team as a first-class identifier.
    """
    features = pd.read_csv(PROCESSED / "features_corrected.csv")
    targets  = pd.read_csv(PROCESSED / "targets.csv")
    meta     = pd.read_csv(PROCESSED / "features_meta.csv")

    meta_slim = meta[
        ["match_id", "winner_side", "blue_team_code", "red_team_code"]
    ].rename(columns={
        "blue_team_code": "blue_team",
        "red_team_code":  "red_team",
    })

    df = features.merge(
        targets[["match_id", "gold_diff_final_api"]],
        on="match_id", how="left",
    ).merge(meta_slim, on="match_id", how="left")
    return df


# ──────────────────────────────────────────────────────────────────────────
# Chart 5 — example_finding_centroid_scatter.png
# ──────────────────────────────────────────────────────────────────────────
def build_example_scatter() -> None:
    df = load_features_with_meta()

    x = df["sp_snap_t420_red_centroid_x"].astype(float)
    y = df["gold_diff_final_api"].astype(float)
    blue_wins = df["winner_side"] == "blue"

    valid = x.notna() & y.notna()
    x, y, blue_wins = x[valid], y[valid], blue_wins[valid]

    # Pearson + Spearman for caption.
    from scipy.stats import pearsonr, spearmanr
    pear_r, pear_p = pearsonr(x, y)
    sp_r, sp_p = spearmanr(x, y)

    fig, ax = plt.subplots(figsize=(11, 7.4))

    # Faint grid as tonal guide (no hard borders).
    ax.grid(True, which="both", color=OUTLINE_VARIANT, alpha=0.22,
            linewidth=0.6)
    ax.axhline(0, color=OUTLINE_VARIANT, alpha=0.5, linewidth=0.9,
               linestyle="--", zorder=1)

    # Scatter: blue wins in primary cyan, red wins in tertiary red.
    blue_mask = blue_wins.to_numpy()
    red_mask = ~blue_mask

    ax.scatter(
        x[blue_mask], y[blue_mask],
        s=170, facecolor=PRIMARY_CONTAINER,
        edgecolor=PRIMARY, linewidth=1.2,
        alpha=0.85, zorder=4, label="Blue win",
    )
    ax.scatter(
        x[red_mask], y[red_mask],
        s=170, facecolor=TERTIARY_CONTAINER,
        edgecolor=TERTIARY, linewidth=1.2,
        alpha=0.85, zorder=4, label="Red win",
    )

    # Outer halo for atmospheric glow.
    ax.scatter(x[blue_mask], y[blue_mask], s=620,
               facecolor=PRIMARY_CONTAINER, alpha=0.12, zorder=3,
               linewidth=0)
    ax.scatter(x[red_mask], y[red_mask], s=620,
               facecolor=TERTIARY_CONTAINER, alpha=0.12, zorder=3,
               linewidth=0)

    # Least-squares trendline.
    coef = np.polyfit(x, y, 1)
    xs = np.linspace(x.min() - 0.02, x.max() + 0.02, 100)
    ys = np.polyval(coef, xs)
    ax.plot(xs, ys, color=PRIMARY, linewidth=1.6, alpha=0.5,
            linestyle="--", zorder=2)

    ax.set_xlabel(
        "sp_snap_t420_red_centroid_x  (normalised x-coord of RED team centroid at 7:00)",
        fontsize=10.5, color=ON_SURFACE_VARIANT, labelpad=10,
    )
    ax.set_ylabel(
        "Final gold differential  (blue − red, from lolesports API)",
        fontsize=10.5, color=ON_SURFACE_VARIANT, labelpad=10,
    )

    # Top caption — tonal authority.
    ax.text(
        0.0, 1.08,
        f"PEARSON r = {pear_r:+.3f}   ·   SPEARMAN r = {sp_r:+.3f}   ·   p = {pear_p:.4f}   ·   n = {len(x)}",
        transform=ax.transAxes,
        fontsize=10, fontweight="bold", color=PRIMARY_DIM,
        ha="left", va="bottom",
    )

    # Legend — no frame, cyan/red markers.
    leg = ax.legend(
        loc="lower right", fontsize=10,
        labelcolor=ON_SURFACE_VARIANT, frameon=False,
    )

    ax.tick_params(axis="both", length=0, labelsize=9,
                   colors=ON_SURFACE_VARIANT)

    fig.savefig(OUT_DIR / "example_finding_centroid_scatter.png", dpi=200)
    plt.close(fig)
    print("  wrote example_finding_centroid_scatter.png")


# ──────────────────────────────────────────────────────────────────────────
# Chart 6 — team_profile_mockup.png
# ──────────────────────────────────────────────────────────────────────────
def _build_per_team_long(df: pd.DataFrame) -> pd.DataFrame:
    """Split each match into a blue-team row and red-team row so every row is
    a (team, match_id, sp_* features) triple with side-agnostic column names.
    """
    blue_cols = [c for c in df.columns if c.startswith("sp_blue_")]
    red_cols  = [c for c in df.columns if c.startswith("sp_red_")]

    # Side-agnostic rename.
    blue_renamed = {c: c.replace("sp_blue_", "sp_") for c in blue_cols}
    red_renamed  = {c: c.replace("sp_red_",  "sp_") for c in red_cols}

    base_cols = ["match_id", "blue_team", "red_team", "winner_side"]

    blue_df = df[base_cols + blue_cols].rename(columns=blue_renamed).copy()
    blue_df["team"] = blue_df["blue_team"]
    blue_df["side"] = "blue"
    blue_df["won"] = blue_df["winner_side"] == "blue"

    red_df = df[base_cols + red_cols].rename(columns=red_renamed).copy()
    red_df["team"] = red_df["red_team"]
    red_df["side"] = "red"
    red_df["won"] = red_df["winner_side"] == "red"

    # Keep only columns that exist in BOTH sides — some features are side-
    # specific (e.g. snap columns with 'blue_centroid' in the middle).
    shared = sorted(
        set(blue_df.columns).intersection(set(red_df.columns))
    )
    return pd.concat([blue_df[shared], red_df[shared]], ignore_index=True)


def build_team_profile_mockup() -> None:
    df = load_features_with_meta()
    long = _build_per_team_long(df)

    # Pick the team with the most games for the mockup.
    team_games = long.groupby("team").size().sort_values(ascending=False)
    target_team = team_games.index[0]
    target_rows = long[long["team"] == target_team]

    # Category mapping — but only map columns that actually survived the
    # side-agnostic rename (keys that start with sp_).
    cat_map = pd.read_csv(PROCESSED / "feature_categories_v2.csv")

    def side_agnostic_name(f: str) -> str:
        return f.replace("sp_blue_", "sp_").replace("sp_red_", "sp_")

    cat_map["feature_sa"] = cat_map["feature"].apply(side_agnostic_name)
    # Drop dupes created by collapsing sp_blue/sp_red into sp_.
    cat_map = cat_map.drop_duplicates(subset=["feature_sa"])
    feat_to_cat = dict(zip(cat_map["feature_sa"], cat_map["category_v2"]))

    numeric_cols = [c for c in long.columns if c.startswith("sp_")]

    # Compute per-game per-category mean for the team vs tournament.
    def category_score_per_row(row, category) -> float:
        cols = [c for c in numeric_cols if feat_to_cat.get(c) == category]
        if not cols:
            return np.nan
        vals = pd.to_numeric(row[cols], errors="coerce").to_numpy()
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            return np.nan
        # Z-score each feature against tournament distribution, then mean.
        # (Done outside for speed — here we use a precomputed z-matrix.)
        return float(np.nan)  # placeholder — replaced below

    categories = sorted(set(cat_map["category_v2"].dropna()))
    # Drop game_state_ocr so the radar reflects strategic play only.
    categories = [c for c in categories if c != "game_state_ocr"]

    # Z-score each numeric feature against the tournament distribution.
    means = long[numeric_cols].mean()
    stds  = long[numeric_cols].std().replace(0, np.nan)
    z = (long[numeric_cols] - means) / stds

    # Per-row category mean = mean of z-scores across features in that category.
    category_to_cols = {
        c: [col for col in numeric_cols if feat_to_cat.get(col) == c]
        for c in categories
    }

    team_scores = {}
    for cat in categories:
        cols = category_to_cols[cat]
        if not cols:
            continue
        # Filter to rows where team == target_team.
        team_z = z.loc[target_rows.index, cols]
        team_scores[cat] = float(team_z.mean().mean())

    # Sort so strongest categories are at the top.
    sorted_cats = sorted(team_scores.items(), key=lambda kv: -kv[1])
    cat_labels = [c.replace("_", " ") for c, _ in sorted_cats]
    cat_values = [v for _, v in sorted_cats]

    fig, ax = plt.subplots(figsize=(12, 7.5))
    ax.set_xlim(-2.1, 2.1)
    ax.set_ylim(-0.6, len(cat_labels) - 0.4)

    # Ghost vertical guides at -1, 0, +1 σ.
    for gx, lbl in [(-1, "−1σ"), (0, "tournament\navg"), (1, "+1σ")]:
        ax.axvline(gx, color=OUTLINE_VARIANT, alpha=0.35,
                   linewidth=0.8, linestyle="--")
        ax.text(gx, len(cat_labels) - 0.45, lbl,
                ha="center", va="bottom", fontsize=8.5,
                color=ON_SURFACE_VARIANT)

    # Horizontal value bars, colored by sign.
    for i, (label, v) in enumerate(zip(cat_labels, cat_values)):
        color = PRIMARY_CONTAINER if v >= 0 else TERTIARY_CONTAINER
        glow = PRIMARY if v >= 0 else TERTIARY
        # Glass background stripe behind every row (tonal separation).
        ax.add_patch(patches.Rectangle(
            (-2.1, i - 0.42),
            4.2, 0.84,
            facecolor=SURFACE_LOW if i % 2 == 0 else SURFACE_HIGH,
            edgecolor="none", alpha=0.55, zorder=0,
        ))
        # Primary bar.
        ax.add_patch(patches.Rectangle(
            (0, i - 0.26),
            v, 0.52,
            facecolor=color, edgecolor="none", zorder=2,
        ))
        # Halo behind the bar for atmospheric glow.
        ax.add_patch(patches.Rectangle(
            (0, i - 0.32),
            v, 0.64,
            facecolor=color, alpha=0.22, zorder=1,
        ))
        # Tip marker.
        ax.scatter([v], [i], s=90, facecolor=glow,
                   edgecolor=color, linewidth=1.4, zorder=4)
        # Value label.
        offset = 0.07 if v >= 0 else -0.07
        ha = "left" if v >= 0 else "right"
        ax.text(v + offset, i, f"{v:+.2f}",
                ha=ha, va="center",
                fontsize=10.2, fontweight="bold", color=ON_SURFACE)

    ax.set_yticks(range(len(cat_labels)))
    ax.set_yticklabels(cat_labels, fontsize=11, color=ON_SURFACE)
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_xticklabels(["−2σ", "−1σ", "0", "+1σ", "+2σ"],
                       fontsize=9, color=ON_SURFACE_VARIANT)
    ax.tick_params(axis="both", length=0)

    # Invisible spines — tonal only.
    for s in ax.spines.values():
        s.set_visible(False)

    # Header block: team name + games + caption.
    team_txt = ax.text(
        -2.05, len(cat_labels) + 0.6, target_team.upper(),
        fontsize=22, fontweight="bold", color=ON_SURFACE,
        ha="left", va="bottom",
    )
    team_txt.set_fontfamily(["Space Grotesk", "DejaVu Sans"])
    cyan_glow(team_txt, color=PRIMARY, blur=4.0)

    ax.text(
        -2.05, len(cat_labels) + 0.25,
        f"TEAM PROFILE · {len(target_rows)} GAMES · z-score vs tournament",
        fontsize=9.5, fontweight="bold", color=PRIMARY_DIM,
        ha="left", va="bottom",
    )

    # Caveat / mockup tag.
    ax.text(
        2.05, len(cat_labels) + 0.25,
        "MOCKUP · first-stand 2026",
        fontsize=8.5, color=ON_SURFACE_VARIANT,
        ha="right", va="bottom", fontstyle="italic",
    )

    fig.subplots_adjust(left=0.20, right=0.97, top=0.88, bottom=0.08)
    fig.savefig(OUT_DIR / "team_profile_mockup.png", dpi=200)
    plt.close(fig)
    print(f"  wrote team_profile_mockup.png (team: {target_team}, "
          f"n_games={len(target_rows)})")


# ──────────────────────────────────────────────────────────────────────────
# Chart 7 — corpus_comparison.png
# ──────────────────────────────────────────────────────────────────────────
def build_corpus_comparison() -> None:
    corpora = [
        ("First Stand 2026",          34,  "n=34",   "CURRENT", PRIMARY_CONTAINER),
        ("LCK or LPL regular split",  135, "~135",   "NEXT",    PRIMARY_DIM),
        ("Multi-tournament pool",     400, "~400",   "STRETCH", SECONDARY_CONTAINER),
    ]

    fig, ax = plt.subplots(figsize=(11, 6.4))

    xs = np.arange(len(corpora))
    values = [n for _, n, _, _, _ in corpora]
    max_val = max(values) * 1.25

    for i, (label, n, count_str, tag, color) in enumerate(corpora):
        bar_w = 0.55
        x0 = i - bar_w / 2

        # Halo.
        ax.add_patch(patches.Rectangle(
            (x0 - 0.015, 0), bar_w + 0.03, n,
            facecolor=color, alpha=0.18, zorder=1,
        ))
        # Primary bar.
        ax.add_patch(patches.Rectangle(
            (x0, 0), bar_w, n,
            facecolor=color, edgecolor="none", zorder=2,
        ))
        # Cyan top accent stripe.
        ax.add_patch(patches.Rectangle(
            (x0, n - max_val * 0.008), bar_w, max_val * 0.008,
            facecolor=PRIMARY, edgecolor="none", zorder=3,
        ))

        # Big value label above the bar.
        val_txt = ax.text(
            i, n + max_val * 0.04, count_str,
            ha="center", va="bottom",
            fontsize=22, fontweight="bold", color=ON_SURFACE,
        )
        val_txt.set_fontfamily(["Space Grotesk", "DejaVu Sans"])
        cyan_glow(val_txt, color=PRIMARY, blur=3.5)

        # Tag above the number.
        ax.text(
            i, n + max_val * 0.125, tag,
            ha="center", va="bottom",
            fontsize=9, fontweight="bold", color=PRIMARY_DIM,
        )

    ax.set_xticks(xs)
    ax.set_xticklabels(
        [c[0] for c in corpora],
        fontsize=11.5, color=ON_SURFACE_VARIANT,
    )
    ax.set_yticks([])
    ax.set_xlim(-0.6, len(corpora) - 0.4)
    ax.set_ylim(0, max_val)
    ax.tick_params(axis="both", length=0)

    ax.text(
        0.0, 1.08,
        "GAMES IN CORPUS · CURRENT vs ACHIEVABLE",
        transform=ax.transAxes,
        fontsize=9.5, fontweight="bold", color=PRIMARY_DIM,
        ha="left", va="bottom",
    )

    fig.savefig(OUT_DIR / "corpus_comparison.png", dpi=200)
    plt.close(fig)
    print("  wrote corpus_comparison.png")


# ──────────────────────────────────────────────────────────────────────────
# Chart 8 — cv_vs_api_temporal.png
# ──────────────────────────────────────────────────────────────────────────
def build_cv_vs_api_temporal() -> None:
    fig, ax = plt.subplots(figsize=(16, 6.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 30)
    ax.set_axis_off()

    # Main horizontal rail.
    rail_y = 20
    ax.add_patch(patches.Rectangle(
        (6, rail_y - 0.25), 88, 0.55,
        facecolor=OUTLINE_VARIANT, edgecolor="none", alpha=0.6,
    ))
    # Subtle cyan pulse line above (primary fixed-dim).
    ax.add_patch(patches.Rectangle(
        (6, rail_y + 0.3), 88, 0.25,
        facecolor=PRIMARY_DIM, edgecolor="none", alpha=0.45,
    ))

    # Two Match-V5 markers (t=240 and t=300).
    api_points = [
        (12, "MATCH-V5", "t = 4:00"),
        (88, "MATCH-V5", "t = 5:00"),
    ]
    for x, label, tstr in api_points:
        # Marker dot.
        ax.add_patch(patches.Circle(
            (x, rail_y + 0.15), 1.35,
            facecolor=PRIMARY_CONTAINER, edgecolor=PRIMARY,
            linewidth=2.0, zorder=5,
        ))
        ax.add_patch(patches.Circle(
            (x, rail_y + 0.15), 2.6,
            facecolor=PRIMARY_CONTAINER, alpha=0.18,
            edgecolor="none", zorder=4,
        ))
        # Labels.
        lbl = ax.text(
            x, rail_y + 3.4, label,
            fontsize=10, fontweight="bold",
            color=PRIMARY_DIM, ha="center", va="bottom",
        )
        lbl.set_fontfamily(["Inter", "DejaVu Sans"])
        t = ax.text(
            x, rail_y + 5.0, tstr,
            fontsize=14, fontweight="bold",
            color=ON_SURFACE, ha="center", va="bottom",
        )
        t.set_fontfamily(["Space Grotesk", "DejaVu Sans"])
        cyan_glow(t, color=PRIMARY, blur=3.0)

    # 60 small tick marks for CV frames — span the full rail.
    tick_y0 = rail_y - 3.5
    tick_y1 = rail_y - 1.0
    rail_xs = np.linspace(7, 93, 60)
    for x in rail_xs:
        ax.plot([x, x], [tick_y0, tick_y1],
                color=PRIMARY_DIM, linewidth=1.3, alpha=0.85, zorder=3)

    # "CV: 1 frame / second × 60" caption under ticks.
    cv_caption = ax.text(
        50, tick_y0 - 2.3,
        "CV  ·  1 FRAME / SECOND  ·  60 SAMPLES",
        fontsize=12, fontweight="bold",
        color=PRIMARY_DIM, ha="center", va="top",
    )
    cv_caption.set_fontfamily(["Inter", "DejaVu Sans"])

    # Gap "blind spot" panel between the two Match-V5 markers.
    gap_x0, gap_x1 = 16, 84
    gap_y0 = rail_y + 7.5
    gap_h = 4.2
    glassy_panel(ax, gap_x0, gap_y0, gap_x1 - gap_x0, gap_h,
                 tier="high", edge_alpha=0.45)

    # Dashed cyan connector from gap panel down to rail.
    ax.plot([50, 50], [gap_y0, rail_y + 0.5],
            color=PRIMARY_DIM, linewidth=1.0, linestyle="--", alpha=0.55)

    # Event text inside the gap panel.
    events_line = "RECALL  →  SHOP  →  ROTATE  →  SET UP DRAGON"
    ev_txt = ax.text(
        50, gap_y0 + gap_h / 2, events_line,
        fontsize=13, fontweight="bold", color=ON_SURFACE,
        ha="center", va="center",
    )
    ev_txt.set_fontfamily(["Space Grotesk", "DejaVu Sans"])
    cyan_glow(ev_txt, color=PRIMARY, blur=3.0)

    ax.text(
        50, gap_y0 + gap_h + 0.8,
        "THE 60-SECOND BLIND SPOT",
        fontsize=10, fontweight="bold",
        color=TERTIARY_CONTAINER, ha="center", va="bottom",
    )

    # Tonal footer — honesty guardrail text.
    foot_txt = ax.text(
        50, 0.5,
        "CV does not beat the PRIVATE Esports API (perfect per-frame feed for top-4 leagues).\n"
        "It beats the PUBLIC Match-V5 API — the only data amateurs and semi-pros actually have.",
        fontsize=9.4, color=ON_SURFACE_VARIANT,
        ha="center", va="bottom", fontstyle="italic",
        linespacing=1.4,
    )

    fig.savefig(OUT_DIR / "cv_vs_api_temporal.png", dpi=200)
    plt.close(fig)
    print("  wrote cv_vs_api_temporal.png")


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────
def main() -> None:
    apply_style()
    print(f"Building charts into {OUT_DIR}")
    build_pipeline_diagram()
    build_strategic_priorities()
    build_category_heatmap_v2()
    build_r2_emergence()
    build_example_scatter()
    build_team_profile_mockup()
    build_corpus_comparison()
    build_cv_vs_api_temporal()
    print("Done.")


if __name__ == "__main__":
    main()
