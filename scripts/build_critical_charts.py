"""Build the three critical charts for the thesis presentation.

Charts produced (all saved to charts/):
  1. category_heatmap.png  - Per-category x window AUC heatmap
  2. r2_emergence.png       - R^2 emergence line chart
  3. window_collapse.png    - Window collapse bar chart
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

# ----------------------------------------------------------------------
# Shared style
# ----------------------------------------------------------------------
mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 14,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.titleweight": "bold",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.dpi": 200,
    }
)

ACCENT = "#1f77b4"        # cool blue - main highlight
ACCENT_DARK = "#0d3b66"   # for emphasis on accent text
WARNING = "#d62728"       # red - only for below baseline / failed
GREY = "#cccccc"
DARKGREY = "#555555"
LIGHTGREY = "#ebebeb"

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "processed"
CHARTS_DIR = REPO_ROOT / "charts"

CATEGORY_LABELS = {
    "strategic_decisions": "Strategic decisions",
    "objective_control": "Objective control",
    "lane_positioning": "Lane positioning",
    "jungle_invasion": "Jungle invasion",
    "team_coordination": "Team coordination",
    "ocr_state": "HUD state (OCR)",
    "temporal_dynamics": "Temporal dynamics",
    "early_strategy": "Early strategy",
}


# ----------------------------------------------------------------------
# Chart 1: category x window heatmap
# ----------------------------------------------------------------------
def build_category_heatmap() -> Path:
    csv_path = DATA_DIR / "analysis" / "category_comparison.csv"
    df = pd.read_csv(csv_path)

    # Reorder rows: strategic_decisions at top, then the rest sorted by Full desc.
    rest = df[df["category"] != "strategic_decisions"].sort_values(
        "Full", ascending=False
    )
    ordered = pd.concat(
        [df[df["category"] == "strategic_decisions"], rest], ignore_index=True
    )

    # Reorder columns: 0-5 min -> 0-10 min -> 0-15 min -> Full
    col_order = ["0-5 min", "0-10 min", "0-15 min", "Full"]
    col_labels = ["First 5 min", "First 10 min", "First 15 min", "Full game"]
    matrix = ordered[col_order].to_numpy()
    row_labels = [CATEGORY_LABELS[c] for c in ordered["category"].tolist()]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    im = ax.imshow(matrix, cmap="Blues", vmin=0.3, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)

    # Move x-axis ticks to top so the "earliest->latest" flow reads naturally
    ax.tick_params(axis="x", which="both", length=0)
    ax.tick_params(axis="y", which="both", length=0)

    # Annotate cells
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            color = "white" if value > 0.75 else "black"
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=color,
                fontsize=13,
                fontweight="bold",
            )

    # Highlight the strategic_decisions row (row 0 after reordering).
    # Draw a thick border slightly inset so it stays fully visible and use
    # a high-contrast accent so the audience cannot miss it.
    strat_idx = row_labels.index("Strategic decisions")
    rect = Rectangle(
        (-0.5 + 0.03, strat_idx - 0.5 + 0.03),
        matrix.shape[1] - 0.06,
        1 - 0.06,
        edgecolor=ACCENT_DARK,
        linewidth=4,
        fill=False,
        zorder=10,
        clip_on=False,
    )
    ax.add_patch(rect)
    # Emphasise the row label itself
    y_labels = ax.get_yticklabels()
    y_labels[strat_idx].set_color(ACCENT_DARK)
    y_labels[strat_idx].set_fontweight("bold")

    # Colorbar, kept sparse
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("ROC-AUC", fontsize=12)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=0)

    # Titles
    fig.suptitle(
        "Only one feature category survives the early-window test",
        fontsize=16,
        fontweight="bold",
        x=0.02,
        y=1.02,
        ha="left",
    )
    ax.set_title(
        "Best ROC-AUC across RF / GBM / SVM per category x window. Baseline = 0.647.",
        fontsize=11,
        color=DARKGREY,
        loc="left",
        pad=12,
        fontweight="normal",
    )

    # Remove remaining spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    out = CHARTS_DIR / "category_heatmap.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ----------------------------------------------------------------------
# Chart 2: R^2 emergence line chart
# ----------------------------------------------------------------------
def build_r2_emergence() -> Path:
    windows = [
        ("analysis_0_300", 5),
        ("analysis_0_600", 10),
        ("analysis_0_900", 15),
        ("analysis", 30),
    ]

    best_r2: dict[int, float] = {}
    for sub, minute in windows:
        path = DATA_DIR / sub / "regression_gold_diff_final_api" / "cv_results.csv"
        df = pd.read_csv(path)
        best_r2[minute] = float(df["r2_mean"].max())

    # Synthetic anchor at x=0 (no information available -> R^2 = 0)
    xs = [0, 5, 10, 15, 30]
    ys = [0.0, best_r2[5], best_r2[10], best_r2[15], best_r2[30]]

    fig, ax = plt.subplots(figsize=(11, 6))

    # Shaded "signal lock-in" zone
    ax.axvspan(10, 15, alpha=0.15, color=ACCENT, zorder=0)

    # Zero reference line
    ax.axhline(
        0,
        color=DARKGREY,
        linestyle="--",
        linewidth=1.2,
        alpha=0.5,
        zorder=1,
    )
    ax.text(
        30.3,
        0.0,
        "no signal",
        color=DARKGREY,
        fontsize=10,
        va="center",
        ha="left",
        alpha=0.8,
    )

    # Main trajectory
    ax.plot(
        xs,
        ys,
        color=ACCENT,
        linewidth=3,
        marker="o",
        markersize=10,
        markeredgecolor="white",
        markeredgewidth=2,
        zorder=3,
    )

    # Annotation 1: signal emerges at minute 10
    ax.annotate(
        "Signal emerges at minute 10\n-> explains 41% of end-game gold",
        xy=(10, best_r2[10]),
        xytext=(4.2, 0.66),
        fontsize=12,
        color=ACCENT_DARK,
        fontweight="bold",
        ha="left",
        va="top",
        arrowprops=dict(
            arrowstyle="->",
            color=ACCENT_DARK,
            lw=1.5,
            connectionstyle="arc3,rad=-0.15",
        ),
    )

    # Annotation 2: half the economy locked in
    ax.annotate(
        "Half the game's economy\nlocked in by minute 15",
        xy=(15, best_r2[15]),
        xytext=(17.5, 0.26),
        fontsize=12,
        color=ACCENT_DARK,
        fontweight="bold",
        ha="left",
        va="center",
        arrowprops=dict(
            arrowstyle="->",
            color=ACCENT_DARK,
            lw=1.5,
            connectionstyle="arc3,rad=0.2",
        ),
    )

    # Annotation 3: plateau at full game (muted)
    ax.annotate(
        "Plateau - last 15 min\nadd only 6 pp",
        xy=(30, best_r2[30]),
        xytext=(22.5, 0.66),
        fontsize=11,
        color=DARKGREY,
        fontweight="normal",
        ha="left",
        va="center",
        arrowprops=dict(
            arrowstyle="->",
            color=DARKGREY,
            lw=1.2,
            alpha=0.7,
            connectionstyle="arc3,rad=-0.2",
        ),
    )

    # Axes setup
    ax.set_xlim(-1.5, 33)
    ax.set_ylim(-0.2, 0.7)
    ax.set_xticks([5, 10, 15, 30])
    ax.set_xticklabels(["First 5 min", "First 10 min", "First 15 min", "Full game"])
    ax.tick_params(axis="x", which="both", length=0)
    ax.set_ylabel("R-squared explaining end-game gold differential")

    # Titles
    fig.suptitle(
        "Early-game positioning shapes the gold lead - but only after minute 10",
        fontsize=15,
        fontweight="bold",
        x=0.02,
        y=1.02,
        ha="left",
    )
    ax.set_title(
        "Best of three regression models per window. "
        "Top-15 features by Spearman correlation. n=34 games.",
        fontsize=11,
        color=DARKGREY,
        loc="left",
        pad=12,
        fontweight="normal",
    )

    out = CHARTS_DIR / "r2_emergence.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ----------------------------------------------------------------------
# Chart 3: Window collapse bar chart
# ----------------------------------------------------------------------
def build_window_collapse() -> Path:
    csv_path = DATA_DIR / "analysis" / "window_comparison.csv"
    df = pd.read_csv(csv_path)

    label_map = {
        "Full game": "Full game",
        "0-15 min": "First 15 min",
        "0-10 min": "First 10 min",
        "0-5 min": "First 5 min",
    }
    order = ["Full game", "0-15 min", "0-10 min", "0-5 min"]
    df = df.set_index("label").loc[order].reset_index()
    display_labels = [label_map[l] for l in df["label"]]
    values = df["gbm_acc"].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = [ACCENT] + [WARNING] * 3
    x_positions = np.arange(len(display_labels))
    bars = ax.bar(
        x_positions,
        values,
        color=colors,
        width=0.6,
        edgecolor="white",
        linewidth=0,
        zorder=3,
    )

    # Baseline line
    baseline = 0.647
    ax.axhline(
        baseline,
        color=DARKGREY,
        linestyle="--",
        linewidth=1.5,
        zorder=2,
    )
    ax.text(
        len(display_labels) - 0.5 + 0.05,
        baseline + 0.012,
        "Always-blue baseline (64.7%)",
        color=DARKGREY,
        fontsize=11,
        ha="right",
        va="bottom",
    )

    # Annotate bars with matching colour
    for bar, value, color in zip(bars, values, colors):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.018,
            f"{value * 100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=15,
            fontweight="bold",
            color=color,
        )

    # "-30 pp lost to hindsight" annotation spanning the gap between
    # the Full-game bar top and the tallest early-window bar top.
    try:
        full_val = float(values[0])
        early_max = float(values[1:].max())
        # Place the bracket off to the right of bar 0, clear of the "58.6%" label
        mid_x = 0.72
        y_high = full_val - 0.015
        y_low = early_max + 0.085
        ax.annotate(
            "",
            xy=(mid_x, y_low),
            xytext=(mid_x, y_high),
            arrowprops=dict(
                arrowstyle="-[,widthB=2.6,lengthB=0.5",
                color=DARKGREY,
                lw=1.8,
            ),
        )
        ax.text(
            mid_x + 0.1,
            (y_high + y_low) / 2,
            "-30 percentage points\nlost to hindsight",
            color=DARKGREY,
            fontsize=12,
            fontweight="bold",
            ha="left",
            va="center",
        )
    except Exception:
        pass

    ax.set_xticks(x_positions)
    ax.set_xticklabels(display_labels)
    ax.tick_params(axis="x", which="both", length=0)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Cross-validated accuracy")

    fig.suptitle(
        "Restricting features to the early game collapses the model",
        fontsize=15,
        fontweight="bold",
        x=0.02,
        y=1.02,
        ha="left",
    )
    ax.set_title(
        "Same gradient boosting model, same features, different time windows. "
        "n=34, 5-fold stratified CV.",
        fontsize=11,
        color=DARKGREY,
        loc="left",
        pad=12,
        fontweight="normal",
    )

    out = CHARTS_DIR / "window_collapse.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    paths = [
        build_category_heatmap(),
        build_r2_emergence(),
        build_window_collapse(),
    ]
    for p in paths:
        size_kb = p.stat().st_size / 1024
        print(f"wrote {p.relative_to(REPO_ROOT)} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
