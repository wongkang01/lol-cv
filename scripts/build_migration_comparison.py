"""
Build a visual comparison of the old rectangle-based zone classifier vs the
new pixel-mask classifier. Loads the pre-computed analysis outputs from both
sides and produces ``charts/zone_mask_migration_diff.png``.

Four-panel figure:
    1. Per-category ROC-AUC delta heatmap (8 categories x 4 windows).
    2. Phase 5 gold-diff R² trajectory (old vs new line chart).
    3. Top-10 feature correlations before / after (rank migration).
    4. Headline delta big-numbers for the reliability slide.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = REPO_ROOT / "data" / "processed"
OUT = REPO_ROOT / "charts" / "zone_mask_migration_diff.png"

ACCENT = "#1f77b4"
ACCENT_DARK = "#0d3b66"
WARNING = "#d62728"
POSITIVE = "#2ca02c"
GREY = "#888888"
LIGHTGREY = "#d8d8d8"

WINDOWS = ["Full", "0-15 min", "0-10 min", "0-5 min"]
WINDOW_DIRS_OLD = {
    "Full":     "analysis",
    "0-15 min": "analysis_0_900",
    "0-10 min": "analysis_0_600",
    "0-5 min":  "analysis_0_300",
}
WINDOW_DIRS_NEW = {
    "Full":     "analysis_corrected",
    "0-15 min": "analysis_corrected_0_900",
    "0-10 min": "analysis_corrected_0_600",
    "0-5 min":  "analysis_corrected_0_300",
}

CATEGORY_ORDER = [
    "jungle_invasion",
    "team_coordination",
    "temporal_dynamics",
    "objective_control",
    "lane_positioning",
    "early_strategy",
    "strategic_decisions",
    "ocr_state",
]


def best_auc_per_category(path: Path) -> dict[str, float]:
    """Read an analysis dir's category_ablation.csv and return the best AUC
    across models per category.
    """
    df = pd.read_csv(path / "category_ablation.csv")
    return df.groupby("category")["roc_auc"].max().to_dict()


def best_r2_per_window(path: Path, target: str = "gold_diff_final_api") -> float:
    """Read a regression cv_results.csv and return best r2_mean."""
    f = path / f"regression_{target}" / "cv_results.csv"
    if not f.exists():
        return float("nan")
    df = pd.read_csv(f)
    return float(df["r2_mean"].max())


def load_top_correlations(path: Path, n: int = 10) -> pd.DataFrame:
    df = pd.read_csv(path / "feature_correlations.csv")
    return df.head(n)


# ── Panel 1: per-category delta heatmap ────────────────────────────────────

def build_panel_category_delta(ax):
    old_data = {
        win: best_auc_per_category(PROCESSED / WINDOW_DIRS_OLD[win])
        for win in WINDOWS
    }
    new_data = {
        win: best_auc_per_category(PROCESSED / WINDOW_DIRS_NEW[win])
        for win in WINDOWS
    }

    delta = np.zeros((len(CATEGORY_ORDER), len(WINDOWS)))
    for i, cat in enumerate(CATEGORY_ORDER):
        for j, win in enumerate(WINDOWS):
            delta[i, j] = new_data[win].get(cat, np.nan) - old_data[win].get(cat, np.nan)

    vmax = np.nanmax(np.abs(delta))
    im = ax.imshow(delta, cmap="RdBu", vmin=-vmax, vmax=+vmax, aspect="auto")

    for i in range(len(CATEGORY_ORDER)):
        for j in range(len(WINDOWS)):
            val = delta[i, j]
            if np.isnan(val):
                continue
            color = "white" if abs(val) > vmax * 0.55 else "black"
            ax.text(
                j, i, f"{val:+.02f}".replace("-0.00", "0.00").replace("+0.00", "0.00"),
                ha="center", va="center", fontsize=10,
                fontweight="bold", color=color,
            )

    ax.set_xticks(range(len(WINDOWS)))
    ax.set_xticklabels(WINDOWS)
    ax.set_yticks(range(len(CATEGORY_ORDER)))
    ax.set_yticklabels([c.replace("_", " ") for c in CATEGORY_ORDER])
    ax.set_title(
        "Per-category AUC delta (new − old)",
        fontsize=13, fontweight="bold", pad=12,
    )

    # Annotation band at the bottom
    ax.set_xlabel("Blue = improved, Red = worsened")


# ── Panel 2: Phase 5 R² trajectory comparison ─────────────────────────────

def build_panel_r2_trajectory(ax):
    xs = [5, 10, 15, 30]  # map "Full game" to x=30 visually
    old_r2 = [
        best_r2_per_window(PROCESSED / WINDOW_DIRS_OLD["0-5 min"]),
        best_r2_per_window(PROCESSED / WINDOW_DIRS_OLD["0-10 min"]),
        best_r2_per_window(PROCESSED / WINDOW_DIRS_OLD["0-15 min"]),
        best_r2_per_window(PROCESSED / WINDOW_DIRS_OLD["Full"]),
    ]
    new_r2 = [
        best_r2_per_window(PROCESSED / WINDOW_DIRS_NEW["0-5 min"]),
        best_r2_per_window(PROCESSED / WINDOW_DIRS_NEW["0-10 min"]),
        best_r2_per_window(PROCESSED / WINDOW_DIRS_NEW["0-15 min"]),
        best_r2_per_window(PROCESSED / WINDOW_DIRS_NEW["Full"]),
    ]

    ax.plot(xs, old_r2, marker="o", markersize=9, linewidth=2.5,
            color=GREY, label="Old (rectangles)",
            markeredgecolor="white", markeredgewidth=1.5)
    ax.plot(xs, new_r2, marker="o", markersize=9, linewidth=3,
            color=ACCENT, label="New (pixel mask)",
            markeredgecolor="white", markeredgewidth=1.5)

    ax.axhline(0, color=GREY, linewidth=0.8, linestyle="--", alpha=0.6)

    # Annotate the biggest delta at 0-10 min
    idx_10 = 1
    ax.annotate(
        f"+{new_r2[idx_10] - old_r2[idx_10]:.2f}",
        xy=(xs[idx_10], new_r2[idx_10]),
        xytext=(xs[idx_10] + 2, new_r2[idx_10] + 0.05),
        fontsize=13, fontweight="bold", color=POSITIVE,
        arrowprops=dict(arrowstyle="->", color=POSITIVE, lw=1.5),
    )
    # Also annotate the 0-5 min delta (sign flip from negative to positive)
    idx_5 = 0
    ax.annotate(
        f"+{new_r2[idx_5] - old_r2[idx_5]:.2f}\n(sign flip)",
        xy=(xs[idx_5], new_r2[idx_5]),
        xytext=(xs[idx_5] + 0.8, new_r2[idx_5] + 0.18),
        fontsize=10, fontweight="bold", color=POSITIVE,
        ha="left",
        arrowprops=dict(arrowstyle="->", color=POSITIVE, lw=1.2),
    )

    ax.set_xticks(xs)
    ax.set_xticklabels(["0-5 min", "0-10 min", "0-15 min", "Full game"])
    ax.set_ylabel("R² (best model, best of 3)")
    ax.set_ylim(-0.3, 0.75)
    ax.set_title(
        "Phase 5: spatial → gold R² trajectory",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.legend(loc="lower right", frameon=False, fontsize=10)


# ── Panel 3: Top correlations rank migration ──────────────────────────────

def build_panel_rank_migration(ax):
    old_corr = load_top_correlations(PROCESSED / "analysis", n=10)
    new_corr = load_top_correlations(PROCESSED / "analysis_corrected", n=10)
    old_all = pd.read_csv(PROCESSED / "analysis" / "feature_correlations.csv")
    new_all = pd.read_csv(PROCESSED / "analysis_corrected" / "feature_correlations.csv")

    # Union of top-10s
    top_features = list(dict.fromkeys(
        list(old_corr["feature"]) + list(new_corr["feature"])
    ))

    rows = []
    for f in top_features:
        old_row = old_all[old_all["feature"] == f]
        new_row = new_all[new_all["feature"] == f]
        old_rank = (old_all.index[old_all["feature"] == f][0] + 1) if len(old_row) else 999
        new_rank = (new_all.index[new_all["feature"] == f][0] + 1) if len(new_row) else 999
        old_r = float(old_row["spearman_r"].iloc[0]) if len(old_row) else 0.0
        new_r = float(new_row["spearman_r"].iloc[0]) if len(new_row) else 0.0
        rows.append((f, old_rank, new_rank, old_r, new_r))

    rows.sort(key=lambda r: r[2])  # sort by new rank
    rows = [r for r in rows if r[2] <= 15 or r[1] <= 10][:12]

    ys = np.arange(len(rows))
    old_ranks = [r[1] for r in rows]
    new_ranks = [r[2] for r in rows]

    # Short feature names
    def short_name(f: str) -> str:
        n = f.replace("sp_", "")
        n = n.replace("_zone_", " : ")
        n = n.replace("_avg_near_count", " near")
        n = n.replace("_baron", " baron")
        n = n.replace("_dragon", " dragon")
        if len(n) > 40:
            n = n[:37] + "..."
        return n

    labels = [short_name(r[0]) for r in rows]

    for i, (old_r, new_r) in enumerate(zip(old_ranks, new_ranks)):
        color_arrow = POSITIVE if new_r < old_r else (WARNING if new_r > old_r else GREY)
        old_display = min(old_r, 30)  # cap display at 30
        new_display = min(new_r, 30)
        ax.plot([old_display, new_display], [i, i],
                color=color_arrow, linewidth=2, alpha=0.5)
        ax.scatter([old_display], [i], s=60, color=GREY, zorder=3,
                   edgecolor="white", linewidth=1.5)
        ax.scatter([new_display], [i], s=60, color=ACCENT_DARK, zorder=3,
                   edgecolor="white", linewidth=1.5)
        # Write rank number next to each dot
        ax.text(old_display - 0.5, i, str(old_r if old_r <= 30 else f"{old_r}"),
                ha="right", va="center", fontsize=8, color=GREY)
        ax.text(new_display + 0.5, i, str(new_r if new_r <= 30 else f"{new_r}"),
                ha="left", va="center", fontsize=8, color=ACCENT_DARK,
                fontweight="bold")

    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0, 35)
    ax.set_xlabel("Rank (1 = strongest, 30 = weaker)")
    ax.set_title(
        "Top-10 feature correlations — old vs new rank",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.invert_yaxis()

    # Legend
    ax.scatter([], [], s=60, color=GREY, label="Old rank", edgecolor="white", linewidth=1.5)
    ax.scatter([], [], s=60, color=ACCENT_DARK, label="New rank", edgecolor="white", linewidth=1.5)
    ax.legend(loc="lower right", fontsize=9, frameon=False)


# ── Panel 4: Headline deltas as big numbers ────────────────────────────────

def build_panel_headline_deltas(ax):
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    deltas = [
        {
            "label": "0-10 min classification AUC",
            "old": 0.760,
            "new": 0.933,
            "format": ".3f",
            "good": True,
        },
        {
            "label": "0-10 min Phase 5 R²",
            "old": 0.406,
            "new": 0.553,
            "format": ".3f",
            "good": True,
        },
        {
            "label": "Unknown-zone coverage",
            "old": 0.1116,
            "new": 0.0107,
            "format": ".2%",
            "good": True,  # lower is better
            "reverse": True,
        },
        {
            "label": "pre3min_invade_secs p-value",
            "old": 0.036,
            "new": 0.114,
            "format": ".3f",
            "good": False,
        },
    ]

    n = len(deltas)
    row_height = 1.0 / n

    for i, d in enumerate(deltas):
        y_top = 1.0 - i * row_height
        y_bot = 1.0 - (i + 1) * row_height
        y_mid = (y_top + y_bot) / 2

        # Background stripe
        ax.add_patch(plt.Rectangle(
            (0, y_bot + 0.01), 1, row_height - 0.02,
            facecolor="#f3f3f3" if i % 2 == 0 else "#ffffff",
            edgecolor="none",
        ))

        # Label
        ax.text(0.02, y_mid + 0.03, d["label"],
                fontsize=11, fontweight="bold", color="#333333",
                va="center", ha="left")

        # Old → new
        fmt = d["format"]
        old_str = f"{d['old']:{fmt}}"
        new_str = f"{d['new']:{fmt}}"

        ax.text(0.02, y_mid - 0.035, f"old: {old_str}",
                fontsize=10, color=GREY, va="center", ha="left")
        ax.text(0.3, y_mid - 0.035, f"new: {new_str}",
                fontsize=10, color=ACCENT_DARK, va="center", ha="left",
                fontweight="bold")

        # Delta arrow with big number
        if d.get("reverse"):
            delta_val = d["old"] - d["new"]
        else:
            delta_val = d["new"] - d["old"]
        color = POSITIVE if d["good"] else WARNING

        delta_str = f"{delta_val:+{fmt}}"
        if delta_val > 0 and not d["good"]:
            delta_str = f"{delta_val:+{fmt}}"

        ax.text(0.98, y_mid, delta_str,
                fontsize=22, fontweight="bold", color=color,
                va="center", ha="right")

    ax.set_title(
        "Headline deltas",
        fontsize=13, fontweight="bold", pad=12,
    )


# ── Main ─────────────────────────────────────────────────────────────────


def main() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "savefig.bbox": "tight",
    })

    fig = plt.figure(figsize=(17, 13))
    gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.28)

    ax_cat = fig.add_subplot(gs[0, 0])
    build_panel_category_delta(ax_cat)

    ax_r2 = fig.add_subplot(gs[0, 1])
    build_panel_r2_trajectory(ax_r2)

    ax_rank = fig.add_subplot(gs[1, 0])
    build_panel_rank_migration(ax_rank)

    ax_head = fig.add_subplot(gs[1, 1])
    build_panel_headline_deltas(ax_head)

    fig.suptitle(
        "Zone classification migration — rectangles → pixel mask",
        fontsize=17, fontweight="bold", y=0.995,
    )
    fig.text(
        0.5, 0.01,
        "Mask: shrink=0.95, shift=(-2/200, +1/200). All numbers best-of-models on 5-fold CV. n=34.",
        ha="center", fontsize=9, color=GREY,
    )

    fig.savefig(OUT, dpi=180, facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
