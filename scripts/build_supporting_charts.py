"""Build the three supporting charts for the thesis presentation.

Charts produced (all saved to charts/):
  1. top_features.png     - Top 5 spatial features by Spearman correlation
  2. yolo_benchmark.png   - YOLOv11 speed/accuracy scatter
  3. cv_results_table.png - 4-model CV results table figure
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
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
DATA_DIR = REPO_ROOT / "data" / "processed" / "analysis"
CHARTS_DIR = REPO_ROOT / "charts"


# ----------------------------------------------------------------------
# Chart 1: Top features bar chart
# ----------------------------------------------------------------------
def build_top_features() -> Path:
    """Horizontal bar chart of top 5 spatial features by Spearman r."""
    df = pd.read_csv(DATA_DIR / "feature_correlations.csv")
    top = df.head(5).copy()

    plain_english = [
        "Time blue spent inside red's bot jungle",
        "Time red spent inside blue's top jungle",
        "Time red spent defending its own bot jungle",
        "Red presence near the dragon",
        "Blue presence near the baron",
    ]
    top["label"] = plain_english
    # Reverse so largest |r| sits at the TOP of the horizontal bars.
    top = top.iloc[::-1].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12.5, 5.6))
    fig.subplots_adjust(top=0.78, left=0.24, right=0.98, bottom=0.14)

    y_positions = range(len(top))
    colors = [ACCENT if r >= 0 else WARNING for r in top["spearman_r"]]
    ax.barh(
        y_positions,
        top["spearman_r"],
        color=colors,
        edgecolor="none",
        height=0.65,
    )

    # Zero reference line.
    ax.axvline(0, color="black", linewidth=0.8)

    # Y-axis labels
    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(top["label"], fontsize=13)

    # X-axis - extra room on the right for annotation callout.
    ax.set_xlim(-1.05, 1.85)
    ax.set_xticks([-1.0, -0.5, 0, 0.5, 1.0])
    ax.set_xlabel("Correlation with blue winning", fontsize=13)

    # Annotate each bar with its r value.
    for i, r in enumerate(top["spearman_r"]):
        if r >= 0:
            ax.text(
                r + 0.02,
                i,
                f"{r:+.2f}",
                va="center",
                ha="left",
                fontsize=13,
                fontweight="bold",
                color=ACCENT_DARK,
            )
        else:
            ax.text(
                r - 0.02,
                i,
                f"{r:+.2f}",
                va="center",
                ha="right",
                fontsize=13,
                fontweight="bold",
                color=WARNING,
            )

    # Remove the left spine tick marks for cleaner look
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=DARKGREY)
    ax.spines["left"].set_color(DARKGREY)
    ax.spines["bottom"].set_color(DARKGREY)

    # Title + subtitle as figure text for reliable left alignment
    fig.text(
        0.02,
        0.97,
        "What was the model actually looking at?",
        fontsize=18,
        fontweight="bold",
        ha="left",
        va="top",
    )
    fig.text(
        0.02,
        0.89,
        "Top 5 features by Spearman correlation, full-game features.\n"
        "Every one measures enemy-territory occupation - a hindsight signal.",
        fontsize=12,
        color=DARKGREY,
        ha="left",
        va="top",
    )

    # Hide the right tick marks past 1.0 visually by limiting xticks above.
    # Storytelling annotation lives in the extra space on the right of the
    # plotting area, with an arrow pointing at the cluster of bars.
    ax.annotate(
        "All five top features measure\n"
        "WHO IS IN WHOSE TERRITORY\n"
        "- the symptom of winning,\n"
        "not the cause of it.",
        xy=(0.92, 3.0),
        xytext=(1.18, 2.0),
        fontsize=11,
        color=ACCENT_DARK,
        ha="left",
        va="center",
        fontweight="bold",
        arrowprops=dict(
            arrowstyle="->",
            color=ACCENT_DARK,
            lw=1.5,
            connectionstyle="arc3,rad=-0.25",
        ),
    )

    out = CHARTS_DIR / "top_features.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ----------------------------------------------------------------------
# Chart 2: YOLOv11 benchmark scatter
# ----------------------------------------------------------------------
def build_yolo_benchmark() -> Path:
    """Scatter of speed (fps) vs F1 for the four YOLOv11 variants."""
    df = pd.read_csv(DATA_DIR / "benchmark.csv")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Draw trade-off frontier first (behind points).
    sorted_df = df.sort_values("fps")
    ax.plot(
        sorted_df["fps"],
        sorted_df["f1_vs_x"],
        color=GREY,
        linewidth=1,
        linestyle="--",
        alpha=0.6,
        zorder=1,
    )

    # Plot non-highlight models first.
    for _, row in df.iterrows():
        if row["model"] == "yolov11m":
            continue
        ax.scatter(
            row["fps"],
            row["f1_vs_x"],
            s=row["size_mb"] * 8,
            color=GREY,
            edgecolor=DARKGREY,
            linewidth=1,
            alpha=0.9,
            zorder=2,
        )

    # Highlight yolov11m.
    m_row = df[df["model"] == "yolov11m"].iloc[0]
    ax.scatter(
        m_row["fps"],
        m_row["f1_vs_x"],
        s=m_row["size_mb"] * 8 * 1.15,
        color=ACCENT,
        edgecolor=ACCENT_DARK,
        linewidth=3,
        zorder=3,
    )

    # Labels for each point, offset so they don't overlap the circle.
    offsets = {
        "yolov11n": (12, 12),
        "yolov11m": (16, 20),
        "yolov11l": (-14, 18),
        "yolov11x": (16, 6),
    }
    for _, row in df.iterrows():
        dx, dy = offsets[row["model"]]
        is_m = row["model"] == "yolov11m"
        ha = "right" if dx < 0 else "left"
        ax.annotate(
            row["model"],
            xy=(row["fps"], row["f1_vs_x"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=13,
            fontweight="bold" if is_m else "normal",
            color=ACCENT_DARK if is_m else DARKGREY,
            ha=ha,
            va="center",
        )

    # Sweet-spot annotation for yolov11m
    ax.annotate(
        "Picked: 95% F1 at 14 fps\non CPU - sweet spot",
        xy=(m_row["fps"], m_row["f1_vs_x"]),
        xytext=(m_row["fps"] + 6, m_row["f1_vs_x"] - 0.055),
        fontsize=11,
        color=ACCENT_DARK,
        ha="left",
        va="center",
        arrowprops=dict(
            arrowstyle="->",
            color=ACCENT_DARK,
            lw=1.2,
            connectionstyle="arc3,rad=0.2",
        ),
    )

    ax.set_xlim(0, 50)
    ax.set_ylim(0.80, 1.02)
    ax.set_xlabel("Inference speed (frames per second on CPU)", fontsize=13)
    ax.set_ylabel("Detection F1 (vs the largest model)", fontsize=13)

    ax.tick_params(colors=DARKGREY)
    ax.spines["left"].set_color(DARKGREY)
    ax.spines["bottom"].set_color(DARKGREY)

    fig.subplots_adjust(top=0.82)
    fig.text(
        0.02,
        0.98,
        "Choosing the right detection model",
        fontsize=18,
        fontweight="bold",
        ha="left",
        va="top",
    )
    fig.text(
        0.02,
        0.91,
        "Larger circles = larger model. "
        "yolov11m matches the largest model's accuracy at 2x the speed.",
        fontsize=12,
        color=DARKGREY,
        ha="left",
        va="top",
    )

    out = CHARTS_DIR / "yolo_benchmark.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ----------------------------------------------------------------------
# Chart 3: CV results table figure
# ----------------------------------------------------------------------
def build_cv_results_table() -> Path:
    """Rendered 4-row CV results table with GBM row highlighted."""
    df = pd.read_csv(DATA_DIR / "cv_results.csv", index_col=0)

    label_map = {
        "random_forest": "Random Forest",
        "mlp": "MLP",
        "gradient_boosting": "Gradient Boosting (chosen)",
        "svm": "SVM (RBF)",
    }

    df = df.sort_values("accuracy", ascending=False)
    rows = []
    for key in df.index:
        row = df.loc[key]
        rows.append(
            {
                "key": key,
                "model": label_map[key],
                "accuracy": f"{row['accuracy']:.3f}",
                "f1": f"{row['f1']:.3f}",
                "roc_auc": f"{row['roc_auc']:.3f}",
            }
        )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Column layout (x positions)
    col_x = {
        "model": 0.4,
        "accuracy": 5.4,
        "f1": 7.0,
        "roc_auc": 8.6,
    }
    col_align = {
        "model": "left",
        "accuracy": "center",
        "f1": "center",
        "roc_auc": "center",
    }

    # Row y positions (top to bottom)
    header_y = 7.2
    row_ys = [6.0, 5.0, 4.0, 3.0]
    row_height = 0.85

    # Header background bottom border
    ax.add_patch(
        Rectangle(
            (0.2, header_y - 0.35),
            9.6,
            0.02,
            facecolor=DARKGREY,
            edgecolor="none",
            zorder=2,
        )
    )

    # Header text
    headers = {
        "model": "Model",
        "accuracy": "Accuracy",
        "f1": "F1",
        "roc_auc": "ROC-AUC",
    }
    for col, text in headers.items():
        ax.text(
            col_x[col],
            header_y,
            text,
            fontsize=14,
            fontweight="bold",
            ha=col_align[col],
            va="center",
            color=ACCENT_DARK,
        )

    # Data rows
    gbm_row_y = None
    for y, row in zip(row_ys, rows):
        is_gbm = row["key"] == "gradient_boosting"
        if is_gbm:
            gbm_row_y = y
            ax.add_patch(
                Rectangle(
                    (0.2, y - row_height / 2),
                    9.6,
                    row_height,
                    facecolor=ACCENT,
                    alpha=0.15,
                    edgecolor="none",
                    zorder=1,
                )
            )

        for col in ["model", "accuracy", "f1", "roc_auc"]:
            ax.text(
                col_x[col],
                y,
                row[col],
                fontsize=14,
                fontweight="bold" if is_gbm else "normal",
                color=ACCENT_DARK if is_gbm else "black",
                ha=col_align[col],
                va="center",
                zorder=3,
            )

    # Title + subtitle drawn as figure text for left alignment
    fig.text(
        0.02,
        0.97,
        "First-attempt classification: 4 models, 5-fold stratified CV",
        fontsize=18,
        fontweight="bold",
        ha="left",
        va="top",
    )
    fig.text(
        0.02,
        0.905,
        "n=34 games. Baseline = 64.7% (always-blue). Gradient Boosting picked as the\n"
        "credible headline because RF's 0.938 is borderline overfit at 137 features.",
        fontsize=12,
        color=DARKGREY,
        ha="left",
        va="top",
    )

    # "+24 percentage points above baseline" annotation pointing at GBM accuracy
    if gbm_row_y is not None:
        ax.annotate(
            "+ 24 percentage points above baseline",
            xy=(col_x["accuracy"], gbm_row_y - 0.35),
            xytext=(col_x["accuracy"] - 1.2, 1.4),
            fontsize=12,
            color=ACCENT_DARK,
            fontweight="bold",
            ha="left",
            va="center",
            arrowprops=dict(
                arrowstyle="->",
                color=ACCENT_DARK,
                lw=1.3,
                connectionstyle="arc3,rad=0.25",
            ),
        )

    out = CHARTS_DIR / "cv_results_table.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ----------------------------------------------------------------------
# Entrypoint
# ----------------------------------------------------------------------
def main() -> None:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [
        build_top_features(),
        build_yolo_benchmark(),
        build_cv_results_table(),
    ]
    for p in outputs:
        size_kb = p.stat().st_size / 1024
        print(f"wrote {p.relative_to(REPO_ROOT)}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
