"""
Static visualizations for the LoL CV analysis pipeline.

All functions return matplotlib Figure objects so they can be
displayed in notebooks or saved to file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from lol_cv.utils import setup_logger

logger = setup_logger("lol_cv.visualization.plots")

# Consistent colour scheme: blue side vs red side
BLUE_COLOR = "#3B82F6"
RED_COLOR = "#EF4444"
SIDE_COLORS = {"blue": BLUE_COLOR, "red": RED_COLOR}


def plot_heatmap(
    heatmap: np.ndarray,
    title: str = "Champion Presence Heatmap",
    cmap: str = "hot",
    minimap_bg: str = None,
    ax: plt.Axes = None,
) -> plt.Figure:
    """Plot a 2D positional heatmap over the minimap.

    Args:
        heatmap: 2D array from SpatialFeatures.generate_heatmap().
        title: Plot title.
        cmap: Matplotlib colormap name.
        minimap_bg: Optional path to a minimap background image.
        ax: Optional axes to draw on.

    Returns:
        Matplotlib Figure.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    else:
        fig = ax.figure

    if minimap_bg:
        bg = plt.imread(minimap_bg)
        ax.imshow(bg, extent=[0, 1, 1, 0], alpha=0.5)

    ax.imshow(heatmap, cmap=cmap, extent=[0, 1, 1, 0], alpha=0.7, interpolation="bilinear")
    ax.set_title(title)
    ax.set_xlabel("X (normalised)")
    ax.set_ylabel("Y (normalised)")
    fig.tight_layout()
    return fig


def plot_trajectory(
    df: pd.DataFrame,
    champions: list[str],
    time_range: tuple[float, float] = None,
    title: str = "Champion Trajectories",
    ax: plt.Axes = None,
) -> plt.Figure:
    """Plot movement trajectories for one or more champions.

    Args:
        df: Position DataFrame with columns [timestamp, champion, x, y].
        champions: Champion names to plot.
        time_range: Optional (start, end) in seconds to filter.
        title: Plot title.

    Returns:
        Matplotlib Figure.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    else:
        fig = ax.figure

    colors = plt.cm.tab10(np.linspace(0, 1, len(champions)))

    for champ, color in zip(champions, colors):
        champ_df = df[df["champion"] == champ].sort_values("timestamp")
        if time_range:
            champ_df = champ_df[
                (champ_df["timestamp"] >= time_range[0])
                & (champ_df["timestamp"] <= time_range[1])
            ]
        if champ_df.empty:
            continue
        ax.plot(champ_df["x"], champ_df["y"], "-", color=color, alpha=0.7, linewidth=1.5, label=champ)
        ax.scatter(champ_df["x"].iloc[0], champ_df["y"].iloc[0], color=color, marker="o", s=60, zorder=5)
        ax.scatter(champ_df["x"].iloc[-1], champ_df["y"].iloc[-1], color=color, marker="x", s=60, zorder=5)

    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)  # Inverted Y to match minimap orientation
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_model_comparison(
    results: dict[str, dict],
    metrics: list[str] = None,
    title: str = "Model Comparison (Cross-Validation)",
) -> plt.Figure:
    """Bar chart comparing ML model performance.

    Args:
        results: Output from WinPredictor.train_and_evaluate().
        metrics: Which metrics to show. Defaults to accuracy, f1, roc_auc.

    Returns:
        Matplotlib Figure.
    """
    metrics = metrics or ["accuracy", "f1", "roc_auc"]
    df = pd.DataFrame(results).T[metrics]

    fig, ax = plt.subplots(figsize=(10, 5))
    df.plot(kind="bar", ax=ax, rot=0)
    ax.set_title(title)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")

    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8, padding=2)

    fig.tight_layout()
    return fig


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 15,
    title: str = "Top Feature Importances",
) -> plt.Figure:
    """Horizontal bar chart of feature importances.

    Args:
        importance_df: DataFrame with columns [feature, importance].
        top_n: Show only the top N features.

    Returns:
        Matplotlib Figure.
    """
    top = importance_df.head(top_n).sort_values("importance")

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
    ax.barh(top["feature"], top["importance"], color="#6366F1")
    ax.set_title(title)
    ax.set_xlabel("Importance")
    fig.tight_layout()
    return fig


def plot_cluster_scatter(
    embeddings_2d: np.ndarray,
    labels: np.ndarray,
    outcomes: np.ndarray = None,
    title: str = "Game State Clusters",
) -> plt.Figure:
    """Scatter plot of clustered embeddings (2D projection).

    Args:
        embeddings_2d: (n_samples, 2) array (e.g. from PCA or t-SNE).
        labels: Cluster labels per sample.
        outcomes: Optional binary outcomes for colouring markers.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(9, 7))

    unique_labels = sorted(set(labels) - {-1})
    colors = plt.cm.Set2(np.linspace(0, 1, len(unique_labels)))

    for cluster_id, color in zip(unique_labels, colors):
        mask = labels == cluster_id
        marker = "o"
        ax.scatter(
            embeddings_2d[mask, 0], embeddings_2d[mask, 1],
            c=[color], label=f"Cluster {cluster_id}", alpha=0.6, s=30, marker=marker,
        )

    # Mark noise points (DBSCAN label -1)
    noise_mask = labels == -1
    if noise_mask.any():
        ax.scatter(
            embeddings_2d[noise_mask, 0], embeddings_2d[noise_mask, 1],
            c="gray", label="Noise", alpha=0.3, s=15, marker="x",
        )

    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


def plot_grouping_timeline(
    grouping_df: pd.DataFrame,
    team_name: str = "Blue",
    threshold: float = None,
    ax: plt.Axes = None,
) -> plt.Figure:
    """Plot team grouping distance over time.

    Args:
        grouping_df: Output from SpatialFeatures.grouping_over_time().
        team_name: Label for the team.
        threshold: Optional horizontal line showing grouping threshold.

    Returns:
        Matplotlib Figure.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    else:
        fig = ax.figure

    ax.plot(grouping_df["timestamp"] / 60, grouping_df["mean_distance"],
            color=BLUE_COLOR if "blue" in team_name.lower() else RED_COLOR, linewidth=1)
    if threshold:
        ax.axhline(y=threshold, color="gray", linestyle="--", alpha=0.5, label="Grouping threshold")
    ax.set_title(f"{team_name} Team Grouping Distance Over Time")
    ax.set_xlabel("Game Time (minutes)")
    ax.set_ylabel("Mean Pairwise Distance")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_ablation_comparison(
    ablation_results: dict[str, dict],
    title: str = "CV vs API Feature Ablation Study",
) -> plt.Figure:
    """Bar chart comparing CV-only, API-only, and combined feature sets.

    Args:
        ablation_results: Output from WinPredictor.ablation_study().

    Returns:
        Matplotlib Figure.
    """
    df = pd.DataFrame(ablation_results).T
    colors = {"cv_only": "#6366F1", "api_only": "#F59E0B", "combined": "#10B981"}

    fig, ax = plt.subplots(figsize=(8, 5))
    df.plot(kind="bar", ax=ax, rot=0, color=[colors.get(c, "#888") for c in df.index])
    ax.set_title(title)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8, padding=2)

    fig.tight_layout()
    return fig
