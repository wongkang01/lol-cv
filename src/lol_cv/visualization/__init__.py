"""
Visualization modules for charts, heatmaps, and dashboards.

- plots: Static matplotlib/seaborn visualizations (heatmaps, trajectories, bar charts)
"""

from lol_cv.visualization.plots import (
    plot_heatmap,
    plot_trajectory,
    plot_model_comparison,
    plot_feature_importance,
    plot_cluster_scatter,
    plot_grouping_timeline,
    plot_ablation_comparison,
    plot_precision_recall_curves,
)

__all__ = [
    "plot_heatmap",
    "plot_trajectory",
    "plot_model_comparison",
    "plot_feature_importance",
    "plot_cluster_scatter",
    "plot_grouping_timeline",
    "plot_ablation_comparison",
    "plot_precision_recall_curves",
]
