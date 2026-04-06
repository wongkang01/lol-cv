"""
Stage 5 — Train classifiers, run ablation, generate insights.

Loads ``data/processed/features.csv`` + ``features_meta.csv``, joins them,
constructs the binary target (1 = blue win), and:

    1. Trains RF/SVM/GBM/MLP with stratified 5-fold CV
    2. Reports accuracy, F1, ROC-AUC per model
    3. Runs ablation: spatial-only vs ocr-only vs combined
    4. Computes feature importance from gradient boosting
    5. Computes Spearman correlation between each feature and outcome
    6. Saves all outputs to data/processed/analysis/

Outputs:
    data/processed/analysis/cv_results.csv         per-model CV metrics
    data/processed/analysis/ablation.csv           feature-set comparison
    data/processed/analysis/feature_importance.csv top features (GBM)
    data/processed/analysis/feature_correlations.csv Spearman vs outcome
    data/processed/analysis/summary.json           overall summary
    data/processed/analysis/plots/*.png            visualizations
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from lol_cv.analysis.classifiers import WinPredictor
from lol_cv.utils import setup_logger, ensure_dir

logger = setup_logger("scripts.run_analysis")

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = REPO_ROOT / "data" / "processed"


def load_data(features_filename: str) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    feats = pd.read_csv(PROCESSED / features_filename).set_index("match_id")
    meta = pd.read_csv(PROCESSED / "features_meta.csv").set_index("match_id")

    # Drop rows missing winner
    meta = meta.dropna(subset=["winner_side"])
    common = feats.index.intersection(meta.index)
    feats = feats.loc[common]
    meta = meta.loc[common]

    y = (meta["winner_side"] == "blue").astype(int)
    y.name = "blue_wins"

    # Drop entirely-NaN columns and fill rest with 0 (safe for trees)
    feats = feats.dropna(axis=1, how="all").fillna(0.0)

    logger.info("Loaded %d games × %d features", *feats.shape)
    logger.info("Class balance: blue_wins=%d red_wins=%d", int(y.sum()), int((1 - y).sum()))
    return feats, y, meta


def split_feature_groups(X: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Partition features by prefix into spatial / temporal / ocr groups."""
    groups = {
        "spatial": X.filter(regex=r"^sp_"),
        "temporal": X.filter(regex=r"^tp_"),
        "ocr": X.filter(regex=r"^ocr_"),
    }
    groups["spatial_temporal"] = pd.concat([groups["spatial"], groups["temporal"]], axis=1)
    groups["combined"] = X
    for k, v in groups.items():
        logger.info("  group %-18s %d features", k, v.shape[1])
    return groups


def train_full_model(X: pd.DataFrame, y: pd.Series, cv_folds: int = 5) -> tuple[WinPredictor, pd.DataFrame]:
    """Train all 4 models and return CV results."""
    predictor = WinPredictor(cv_folds=cv_folds, random_state=42)
    results = predictor.train_and_evaluate(X, y)
    df = predictor.results_to_dataframe()
    return predictor, df


def run_ablation(X: pd.DataFrame, y: pd.Series, groups: dict[str, pd.DataFrame], cv_folds: int = 5) -> pd.DataFrame:
    """Compare spatial-only / ocr-only / combined feature sets across models.

    If the OCR feature group is empty, the ``ocr_only`` ablation is skipped
    entirely (a warning is logged once) rather than being silently substituted
    with a placeholder column — substitution would make the ``ocr_only`` row
    meaningless. In that case only ``spatial_only`` and ``combined`` rows are
    produced per model.
    """
    rows = []
    ocr_empty = groups["ocr"].empty
    if ocr_empty:
        logger.warning("Skipping OCR-only ablation: no OCR features in matrix")

    for model in ["random_forest", "gradient_boosting", "svm"]:
        pred = WinPredictor(models=[model], cv_folds=cv_folds, random_state=42)
        if ocr_empty:
            # Call ablation_study with a tiny placeholder X_ocr (one spatial
            # column) just to satisfy the signature, then drop the ocr_only
            # entry from the result so it never reaches the output CSV.
            placeholder = groups["spatial"].iloc[:, :1]
            ab = pred.ablation_study(
                X_spatial=groups["spatial"],
                X_ocr=placeholder,
                X_combined=groups["combined"],
                y=y,
                model_name=model,
            )
            ab.pop("ocr_only", None)
        else:
            ab = pred.ablation_study(
                X_spatial=groups["spatial"],
                X_ocr=groups["ocr"],
                X_combined=groups["combined"],
                y=y,
                model_name=model,
            )
        for label, m in ab.items():
            rows.append({
                "model": model, "feature_set": label,
                "accuracy": m["accuracy"], "f1": m["f1"], "roc_auc": m["roc_auc"],
            })
    return pd.DataFrame(rows)


def feature_importance_table(
    predictor: WinPredictor, feature_names: list[str], model_name: str = "gradient_boosting"
) -> pd.DataFrame:
    return predictor.feature_importance(model_name, feature_names=feature_names)


def feature_correlations(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    rows = []
    for col in X.columns:
        try:
            r, p = spearmanr(X[col], y)
        except Exception:
            r, p = np.nan, np.nan
        rows.append({"feature": col, "spearman_r": r, "p_value": p, "abs_r": abs(r) if r is not None else 0})
    df = pd.DataFrame(rows).sort_values("abs_r", ascending=False).reset_index(drop=True)
    return df


def plot_cv_results(cv_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    metrics = ["accuracy", "f1", "roc_auc"]
    x = np.arange(len(cv_df.index))
    width = 0.25
    for i, m in enumerate(metrics):
        ax.bar(x + i * width, cv_df[m], width=width, label=m)
    ax.set_xticks(x + width)
    ax.set_xticklabels(cv_df.index, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score (5-fold CV mean)")
    ax.set_title("Win prediction performance by model")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_ablation(ab_df: pd.DataFrame, out_path: Path) -> None:
    """Bar-plot ROC-AUC by model × feature-set.

    Handles ablation frames where some feature sets are absent (e.g. the
    ``ocr_only`` rows are dropped when no OCR features are in the matrix) by
    only including the feature sets that are actually present in ``ab_df``.
    """
    if ab_df.empty:
        logger.warning("plot_ablation: ablation frame is empty — skipping plot")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    pivot = ab_df.pivot(index="model", columns="feature_set", values="roc_auc")

    # Preserve a sensible column order when present, but don't assume all
    # three ablation sets ran for every model.
    preferred_order = ["spatial_only", "ocr_only", "combined"]
    ordered_cols = [c for c in preferred_order if c in pivot.columns]
    extra_cols = [c for c in pivot.columns if c not in preferred_order]
    pivot = pivot[ordered_cols + extra_cols]

    pivot.plot(kind="bar", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("ROC-AUC (5-fold CV)")
    ax.set_title("Ablation: feature-set contribution to win prediction")
    ax.legend(title="Feature set")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_top_features(imp_df: pd.DataFrame, out_path: Path, top_k: int = 20) -> None:
    top = imp_df.head(top_k).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top["feature"], top["importance"])
    ax.set_xlabel("Gradient boosting importance")
    ax.set_title(f"Top {top_k} features for predicting blue side win")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run win-prediction analysis on a features file.")
    parser.add_argument(
        "--features",
        type=str,
        default="features.csv",
        help="Name of the features CSV under data/processed/ (default: features.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis",
        help="Name of the output directory under data/processed/ (default: analysis)",
    )
    args = parser.parse_args()

    logger.info("Loading features from %s, output to %s", args.features, args.output_dir)

    if not (PROCESSED / args.features).exists():
        sys.exit(f"Missing data/processed/{args.features} — run scripts/run_features.py first")

    analysis_dir = PROCESSED / args.output_dir
    plots_dir = analysis_dir / "plots"
    ensure_dir(analysis_dir)
    ensure_dir(plots_dir)

    X, y, meta = load_data(args.features)
    groups = split_feature_groups(X)

    # Adapt CV fold count to dataset size.
    #
    # Rules (based on the minority-class count):
    #   min_class < 2   -> fatal: cannot stratify a CV at all
    #   min_class == 2  -> k=2 (warn that fold-level metrics will be very
    #                      noisy and AUC may be undefined in some folds)
    #   2 < min_class < 5 -> k=min_class (so 3->3, 4->4)
    #   min_class >= 5  -> k=5 (full behaviour)
    min_class = int(min(y.sum(), (1 - y).sum()))
    if min_class < 2:
        sys.exit(
            "Cannot run CV with fewer than 2 samples of the minority class — got %d"
            % min_class
        )
    if min_class == 2:
        cv_folds = 2
        logger.warning(
            "Minority class has only 2 samples — falling back to cv_folds=2. "
            "Fold-level metrics will be very noisy and ROC-AUC may be "
            "undefined for some folds."
        )
    elif min_class >= 5:
        cv_folds = 5
    else:
        cv_folds = min_class
    logger.info("Using cv_folds=%d (min class size %d)", cv_folds, min_class)

    # ── 1. CV training on all features ──
    logger.info("\n=== Training all models on combined features ===")
    predictor, cv_df = train_full_model(X, y, cv_folds=cv_folds)
    cv_df.to_csv(analysis_dir / "cv_results.csv")
    plot_cv_results(cv_df, plots_dir / "cv_results.png")
    logger.info("\n%s", cv_df.to_string())

    # ── 2. Ablation ──
    logger.info("\n=== Ablation ===")
    ab_df = run_ablation(X, y, groups, cv_folds=cv_folds)
    ab_df.to_csv(analysis_dir / "ablation.csv", index=False)
    plot_ablation(ab_df, plots_dir / "ablation.png")
    logger.info("\n%s", ab_df.to_string())

    # ── 3. Feature importance (GBM trained on combined) ──
    logger.info("\n=== Feature importance (gradient boosting) ===")
    imp_df = feature_importance_table(predictor, list(X.columns), "gradient_boosting")
    imp_df.to_csv(analysis_dir / "feature_importance.csv", index=False)
    plot_top_features(imp_df, plots_dir / "feature_importance_top20.png")
    logger.info("\nTop 20 features:\n%s", imp_df.head(20).to_string(index=False))

    # ── 4. Spearman correlations ──
    logger.info("\n=== Spearman correlations ===")
    corr_df = feature_correlations(X, y)
    corr_df.to_csv(analysis_dir / "feature_correlations.csv", index=False)
    logger.info("\nTop 15 |corr| with outcome:\n%s",
                corr_df.head(15)[["feature", "spearman_r", "p_value"]].to_string(index=False))

    # ── 5. Summary ──
    # Pick best model by AUC if available, else by accuracy
    auc_col = cv_df["roc_auc"].dropna()
    if not auc_col.empty:
        best_model = auc_col.idxmax()
        ranking_metric = "roc_auc"
    else:
        best_model = cv_df["accuracy"].idxmax()
        ranking_metric = "accuracy"

    ab_auc = ab_df["roc_auc"].dropna()
    ablation_best = ab_df.loc[ab_auc.idxmax(), "feature_set"] if not ab_auc.empty else (
        ab_df.loc[ab_df["accuracy"].idxmax(), "feature_set"] if not ab_df.empty else None
    )

    summary = {
        "n_games": int(len(y)),
        "blue_wins": int(y.sum()),
        "red_wins": int((1 - y).sum()),
        "blue_win_rate": float(y.mean()),
        "cv_folds": int(cv_folds),
        "n_features_total": int(X.shape[1]),
        "n_features_spatial": int(groups["spatial"].shape[1]),
        "n_features_temporal": int(groups["temporal"].shape[1]),
        "n_features_ocr": int(groups["ocr"].shape[1]),
        "best_model": best_model,
        "best_model_auc": float(cv_df.loc[best_model, "roc_auc"]) if not pd.isna(cv_df.loc[best_model, "roc_auc"]) else None,
        "best_model_acc": float(cv_df.loc[best_model, "accuracy"]),
        "ranking_metric": ranking_metric,
        "ablation_best_set": ablation_best,
        "top_5_features": imp_df.head(5)["feature"].tolist(),
        "top_5_correlations": corr_df.head(5)[["feature", "spearman_r", "p_value"]].to_dict("records"),
    }
    (analysis_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    logger.info("\n=== Summary ===")
    auc_str = f"{summary['best_model_auc']:.3f}" if summary["best_model_auc"] is not None else "n/a"
    logger.info("Best model: %s — AUC=%s, Acc=%.3f",
                summary["best_model"], auc_str, summary["best_model_acc"])
    logger.info("All outputs in %s", analysis_dir)


if __name__ == "__main__":
    main()
