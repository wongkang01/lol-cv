"""
Re-run per-category ablation using the *new* taxonomy in
``data/processed/feature_categories_v2.csv``.

For each of the four time-window feature files (``features_corrected.csv``,
``features_corrected_0_300.csv``, ``features_corrected_0_600.csv``,
``features_corrected_0_900.csv``) and for each category in the new
taxonomy, we:

    1. Load X, y via the same helper logic used by ``scripts/run_analysis.py``
       (same NaN handling, same target construction).
    2. Subset to the category's features (that actually exist in this window).
    3. Train RF / GBM / SVM with 5-fold stratified CV (the same three models
       ``run_analysis.py`` uses for category ablation).
    4. Record accuracy / f1 / roc_auc per (window, category, model).

Output lives in ``data/processed/analysis_corrected_v2/`` and includes:
    - per_window_category_models.csv   (long format — one row per W/C/M)
    - per_window_category_best.csv     (best-AUC-per-(window,category), used
      for the summary table in the report)
    - best_auc_wide.csv                (categories x windows wide table)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lol_cv.analysis.classifiers import WinPredictor  # noqa: E402
from lol_cv.utils import setup_logger  # noqa: E402

logger = setup_logger("scripts.rerun_categories_v2")

PROCESSED = REPO_ROOT / "data" / "processed"
CATEGORY_CSV = PROCESSED / "feature_categories_v2.csv"
OUT_DIR = PROCESSED / "analysis_corrected_v2"

WINDOWS = [
    ("0-5 min",  "features_corrected_0_300.csv"),
    ("0-10 min", "features_corrected_0_600.csv"),
    ("0-15 min", "features_corrected_0_900.csv"),
    ("Full",     "features_corrected.csv"),
]

MODELS = ["random_forest", "gradient_boosting", "svm"]


def _apply_feature_nan_handling(feats: pd.DataFrame) -> pd.DataFrame:
    """Mirror the NaN handling used by run_analysis.py so AUCs are comparable."""
    feats = feats.dropna(axis=1, how="all")
    rare_event_cols = [
        c for c in feats.columns
        if c.startswith("sp_early_") or c.startswith("sp_strat_")
    ]
    for col in rare_event_cols:
        feats[f"{col}_missing"] = feats[col].isna().astype(int)
        if any(s in col for s in ["_count", "_frames", "_secs", "_recalls"]):
            fill_value: float = 0
        else:
            fill_value = -1
        feats[col] = feats[col].fillna(fill_value)
    feats = feats.fillna(0.0)
    return feats


def _load_xy(features_filename: str) -> tuple[pd.DataFrame, pd.Series]:
    feats = pd.read_csv(PROCESSED / features_filename).set_index("match_id")
    meta = pd.read_csv(PROCESSED / "features_meta.csv").set_index("match_id")
    meta = meta.dropna(subset=["winner_side"])
    common = feats.index.intersection(meta.index)
    feats = feats.loc[common]
    meta = meta.loc[common]
    y = (meta["winner_side"] == "blue").astype(int)
    feats = _apply_feature_nan_handling(feats)
    return feats, y


def main() -> None:
    if not CATEGORY_CSV.exists():
        sys.exit(
            f"Missing {CATEGORY_CSV}. Run scripts/build_categories_v2.py first."
        )

    cat_df = pd.read_csv(CATEGORY_CSV)
    logger.info("Loaded v2 category map: %d features, %d categories",
                len(cat_df), cat_df["category_v2"].nunique())

    # feature -> category_v2
    feat_to_cat = dict(zip(cat_df["feature"], cat_df["category_v2"]))
    all_categories = sorted(cat_df["category_v2"].unique())

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    long_rows: list[dict] = []

    for window_label, fname in WINDOWS:
        if not (PROCESSED / fname).exists():
            logger.warning("Skipping %s — %s not found", window_label, fname)
            continue

        X_all, y = _load_xy(fname)
        logger.info("[%s] Loaded %d games x %d features",
                    window_label, X_all.shape[0], X_all.shape[1])

        # The NaN handler may add ``_missing`` sibling columns for rare-event
        # features. Bucket these into the same category as the parent.
        for col in X_all.columns:
            if col in feat_to_cat:
                continue
            if col.endswith("_missing"):
                parent = col[: -len("_missing")]
                if parent in feat_to_cat:
                    feat_to_cat[col] = feat_to_cat[parent]

        for cat in all_categories:
            cols = [c for c in X_all.columns if feat_to_cat.get(c) == cat]
            if not cols:
                logger.warning("[%s] category %s empty — skipping", window_label, cat)
                continue

            X_cat = X_all[cols]
            logger.info("[%s] category %-28s %3d features",
                        window_label, cat, X_cat.shape[1])

            for model in MODELS:
                pred = WinPredictor(models=[model], cv_folds=5, random_state=42)
                metrics = pred.train_and_evaluate(X_cat, y)[model]
                long_rows.append({
                    "window": window_label,
                    "category": cat,
                    "model": model,
                    "accuracy": metrics["accuracy"],
                    "f1": metrics["f1"],
                    "roc_auc": metrics["roc_auc"],
                    "n_features": int(X_cat.shape[1]),
                })

    long_df = pd.DataFrame(long_rows)
    long_df.to_csv(OUT_DIR / "per_window_category_models.csv", index=False)
    logger.info("Wrote %s", OUT_DIR / "per_window_category_models.csv")

    # Best model per (window, category) by AUC.
    best_df = (
        long_df.sort_values("roc_auc", ascending=False)
        .drop_duplicates(["window", "category"])
        .reset_index(drop=True)
    )
    best_df = best_df[["window", "category", "model", "roc_auc", "accuracy", "f1", "n_features"]]
    best_df.to_csv(OUT_DIR / "per_window_category_best.csv", index=False)
    logger.info("Wrote %s", OUT_DIR / "per_window_category_best.csv")

    # Wide: categories x windows, values = best AUC.
    wide = best_df.pivot(index="category", columns="window", values="roc_auc")
    # Preserve the requested column order.
    wide = wide[[w for w, _ in WINDOWS if w in wide.columns]]
    # Sort categories by mean AUC across windows (descending).
    wide["avg"] = wide.mean(axis=1)
    wide = wide.sort_values("avg", ascending=False)
    wide.to_csv(OUT_DIR / "best_auc_wide.csv")
    logger.info("Wrote %s", OUT_DIR / "best_auc_wide.csv")

    # Print a quick summary for the run log.
    print("\n=== Per-category best AUC (categories x windows) ===")
    print(wide.round(3).to_string())

    print("\n=== Top 3 categories per window ===")
    for window_label, _ in WINDOWS:
        if window_label not in wide.columns:
            continue
        top3 = wide[window_label].dropna().sort_values(ascending=False).head(3)
        print(f"\n{window_label}:")
        for cat, auc in top3.items():
            print(f"  {cat:30s} AUC={auc:.3f}")


if __name__ == "__main__":
    main()
