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
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from lol_cv.analysis.classifiers import WinPredictor
from lol_cv.utils import setup_logger, ensure_dir

logger = setup_logger("scripts.run_analysis")

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = REPO_ROOT / "data" / "processed"


def _apply_feature_nan_handling(feats: pd.DataFrame) -> pd.DataFrame:
    """Apply the rare-event-aware NaN handling used by both classification
    and regression paths.

    - Drops entirely-NaN columns.
    - For ``sp_early_*``/``sp_strat_*`` columns adds a parallel
      ``<col>_missing`` indicator (NaN -> 1) and fills the original with
      either ``0`` (count-like features) or ``-1`` (continuous sentinel).
    - Fills everything else with ``0.0``.

    The behaviour is intentionally identical to the logic that used to live
    inline in ``load_data`` — regression reuses the same helper so that
    feature semantics stay consistent across modes.
    """
    feats = feats.dropna(axis=1, how="all")

    rare_event_cols = [
        c for c in feats.columns
        if c.startswith("sp_early_") or c.startswith("sp_strat_")
    ]

    missing_indicators_added = 0
    for col in rare_event_cols:
        indicator = feats[col].isna().astype(int)
        feats[f"{col}_missing"] = indicator
        missing_indicators_added += 1

        # Heuristic: counts use 0, everything else uses -1 sentinel.
        if any(s in col for s in ["_count", "_frames", "_secs", "_recalls"]):
            fill_value: float = 0
        else:
            fill_value = -1
        feats[col] = feats[col].fillna(fill_value)

    logger.info(
        "Added %d missingness indicators for rare-event sp_early_*/sp_strat_* columns",
        missing_indicators_added,
    )

    # For everything else (full-game sp_*, tp_*, ocr_*, meta), use legacy 0-fill.
    feats = feats.fillna(0.0)
    return feats


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

    # ── Smarter NaN handling for rare-event features ──
    #
    # Rare-event features (``sp_early_*`` / ``sp_strat_*``) have meaningful
    # missingness: a NaN means "the event never happened" (e.g. the mid laner
    # never roamed), which is NOT the same as "the event happened at t=0".
    # Filling these with 0 would tell the model "roamed instantly" instead of
    # "never roamed" — destroying signal for rare events.
    feats = _apply_feature_nan_handling(feats)

    logger.info("Loaded %d games × %d features", *feats.shape)
    logger.info("Class balance: blue_wins=%d red_wins=%d", int(y.sum()), int((1 - y).sum()))
    return feats, y, meta


def load_data_regression(
    features_filename: str, target_name: str
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load features and a continuous regression target keyed on match_id.

    Behaves like ``load_data`` for the feature matrix (same NaN handling,
    same meta join) but joins against ``data/processed/targets.csv`` and
    returns ``y = targets[target_name]``. Rows where the target is NaN are
    dropped so the downstream CV runner never sees a NaN in ``y``.

    ``sys.exit`` is raised (with a helpful message) if ``targets.csv`` does
    not exist or does not contain the requested column.
    """
    targets_path = PROCESSED / "targets.csv"
    if not targets_path.exists():
        sys.exit(
            f"Missing {targets_path} — build_targets.py has not produced "
            "targets.csv yet. Re-run once the targets agent finishes, or "
            "drop the --regression-target flag to use classification mode."
        )

    targets = pd.read_csv(targets_path).set_index("match_id")
    if target_name not in targets.columns:
        sys.exit(
            f"Target column '{target_name}' not found in targets.csv. "
            f"Available: {sorted(targets.columns)}"
        )

    feats = pd.read_csv(PROCESSED / features_filename).set_index("match_id")
    meta_path = PROCESSED / "features_meta.csv"
    if meta_path.exists():
        meta = pd.read_csv(meta_path).set_index("match_id")
    else:
        meta = pd.DataFrame(index=feats.index)

    # Join on the 3-way intersection of feats / meta / targets, then drop
    # rows where the target is NaN (some games may be missing e.g.
    # gold_diff_t1200 because they ended early).
    common = feats.index.intersection(targets.index)
    if not meta.empty:
        common = common.intersection(meta.index)
    feats = feats.loc[common]
    y_full = targets.loc[common, target_name]
    mask = y_full.notna()
    feats = feats.loc[mask]
    y = y_full.loc[mask].astype(float)
    y.name = target_name
    if not meta.empty:
        meta = meta.loc[feats.index]

    dropped = int((~mask).sum())
    if dropped:
        logger.info(
            "Dropped %d rows with NaN target (%s)", dropped, target_name
        )

    feats = _apply_feature_nan_handling(feats)

    logger.info(
        "Loaded %d games × %d features for regression target '%s'",
        feats.shape[0], feats.shape[1], target_name,
    )
    logger.info(
        "Target stats: mean=%.3f std=%.3f min=%.3f max=%.3f",
        float(y.mean()), float(y.std(ddof=0)), float(y.min()), float(y.max()),
    )
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


# ── Strategic category ablation ──
#
# Maps each category to a predicate over column names. A column may match more
# than one category (e.g. a jungle-lane positioning column) — that's expected
# and fine: it just shows up in both ablations. The two rare-event categories
# (``early_strategy``/``strategic_decisions``) are deliberately mutually
# exclusive with the three pattern-based full-game categories.
CATEGORIES = {
    "jungle_invasion": lambda c: (
        "jungle" in c
        and c.startswith("sp_")
        and not c.startswith("sp_early_")
        and not c.startswith("sp_strat_")
    ),
    "objective_control": lambda c: (
        any(o in c for o in ["baron", "dragon", "herald", "_pit"])
        and c.startswith("sp_")
    ),
    "lane_positioning": lambda c: (
        any(l in c for l in ["_lane", "_river_", "_base"])
        and c.startswith("sp_")
        and "jungle" not in c
    ),
    "team_coordination": lambda c: (
        any(g in c for g in ["grouping", "convergence", "synced_recalls", "asymmetry"])
        and c.startswith("sp_")
    ),
    "early_strategy": lambda c: c.startswith("sp_early_"),
    "strategic_decisions": lambda c: c.startswith("sp_strat_"),
    "temporal_dynamics": lambda c: c.startswith("tp_"),
    "ocr_state": lambda c: c.startswith("ocr_"),
}


def category_ablation(
    X: pd.DataFrame, y: pd.Series, cv_folds: int = 5
) -> pd.DataFrame:
    """Rank strategic categories by predictive contribution.

    For each category defined in ``CATEGORIES`` above, build the subset
    ``X_cat = X[matching columns]`` and train RF / GBM / SVM with
    stratified ``cv_folds``-fold CV. Records one row per (category, model)
    with accuracy, f1, roc_auc and the feature count.

    Empty categories (no columns match in the current matrix) are skipped
    with a warning so the function stays backward compatible while new
    feature families (e.g. ``sp_strat_*``) are still being added.

    Returns a DataFrame sorted by ``roc_auc`` descending.
    """
    rows: list[dict] = []
    for cat_name, predicate in CATEGORIES.items():
        cols = [c for c in X.columns if predicate(c)]
        if not cols:
            logger.warning(
                "category_ablation: category %s is empty — skipping", cat_name
            )
            continue

        X_cat = X[cols]
        logger.info(
            "category_ablation: %-22s %3d features", cat_name, X_cat.shape[1]
        )

        for model in ["random_forest", "gradient_boosting", "svm"]:
            pred = WinPredictor(
                models=[model], cv_folds=cv_folds, random_state=42
            )
            metrics = pred.train_and_evaluate(X_cat, y)[model]
            rows.append({
                "category": cat_name,
                "model": model,
                "accuracy": metrics["accuracy"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
                "n_features": int(X_cat.shape[1]),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        logger.warning("category_ablation: no non-empty categories — returning empty frame")
        return df
    return df.sort_values("roc_auc", ascending=False).reset_index(drop=True)


def plot_category_ablation(cat_df: pd.DataFrame, out_path: Path) -> None:
    """Bar-plot category_ablation: categories on x-axis, AUC on y-axis,
    grouped by model. Feature counts annotated above each bar.

    Styled to match ``plot_ablation`` for visual consistency.
    """
    if cat_df.empty:
        logger.warning("plot_category_ablation: frame is empty — skipping plot")
        return

    fig, ax = plt.subplots(figsize=(11, 6))
    pivot_auc = cat_df.pivot(index="category", columns="model", values="roc_auc")
    pivot_n = cat_df.pivot(index="category", columns="model", values="n_features")

    # Order categories by best AUC across models (descending) so the plot
    # mirrors the CSV's "most predictive category first" ordering.
    ordering = pivot_auc.max(axis=1).sort_values(ascending=False).index
    pivot_auc = pivot_auc.loc[ordering]
    pivot_n = pivot_n.loc[ordering]

    pivot_auc.plot(kind="bar", ax=ax)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("ROC-AUC (CV mean)")
    ax.set_title("Per-category ablation: strategic component contribution")
    ax.legend(title="Model")
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")

    # Annotate each bar with the feature count for its (category, model).
    # ``matplotlib`` renders bars model-by-model; we iterate in the same
    # order so the labels line up.
    n_cats = pivot_auc.shape[0]
    n_models = pivot_auc.shape[1]
    for m_idx, model_name in enumerate(pivot_auc.columns):
        container = ax.containers[m_idx]
        for c_idx, bar in enumerate(container):
            n_feat = pivot_n.iloc[c_idx, m_idx]
            if pd.isna(n_feat):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"n={int(n_feat)}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=0,
            )
        # Use n_models just to silence unused-variable lint in case of future edits.
        _ = n_cats, n_models

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


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


def feature_correlations_regression(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Compute Pearson AND Spearman correlations between each feature and a
    continuous target.

    Returns a frame with columns
    ``[feature, pearson_r, pearson_p, spearman_r, spearman_p, abs_max_r]``
    sorted by ``abs_max_r`` descending, where ``abs_max_r = max(|pearson|,
    |spearman|)``. Constant / zero-variance columns produce NaN without
    raising (scipy warnings are silenced at the call site).
    """
    import warnings as _warnings

    rows: list[dict] = []
    y_vals = np.asarray(y, dtype=float)
    for col in X.columns:
        x_vals = np.asarray(X[col], dtype=float)
        try:
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                if np.nanstd(x_vals) == 0 or np.nanstd(y_vals) == 0:
                    pr, pp = np.nan, np.nan
                    sr, sp = np.nan, np.nan
                else:
                    pr, pp = pearsonr(x_vals, y_vals)
                    sr, sp = spearmanr(x_vals, y_vals)
        except Exception:
            pr, pp, sr, sp = np.nan, np.nan, np.nan, np.nan

        abs_pr = 0.0 if pr is None or np.isnan(pr) else abs(pr)
        abs_sr = 0.0 if sr is None or np.isnan(sr) else abs(sr)
        rows.append({
            "feature": col,
            "pearson_r": pr,
            "pearson_p": pp,
            "spearman_r": sr,
            "spearman_p": sp,
            "abs_max_r": max(abs_pr, abs_sr),
        })
    df = pd.DataFrame(rows).sort_values("abs_max_r", ascending=False).reset_index(drop=True)
    return df


def _build_regression_models(random_state: int = 42) -> dict:
    """Return the regression models used in regression mode.

    - ``linear_regression``: unregularised baseline. Useful for comparison
      but unstable when n is comparable to the number of features.
    - ``ridge_regression``: L2-regularised linear model wrapped in a
      ``StandardScaler`` so the regularisation strength applies uniformly
      across features. Critical for n≪d regimes.
    - ``gradient_boosting_regressor``: non-linear baseline. Tree-based so
      it handles unscaled features and missing-indicator columns natively.

    Kept as a tiny helper so the test suite can import and reuse it.
    """
    return {
        "linear_regression": LinearRegression(),
        "ridge_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0, random_state=random_state)),
        ]),
        "gradient_boosting_regressor": GradientBoostingRegressor(
            random_state=random_state
        ),
    }


def _select_top_k_by_correlation(
    X: pd.DataFrame, y: pd.Series, top_k: int
) -> tuple[pd.DataFrame, list[str]]:
    """Restrict X to the top-k features by absolute Spearman correlation.

    Used by the regression pipeline to keep n ≫ d when fitting models on
    a small dataset. Constant columns and missingness indicators are
    excluded before ranking.
    """
    candidate_cols = [
        c for c in X.columns
        if not c.endswith("_missing") and X[c].nunique(dropna=False) > 1
    ]
    if not candidate_cols:
        logger.warning("No non-constant features available for correlation ranking")
        return X, list(X.columns)

    scores: list[tuple[str, float]] = []
    for col in candidate_cols:
        try:
            r, _ = spearmanr(X[col], y)
        except Exception:
            r = 0.0
        if r is None or np.isnan(r):
            r = 0.0
        scores.append((col, abs(float(r))))
    scores.sort(key=lambda kv: kv[1], reverse=True)
    selected = [c for c, _ in scores[:top_k]]
    return X[selected], selected


def run_regression_cv(
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Run K-fold cross validation for both regression models.

    Uses ``KFold`` (NOT stratified — the target is continuous). For each
    model, computes per-fold R² and MAE and reports mean/std. Also returns
    the out-of-fold predictions (via ``cross_val_predict``) keyed by model
    name so the caller can build scatter plots.

    Returns
    -------
    cv_df : pd.DataFrame
        Indexed by model name, columns
        ``[r2_mean, r2_std, mae_mean, mae_std, n_samples]``.
    oof_preds : dict[str, np.ndarray]
        Out-of-fold predictions per model, aligned to ``y.index``.
    """
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    results: list[dict] = []
    oof_preds: dict[str, np.ndarray] = {}
    X_vals = X.values
    y_vals = y.values.astype(float)

    for name, model in _build_regression_models(random_state=random_state).items():
        r2s: list[float] = []
        maes: list[float] = []
        for train_idx, test_idx in kf.split(X_vals):
            model.fit(X_vals[train_idx], y_vals[train_idx])
            pred = model.predict(X_vals[test_idx])
            r2s.append(r2_score(y_vals[test_idx], pred))
            maes.append(mean_absolute_error(y_vals[test_idx], pred))

        # Separate call for out-of-fold predictions so we can plot actual
        # vs. predicted later without interfering with the per-fold scoring
        # above.
        oof = cross_val_predict(
            _build_regression_models(random_state=random_state)[name],
            X_vals,
            y_vals,
            cv=KFold(n_splits=cv_folds, shuffle=True, random_state=random_state),
        )
        oof_preds[name] = oof

        results.append({
            "model": name,
            "r2_mean": float(np.mean(r2s)),
            "r2_std": float(np.std(r2s, ddof=0)),
            "mae_mean": float(np.mean(maes)),
            "mae_std": float(np.std(maes, ddof=0)),
            "n_samples": int(len(y)),
        })
        logger.info(
            "  %s: R²=%.3f±%.3f, MAE=%.3f±%.3f",
            name, results[-1]["r2_mean"], results[-1]["r2_std"],
            results[-1]["mae_mean"], results[-1]["mae_std"],
        )

    cv_df = pd.DataFrame(results).set_index("model")
    return cv_df, oof_preds


def plot_regression_correlations_top20(
    corr_df: pd.DataFrame, out_path: Path, top_k: int = 20
) -> None:
    """Horizontal bar plot of the top features ranked by ``abs_max_r``.

    The x-axis shows signed Spearman r (not |r|) so the reader can still
    see direction, but ranking uses ``abs_max_r`` per the spec.
    """
    if corr_df.empty:
        logger.warning("plot_regression_correlations_top20: empty frame — skipping")
        return
    top = corr_df.head(top_k).iloc[::-1]  # reverse so largest is at top
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = ["#1f77b4" if r >= 0 else "#d62728" for r in top["spearman_r"].fillna(0)]
    ax.barh(top["feature"], top["spearman_r"].fillna(0), color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Spearman r vs target")
    ax.set_title(f"Top {top_k} features by |correlation| with target")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_regression_scatter(
    y_true: pd.Series, y_pred: np.ndarray, model_name: str, out_path: Path
) -> None:
    """Scatter of out-of-fold predicted vs actual target values."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolor="k", linewidth=0.3)
    lo = float(min(y_true.min(), np.min(y_pred)))
    hi = float(max(y_true.max(), np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi], "r--", lw=1, label="y = x")
    ax.set_xlabel(f"Actual ({y_true.name})")
    ax.set_ylabel("Predicted (out-of-fold)")
    ax.set_title(f"{model_name}: predicted vs actual")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _sanitise_target_name(target: str) -> str:
    """Make a target column name safe to use as a directory component.

    Replaces any character that is not alphanumeric, underscore, dash or dot
    with ``_``. Idempotent.
    """
    return re.sub(r"[^A-Za-z0-9_.\-]", "_", target)


def run_regression_mode(args: argparse.Namespace) -> None:
    """End-to-end regression pipeline for a single continuous target.

    Mirrors the classification ``main()`` structure but uses KFold CV,
    Pearson+Spearman correlations, and GradientBoostingRegressor feature
    importances instead of the classifier-based outputs. Called from
    ``main()`` when ``--regression-target`` is supplied.
    """
    target = args.regression_target
    logger.info(
        "Regression mode: target=%s, features=%s, output_dir=%s, top_k=%s",
        target, args.features, args.output_dir, args.regression_top_k,
    )

    if not (PROCESSED / args.features).exists():
        sys.exit(
            f"Missing data/processed/{args.features} — run scripts/run_features.py first"
        )

    X_full, y, _ = load_data_regression(args.features, target)
    n = len(y)
    if n < 4:
        sys.exit(
            f"Cannot run regression CV with only {n} samples — need at least 4."
        )
    if n < 10:
        cv_folds = max(2, n // 4)
        logger.warning(
            "Only %d samples — falling back to cv_folds=%d", n, cv_folds
        )
    else:
        cv_folds = 5
    logger.info("Using cv_folds=%d (n=%d)", cv_folds, n)

    safe_target = _sanitise_target_name(target)
    out_dir = PROCESSED / args.output_dir / f"regression_{safe_target}"
    plots_dir = out_dir / "plots"
    ensure_dir(out_dir)
    ensure_dir(plots_dir)

    # ── 1. Correlations (Pearson + Spearman) on the FULL feature matrix ──
    # Compute correlations against the unfiltered matrix so the diagnostic
    # CSV reflects every feature's relationship to the target — not just the
    # top-k that the models will see.
    logger.info("\n=== Feature correlations (Pearson + Spearman) ===")
    corr_df = feature_correlations_regression(X_full, y)
    corr_df.to_csv(out_dir / "feature_correlations.csv", index=False)
    logger.info(
        "\nTop 15 |r| with target:\n%s",
        corr_df.head(15)[["feature", "pearson_r", "spearman_r", "abs_max_r"]].to_string(index=False),
    )
    plot_regression_correlations_top20(
        corr_df, plots_dir / "correlations_top20.png"
    )

    # ── 1b. Filter features for the model fits ──
    # n=34 vs ~210 features = guaranteed overfit. Restrict to the top-k by
    # |Spearman r| against the target before fitting any model. This is a
    # standard small-n regression workflow.
    X, selected_features = _select_top_k_by_correlation(
        X_full, y, top_k=args.regression_top_k
    )
    logger.info(
        "Filtered features for modelling: %d → %d (top_k=%d)",
        X_full.shape[1], X.shape[1], args.regression_top_k,
    )

    # ── 2. Cross-validated regression models ──
    logger.info("\n=== Training regression models (KFold CV) ===")
    cv_df, oof_preds = run_regression_cv(X, y, cv_folds=cv_folds)
    cv_df.to_csv(out_dir / "cv_results.csv")
    logger.info("\n%s", cv_df.to_string())

    # Pick best model by R² mean.
    best_model = str(cv_df["r2_mean"].idxmax())
    best_r2 = float(cv_df.loc[best_model, "r2_mean"])
    logger.info("Best model: %s (R²=%.3f)", best_model, best_r2)

    # ── 3. Feature importance (refit GBR on full data) ──
    logger.info("\n=== Feature importance (GradientBoostingRegressor) ===")
    gbr = GradientBoostingRegressor(random_state=42)
    gbr.fit(X.values, y.values.astype(float))
    imp_df = pd.DataFrame({
        "feature": list(X.columns),
        "importance": gbr.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    imp_df.to_csv(out_dir / "feature_importance.csv", index=False)
    logger.info("\nTop 20 features:\n%s", imp_df.head(20).to_string(index=False))

    # ── 4. Scatter plot for best model ──
    plot_regression_scatter(
        y, oof_preds[best_model], best_model,
        plots_dir / f"regression_scatter_{best_model}.png",
    )

    # ── 5. Summary ──
    summary = {
        "n_samples": int(n),
        "target": target,
        "cv_folds": int(cv_folds),
        "best_model": best_model,
        "best_r2": best_r2,
        "models": {
            name: {
                "r2_mean": float(cv_df.loc[name, "r2_mean"]),
                "r2_std": float(cv_df.loc[name, "r2_std"]),
                "mae_mean": float(cv_df.loc[name, "mae_mean"]),
                "mae_std": float(cv_df.loc[name, "mae_std"]),
            }
            for name in cv_df.index
        },
        "top_5_correlations": corr_df.head(5)[
            ["feature", "pearson_r", "spearman_r", "abs_max_r"]
        ].to_dict("records"),
        "top_5_features_by_importance": imp_df.head(5).to_dict("records"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    logger.info("\n=== Regression summary ===")
    logger.info("Target: %s", target)
    logger.info("n_samples=%d, best_model=%s, best_r2=%.3f", n, best_model, best_r2)
    logger.info("All outputs in %s", out_dir)


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
    parser.add_argument(
        "--regression-target",
        type=str,
        default=None,
        help=(
            "If set, run in regression mode against the named column from "
            "data/processed/targets.csv (e.g. gold_diff_t600). Defaults to "
            "classification mode."
        ),
    )
    parser.add_argument(
        "--regression-top-k",
        type=int,
        default=15,
        help=(
            "In regression mode, restrict the model fits to the top-K "
            "features ranked by |Spearman r| against the target. With "
            "n=34 a value of 10–20 keeps the n≫d safety margin. Set to a "
            "very large number to disable filtering. Default: 15."
        ),
    )
    args = parser.parse_args()

    if args.regression_target is not None:
        run_regression_mode(args)
        return

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

    # ── 2b. Per-category ablation ──
    logger.info("\n=== Per-category ablation ===")
    cat_df = category_ablation(X, y, cv_folds=cv_folds)
    cat_df.to_csv(analysis_dir / "category_ablation.csv", index=False)
    plot_category_ablation(cat_df, plots_dir / "category_ablation.png")
    if not cat_df.empty:
        logger.info("\n%s", cat_df.to_string())
        # Rank categories by best AUC across models.
        best_per_cat = (
            cat_df.groupby("category")["roc_auc"].max().sort_values(ascending=False)
        )
        logger.info("\nCategory ranking (best AUC per category):\n%s",
                    best_per_cat.to_string())
    else:
        logger.warning("Per-category ablation produced an empty frame")

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
