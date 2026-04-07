"""Unit tests for pure helpers in ``scripts/run_analysis.py``.

Currently covers:
    - ``split_feature_groups`` — partitioning by column-name prefix
    - ``feature_correlations`` — Spearman correlation ranking table
    - ``feature_correlations_regression`` — Pearson+Spearman vs continuous target
    - ``load_data_regression`` — reads features + targets and drops NaN targets
    - ``run_regression_cv`` — KFold CV over linear + GBR regressors
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
# The package lives under ``src/`` but isn't installed, so make it importable
# for both ``scripts.run_analysis`` (which imports ``lol_cv``) and the tests
# themselves.
sys.path.insert(0, str(_REPO_ROOT / "src"))

from scripts.run_analysis import (  # noqa: E402
    feature_correlations,
    feature_correlations_regression,
    load_data_regression,
    run_regression_cv,
    split_feature_groups,
)
import scripts.run_analysis as run_analysis  # noqa: E402


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def mixed_frame() -> pd.DataFrame:
    """Synthetic feature frame with sp_/tp_/ocr_ columns plus an unprefixed one."""
    rng = np.random.default_rng(0)
    n = 20
    return pd.DataFrame({
        "sp_zone_count": rng.integers(0, 5, n),
        "sp_team_spread": rng.random(n),
        "sp_convergence_dragon": rng.random(n),
        "tp_event_count_early": rng.integers(0, 10, n),
        "tp_gold_slope_mid": rng.random(n),
        "ocr_final_gold_diff": rng.integers(-5000, 5000, n),
        "ocr_game_duration": rng.integers(800, 2200, n),
        "unprefixed_meta": rng.random(n),  # must land in `combined` only
    })


# ── split_feature_groups ────────────────────────────────────────────


class TestSplitFeatureGroups:
    def test_returns_expected_keys(self, mixed_frame):
        groups = split_feature_groups(mixed_frame)
        assert set(groups.keys()) == {
            "spatial",
            "temporal",
            "ocr",
            "spatial_temporal",
            "combined",
        }

    def test_spatial_columns(self, mixed_frame):
        groups = split_feature_groups(mixed_frame)
        assert list(groups["spatial"].columns) == [
            "sp_zone_count",
            "sp_team_spread",
            "sp_convergence_dragon",
        ]
        assert groups["spatial"].shape[1] == 3

    def test_temporal_columns(self, mixed_frame):
        groups = split_feature_groups(mixed_frame)
        assert list(groups["temporal"].columns) == [
            "tp_event_count_early",
            "tp_gold_slope_mid",
        ]
        assert groups["temporal"].shape[1] == 2

    def test_ocr_columns(self, mixed_frame):
        groups = split_feature_groups(mixed_frame)
        assert list(groups["ocr"].columns) == [
            "ocr_final_gold_diff",
            "ocr_game_duration",
        ]
        assert groups["ocr"].shape[1] == 2

    def test_spatial_temporal_concat(self, mixed_frame):
        groups = split_feature_groups(mixed_frame)
        assert groups["spatial_temporal"].shape[1] == 5
        assert set(groups["spatial_temporal"].columns) == {
            "sp_zone_count",
            "sp_team_spread",
            "sp_convergence_dragon",
            "tp_event_count_early",
            "tp_gold_slope_mid",
        }

    def test_combined_is_full_frame(self, mixed_frame):
        groups = split_feature_groups(mixed_frame)
        assert groups["combined"].shape == mixed_frame.shape
        assert list(groups["combined"].columns) == list(mixed_frame.columns)

    def test_unprefixed_column_only_in_combined(self, mixed_frame):
        """A column with no recognised prefix must not leak into sp/tp/ocr groups."""
        groups = split_feature_groups(mixed_frame)
        assert "unprefixed_meta" not in groups["spatial"].columns
        assert "unprefixed_meta" not in groups["temporal"].columns
        assert "unprefixed_meta" not in groups["ocr"].columns
        assert "unprefixed_meta" not in groups["spatial_temporal"].columns
        assert "unprefixed_meta" in groups["combined"].columns

    def test_empty_ocr_group(self):
        """Frame with no ocr_ columns produces an empty ocr group."""
        df = pd.DataFrame({
            "sp_a": [1, 2, 3],
            "tp_b": [4, 5, 6],
        })
        groups = split_feature_groups(df)
        assert groups["ocr"].shape[1] == 0
        assert groups["ocr"].empty
        assert groups["spatial"].shape[1] == 1
        assert groups["temporal"].shape[1] == 1
        assert groups["combined"].shape[1] == 2


# ── feature_correlations ────────────────────────────────────────────


class TestFeatureCorrelations:
    def test_output_schema(self):
        rng = np.random.default_rng(1)
        n = 30
        X = pd.DataFrame({
            "a": rng.random(n),
            "b": rng.random(n),
            "c": rng.random(n),
        })
        y = pd.Series(rng.integers(0, 2, n), name="blue_wins")
        corr = feature_correlations(X, y)

        assert list(corr.columns) == ["feature", "spearman_r", "p_value", "abs_r"]
        assert len(corr) == 3
        assert set(corr["feature"]) == {"a", "b", "c"}

    def test_sorted_by_abs_r_descending(self):
        n = 20
        y = pd.Series(np.arange(n) % 2, name="y")
        X = pd.DataFrame({
            # Perfectly anti-correlated with y (flip the bits)
            "strong": 1 - y.values,
            # Middling correlation: partially matches y
            "medium": np.concatenate([y.values[:n // 2], y.values[:n // 2][::-1]]),
            # Uncorrelated noise
            "noise": np.sin(np.arange(n) * 3.7),
        })
        corr = feature_correlations(X, y)

        # Output must be sorted descending on abs_r.
        assert corr["abs_r"].is_monotonic_decreasing

        # The perfectly (anti-)correlated column should top the ranking.
        assert corr.iloc[0]["feature"] == "strong"
        assert corr.iloc[0]["abs_r"] == pytest.approx(1.0)

    def test_constant_column_does_not_crash(self):
        """Zero-variance columns produce NaN spearman_r without raising."""
        n = 15
        y = pd.Series(np.arange(n) % 2, name="y")
        X = pd.DataFrame({
            "constant": np.ones(n) * 5.0,
            "varying": np.arange(n, dtype=float),
        })
        # scipy.stats.spearmanr warns on constant input; silence it for
        # this assertion block.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr = feature_correlations(X, y)

        assert len(corr) == 2
        constant_row = corr[corr["feature"] == "constant"].iloc[0]
        assert np.isnan(constant_row["spearman_r"])
        # abs_r is derived from spearman_r; should also be NaN (or at
        # least not raise) — the sort must still succeed.
        assert np.isnan(constant_row["abs_r"]) or constant_row["abs_r"] == 0

    def test_constant_and_anticorrelated_edge_case(self):
        """With one constant column and one perfectly anti-correlated column
        the perfectly correlated one must rank first.
        """
        n = 16
        y = pd.Series(np.arange(n) % 2, name="y")
        X = pd.DataFrame({
            "flat": np.full(n, 3.14),
            "anti": (1 - y.values).astype(float),
        })
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr = feature_correlations(X, y)

        # `anti` has |r|=1, `flat` has NaN (treated as 0 for sorting in
        # the helper). The anti-correlated column must be first.
        assert corr.iloc[0]["feature"] == "anti"
        assert corr.iloc[0]["abs_r"] == pytest.approx(1.0)


# ── Regression path ─────────────────────────────────────────────────


def _build_synthetic_regression_frame(n: int = 10, seed: int = 0):
    """Build a tiny (n, 3) features frame + noisy-linear target."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)
    y = 2.5 * x1 + 0.1 * rng.normal(size=n)  # strong positive dependence on x1
    X = pd.DataFrame({"feat_a": x1, "feat_b": x2, "feat_c": x3})
    y_series = pd.Series(y, name="synthetic_target")
    return X, y_series


class TestFeatureCorrelationsRegression:
    def test_sign_of_strong_feature(self):
        X, y = _build_synthetic_regression_frame(n=60, seed=1)
        corr = feature_correlations_regression(X, y)
        # Schema.
        assert list(corr.columns) == [
            "feature", "pearson_r", "pearson_p",
            "spearman_r", "spearman_p", "abs_max_r",
        ]
        assert len(corr) == 3
        # abs_max_r sorted descending.
        assert corr["abs_max_r"].is_monotonic_decreasing

        # The strongly-related feature should rank first and have a
        # positive pearson_r (target was built as +2.5 * feat_a + noise).
        top = corr.iloc[0]
        assert top["feature"] == "feat_a"
        assert top["pearson_r"] > 0.8
        assert top["spearman_r"] > 0.8

    def test_constant_column_returns_nan(self):
        n = 20
        X = pd.DataFrame({
            "flat": np.full(n, 7.0),
            "varying": np.arange(n, dtype=float),
        })
        y = pd.Series(np.arange(n, dtype=float), name="t")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr = feature_correlations_regression(X, y)

        flat_row = corr[corr["feature"] == "flat"].iloc[0]
        assert np.isnan(flat_row["pearson_r"])
        assert np.isnan(flat_row["spearman_r"])
        # Varying column perfectly linear in y -> pearson_r ~ 1.
        varying_row = corr[corr["feature"] == "varying"].iloc[0]
        assert varying_row["pearson_r"] == pytest.approx(1.0)


class TestLoadDataRegression:
    def test_drops_nan_target_rows(self, tmp_path, monkeypatch):
        # Point the script's PROCESSED constant at a temporary directory
        # with minimal features.csv / features_meta.csv / targets.csv files.
        monkeypatch.setattr(run_analysis, "PROCESSED", tmp_path)

        n = 10
        match_ids = [f"g_{i}" for i in range(n)]
        feats = pd.DataFrame({
            "match_id": match_ids,
            "sp_foo": np.arange(n, dtype=float),
            "tp_bar": np.arange(n, dtype=float) * 0.5,
            "ocr_baz": np.arange(n, dtype=float) + 2,
        })
        feats.to_csv(tmp_path / "features.csv", index=False)

        meta = pd.DataFrame({
            "match_id": match_ids,
            "winner_side": ["blue"] * n,  # ignored by regression path
        })
        meta.to_csv(tmp_path / "features_meta.csv", index=False)

        # 3 of 10 games have a NaN target and must be dropped.
        target_vals = np.arange(n, dtype=float)
        target_vals[[2, 5, 8]] = np.nan
        targets = pd.DataFrame({
            "match_id": match_ids,
            "gold_diff_t600": target_vals,
        })
        targets.to_csv(tmp_path / "targets.csv", index=False)

        X, y, _ = load_data_regression("features.csv", "gold_diff_t600")

        assert len(X) == 7
        assert len(y) == 7
        assert y.name == "gold_diff_t600"
        assert not y.isna().any()
        # Dropped rows should not appear in the index.
        dropped = {"g_2", "g_5", "g_8"}
        assert dropped.isdisjoint(set(X.index))

    def test_missing_targets_file_exits(self, tmp_path, monkeypatch):
        monkeypatch.setattr(run_analysis, "PROCESSED", tmp_path)
        feats = pd.DataFrame({
            "match_id": ["g_0"],
            "sp_foo": [1.0],
        })
        feats.to_csv(tmp_path / "features.csv", index=False)

        with pytest.raises(SystemExit, match="targets.csv"):
            load_data_regression("features.csv", "gold_diff_t600")

    def test_missing_target_column_exits(self, tmp_path, monkeypatch):
        monkeypatch.setattr(run_analysis, "PROCESSED", tmp_path)
        match_ids = ["g_0", "g_1"]
        pd.DataFrame({
            "match_id": match_ids,
            "sp_foo": [1.0, 2.0],
        }).to_csv(tmp_path / "features.csv", index=False)
        pd.DataFrame({
            "match_id": match_ids,
            "winner_side": ["blue", "red"],
        }).to_csv(tmp_path / "features_meta.csv", index=False)
        pd.DataFrame({
            "match_id": match_ids,
            "gold_diff_t600": [100.0, -100.0],
        }).to_csv(tmp_path / "targets.csv", index=False)

        with pytest.raises(SystemExit, match="not found in targets.csv"):
            load_data_regression("features.csv", "nonexistent_target")


class TestRunRegressionCv:
    def test_returns_non_empty_results(self):
        # Use a slightly larger frame so 2-fold CV is well-defined and
        # linear regression can actually fit something meaningful.
        X, y = _build_synthetic_regression_frame(n=40, seed=2)
        cv_df, oof = run_regression_cv(X, y, cv_folds=2, random_state=42)

        # Shape / schema.
        assert not cv_df.empty
        assert set(cv_df.columns) == {"r2_mean", "r2_std", "mae_mean", "mae_std", "n_samples"}
        # Phase 5 added `ridge_regression` to the model dict so the regression
        # pipeline can handle n≪d regimes. Keep the set assertion exhaustive so
        # future additions force a test update.
        assert set(cv_df.index) == {
            "linear_regression",
            "ridge_regression",
            "gradient_boosting_regressor",
        }

        # Both linear models should have a sensible R² (target is almost
        # linear in feat_a so linear/ridge in particular should be ~1).
        assert cv_df.loc["linear_regression", "r2_mean"] > 0.8
        assert cv_df.loc["ridge_regression", "r2_mean"] > 0.8

        # Out-of-fold predictions must exist for every model and have
        # the right length.
        assert set(oof.keys()) == {
            "linear_regression",
            "ridge_regression",
            "gradient_boosting_regressor",
        }
        for name, arr in oof.items():
            assert len(arr) == len(y), f"{name} oof length mismatch"


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
