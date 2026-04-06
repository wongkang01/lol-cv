"""Unit tests for pure helpers in ``scripts/run_analysis.py``.

Currently covers:
    - ``split_feature_groups`` — partitioning by column-name prefix
    - ``feature_correlations`` — Spearman correlation ranking table
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_analysis import feature_correlations, split_feature_groups  # noqa: E402


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


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
