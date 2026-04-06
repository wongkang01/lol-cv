"""Unit tests for the pipeline orchestrator's pure-logic methods."""

import numpy as np
import pandas as pd
import pytest

from lol_cv.pipeline import Pipeline


@pytest.fixture
def pipe():
    return Pipeline(config_path="configs/default.yaml")


# ── _ocr_to_events ────────────────────────────────────────────────────


class TestOcrToEvents:
    def test_empty_dataframe(self, pipe):
        df = pd.DataFrame()
        result = pipe._ocr_to_events(df)
        assert len(result) == 0
        assert list(result.columns) == ["timestamp", "event_type", "team"]

    def test_missing_game_time_seconds_column(self, pipe):
        df = pd.DataFrame({"blue_kills": [1, 2], "red_kills": [0, 1]})
        result = pipe._ocr_to_events(df)
        assert len(result) == 0
        assert list(result.columns) == ["timestamp", "event_type", "team"]

    def test_increasing_kills(self, pipe):
        df = pd.DataFrame({
            "game_time_seconds": [60, 120, 180],
            "blue_kills": [0, 2, 3],
            "red_kills": [0, 0, 1],
        })
        result = pipe._ocr_to_events(df)
        blue_events = result[result["team"] == "blue"]
        red_events = result[result["team"] == "red"]
        assert len(blue_events) == 3  # 0->2 = 2 kills, 2->3 = 1 kill
        assert len(red_events) == 1  # 0->1 = 1 kill
        assert all(blue_events["event_type"] == "kill")
        assert all(red_events["event_type"] == "kill")

    def test_kill_timestamps(self, pipe):
        df = pd.DataFrame({
            "game_time_seconds": [60, 120, 180],
            "blue_kills": [0, 2, 3],
            "red_kills": [0, 0, 0],
        })
        result = pipe._ocr_to_events(df)
        # 2 kills at t=120, 1 kill at t=180
        assert result.iloc[0]["timestamp"] == 120
        assert result.iloc[1]["timestamp"] == 120
        assert result.iloc[2]["timestamp"] == 180

    def test_no_kill_increases_after_initial(self, pipe):
        """When kills stay constant after the first observation, no new events
        are emitted beyond the initial jump from prev=0."""
        df = pd.DataFrame({
            "game_time_seconds": [60, 120, 180],
            "blue_kills": [0, 0, 0],
            "red_kills": [0, 0, 0],
        })
        result = pipe._ocr_to_events(df)
        assert len(result) == 0

    def test_nan_kills_handled(self, pipe):
        df = pd.DataFrame({
            "game_time_seconds": [60, 120, 180],
            "blue_kills": [0, np.nan, 2],
            "red_kills": [np.nan, 1, 1],
        })
        result = pipe._ocr_to_events(df)
        # NaN values should be skipped (treated as None, not > prev)
        # blue: 0 -> NaN (skip) -> 2 (2 kills)
        # red: NaN (skip) -> 1 (1 kill) -> 1 (no increase)
        blue_events = result[result["team"] == "blue"]
        red_events = result[result["team"] == "red"]
        assert len(blue_events) == 2
        assert len(red_events) == 1

    def test_unsorted_timestamps(self, pipe):
        df = pd.DataFrame({
            "game_time_seconds": [180, 60, 120],
            "blue_kills": [3, 0, 2],
            "red_kills": [0, 0, 0],
        })
        result = pipe._ocr_to_events(df)
        # Should sort by game_time_seconds first, so: 0->2->3
        assert len(result) == 3
        assert result.iloc[0]["timestamp"] == 120

    def test_nan_game_time_seconds_rows_dropped(self, pipe):
        df = pd.DataFrame({
            "game_time_seconds": [60, np.nan, 180],
            "blue_kills": [0, 5, 2],
            "red_kills": [0, 0, 0],
        })
        result = pipe._ocr_to_events(df)
        # Row with NaN game_time_seconds is dropped; 0->2 = 2 kills
        assert len(result) == 2


# ── _ocr_aggregate_features ───────────────────────────────────────────


class TestOcrAggregateFeatures:
    def test_empty_dataframe(self, pipe):
        df = pd.DataFrame()
        result = pipe._ocr_aggregate_features(df)
        assert result == {}

    def test_game_duration(self, pipe):
        df = pd.DataFrame({
            "game_time_seconds": [60, 120, 1800],
        })
        result = pipe._ocr_aggregate_features(df)
        assert result["game_duration_seconds"] == pytest.approx(1800.0)

    def test_final_kills(self, pipe):
        df = pd.DataFrame({
            "game_time_seconds": [60, 120, 180],
            "blue_kills": [0, 3, 7],
            "red_kills": [1, 2, 5],
        })
        result = pipe._ocr_aggregate_features(df)
        assert result["final_blue_kills"] == pytest.approx(7.0)
        assert result["final_red_kills"] == pytest.approx(5.0)

    def test_gold_diff_stats(self, pipe):
        df = pd.DataFrame({
            "game_time_seconds": [60, 120, 180],
            "blue_gold": [1000, 2000, 3000],
            "red_gold": [1000, 1500, 2500],
        })
        result = pipe._ocr_aggregate_features(df)
        # Gold diffs: 0, 500, 500
        assert result["final_gold_diff"] == pytest.approx(500.0)
        assert result["max_gold_lead_blue"] == pytest.approx(500.0)
        assert result["max_gold_lead_red"] == pytest.approx(0.0)
        assert result["mean_gold_diff"] == pytest.approx(1000 / 3)

    def test_missing_columns_partial(self, pipe):
        df = pd.DataFrame({
            "game_time_seconds": [60, 120],
        })
        result = pipe._ocr_aggregate_features(df)
        assert "game_duration_seconds" in result
        assert "final_blue_kills" not in result
        assert "final_gold_diff" not in result

    def test_single_row(self, pipe):
        df = pd.DataFrame({
            "game_time_seconds": [300],
            "blue_kills": [5],
            "red_kills": [3],
            "blue_gold": [5000],
            "red_gold": [4000],
        })
        result = pipe._ocr_aggregate_features(df)
        assert result["game_duration_seconds"] == pytest.approx(300.0)
        assert result["final_blue_kills"] == pytest.approx(5.0)
        assert result["final_red_kills"] == pytest.approx(3.0)
        assert result["final_gold_diff"] == pytest.approx(1000.0)

    def test_nan_values_in_gold(self, pipe):
        df = pd.DataFrame({
            "game_time_seconds": [60, 120, 180],
            "blue_gold": [1000, np.nan, 3000],
            "red_gold": [1000, np.nan, 2000],
        })
        result = pipe._ocr_aggregate_features(df)
        # Only rows where both blue_gold and red_gold are non-NaN: rows 0 and 2
        assert result["final_gold_diff"] == pytest.approx(1000.0)
        assert result["max_gold_lead_blue"] == pytest.approx(1000.0)


# ── identify_key_moments ──────────────────────────────────────────────


class TestIdentifyKeyMoments:
    def test_empty_dataframe(self, pipe):
        df = pd.DataFrame()
        result = pipe.identify_key_moments(df)
        assert result == []

    def test_missing_game_time_seconds(self, pipe):
        df = pd.DataFrame({"blue_kills": [1, 2]})
        result = pipe.identify_key_moments(df)
        assert result == []

    def test_gold_swing_detected(self, pipe):
        """Gold swing >1000g in 60s should be detected."""
        df = pd.DataFrame({
            "game_time_seconds": [100, 130, 160],
            "blue_gold": [5000, 5000, 6500],
            "red_gold": [5000, 5000, 5000],
            "blue_kills": [0, 0, 0],
            "red_kills": [0, 0, 0],
        })
        result = pipe.identify_key_moments(df)
        gold_swings = [m for m in result if m["type"] == "gold_swing"]
        assert len(gold_swings) >= 1
        assert gold_swings[0]["timestamp"] == pytest.approx(100.0)

    def test_gold_swing_not_detected_below_threshold(self, pipe):
        """Gold swing <1000g should NOT be detected."""
        df = pd.DataFrame({
            "game_time_seconds": [100, 130, 160],
            "blue_gold": [5000, 5200, 5500],
            "red_gold": [5000, 5000, 4800],
            "blue_kills": [0, 0, 0],
            "red_kills": [0, 0, 0],
        })
        result = pipe.identify_key_moments(df)
        gold_swings = [m for m in result if m["type"] == "gold_swing"]
        assert len(gold_swings) == 0

    def test_kill_burst_detected(self, pipe):
        """3+ kills in 30s should be detected."""
        df = pd.DataFrame({
            "game_time_seconds": [100, 110, 120, 125],
            "blue_kills": [0, 1, 2, 3],
            "red_kills": [0, 0, 0, 0],
            "blue_gold": [5000, 5000, 5000, 5000],
            "red_gold": [5000, 5000, 5000, 5000],
        })
        result = pipe.identify_key_moments(df)
        kill_bursts = [m for m in result if m["type"] == "kill_burst"]
        assert len(kill_bursts) >= 1

    def test_kill_burst_not_detected_below_threshold(self, pipe):
        """2 kills in 30s should NOT be detected."""
        df = pd.DataFrame({
            "game_time_seconds": [100, 110, 120, 125],
            "blue_kills": [0, 0, 1, 2],
            "red_kills": [0, 0, 0, 0],
            "blue_gold": [5000, 5000, 5000, 5000],
            "red_gold": [5000, 5000, 5000, 5000],
        })
        result = pipe.identify_key_moments(df)
        kill_bursts = [m for m in result if m["type"] == "kill_burst"]
        assert len(kill_bursts) == 0

    def test_moments_sorted_by_timestamp(self, pipe):
        """Returned moments should be sorted by timestamp."""
        df = pd.DataFrame({
            "game_time_seconds": list(range(0, 600, 10)),
            "blue_gold": [5000 + i * 100 for i in range(60)],
            "red_gold": [5000] * 60,
            "blue_kills": list(range(60)),
            "red_kills": [0] * 60,
        })
        result = pipe.identify_key_moments(df)
        timestamps = [m["timestamp"] for m in result]
        assert timestamps == sorted(timestamps)

    def test_gold_swing_outside_window_not_detected(self, pipe):
        """Gold swing >1000g but spread over >60s should not be detected."""
        df = pd.DataFrame({
            "game_time_seconds": [100, 200],
            "blue_gold": [5000, 6500],
            "red_gold": [5000, 5000],
            "blue_kills": [0, 0],
            "red_kills": [0, 0],
        })
        result = pipe.identify_key_moments(df)
        gold_swings = [m for m in result if m["type"] == "gold_swing"]
        # 200 - 100 = 100s > 60s window, so no detection
        assert len(gold_swings) == 0


# ── _deduplicate_moments ──────────────────────────────────────────────


class TestDeduplicateMoments:
    def test_empty_list(self):
        result = Pipeline._deduplicate_moments([])
        assert result == []

    def test_no_duplicates(self):
        moments = [
            {"timestamp": 100, "type": "gold_swing", "description": "a"},
            {"timestamp": 200, "type": "kill_burst", "description": "b"},
        ]
        result = Pipeline._deduplicate_moments(moments, merge_window=30)
        assert len(result) == 2

    def test_same_type_within_window_merged(self):
        moments = [
            {"timestamp": 100, "type": "gold_swing", "description": "a"},
            {"timestamp": 115, "type": "gold_swing", "description": "b"},
            {"timestamp": 125, "type": "gold_swing", "description": "c"},
        ]
        result = Pipeline._deduplicate_moments(moments, merge_window=30)
        assert len(result) == 1
        assert result[0]["timestamp"] == 100

    def test_different_types_at_same_time_kept(self):
        moments = [
            {"timestamp": 100, "type": "gold_swing", "description": "a"},
            {"timestamp": 100, "type": "kill_burst", "description": "b"},
        ]
        result = Pipeline._deduplicate_moments(moments, merge_window=30)
        assert len(result) == 2

    def test_same_type_outside_window_kept(self):
        moments = [
            {"timestamp": 100, "type": "gold_swing", "description": "a"},
            {"timestamp": 200, "type": "gold_swing", "description": "b"},
        ]
        result = Pipeline._deduplicate_moments(moments, merge_window=30)
        assert len(result) == 2

    def test_merge_window_boundary(self):
        """Moments exactly at the merge window boundary should be merged."""
        moments = [
            {"timestamp": 100, "type": "gold_swing", "description": "a"},
            {"timestamp": 130, "type": "gold_swing", "description": "b"},
        ]
        # 130 - 100 = 30, which is NOT > 30, so should be merged
        result = Pipeline._deduplicate_moments(moments, merge_window=30)
        assert len(result) == 1

    def test_merge_window_just_outside(self):
        """Moments just outside the merge window should both be kept."""
        moments = [
            {"timestamp": 100, "type": "gold_swing", "description": "a"},
            {"timestamp": 131, "type": "gold_swing", "description": "b"},
        ]
        # 131 - 100 = 31 > 30, so both kept
        result = Pipeline._deduplicate_moments(moments, merge_window=30)
        assert len(result) == 2

    def test_mixed_types_with_merging(self):
        moments = [
            {"timestamp": 100, "type": "gold_swing", "description": "gs1"},
            {"timestamp": 105, "type": "kill_burst", "description": "kb1"},
            {"timestamp": 110, "type": "gold_swing", "description": "gs2"},
            {"timestamp": 200, "type": "gold_swing", "description": "gs3"},
            {"timestamp": 210, "type": "kill_burst", "description": "kb2"},
        ]
        result = Pipeline._deduplicate_moments(moments, merge_window=30)
        types = sorted([(m["type"], m["timestamp"]) for m in result])
        # gold_swing: 100 kept, 110 merged (within 30s), 200 kept => 2
        # kill_burst: 105 kept, 210 kept (>30s apart) => 2
        assert len(result) == 4


# ── build_feature_matrix ──────────────────────────────────────────────


class TestBuildFeatureMatrix:
    def test_normal_case(self, pipe):
        features = [
            {"a": 1, "b": 2, "c": 3},
            {"a": 4, "b": 5, "c": 6},
            {"a": 7, "b": 8, "c": 9},
        ]
        outcomes = [1, 0, 1]
        X, y = pipe.build_feature_matrix(features, outcomes)
        assert X.shape == (3, 3)
        assert len(y) == 3
        assert list(y) == [1, 0, 1]

    def test_nan_filling(self, pipe):
        features = [
            {"a": 1, "b": 2},
            {"a": 3, "c": 4},  # missing "b", has "c" instead
        ]
        outcomes = [1, 0]
        X, y = pipe.build_feature_matrix(features, outcomes)
        # Both dicts contribute columns a, b, c
        assert X.shape == (2, 3)
        # Missing values should be filled with 0
        assert X.loc[0, "c"] == pytest.approx(0.0)
        assert X.loc[1, "b"] == pytest.approx(0.0)

    def test_all_nan_columns_dropped(self, pipe):
        features = [
            {"a": 1, "b": np.nan},
            {"a": 2, "b": np.nan},
        ]
        outcomes = [1, 0]
        X, y = pipe.build_feature_matrix(features, outcomes)
        assert "b" not in X.columns
        assert X.shape == (2, 1)

    def test_outcomes_mapped(self, pipe):
        features = [{"x": 10}, {"x": 20}]
        outcomes = [0, 1]
        X, y = pipe.build_feature_matrix(features, outcomes)
        assert y.name == "outcome"
        assert list(y) == [0, 1]

    def test_single_match(self, pipe):
        features = [{"feat1": 42, "feat2": 3.14}]
        outcomes = [1]
        X, y = pipe.build_feature_matrix(features, outcomes)
        assert X.shape == (1, 2)
        assert y.iloc[0] == 1

    def test_partial_nan_filled_not_dropped(self, pipe):
        """Columns with some (but not all) NaN should be filled, not dropped."""
        features = [
            {"a": 1, "b": 2},
            {"a": 3, "b": np.nan},
            {"a": 5, "b": 6},
        ]
        outcomes = [1, 0, 1]
        X, y = pipe.build_feature_matrix(features, outcomes)
        assert "b" in X.columns
        assert X.loc[1, "b"] == pytest.approx(0.0)
