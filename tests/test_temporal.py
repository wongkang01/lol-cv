"""Unit tests for temporal feature engineering."""

import numpy as np
import pandas as pd
import pytest

from lol_cv.features.temporal import TemporalFeatures


@pytest.fixture
def temporal():
    return TemporalFeatures(early_game_end=900, mid_game_end=1500)


# ── Game Phase Detection ────────────────────────────────────────────


class TestDetectGamePhase:
    def test_early_game(self, temporal):
        assert temporal.detect_game_phase(0) == "early"
        assert temporal.detect_game_phase(300) == "early"
        assert temporal.detect_game_phase(899) == "early"

    def test_mid_game(self, temporal):
        assert temporal.detect_game_phase(900) == "mid"
        assert temporal.detect_game_phase(1200) == "mid"
        assert temporal.detect_game_phase(1499) == "mid"

    def test_late_game(self, temporal):
        assert temporal.detect_game_phase(1500) == "late"
        assert temporal.detect_game_phase(2400) == "late"

    def test_custom_boundaries(self):
        t = TemporalFeatures(early_game_end=600, mid_game_end=1200)
        assert t.detect_game_phase(500) == "early"
        assert t.detect_game_phase(800) == "mid"
        assert t.detect_game_phase(1300) == "late"


# ── Event Sequence ──────────────────────────────────────────────────


class TestEventSequence:
    def test_filters_and_sorts(self, temporal):
        events = pd.DataFrame({
            "timestamp": [500, 200, 1000],
            "event_type": ["kill", "dragon", "unknown_event"],
            "team": ["blue", "red", "blue"],
        })
        seq = temporal.event_sequence(events)
        assert len(seq) == 2  # unknown_event filtered out
        assert seq.iloc[0]["event_type"] == "dragon"  # sorted by time
        assert seq.iloc[1]["event_type"] == "kill"

    def test_phase_annotation(self, temporal):
        events = pd.DataFrame({
            "timestamp": [100, 1000, 2000],
            "event_type": ["kill", "tower", "baron"],
            "team": ["blue", "red", "blue"],
        })
        seq = temporal.event_sequence(events)
        assert list(seq["game_phase"]) == ["early", "mid", "late"]


# ── Event Counts by Phase ───────────────────────────────────────────


class TestEventCountsByPhase:
    def test_counts_per_phase(self, temporal):
        events = pd.DataFrame({
            "timestamp": [100, 200, 500, 1000, 1600],
            "event_type": ["kill", "kill", "tower", "dragon", "baron"],
            "team": ["blue"] * 5,
        })
        counts = temporal.event_counts_by_phase(events)
        assert counts["early"]["kill"] == 2
        assert counts["early"]["tower"] == 1
        assert counts["mid"]["dragon"] == 1
        assert counts["late"]["baron"] == 1


# ── First Event Timing ──────────────────────────────────────────────


class TestFirstEventTiming:
    def test_first_timings(self, temporal):
        events = pd.DataFrame({
            "timestamp": [100, 300, 200],
            "event_type": ["kill", "kill", "dragon"],
            "team": ["blue", "red", "blue"],
        })
        timings = temporal.first_event_timing(events)
        assert timings["first_kill_time"] == 100.0
        assert timings["first_dragon_time"] == 200.0
        assert timings["first_baron_time"] is None


# ── Event Tempo ─────────────────────────────────────────────────────


class TestEventTempo:
    def test_tempo_windows(self, temporal):
        events = pd.DataFrame({
            "timestamp": [10, 50, 100, 310],
            "event_type": ["kill", "kill", "kill", "tower"],
            "team": ["blue"] * 4,
        })
        tempo = temporal.event_tempo(events, window=300)
        assert len(tempo) == 2
        assert tempo.iloc[0]["event_count"] == 3
        assert tempo.iloc[1]["event_count"] == 1

    def test_empty_events(self, temporal):
        events = pd.DataFrame(columns=["timestamp", "event_type", "team"])
        tempo = temporal.event_tempo(events)
        assert len(tempo) == 0


# ── Gold Diff Slope ────────────────────────────────────────────────


class TestGoldDiffSlope:
    """Tests for TemporalFeatures.gold_diff_slope."""

    def test_positive_slope(self, temporal):
        """Blue gaining gold over time -> positive slope."""
        ocr_df = pd.DataFrame({
            "timestamp": [100, 200, 300, 400, 500],
            "gold_diff": [0, 100, 200, 300, 400],
        })
        slope = temporal.gold_diff_slope(ocr_df)
        assert slope > 0

    def test_negative_slope(self, temporal):
        """Red gaining gold over time -> negative slope."""
        ocr_df = pd.DataFrame({
            "timestamp": [100, 200, 300, 400, 500],
            "gold_diff": [400, 300, 200, 100, 0],
        })
        slope = temporal.gold_diff_slope(ocr_df)
        assert slope < 0

    def test_flat_slope(self, temporal):
        """Constant gold diff -> slope near zero."""
        ocr_df = pd.DataFrame({
            "timestamp": [100, 200, 300, 400],
            "gold_diff": [500, 500, 500, 500],
        })
        slope = temporal.gold_diff_slope(ocr_df)
        assert slope == pytest.approx(0.0, abs=1e-6)

    def test_insufficient_data(self, temporal):
        """Fewer than 2 rows -> NaN."""
        ocr_df = pd.DataFrame({
            "timestamp": [100],
            "gold_diff": [500],
        })
        slope = temporal.gold_diff_slope(ocr_df)
        assert np.isnan(slope)

    def test_phase_filtering(self, temporal):
        """Only data in the specified phase should be used for the fit."""
        # Early phase is 0-900. Include data spanning early and mid.
        # Early data: positive trend. Mid data: negative trend.
        ocr_df = pd.DataFrame({
            "timestamp": [100, 200, 300, 400, 1000, 1100, 1200],
            "gold_diff": [0, 100, 200, 300, 300, 200, 100],
        })
        early_slope = temporal.gold_diff_slope(ocr_df, phase="early")
        assert early_slope > 0
        mid_slope = temporal.gold_diff_slope(ocr_df, phase="mid")
        assert mid_slope < 0

    def test_invalid_phase(self, temporal):
        """Unknown phase name raises ValueError."""
        ocr_df = pd.DataFrame({
            "timestamp": [100, 200],
            "gold_diff": [0, 100],
        })
        with pytest.raises(ValueError, match="Unknown phase"):
            temporal.gold_diff_slope(ocr_df, phase="super_late")


# ── Lane Assignment Stability ─────────────────────────────────────


class TestLaneAssignmentStability:
    """Tests for TemporalFeatures.lane_assignment_stability."""

    def test_always_in_zone(self, temporal):
        """Champion always in expected zone -> 1.0."""
        # mid_lane zone is (0.3, 0.3, 0.7, 0.7). Place champion squarely inside.
        df = pd.DataFrame({
            "timestamp": [0, 1, 2, 3, 4],
            "champion": ["Ahri"] * 5,
            "x": [0.5, 0.5, 0.5, 0.5, 0.5],
            "y": [0.5, 0.5, 0.5, 0.5, 0.5],
            "confidence": [0.9] * 5,
        })
        stability = temporal.lane_assignment_stability(df, "Ahri", "mid_lane")
        assert stability == pytest.approx(1.0)

    def test_never_in_zone(self, temporal):
        """Champion never in expected zone -> 0.0."""
        # Place champion in blue_base (0.0-0.2, 0.8-1.0), expect mid_lane.
        df = pd.DataFrame({
            "timestamp": [0, 1, 2],
            "champion": ["Ahri"] * 3,
            "x": [0.1, 0.1, 0.1],
            "y": [0.9, 0.9, 0.9],
            "confidence": [0.9] * 3,
        })
        stability = temporal.lane_assignment_stability(df, "Ahri", "mid_lane")
        assert stability == pytest.approx(0.0)

    def test_half_in_zone(self, temporal):
        """Champion in zone half the time -> 0.5."""
        # 2 timestamps in mid_lane (0.5, 0.5), 2 timestamps in blue_base (0.1, 0.9).
        df = pd.DataFrame({
            "timestamp": [0, 1, 2, 3],
            "champion": ["Ahri"] * 4,
            "x": [0.5, 0.5, 0.1, 0.1],
            "y": [0.5, 0.5, 0.9, 0.9],
            "confidence": [0.9] * 4,
        })
        stability = temporal.lane_assignment_stability(df, "Ahri", "mid_lane")
        assert stability == pytest.approx(0.5)

    def test_missing_champion(self, temporal):
        """Champion not found in data -> NaN."""
        df = pd.DataFrame({
            "timestamp": [0, 1],
            "champion": ["Jinx", "Jinx"],
            "x": [0.5, 0.5],
            "y": [0.5, 0.5],
            "confidence": [0.9, 0.9],
        })
        stability = temporal.lane_assignment_stability(df, "Ahri", "mid_lane")
        assert np.isnan(stability)
