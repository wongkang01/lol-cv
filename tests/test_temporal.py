"""Unit tests for temporal feature engineering."""

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
