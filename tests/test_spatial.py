"""Unit tests for spatial feature engineering."""

import numpy as np
import pandas as pd
import pytest

from lol_cv.features.spatial import (
    ZONES,
    ZONE_PRIORITY,
    OBJECTIVES,
    SpatialFeatures,
)


@pytest.fixture
def spatial():
    return SpatialFeatures(grouping_threshold=0.15)


# ── Zone Classification ─────────────────────────────────────────────


class TestClassifyZone:
    def test_blue_base(self, spatial):
        assert spatial.classify_zone(0.1, 0.9) == "blue_base"

    def test_red_base(self, spatial):
        assert spatial.classify_zone(0.9, 0.1) == "red_base"

    def test_dragon_pit(self, spatial):
        """Dragon pit should win over overlapping river_bot / bot_jungle_blue."""
        assert spatial.classify_zone(0.62, 0.78) == "dragon_pit"

    def test_baron_pit(self, spatial):
        assert spatial.classify_zone(0.38, 0.22) == "baron_pit"

    def test_mid_lane(self, spatial):
        assert spatial.classify_zone(0.5, 0.5) == "mid_lane"

    def test_unknown_region(self, spatial):
        """A point outside all zone bounds returns 'unknown'."""
        # (0.42, 0.0) is in a gap between top_lane and top_jungle_red
        assert spatial.classify_zone(0.42, 0.0) == "unknown"

    def test_priority_covers_all_zones(self):
        """Every zone in ZONES must appear in ZONE_PRIORITY."""
        assert set(ZONE_PRIORITY) == set(ZONES.keys())

    def test_objective_pits_before_jungle(self, spatial):
        """Points in dragon/baron pit should not be classified as jungle."""
        # Dragon pit center
        zone = spatial.classify_zone(0.625, 0.775)
        assert zone == "dragon_pit"
        # Baron pit center
        zone = spatial.classify_zone(0.375, 0.225)
        assert zone == "baron_pit"


# ── Zone Transitions ────────────────────────────────────────────────


class TestZoneTransitions:
    def test_no_transitions_single_zone(self, spatial):
        """Champion staying in one zone has 0 transitions."""
        df = pd.DataFrame({
            "timestamp": [0, 1, 2],
            "champion": ["Jinx"] * 3,
            "x": [0.5, 0.5, 0.5],
            "y": [0.5, 0.5, 0.5],
            "confidence": [0.9] * 3,
        })
        transitions = spatial.zone_transitions(df, "Jinx")
        assert len(transitions) == 0

    def test_transitions_between_zones(self, spatial):
        """Champion moving from blue_base to mid should produce a transition."""
        df = pd.DataFrame({
            "timestamp": [0, 1, 2, 3],
            "champion": ["Jinx"] * 4,
            "x": [0.1, 0.1, 0.5, 0.5],
            "y": [0.9, 0.9, 0.5, 0.5],
            "confidence": [0.9] * 4,
        })
        transitions = spatial.zone_transitions(df, "Jinx")
        assert len(transitions) == 1
        assert transitions.iloc[0]["from_zone"] == "blue_base"
        assert transitions.iloc[0]["to_zone"] == "mid_lane"

    def test_transition_count(self, spatial):
        df = pd.DataFrame({
            "timestamp": [0, 1, 2],
            "champion": ["Jinx"] * 3,
            "x": [0.1, 0.5, 0.9],
            "y": [0.9, 0.5, 0.1],
            "confidence": [0.9] * 3,
        })
        assert spatial.transition_count(df, "Jinx") == 2

    def test_empty_dataframe(self, spatial):
        df = pd.DataFrame(columns=["timestamp", "champion", "x", "y", "confidence"])
        transitions = spatial.zone_transitions(df, "Jinx")
        assert len(transitions) == 0


# ── Team Grouping ───────────────────────────────────────────────────


class TestTeamGrouping:
    def test_grouped_together(self, spatial):
        """Champions at the same position should have distance ~0."""
        df = pd.DataFrame({
            "timestamp": [10.0, 10.0, 10.0],
            "champion": ["Jinx", "Thresh", "Ezreal"],
            "x": [0.5, 0.5, 0.5],
            "y": [0.5, 0.5, 0.5],
            "confidence": [0.9] * 3,
        })
        dist = spatial.team_grouping_distance(df, ["Jinx", "Thresh", "Ezreal"], 10.0)
        assert dist == pytest.approx(0.0)

    def test_spread_out(self, spatial):
        """Champions far apart should have a larger distance."""
        df = pd.DataFrame({
            "timestamp": [10.0, 10.0],
            "champion": ["Jinx", "Thresh"],
            "x": [0.0, 1.0],
            "y": [0.0, 1.0],
            "confidence": [0.9] * 2,
        })
        dist = spatial.team_grouping_distance(df, ["Jinx", "Thresh"], 10.0)
        assert dist == pytest.approx(np.sqrt(2), abs=0.01)

    def test_single_champion_returns_nan(self, spatial):
        df = pd.DataFrame({
            "timestamp": [10.0],
            "champion": ["Jinx"],
            "x": [0.5],
            "y": [0.5],
            "confidence": [0.9],
        })
        dist = spatial.team_grouping_distance(df, ["Jinx"], 10.0)
        assert np.isnan(dist)


# ── Zone Occupancy ──────────────────────────────────────────────────


class TestZoneOccupancy:
    def test_single_zone_100_pct(self, spatial):
        df = pd.DataFrame({
            "timestamp": [0, 1, 2],
            "champion": ["Jinx"] * 3,
            "x": [0.5, 0.5, 0.5],
            "y": [0.5, 0.5, 0.5],
            "confidence": [0.9] * 3,
        })
        occ = spatial.zone_occupancy(df, "Jinx")
        assert occ["mid_lane"] == pytest.approx(1.0)
        assert occ["top_lane"] == pytest.approx(0.0)

    def test_missing_champion(self, spatial):
        df = pd.DataFrame(columns=["timestamp", "champion", "x", "y", "confidence"])
        occ = spatial.zone_occupancy(df, "Jinx")
        assert all(v == 0.0 for v in occ.values())
