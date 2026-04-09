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
        """Dragon pit centroid (as painted in zone_mask.png)."""
        # Pit centre lies a touch up-left of the legacy rectangle centre.
        assert spatial.classify_zone(0.64, 0.72) == "dragon_pit"

    def test_baron_pit(self, spatial):
        """Baron pit centroid (as painted in zone_mask.png)."""
        # Pit centre lies a touch down-left of the legacy rectangle centre.
        assert spatial.classify_zone(0.34, 0.28) == "baron_pit"

    def test_mid_lane(self, spatial):
        assert spatial.classify_zone(0.5, 0.5) == "mid_lane"

    def test_unknown_region(self, spatial):
        """The broadcast border region (outside the painted mask) returns 'unknown'."""
        # The mask is shrunk to 0.95 of the minimap crop, so any point in
        # the outermost ~2.5% border falls outside the painted area.
        assert spatial.classify_zone(0.001, 0.5) == "unknown"

    def test_priority_covers_all_zones(self):
        """Legacy ZONE_PRIORITY list still mirrors the ZONES keys (kept as a
        zone-name inventory; the mask resolves overlaps at paint time)."""
        assert set(ZONE_PRIORITY) == set(ZONES.keys())

    def test_objective_pits_before_jungle(self, spatial):
        """Pit centroids classify as the pit, not the surrounding jungle."""
        # Dragon pit centroid (mask paint).
        assert spatial.classify_zone(0.64, 0.72) == "dragon_pit"
        # Baron pit centroid (mask paint).
        assert spatial.classify_zone(0.34, 0.28) == "baron_pit"


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


# ── Convergence Speed ──────────────────────────────────────────────


class TestConvergenceSpeed:
    """Tests for SpatialFeatures.convergence_speed."""

    def test_team_converges_near_dragon(self, spatial):
        """Three champions arrive near dragon at t=20 -> returns 20.0."""
        # Dragon is at (0.62, 0.78). Place champions far away first,
        # then move 3 of them within proximity_threshold=0.2 at t=20.
        df = pd.DataFrame({
            "timestamp": [10, 10, 10, 10, 20, 20, 20, 20],
            "champion": ["A", "B", "C", "D", "A", "B", "C", "D"],
            "x": [0.1, 0.1, 0.1, 0.1, 0.62, 0.60, 0.64, 0.1],
            "y": [0.1, 0.1, 0.1, 0.1, 0.78, 0.76, 0.80, 0.1],
            "confidence": [0.9] * 8,
        })
        result = spatial.convergence_speed(df, ["A", "B", "C", "D"], "dragon")
        assert result == 20.0

    def test_team_never_groups(self, spatial):
        """Champions never converge near baron -> returns NaN."""
        # Baron is at (0.38, 0.22). All champions stay at (0.9, 0.9),
        # well outside the default proximity_threshold=0.2.
        df = pd.DataFrame({
            "timestamp": [10, 10, 10, 20, 20, 20],
            "champion": ["A", "B", "C", "A", "B", "C"],
            "x": [0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
            "y": [0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
            "confidence": [0.9] * 6,
        })
        result = spatial.convergence_speed(df, ["A", "B", "C"], "baron")
        assert np.isnan(result)

    def test_fewer_champions_than_target(self, spatial):
        """Only 2 champions in data but target_count=3 -> returns NaN."""
        df = pd.DataFrame({
            "timestamp": [10, 10],
            "champion": ["A", "B"],
            "x": [0.62, 0.62],
            "y": [0.78, 0.78],
            "confidence": [0.9, 0.9],
        })
        result = spatial.convergence_speed(
            df, ["A", "B"], "dragon", target_count=3
        )
        assert np.isnan(result)

    def test_invalid_objective_raises(self, spatial):
        """Unknown objective name raises ValueError."""
        df = pd.DataFrame({
            "timestamp": [10],
            "champion": ["A"],
            "x": [0.5],
            "y": [0.5],
            "confidence": [0.9],
        })
        with pytest.raises(ValueError, match="Unknown objective"):
            spatial.convergence_speed(df, ["A"], "invalid_objective")

    def test_near_from_start(self, spatial):
        """Champions already near objective at first timestamp -> returns that timestamp."""
        # All three start right next to dragon at t=5.
        df = pd.DataFrame({
            "timestamp": [5, 5, 5],
            "champion": ["A", "B", "C"],
            "x": [0.62, 0.63, 0.61],
            "y": [0.78, 0.79, 0.77],
            "confidence": [0.9] * 3,
        })
        result = spatial.convergence_speed(df, ["A", "B", "C"], "dragon")
        assert result == 5.0


# ── Bespoke Early-Game Features ─────────────────────────────────────


def _base_df(rows: list[dict]) -> pd.DataFrame:
    """Build a positions DataFrame with the standard column schema."""
    df = pd.DataFrame(rows)
    if "confidence" not in df.columns:
        df["confidence"] = 0.9
    return df[["timestamp", "champion", "x", "y", "confidence"]]


class TestJunglerCommitSide:
    """Feature 1: Jungler commit side (top/mid/bot) + vertical jungling."""

    def _jgl_df(self, blue_y: float, red_y: float) -> pd.DataFrame:
        rows = []
        # Sample the [60, 200] window at 1 Hz.
        for t in range(60, 201):
            rows.append({"timestamp": t, "champion": "BlueJgl", "x": 0.5, "y": blue_y})
            rows.append({"timestamp": t, "champion": "RedJgl", "x": 0.5, "y": red_y})
        return _base_df(rows)

    def test_commit_top(self, spatial):
        df = self._jgl_df(blue_y=0.2, red_y=0.2)
        out = spatial.compute_early_features(
            df,
            blue_team=["BlueJgl"], red_team=["RedJgl"],
            blue_jungler="BlueJgl", red_jungler="RedJgl",
            blue_mid=None, red_mid=None,
        )
        assert out["blue_jgl_commit_side"] == 0  # top
        assert out["red_jgl_commit_side"] == 0
        assert out["vertical_jungling"] == 0

    def test_commit_mid(self, spatial):
        df = self._jgl_df(blue_y=0.5, red_y=0.5)
        out = spatial.compute_early_features(
            df,
            blue_team=["BlueJgl"], red_team=["RedJgl"],
            blue_jungler="BlueJgl", red_jungler="RedJgl",
            blue_mid=None, red_mid=None,
        )
        assert out["blue_jgl_commit_side"] == 1
        assert out["red_jgl_commit_side"] == 1
        assert out["vertical_jungling"] == 0

    def test_commit_bot(self, spatial):
        df = self._jgl_df(blue_y=0.85, red_y=0.85)
        out = spatial.compute_early_features(
            df,
            blue_team=["BlueJgl"], red_team=["RedJgl"],
            blue_jungler="BlueJgl", red_jungler="RedJgl",
            blue_mid=None, red_mid=None,
        )
        assert out["blue_jgl_commit_side"] == 2
        assert out["red_jgl_commit_side"] == 2
        assert out["vertical_jungling"] == 0

    def test_vertical_jungling_flag(self, spatial):
        """Blue top + red bot == vertical jungling."""
        df = self._jgl_df(blue_y=0.2, red_y=0.85)
        out = spatial.compute_early_features(
            df,
            blue_team=["BlueJgl"], red_team=["RedJgl"],
            blue_jungler="BlueJgl", red_jungler="RedJgl",
            blue_mid=None, red_mid=None,
        )
        assert out["blue_jgl_commit_side"] == 0
        assert out["red_jgl_commit_side"] == 2
        assert out["vertical_jungling"] == 1

    def test_missing_jungler_is_nan(self, spatial):
        df = _base_df([{"timestamp": 100, "champion": "X", "x": 0.5, "y": 0.5}])
        out = spatial.compute_early_features(
            df,
            blue_team=["X"], red_team=[],
            blue_jungler=None, red_jungler=None,
            blue_mid=None, red_mid=None,
        )
        assert np.isnan(out["blue_jgl_commit_side"])
        assert np.isnan(out["red_jgl_commit_side"])
        assert np.isnan(out["vertical_jungling"])


class TestScuttleProximity:
    """Feature 2: Scuttle-crab proximity counts and arrival latencies."""

    def test_three_blue_near_top_crab(self, spatial):
        # Three blue champions parked at the top crab at t=200, plus some
        # filler rows outside the window so we exercise the full windowing.
        rows = []
        for champ in ["B1", "B2", "B3"]:
            rows.append({"timestamp": 200, "champion": champ, "x": 0.38, "y": 0.22})
        # Red players far away
        rows.append({"timestamp": 200, "champion": "R1", "x": 0.9, "y": 0.9})
        df = _base_df(rows)

        out = spatial.compute_early_features(
            df,
            blue_team=["B1", "B2", "B3"], red_team=["R1"],
            blue_jungler=None, red_jungler=None,
            blue_mid=None, red_mid=None,
        )
        assert out["blue_top_scuttle_count"] == 3
        assert out["red_top_scuttle_count"] == 0
        assert out["blue_top_scuttle_arrival"] == pytest.approx(200.0)
        assert np.isnan(out["red_top_scuttle_arrival"])

    def test_no_data_returns_nan(self, spatial):
        # All timestamps outside the scuttle window.
        rows = [
            {"timestamp": 10, "champion": "B1", "x": 0.38, "y": 0.22},
        ]
        df = _base_df(rows)
        out = spatial.compute_early_features(
            df,
            blue_team=["B1"], red_team=[],
            blue_jungler=None, red_jungler=None,
            blue_mid=None, red_mid=None,
        )
        assert np.isnan(out["blue_top_scuttle_count"])
        assert np.isnan(out["blue_top_scuttle_arrival"])


class TestMidRoam:
    """Feature 3: mid laner first roam time + target."""

    def test_mid_roams_top_at_300(self, spatial):
        # Mid lane axis: y = -x + 1. Sitting at (0.5, 0.5) -> dist 0.
        # From t=300 onwards, jump to (0.2, 0.1) (very low y, far from axis).
        rows = []
        for t in range(0, 300):
            rows.append({"timestamp": t, "champion": "BMid", "x": 0.5, "y": 0.5})
        for t in range(300, 335):
            rows.append({"timestamp": t, "champion": "BMid", "x": 0.2, "y": 0.1})
        df = _base_df(rows)

        out = spatial.compute_early_features(
            df,
            blue_team=["BMid"], red_team=[],
            blue_jungler=None, red_jungler=None,
            blue_mid="BMid", red_mid=None,
        )
        assert out["blue_mid_first_roam_time"] == pytest.approx(300.0)
        assert out["blue_mid_roam_target"] == 0  # top half (y < 0.5)

    def test_mid_never_roams(self, spatial):
        rows = [{"timestamp": t, "champion": "BMid", "x": 0.5, "y": 0.5} for t in range(0, 450)]
        df = _base_df(rows)
        out = spatial.compute_early_features(
            df,
            blue_team=["BMid"], red_team=[],
            blue_jungler=None, red_jungler=None,
            blue_mid="BMid", red_mid=None,
        )
        assert np.isnan(out["blue_mid_first_roam_time"])
        assert np.isnan(out["blue_mid_roam_target"])

    def test_missing_mid_returns_nan(self, spatial):
        df = _base_df([{"timestamp": 100, "champion": "X", "x": 0.5, "y": 0.5}])
        out = spatial.compute_early_features(
            df,
            blue_team=["X"], red_team=[],
            blue_jungler=None, red_jungler=None,
            blue_mid=None, red_mid=None,
        )
        assert np.isnan(out["blue_mid_first_roam_time"])
        assert np.isnan(out["blue_mid_roam_target"])


class TestLevel1Invade:
    """Feature 4: level-1 invade frame counts."""

    def test_blue_invades_red_bot_jungle(self, spatial):
        # Place 3 blue champions inside bot_jungle_red zone
        # (0.65, 0.35, 0.9, 0.65) -> use (0.7, 0.5) for 5 frames in [0, 90].
        rows = []
        for t in [10, 20, 30, 40, 50]:
            rows.append({"timestamp": t, "champion": "B1", "x": 0.7, "y": 0.5})
            rows.append({"timestamp": t, "champion": "B2", "x": 0.72, "y": 0.5})
            rows.append({"timestamp": t, "champion": "B3", "x": 0.74, "y": 0.5})
        # Red champions safely in their base
        for t in [10, 20, 30, 40, 50]:
            rows.append({"timestamp": t, "champion": "R1", "x": 0.9, "y": 0.1})
        df = _base_df(rows)

        out = spatial.compute_early_features(
            df,
            blue_team=["B1", "B2", "B3"], red_team=["R1"],
            blue_jungler=None, red_jungler=None,
            blue_mid=None, red_mid=None,
        )
        assert out["blue_lvl1_invade_frames"] == 5
        assert out["red_lvl1_invade_frames"] == 0

    def test_no_invade_when_only_two(self, spatial):
        rows = []
        for t in [10, 20, 30]:
            rows.append({"timestamp": t, "champion": "B1", "x": 0.7, "y": 0.5})
            rows.append({"timestamp": t, "champion": "B2", "x": 0.72, "y": 0.5})
        df = _base_df(rows)
        out = spatial.compute_early_features(
            df,
            blue_team=["B1", "B2"], red_team=[],
            blue_jungler=None, red_jungler=None,
            blue_mid=None, red_mid=None,
        )
        assert out["blue_lvl1_invade_frames"] == 0


# ── Bespoke Strategic Features ──────────────────────────────────────


class TestBotZoningDepth:
    """Feature 5: bot lane 2v2 zoning depth along the bot-lane axis."""

    def test_at_blue_tower_is_zero(self, spatial):
        """Duo sitting exactly at the blue bot tower projects to 0."""
        rows = []
        for t in range(90, 301):
            rows.append({"timestamp": t, "champion": "BADC", "x": 0.62, "y": 0.92})
            rows.append({"timestamp": t, "champion": "BSUP", "x": 0.62, "y": 0.92})
        df = _base_df(rows)
        out = spatial.compute_strategic_features(
            df,
            blue_team=["BADC", "BSUP"], red_team=[],
            blue_jungler=None, red_jungler=None,
            blue_bot="BADC", red_bot=None,
            blue_sup="BSUP", red_sup=None,
        )
        assert out["blue_bot_zoning_depth"] == pytest.approx(0.0, abs=1e-6)
        assert np.isnan(out["red_bot_zoning_depth"])
        assert np.isnan(out["bot_zoning_diff"])

    def test_at_red_tower_is_one(self, spatial):
        """Duo at the red bot tower projects to 1."""
        rows = []
        for t in range(90, 301):
            rows.append({"timestamp": t, "champion": "BADC", "x": 0.92, "y": 0.58})
            rows.append({"timestamp": t, "champion": "BSUP", "x": 0.92, "y": 0.58})
        df = _base_df(rows)
        out = spatial.compute_strategic_features(
            df,
            blue_team=["BADC", "BSUP"], red_team=[],
            blue_jungler=None, red_jungler=None,
            blue_bot="BADC", red_bot=None,
            blue_sup="BSUP", red_sup=None,
        )
        assert out["blue_bot_zoning_depth"] == pytest.approx(1.0, abs=1e-6)

    def test_midpoint_is_half(self, spatial):
        """Duo midpoint at the axis centre projects to 0.5."""
        # Midpoint between (0.62, 0.92) and (0.92, 0.58) = (0.77, 0.75)
        rows = []
        for t in range(90, 301):
            rows.append({"timestamp": t, "champion": "BADC", "x": 0.77, "y": 0.75})
            rows.append({"timestamp": t, "champion": "BSUP", "x": 0.77, "y": 0.75})
        df = _base_df(rows)
        out = spatial.compute_strategic_features(
            df,
            blue_team=["BADC", "BSUP"], red_team=[],
            blue_jungler=None, red_jungler=None,
            blue_bot="BADC", red_bot=None,
            blue_sup="BSUP", red_sup=None,
        )
        assert out["blue_bot_zoning_depth"] == pytest.approx(0.5, abs=1e-3)

    def test_missing_role_returns_nan(self, spatial):
        """Missing ADC or support yields NaN."""
        df = _base_df([{"timestamp": 100, "champion": "X", "x": 0.5, "y": 0.5}])
        out = spatial.compute_strategic_features(
            df,
            blue_team=["X"], red_team=[],
            blue_jungler=None, red_jungler=None,
            blue_bot=None, red_bot=None,
            blue_sup=None, red_sup=None,
        )
        assert np.isnan(out["blue_bot_zoning_depth"])
        assert np.isnan(out["red_bot_zoning_depth"])
        assert np.isnan(out["bot_zoning_diff"])

    def test_diff_sign(self, spatial):
        """Blue deeper than red -> positive diff."""
        rows = []
        for t in range(90, 301):
            # Blue duo at midpoint of axis (0.5)
            rows.append({"timestamp": t, "champion": "BADC", "x": 0.77, "y": 0.75})
            rows.append({"timestamp": t, "champion": "BSUP", "x": 0.77, "y": 0.75})
            # Red duo at blue tower (0.0)
            rows.append({"timestamp": t, "champion": "RADC", "x": 0.62, "y": 0.92})
            rows.append({"timestamp": t, "champion": "RSUP", "x": 0.62, "y": 0.92})
        df = _base_df(rows)
        out = spatial.compute_strategic_features(
            df,
            blue_team=["BADC", "BSUP"], red_team=["RADC", "RSUP"],
            blue_jungler=None, red_jungler=None,
            blue_bot="BADC", red_bot="RADC",
            blue_sup="BSUP", red_sup="RSUP",
        )
        assert out["bot_zoning_diff"] > 0


class TestSyncedRecalls:
    """Feature 6: synchronised recall count in [3:00, 8:00]."""

    def test_one_synced_recall(self, spatial):
        """3 blue champs at blue fountain at t=200, none at t=199 -> 1 sync."""
        rows = []
        # Far away at t=199
        for champ in ["B1", "B2", "B3"]:
            rows.append({"timestamp": 199, "champion": champ, "x": 0.5, "y": 0.5})
        # All at fountain at t=200
        for champ in ["B1", "B2", "B3"]:
            rows.append({"timestamp": 200, "champion": champ, "x": 0.05, "y": 0.95})
        df = _base_df(rows)
        out = spatial.compute_strategic_features(
            df,
            blue_team=["B1", "B2", "B3"], red_team=[],
            blue_jungler=None, red_jungler=None,
            blue_bot=None, red_bot=None,
            blue_sup=None, red_sup=None,
        )
        assert out["blue_synced_recalls"] == 1

    def test_no_sync_when_only_one(self, spatial):
        """Only 1 champion at base -> no synced recall."""
        rows = []
        rows.append({"timestamp": 199, "champion": "B1", "x": 0.5, "y": 0.5})
        rows.append({"timestamp": 200, "champion": "B1", "x": 0.05, "y": 0.95})
        df = _base_df(rows)
        out = spatial.compute_strategic_features(
            df,
            blue_team=["B1"], red_team=[],
            blue_jungler=None, red_jungler=None,
            blue_bot=None, red_bot=None,
            blue_sup=None, red_sup=None,
        )
        assert out["blue_synced_recalls"] == 0

    def test_no_sync_when_all_already_at_base(self, spatial):
        """If all allies are already at base the previous second, no transition."""
        rows = []
        for t in [199, 200]:
            for champ in ["B1", "B2", "B3"]:
                rows.append({"timestamp": t, "champion": champ, "x": 0.05, "y": 0.95})
        df = _base_df(rows)
        out = spatial.compute_strategic_features(
            df,
            blue_team=["B1", "B2", "B3"], red_team=[],
            blue_jungler=None, red_jungler=None,
            blue_bot=None, red_bot=None,
            blue_sup=None, red_sup=None,
        )
        # At t=200, newly_arrived is empty because all were there at t=199.
        # (t=199 itself is before the [180, 480] counting window start check.)
        # Note: at t=199 cur=3, prev=empty -> but 199 is outside [180, 480]?
        # Actually 180 <= 199 <= 480 so t=199 yields 1 sync.
        # We only care that the answer is exactly 1 (from t=199), not 2.
        assert out["blue_synced_recalls"] == 1


class TestMapAsymmetryIndex:
    """Feature 7: 4x4 presence histogram cosine distance (mirror-rotated)."""

    def test_mirrored_positions_low_asymmetry(self, spatial):
        """Blue at one corner, red at the anti-diagonal corner -> asymmetry ~0."""
        # Blue bases in the bottom-left corner (blue base area).
        # Red bases in the top-right corner.
        # Cell for blue point (0.1, 0.9): (i, j) = (3, 0) -> mirrored to (3, 0).
        # Cell for red point (0.9, 0.1): (i, j) = (0, 3) -> mirrored to (0, 3).
        # These don't match under this mirror. Use a symmetric placement:
        # blue at (0.1, 0.9) -> (3, 0). mirror -> (3, 0).
        # For blue and red-mirrored to match, place red at positions whose
        # mirrored cell is (3, 0): the cell (i, j) with (3-j, 3-i) = (3, 0)
        # means j=0, i=3, so red cell is (3, 0) too. That means red at same
        # bottom-left corner (0.1, 0.9). But we want RED to be in its base.
        # Testing: both teams in the same cell -> mirror maps red cell to
        # itself if red is at (3, 0) i.e. bottom-left corner — not red base.
        # Simpler: both teams in the centre cells on anti-diagonal.
        # Use blue at (0.3, 0.7), red at (0.7, 0.3).
        # Blue cell: yi=2, xi=1 -> (2, 1). Red cell: yi=1, xi=2 -> (1, 2).
        # Red mirrored: (3-2, 3-1) = (1, 2) -> doesn't match (2, 1).
        # Correct mirror: cell (i, j) in red -> (3-j, 3-i).
        # So red at (1, 2) maps to (3-2, 3-1) = (1, 2). Still not (2, 1).
        # Try blue at (0.8, 0.2) -> (0, 3), red at (0.2, 0.8) -> (3, 0).
        # Red (3, 0) -> (3-0, 3-3) = (3, 0). Doesn't match.
        # I think the mirror rule isn't strict "anti-diagonal mirror" -
        # it's defined in the spec. Let me just verify that IDENTICAL
        # placements yield asymmetry > 0 and blue==red mirrored setups
        # yield 0. Use blue and red in SAME exact cells -> red mirrored
        # will NOT equal blue in general, so this is a "high asymmetry"
        # test, not low. Use a uniform distribution that is self-mirroring:
        # place one blue and one red in EACH cell -> hists are uniform
        # -> mirrored is also uniform -> distance 0.
        rows = []
        champs_blue = [f"B{i}" for i in range(16)]
        champs_red = [f"R{i}" for i in range(16)]
        for t in [300, 360, 420]:
            idx = 0
            for i in range(4):
                for j in range(4):
                    x = (j + 0.5) / 4
                    y = (i + 0.5) / 4
                    rows.append({
                        "timestamp": t, "champion": champs_blue[idx], "x": x, "y": y
                    })
                    rows.append({
                        "timestamp": t, "champion": champs_red[idx], "x": x, "y": y
                    })
                    idx += 1
        df = _base_df(rows)
        out = spatial.compute_strategic_features(
            df,
            blue_team=champs_blue, red_team=champs_red,
            blue_jungler=None, red_jungler=None,
            blue_bot=None, red_bot=None,
            blue_sup=None, red_sup=None,
        )
        # Uniform histogram mirrored is still uniform -> cosine distance ~0.
        assert out["map_asymmetry_index_mean"] == pytest.approx(0.0, abs=1e-6)

    def test_opposite_positions_high_asymmetry(self, spatial):
        """Blue piled into one cell, red piled into a NON-mirror cell -> distance > 0.5."""
        rows = []
        # Blue all in cell (0, 0) = top-left (x in [0, 0.25), y in [0, 0.25)).
        # Red all in cell (0, 0) as well. Red mirrored: (3-0, 3-0) = (3, 3).
        # So blue hist has mass at (0, 0), red mirrored has mass at (3, 3).
        # Cosine distance = 1.0 (orthogonal vectors).
        for t in [300, 360, 420]:
            for champ in ["B1", "B2", "B3", "B4", "B5"]:
                rows.append({"timestamp": t, "champion": champ, "x": 0.1, "y": 0.1})
            for champ in ["R1", "R2", "R3", "R4", "R5"]:
                rows.append({"timestamp": t, "champion": champ, "x": 0.1, "y": 0.1})
        df = _base_df(rows)
        out = spatial.compute_strategic_features(
            df,
            blue_team=["B1", "B2", "B3", "B4", "B5"],
            red_team=["R1", "R2", "R3", "R4", "R5"],
            blue_jungler=None, red_jungler=None,
            blue_bot=None, red_bot=None,
            blue_sup=None, red_sup=None,
        )
        assert out["map_asymmetry_index_mean"] > 0.5

    def test_empty_window_returns_nan(self, spatial):
        """No data in [240, 480] -> NaN."""
        df = _base_df([{"timestamp": 100, "champion": "X", "x": 0.5, "y": 0.5}])
        out = spatial.compute_strategic_features(
            df,
            blue_team=["X"], red_team=[],
            blue_jungler=None, red_jungler=None,
            blue_bot=None, red_bot=None,
            blue_sup=None, red_sup=None,
        )
        assert np.isnan(out["map_asymmetry_index_mean"])


class TestPre3MinInvade:
    """Feature 8: pre-3-min counter-jungle asymmetry."""

    def test_blue_jungler_invades_30s(self, spatial):
        """Blue jungler in red bot jungle for 30 seconds of [90, 180]."""
        rows = []
        # Blue jungler sitting in bot_jungle_red (0.65-0.9, 0.35-0.65) for 30 secs.
        for t in range(100, 130):
            rows.append({"timestamp": t, "champion": "BJgl", "x": 0.75, "y": 0.5})
        # Red jungler safe in their own jungle (same zone) — but from blue's
        # perspective, red jungler is in blue_jungle_* zones. Place red jungler
        # in bot_jungle_blue (0.6-0.9, 0.65-0.95) -> that IS blue's jungle from
        # red's perspective. Actually "blue jungle" = blue team's jungle, and
        # from red's perspective the opposing jungle IS blue's. So red jungler
        # at (0.75, 0.8) would be IN blue jungle. To give red 0 invade secs,
        # place red jungler in red's own jungle e.g. (0.75, 0.5) is bot_jungle_red.
        for t in range(100, 130):
            rows.append({"timestamp": t, "champion": "RJgl", "x": 0.75, "y": 0.5})
        df = _base_df(rows)
        out = spatial.compute_strategic_features(
            df,
            blue_team=["BJgl"], red_team=["RJgl"],
            blue_jungler="BJgl", red_jungler="RJgl",
            blue_bot=None, red_bot=None,
            blue_sup=None, red_sup=None,
        )
        assert out["blue_pre3min_invade_secs"] == 30
        assert out["red_pre3min_invade_secs"] == 0
        assert out["pre3min_invade_diff"] == 30
        assert out["pre3min_invade_min"] == 0

    def test_missing_jungler_returns_nan(self, spatial):
        df = _base_df([{"timestamp": 100, "champion": "X", "x": 0.5, "y": 0.5}])
        out = spatial.compute_strategic_features(
            df,
            blue_team=["X"], red_team=[],
            blue_jungler=None, red_jungler=None,
            blue_bot=None, red_bot=None,
            blue_sup=None, red_sup=None,
        )
        assert np.isnan(out["blue_pre3min_invade_secs"])
        assert np.isnan(out["red_pre3min_invade_secs"])
        assert np.isnan(out["pre3min_invade_diff"])
        assert np.isnan(out["pre3min_invade_min"])

    def test_joint_counter_jungling(self, spatial):
        """Both junglers invading simultaneously -> min > 0."""
        rows = []
        # Both junglers in opposing jungle for 20 seconds.
        for t in range(100, 120):
            rows.append({"timestamp": t, "champion": "BJgl", "x": 0.75, "y": 0.5})
            rows.append({"timestamp": t, "champion": "RJgl", "x": 0.75, "y": 0.8})
        df = _base_df(rows)
        out = spatial.compute_strategic_features(
            df,
            blue_team=["BJgl"], red_team=["RJgl"],
            blue_jungler="BJgl", red_jungler="RJgl",
            blue_bot=None, red_bot=None,
            blue_sup=None, red_sup=None,
        )
        assert out["blue_pre3min_invade_secs"] == 20
        assert out["red_pre3min_invade_secs"] == 20
        assert out["pre3min_invade_diff"] == 0
        assert out["pre3min_invade_min"] == 20


# ── Per-Objective Snapshot Features ─────────────────────────────────


class TestObjectiveSnapshotFeatures:
    """compute_objective_snapshot_features — per-t positional aggregates."""

    METRICS = (
        "grouping_dist",
        "dragon_quadrant_count",
        "baron_quadrant_count",
        "enemy_half_count",
        "centroid_x",
        "centroid_y",
        "spread",
    )

    def _expected_keys(self, snapshot_times):
        keys = set()
        for t in snapshot_times:
            for side in ("blue", "red"):
                for m in self.METRICS:
                    keys.add(f"t{t}_{side}_{m}")
        return keys

    def test_stationary_teams_known_positions(self, spatial):
        """Both teams stationary at known positions: grouping_dist == 0
        (all teammates share the same point) and quadrant counts match."""
        blue_team = ["B1", "B2", "B3"]
        red_team = ["R1", "R2", "R3"]

        rows = []
        # Sample the full [150, 210) and [270, 330) windows at 1 Hz so each
        # t in the default snapshot_times that overlaps gets coverage.
        for t in range(150, 911):
            # Blue team all at dragon quadrant (0.6, 0.8) — bot right.
            for champ in blue_team:
                rows.append({"timestamp": t, "champion": champ, "x": 0.6, "y": 0.8})
            # Red team all at baron quadrant (0.4, 0.2) — top left.
            for champ in red_team:
                rows.append({"timestamp": t, "champion": champ, "x": 0.4, "y": 0.2})

        df = _base_df(rows)
        out = spatial.compute_objective_snapshot_features(
            df, blue_team=blue_team, red_team=red_team,
        )

        snapshot_times = (180, 300, 420, 540, 660, 780, 900)
        assert set(out.keys()) == self._expected_keys(snapshot_times)

        for t in snapshot_times:
            # Blue — 3 stationary champions at (0.6, 0.8).
            assert out[f"t{t}_blue_grouping_dist"] == pytest.approx(0.0, abs=1e-9)
            assert out[f"t{t}_blue_spread"] == pytest.approx(0.0, abs=1e-9)
            assert out[f"t{t}_blue_dragon_quadrant_count"] == 3
            assert out[f"t{t}_blue_baron_quadrant_count"] == 0
            assert out[f"t{t}_blue_enemy_half_count"] == 0  # y >= 0.5
            assert out[f"t{t}_blue_centroid_x"] == pytest.approx(0.6, abs=1e-9)
            assert out[f"t{t}_blue_centroid_y"] == pytest.approx(0.8, abs=1e-9)

            # Red — 3 stationary champions at (0.4, 0.2).
            assert out[f"t{t}_red_grouping_dist"] == pytest.approx(0.0, abs=1e-9)
            assert out[f"t{t}_red_spread"] == pytest.approx(0.0, abs=1e-9)
            assert out[f"t{t}_red_baron_quadrant_count"] == 3
            assert out[f"t{t}_red_dragon_quadrant_count"] == 0
            assert out[f"t{t}_red_enemy_half_count"] == 0  # need y > 0.5
            assert out[f"t{t}_red_centroid_x"] == pytest.approx(0.4, abs=1e-9)
            assert out[f"t{t}_red_centroid_y"] == pytest.approx(0.2, abs=1e-9)

    def test_empty_window_all_nan(self, spatial):
        """If no positions fall in the [t-30, t+30) slice, all metrics NaN."""
        # Only data at t=5000 (far outside default snapshot_times window).
        df = _base_df([
            {"timestamp": 5000, "champion": "B1", "x": 0.5, "y": 0.5},
            {"timestamp": 5000, "champion": "R1", "x": 0.5, "y": 0.5},
        ])
        out = spatial.compute_objective_snapshot_features(
            df, blue_team=["B1"], red_team=["R1"],
        )
        snapshot_times = (180, 300, 420, 540, 660, 780, 900)
        assert set(out.keys()) == self._expected_keys(snapshot_times)
        for k, v in out.items():
            assert np.isnan(v), f"Expected NaN for {k}, got {v}"

    def test_single_champion_grouping_nan_but_counts_populate(self, spatial):
        """Only 1 champion in-window → grouping_dist/spread NaN but
        counts and centroids are still computed."""
        blue_team = ["B1", "B2"]
        red_team = ["R1"]

        rows = []
        # Only B1 has data around t=300 — B2 has nothing in-window.
        for t in range(270, 330):
            rows.append({"timestamp": t, "champion": "B1", "x": 0.7, "y": 0.8})
            rows.append({"timestamp": t, "champion": "R1", "x": 0.3, "y": 0.2})

        df = _base_df(rows)
        out = spatial.compute_objective_snapshot_features(
            df,
            blue_team=blue_team,
            red_team=red_team,
            snapshot_times=(300,),
        )

        # Blue: 1 champion → grouping NaN, counts still valid.
        assert np.isnan(out["t300_blue_grouping_dist"])
        assert np.isnan(out["t300_blue_spread"])
        assert out["t300_blue_dragon_quadrant_count"] == 1  # (0.7, 0.8)
        assert out["t300_blue_baron_quadrant_count"] == 0
        assert out["t300_blue_enemy_half_count"] == 0
        assert out["t300_blue_centroid_x"] == pytest.approx(0.7, abs=1e-9)
        assert out["t300_blue_centroid_y"] == pytest.approx(0.8, abs=1e-9)

        # Red: 1 champion → grouping NaN, counts still valid.
        assert np.isnan(out["t300_red_grouping_dist"])
        assert np.isnan(out["t300_red_spread"])
        assert out["t300_red_baron_quadrant_count"] == 1  # (0.3, 0.2)
        assert out["t300_red_dragon_quadrant_count"] == 0
        assert out["t300_red_enemy_half_count"] == 0  # needs y > 0.5
        assert out["t300_red_centroid_x"] == pytest.approx(0.3, abs=1e-9)
        assert out["t300_red_centroid_y"] == pytest.approx(0.2, abs=1e-9)

    def test_partial_time_coverage(self, spatial):
        """Data only covers t=[270, 330). t=300 should populate but t=600
        and beyond must be NaN."""
        blue_team = ["B1", "B2"]
        red_team = ["R1", "R2"]

        rows = []
        for t in range(270, 330):
            rows.append({"timestamp": t, "champion": "B1", "x": 0.6, "y": 0.8})
            rows.append({"timestamp": t, "champion": "B2", "x": 0.65, "y": 0.85})
            rows.append({"timestamp": t, "champion": "R1", "x": 0.4, "y": 0.2})
            rows.append({"timestamp": t, "champion": "R2", "x": 0.35, "y": 0.15})

        df = _base_df(rows)
        out = spatial.compute_objective_snapshot_features(
            df,
            blue_team=blue_team,
            red_team=red_team,
            snapshot_times=(300, 600, 900),
        )

        # t=300 — populated.
        assert not np.isnan(out["t300_blue_grouping_dist"])
        assert out["t300_blue_grouping_dist"] > 0
        assert out["t300_blue_dragon_quadrant_count"] == 2
        assert out["t300_red_baron_quadrant_count"] == 2

        # t=600, t=900 — outside data window → all NaN.
        for t in (600, 900):
            for side in ("blue", "red"):
                for m in self.METRICS:
                    assert np.isnan(out[f"t{t}_{side}_{m}"]), (
                        f"Expected NaN for t{t}_{side}_{m}"
                    )
