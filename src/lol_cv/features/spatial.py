"""
Spatial feature engineering from champion position data.

Converts raw (x, y, timestamp) tracking data into features focused on:
- Zone transitions and rotation patterns
- Team grouping near objectives (combined with objective timers)
- Positional heatmaps
- Objective approach timing and convergence

NOTE: Movement speed is intentionally excluded — minimap icons move in
discrete steps rather than smoothly, making per-second speed unreliable.
Zone transitions capture the same macro-movement information more robustly.

All coordinates are normalised to [0, 1] relative to minimap size
(as output by MinimapTracker). Zone boundaries use these normalised coords.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

from lol_cv.utils import setup_logger

logger = setup_logger("lol_cv.features.spatial")

# LoL minimap zone definitions in normalised [0, 1] coordinates.
# The minimap is oriented with blue base at bottom-left, red base at top-right.
# Each zone is defined as (x_min, y_min, x_max, y_max).
ZONES = {
    "top_lane": (0.0, 0.0, 0.2, 0.55),
    "mid_lane": (0.3, 0.3, 0.7, 0.7),
    "bot_lane": (0.45, 0.8, 1.0, 1.0),
    "top_jungle_blue": (0.1, 0.35, 0.35, 0.65),
    "top_jungle_red": (0.1, 0.05, 0.4, 0.35),
    "bot_jungle_blue": (0.6, 0.65, 0.9, 0.95),
    "bot_jungle_red": (0.65, 0.35, 0.9, 0.65),
    "dragon_pit": (0.55, 0.7, 0.7, 0.85),
    "baron_pit": (0.3, 0.15, 0.45, 0.3),
    "river_top": (0.2, 0.15, 0.45, 0.4),
    "river_bot": (0.55, 0.6, 0.8, 0.85),
    "blue_base": (0.0, 0.8, 0.2, 1.0),
    "red_base": (0.8, 0.0, 1.0, 0.2),
}

# Priority ordering for zone classification when zones overlap.
# Checked first-to-last: bases > objective pits > river > lanes > jungle.
ZONE_PRIORITY = [
    "blue_base", "red_base",
    "dragon_pit", "baron_pit",
    "river_top", "river_bot",
    "top_lane", "mid_lane", "bot_lane",
    "top_jungle_blue", "top_jungle_red", "bot_jungle_blue", "bot_jungle_red",
]

# Key objective locations (normalised coordinates)
OBJECTIVES = {
    "dragon": (0.62, 0.78),
    "baron": (0.38, 0.22),
    "herald": (0.38, 0.22),  # Same pit as baron
    "blue_top_tower_outer": (0.08, 0.42),
    "blue_mid_tower_outer": (0.35, 0.65),
    "blue_bot_tower_outer": (0.62, 0.92),
    "red_top_tower_outer": (0.38, 0.08),
    "red_mid_tower_outer": (0.65, 0.35),
    "red_bot_tower_outer": (0.92, 0.58),
}


class SpatialFeatures:
    """Compute spatial features from champion position trajectories.

    Expects input as a pandas DataFrame with columns:
    [timestamp, champion, x, y, confidence] — as produced by
    MinimapTracker.positions_to_dataframe().
    """

    def __init__(self, grouping_threshold: float = 0.15):
        """
        Args:
            grouping_threshold: Max normalised distance to consider champions
                                as "grouped" together (default 0.15 ≈ 200px on 512 minimap).
        """
        self.grouping_threshold = grouping_threshold

    # ── Zone Classification & Transitions ────────────────────────────

    def classify_zone(self, x: float, y: float) -> str:
        """Determine which map zone a position falls in.

        Zones are checked in priority order (bases > pits > river > lanes >
        jungle) so that overlapping regions resolve deterministically.

        Returns:
            Zone name, or 'unknown' if no zone matches.
        """
        for zone_name in ZONE_PRIORITY:
            x_min, y_min, x_max, y_max = ZONES[zone_name]
            if x_min <= x <= x_max and y_min <= y <= y_max:
                return zone_name
        return "unknown"

    def zone_occupancy(self, df: pd.DataFrame, champion: str) -> dict[str, float]:
        """Fraction of time a champion spends in each zone.

        Returns:
            Dict mapping zone names to occupancy fractions (0-1).
        """
        champ_df = df[df["champion"] == champion]
        if champ_df.empty:
            return {z: 0.0 for z in ZONES}

        zones = champ_df.apply(lambda r: self.classify_zone(r["x"], r["y"]), axis=1)
        counts = zones.value_counts(normalize=True)
        return {z: counts.get(z, 0.0) for z in list(ZONES.keys()) + ["unknown"]}

    def zone_transitions(self, df: pd.DataFrame, champion: str) -> pd.DataFrame:
        """Detect zone-to-zone transitions (rotations) for a champion.

        Returns:
            DataFrame with columns [timestamp, from_zone, to_zone].
        """
        champ_df = df[df["champion"] == champion].sort_values("timestamp").copy()
        if len(champ_df) < 2:
            return pd.DataFrame(columns=["timestamp", "from_zone", "to_zone"])

        champ_df["zone"] = champ_df.apply(
            lambda r: self.classify_zone(r["x"], r["y"]), axis=1
        )
        prev_zone = champ_df["zone"].iloc[0]
        transitions = []
        for _, row in champ_df.iloc[1:].iterrows():
            if row["zone"] != prev_zone:
                transitions.append({
                    "timestamp": row["timestamp"],
                    "from_zone": prev_zone,
                    "to_zone": row["zone"],
                })
                prev_zone = row["zone"]

        return pd.DataFrame(transitions)

    def transition_count(self, df: pd.DataFrame, champion: str) -> int:
        """Total number of zone transitions for a champion."""
        return len(self.zone_transitions(df, champion))

    def unique_zones_visited(self, df: pd.DataFrame, champion: str) -> int:
        """Number of unique zones a champion visited during the game."""
        champ_df = df[df["champion"] == champion]
        if champ_df.empty:
            return 0
        zones = champ_df.apply(lambda r: self.classify_zone(r["x"], r["y"]), axis=1)
        return zones.nunique()

    # ── Team Grouping ────────────────────────────────────────────────

    def team_grouping_distance(self, df: pd.DataFrame, champions: list[str], timestamp: float) -> float:
        """Mean pairwise distance between team members at a given timestamp.

        Lower values indicate tighter grouping (e.g. during teamfights
        or objective setups).

        Args:
            champions: List of champion names on the same team.
            timestamp: Game time in seconds.

        Returns:
            Mean pairwise euclidean distance (normalised coords), or NaN.
        """
        snap = df[(df["timestamp"] == timestamp) & (df["champion"].isin(champions))]
        if len(snap) < 2:
            return np.nan

        coords = snap[["x", "y"]].values
        distances = pdist(coords, metric="euclidean")
        return float(np.mean(distances))

    def grouping_over_time(
        self, df: pd.DataFrame, champions: list[str]
    ) -> pd.DataFrame:
        """Track team grouping distance over all timestamps.

        Returns:
            DataFrame with columns [timestamp, mean_distance, is_grouped].
        """
        timestamps = sorted(df["timestamp"].unique())
        rows = []
        for ts in timestamps:
            dist = self.team_grouping_distance(df, champions, ts)
            rows.append({
                "timestamp": ts,
                "mean_distance": dist,
                "is_grouped": dist < self.grouping_threshold if not np.isnan(dist) else False,
            })
        return pd.DataFrame(rows)

    def grouping_near_objective(
        self,
        df: pd.DataFrame,
        champions: list[str],
        objective: str,
        proximity_threshold: float = 0.2,
    ) -> pd.DataFrame:
        """Track how many team members are grouped near an objective over time.

        Combines team grouping with objective proximity — the key feature
        for correlating with objective timers and gold differences.

        Args:
            champions: Team champion list.
            objective: Objective name (e.g. 'dragon', 'baron').
            proximity_threshold: Max distance from objective to be "near" it.

        Returns:
            DataFrame with columns [timestamp, near_count, team_near_pct, grouped_near].
        """
        if objective not in OBJECTIVES:
            raise ValueError(f"Unknown objective: {objective}")

        ox, oy = OBJECTIVES[objective]
        timestamps = sorted(df["timestamp"].unique())
        rows = []

        for ts in timestamps:
            snap = df[(df["timestamp"] == ts) & (df["champion"].isin(champions))]
            if snap.empty:
                rows.append({"timestamp": ts, "near_count": 0, "team_near_pct": 0.0, "grouped_near": False})
                continue

            dists = np.sqrt((snap["x"] - ox) ** 2 + (snap["y"] - oy) ** 2)
            near_count = int((dists < proximity_threshold).sum())
            team_near_pct = near_count / len(champions)

            # "Grouped near" = 3+ members within proximity of objective
            grouped_near = near_count >= 3

            rows.append({
                "timestamp": ts,
                "near_count": near_count,
                "team_near_pct": team_near_pct,
                "grouped_near": grouped_near,
            })

        return pd.DataFrame(rows)

    def convergence_speed(
        self,
        df: pd.DataFrame,
        champions: list[str],
        objective: str,
        target_count: int = 3,
        proximity_threshold: float = 0.2,
    ) -> float:
        """Time (seconds) for target_count+ team members to arrive within proximity of objective.

        Measures how quickly a team converges on an objective — a key coordination metric.
        Returns NaN if team never reaches the target grouping count.

        Args:
            df: Position DataFrame with columns [timestamp, champion, x, y].
            champions: List of champion names on the same team.
            objective: Objective name (must be in OBJECTIVES dict).
            target_count: Minimum number of team members required near the objective.
            proximity_threshold: Max normalised distance from objective to be "near" it.

        Returns:
            First timestamp (seconds) at which target_count+ members are near the
            objective, or NaN if this never happens.
        """
        if objective not in OBJECTIVES:
            raise ValueError(f"Unknown objective: {objective}")

        ox, oy = OBJECTIVES[objective]
        timestamps = sorted(df["timestamp"].unique())

        for ts in timestamps:
            snap = df[(df["timestamp"] == ts) & (df["champion"].isin(champions))]
            if snap.empty:
                continue
            dists = np.sqrt((snap["x"] - ox) ** 2 + (snap["y"] - oy) ** 2)
            near_count = int((dists < proximity_threshold).sum())
            if near_count >= target_count:
                return float(ts)

        return float("nan")

    # ── Heatmaps ─────────────────────────────────────────────────────

    def generate_heatmap(
        self, df: pd.DataFrame, champion: str = None, resolution: int = 64
    ) -> np.ndarray:
        """Generate a 2D positional heatmap.

        Args:
            df: Position DataFrame.
            champion: Filter to a specific champion (None = all).
            resolution: Grid resolution (default 64x64).

        Returns:
            2D numpy array of shape (resolution, resolution) with visit counts.
        """
        subset = df[df["champion"] == champion] if champion else df
        if subset.empty:
            return np.zeros((resolution, resolution))

        x_bins = np.clip((subset["x"] * resolution).astype(int), 0, resolution - 1)
        y_bins = np.clip((subset["y"] * resolution).astype(int), 0, resolution - 1)

        heatmap = np.zeros((resolution, resolution))
        for xb, yb in zip(x_bins, y_bins):
            heatmap[yb, xb] += 1

        return heatmap

    # ── Objective Approach ───────────────────────────────────────────

    def distance_to_objective(self, x: float, y: float, objective: str) -> float:
        """Euclidean distance from a position to an objective."""
        if objective not in OBJECTIVES:
            raise ValueError(f"Unknown objective: {objective}. Choose from {list(OBJECTIVES.keys())}")
        ox, oy = OBJECTIVES[objective]
        return np.sqrt((x - ox) ** 2 + (y - oy) ** 2)

    def objective_approach_timing(
        self, df: pd.DataFrame, champions: list[str], objective: str
    ) -> pd.DataFrame:
        """Track team distance to an objective over time.

        Useful for detecting when a team starts converging toward
        dragon, baron, etc. — especially when combined with OCR-extracted
        objective timers.

        Returns:
            DataFrame with columns [timestamp, mean_distance, approaching].
        """
        timestamps = sorted(df["timestamp"].unique())
        rows = []
        prev_dist = None

        for ts in timestamps:
            snap = df[(df["timestamp"] == ts) & (df["champion"].isin(champions))]
            if snap.empty:
                continue
            distances = snap.apply(
                lambda r: self.distance_to_objective(r["x"], r["y"], objective), axis=1
            )
            mean_dist = distances.mean()
            approaching = mean_dist < prev_dist if prev_dist is not None else False
            rows.append({
                "timestamp": ts,
                "mean_distance": mean_dist,
                "approaching": approaching,
            })
            prev_dist = mean_dist

        return pd.DataFrame(rows)

    # ── Bespoke Early-Game Features ──────────────────────────────────

    def compute_early_features(
        self,
        df: pd.DataFrame,
        blue_team: list[str],
        red_team: list[str],
        blue_jungler: str | None,
        red_jungler: str | None,
        blue_mid: str | None,
        red_mid: str | None,
    ) -> dict:
        """Compute bespoke early-game (first ~5-10 min) rare-event features.

        Independent of :meth:`compute_all` — built to produce high-information
        signals in the early window where generic zone-occupancy features
        collapse to baseline.

        Features computed:
            * Jungler commit side (top/mid/bot) in the 1:00-3:20 window
              and a "vertical jungling" indicator.
            * First-scuttle proximity counts and earliest arrival latencies
              per side, per crab, around 3:15.
            * Mid-laner first roam time and the side they roamed to.
            * Level-1 invade frames per side in [0, 90].

        Gracefully returns NaN for any feature whose required input data
        (role assignment, time window) is missing.

        Args:
            df: Position DataFrame with columns [timestamp, champion, x, y].
            blue_team: Full list of blue-side champion names.
            red_team: Full list of red-side champion names.
            blue_jungler: Blue jungler champion name (or None).
            red_jungler: Red jungler champion name (or None).
            blue_mid: Blue mid-laner champion name (or None).
            red_mid: Red mid-laner champion name (or None).

        Returns:
            Flat dict of feature_name -> value (numeric or NaN). The caller
            is expected to prefix the keys with ``sp_early_`` when writing
            into the feature matrix.
        """
        features: dict = {}

        # ── Feature 1: Jungler commit side ─────────────────────────────
        def _jgl_commit_side(jungler: str | None) -> float:
            if jungler is None:
                return float("nan")
            sub = df[
                (df["champion"] == jungler)
                & (df["timestamp"] >= 60)
                & (df["timestamp"] <= 200)
            ]
            if sub.empty:
                return float("nan")
            mean_y = float(sub["y"].mean())
            if mean_y < 0.4:
                return 0.0  # top
            if mean_y <= 0.6:
                return 1.0  # mid
            return 2.0  # bot

        blue_side = _jgl_commit_side(blue_jungler)
        red_side = _jgl_commit_side(red_jungler)
        features["blue_jgl_commit_side"] = blue_side
        features["red_jgl_commit_side"] = red_side

        # Vertical jungling: opposite top/bot sides (0 vs 2 or 2 vs 0)
        if np.isnan(blue_side) or np.isnan(red_side):
            features["vertical_jungling"] = float("nan")
        else:
            features["vertical_jungling"] = float(
                {blue_side, red_side} == {0.0, 2.0}
            )

        # ── Feature 2: First scuttle proximity at 3:15 ─────────────────
        top_crab = (0.38, 0.22)
        bot_crab = (0.62, 0.78)
        scuttle_radius = 0.12

        def _scuttle_counts_max(team: list[str], crab: tuple[float, float]) -> float:
            win = df[
                (df["champion"].isin(team))
                & (df["timestamp"] >= 185)
                & (df["timestamp"] <= 215)
            ]
            if win.empty:
                return float("nan")
            cx, cy = crab
            dists = np.sqrt((win["x"] - cx) ** 2 + (win["y"] - cy) ** 2)
            win = win.assign(_d=dists.values)
            near = win[win["_d"] < scuttle_radius]
            if near.empty:
                return 0.0
            return float(near.groupby("timestamp").size().max())

        def _scuttle_arrival(team: list[str], crab: tuple[float, float]) -> float:
            win = df[
                (df["champion"].isin(team))
                & (df["timestamp"] >= 180)
                & (df["timestamp"] <= 240)
            ]
            if win.empty:
                return float("nan")
            cx, cy = crab
            dists = np.sqrt((win["x"] - cx) ** 2 + (win["y"] - cy) ** 2)
            near = win[dists.values < scuttle_radius]
            if near.empty:
                return float("nan")
            return float(near["timestamp"].min())

        features["blue_top_scuttle_count"] = _scuttle_counts_max(blue_team, top_crab)
        features["red_top_scuttle_count"] = _scuttle_counts_max(red_team, top_crab)
        features["blue_bot_scuttle_count"] = _scuttle_counts_max(blue_team, bot_crab)
        features["red_bot_scuttle_count"] = _scuttle_counts_max(red_team, bot_crab)

        features["blue_top_scuttle_arrival"] = _scuttle_arrival(blue_team, top_crab)
        features["red_top_scuttle_arrival"] = _scuttle_arrival(red_team, top_crab)
        features["blue_bot_scuttle_arrival"] = _scuttle_arrival(blue_team, bot_crab)
        features["red_bot_scuttle_arrival"] = _scuttle_arrival(red_team, bot_crab)

        # ── Feature 3: Mid laner first roam timing ─────────────────────
        # Mid lane axis: diagonal from (0.18, 0.82) -> (0.82, 0.18).
        # This is the line y = -x + 1, i.e. x + y - 1 = 0.
        # Perpendicular distance = |x + y - 1| / sqrt(2).
        def _mid_roam(mid: str | None) -> tuple[float, float]:
            if mid is None:
                return float("nan"), float("nan")
            sub = df[
                (df["champion"] == mid)
                & (df["timestamp"] < 450)
            ].sort_values("timestamp").reset_index(drop=True)
            if sub.empty:
                return float("nan"), float("nan")

            dists = np.abs(sub["x"].values + sub["y"].values - 1.0) / np.sqrt(2.0)
            timestamps = sub["timestamp"].values
            ys = sub["y"].values

            # Find first contiguous window of >= 15 consecutive seconds
            # where distance > 0.15. "Consecutive" means successive rows
            # have adjacent timestamps (gap <= 1s tolerance, since upstream
            # samples at 1 Hz).
            in_run = False
            run_start_idx = 0
            run_start_ts = 0.0
            last_ts = 0.0
            for i, (ts, d) in enumerate(zip(timestamps, dists)):
                if d > 0.15:
                    if not in_run:
                        in_run = True
                        run_start_idx = i
                        run_start_ts = float(ts)
                    elif ts - last_ts > 2.0:
                        # Gap in samples — reset the run
                        run_start_idx = i
                        run_start_ts = float(ts)
                    if ts - run_start_ts >= 15.0:
                        # Found it. roam_target based on y within run.
                        run_y = ys[run_start_idx : i + 1]
                        mean_y = float(np.mean(run_y))
                        target = 0.0 if mean_y < 0.5 else 1.0
                        return run_start_ts, target
                    last_ts = float(ts)
                else:
                    in_run = False

            return float("nan"), float("nan")

        blue_mid_time, blue_mid_target = _mid_roam(blue_mid)
        red_mid_time, red_mid_target = _mid_roam(red_mid)
        features["blue_mid_first_roam_time"] = blue_mid_time
        features["blue_mid_roam_target"] = blue_mid_target
        features["red_mid_first_roam_time"] = red_mid_time
        features["red_mid_roam_target"] = red_mid_target

        # ── Feature 4: Level-1 invade frames ───────────────────────────
        blue_jungle_zones = ("bot_jungle_blue", "top_jungle_blue")
        red_jungle_zones = ("bot_jungle_red", "top_jungle_red")

        def _in_any_zone(x: float, y: float, zone_names: tuple[str, ...]) -> bool:
            for zn in zone_names:
                x_min, y_min, x_max, y_max = ZONES[zn]
                if x_min <= x <= x_max and y_min <= y <= y_max:
                    return True
            return False

        lvl1 = df[(df["timestamp"] >= 0) & (df["timestamp"] <= 90)]
        if lvl1.empty:
            features["blue_lvl1_invade_frames"] = float("nan")
            features["red_lvl1_invade_frames"] = float("nan")
        else:
            blue_rows = lvl1[lvl1["champion"].isin(blue_team)]
            red_rows = lvl1[lvl1["champion"].isin(red_team)]

            def _invade_frame_count(team_rows: pd.DataFrame, enemy_zones: tuple[str, ...]) -> int:
                if team_rows.empty:
                    return 0
                in_enemy = team_rows.apply(
                    lambda r: _in_any_zone(r["x"], r["y"], enemy_zones), axis=1
                )
                if in_enemy.empty:
                    return 0
                counts = (
                    team_rows[in_enemy].groupby("timestamp").size()
                )
                return int((counts >= 3).sum())

            features["blue_lvl1_invade_frames"] = float(
                _invade_frame_count(blue_rows, red_jungle_zones)
            )
            features["red_lvl1_invade_frames"] = float(
                _invade_frame_count(red_rows, blue_jungle_zones)
            )

        logger.info("Computed %d early-game spatial features", len(features))
        return features

    # ── Aggregated Feature Vector ────────────────────────────────────

    def compute_all(
        self, df: pd.DataFrame, blue_team: list[str], red_team: list[str]
    ) -> dict:
        """Compute a full spatial feature vector for a game.

        Focuses on zone transitions, team grouping, and objective proximity —
        the features most likely to correlate with match outcome.

        Returns a flat dict suitable for use as a single row in an ML dataset.
        """
        features = {}

        for side, team in [("blue", blue_team), ("red", red_team)]:
            # Zone transitions (rotations)
            transitions = [self.transition_count(df, c) for c in team]
            features[f"{side}_total_transitions"] = int(np.sum(transitions))
            features[f"{side}_avg_transitions"] = float(np.mean(transitions))

            unique_zones = [self.unique_zones_visited(df, c) for c in team]
            features[f"{side}_avg_zones_visited"] = float(np.mean(unique_zones))

            # Team grouping
            grouping = self.grouping_over_time(df, team)
            features[f"{side}_avg_grouping_dist"] = grouping["mean_distance"].mean()
            features[f"{side}_grouped_pct"] = grouping["is_grouped"].mean()

            # Grouping near major objectives
            for obj in ["dragon", "baron"]:
                obj_grouping = self.grouping_near_objective(df, team, obj)
                features[f"{side}_{obj}_grouped_near_pct"] = obj_grouping["grouped_near"].mean()
                features[f"{side}_{obj}_avg_near_count"] = obj_grouping["near_count"].mean()

            # Convergence speed for major objectives
            for obj in ["dragon", "baron"]:
                features[f"{side}_{obj}_convergence_speed"] = self.convergence_speed(df, team, obj)

            # Zone occupancy (averaged across team)
            zone_totals = {z: 0.0 for z in ZONES}
            for champ in team:
                occ = self.zone_occupancy(df, champ)
                for z in ZONES:
                    zone_totals[z] += occ.get(z, 0.0)
            for z in ZONES:
                features[f"{side}_zone_{z}"] = zone_totals[z] / max(len(team), 1)

        logger.info("Computed %d spatial features", len(features))
        return features
