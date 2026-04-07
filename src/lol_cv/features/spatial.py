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

    # ── Bespoke Strategic (non-jungle-proximity) Features ────────────

    def compute_strategic_features(
        self,
        df: pd.DataFrame,
        blue_team: list[str],
        red_team: list[str],
        blue_jungler: str | None,
        red_jungler: str | None,
        blue_bot: str | None,
        red_bot: str | None,
        blue_sup: str | None,
        red_sup: str | None,
    ) -> dict:
        """Compute non-jungle-proximity strategic features.

        These complement :meth:`compute_early_features` (which is focused on
        jungle proximity) by measuring distinct strategic dimensions:

            * Feature 5 — Bot lane 2v2 zoning depth: how far the ADC+support
              duo pushes along the bot-lane axis in [1:30, 5:00].
            * Feature 6 — Synchronised recall count: simultaneous base
              returns (2+ allies at fountain, with at least one arriving
              this second) in [3:00, 8:00]. Measures team tempo.
            * Feature 7 — Map presence asymmetry index: cosine distance
              between blue's 4×4 presence histogram and the mirrored red
              histogram, averaged over [4:00, 8:00]. Sign-agnostic — a
              single number per game.
            * Feature 8 — Pre-3-min counter-jungle: seconds each jungler
              spends in the opposing jungle in [1:30, 3:00], plus diff and
              min (joint counter-jungling).

        Gracefully returns NaN for any feature whose required input data
        (role assignments, time window) is missing.

        Returns:
            Flat dict of feature_name -> value. The caller is expected to
            prefix the keys with ``sp_strat_`` when writing into the
            feature matrix.
        """
        features: dict = {}

        # ── Feature 5: Bot lane 2v2 zoning depth ──────────────────────
        # Axis: blue bot tower (0.62, 0.92) -> red bot tower (0.92, 0.58).
        # Project midpoint(ADC, sup) onto the axis as scalar in [0, 1].
        bot_axis_start = np.array([0.62, 0.92])
        bot_axis_end = np.array([0.92, 0.58])
        axis_vec = bot_axis_end - bot_axis_start
        axis_len_sq = float(np.dot(axis_vec, axis_vec))

        def _bot_zoning_depth(adc: str | None, sup: str | None) -> float:
            if adc is None or sup is None or axis_len_sq == 0.0:
                return float("nan")
            win = df[
                (df["champion"].isin([adc, sup]))
                & (df["timestamp"] >= 90)
                & (df["timestamp"] <= 300)
            ]
            if win.empty:
                return float("nan")

            # For each timestamp with BOTH ADC and sup present, compute the
            # midpoint and project it onto the bot-lane axis.
            grouped = win.groupby("timestamp")
            projections: list[float] = []
            for _, snap in grouped:
                if snap["champion"].nunique() < 2:
                    continue
                adc_row = snap[snap["champion"] == adc]
                sup_row = snap[snap["champion"] == sup]
                if adc_row.empty or sup_row.empty:
                    continue
                mid = np.array([
                    (float(adc_row["x"].iloc[0]) + float(sup_row["x"].iloc[0])) / 2.0,
                    (float(adc_row["y"].iloc[0]) + float(sup_row["y"].iloc[0])) / 2.0,
                ])
                rel = mid - bot_axis_start
                t = float(np.dot(rel, axis_vec) / axis_len_sq)
                t = max(0.0, min(1.0, t))
                projections.append(t)

            if not projections:
                return float("nan")
            return float(np.mean(projections))

        blue_bot_depth = _bot_zoning_depth(blue_bot, blue_sup)
        red_bot_depth = _bot_zoning_depth(red_bot, red_sup)
        features["blue_bot_zoning_depth"] = blue_bot_depth
        features["red_bot_zoning_depth"] = red_bot_depth
        if np.isnan(blue_bot_depth) or np.isnan(red_bot_depth):
            features["bot_zoning_diff"] = float("nan")
        else:
            features["bot_zoning_diff"] = blue_bot_depth - red_bot_depth

        # ── Feature 6: Synchronised recall count ──────────────────────
        blue_fountain = (0.05, 0.95)
        red_fountain = (0.95, 0.05)
        fountain_radius = 0.10

        def _at_base_per_second(team: list[str], fountain: tuple[float, float]) -> dict[int, set]:
            """Return dict: timestamp -> set of champions currently at base."""
            fx, fy = fountain
            win = df[
                (df["champion"].isin(team))
                & (df["timestamp"] >= 179)  # need t=179 to test transitions at t=180
                & (df["timestamp"] <= 480)
            ]
            if win.empty:
                return {}
            dists = np.sqrt((win["x"] - fx) ** 2 + (win["y"] - fy) ** 2)
            at_base = win[dists.values < fountain_radius]
            out: dict[int, set] = {}
            for ts, grp in at_base.groupby("timestamp"):
                out[int(ts)] = set(grp["champion"].unique())
            return out

        def _synced_recalls(team: list[str], fountain: tuple[float, float]) -> float:
            if not team:
                return float("nan")
            win_check = df[
                (df["champion"].isin(team))
                & (df["timestamp"] >= 179)
                & (df["timestamp"] <= 480)
            ]
            if win_check.empty:
                return float("nan")
            at_base = _at_base_per_second(team, fountain)
            count = 0
            for t in range(180, 481):
                cur = at_base.get(t, set())
                prev = at_base.get(t - 1, set())
                newly_arrived = cur - prev
                if len(cur) >= 2 and len(newly_arrived) >= 1:
                    count += 1
            return float(count)

        features["blue_synced_recalls"] = _synced_recalls(blue_team, blue_fountain)
        features["red_synced_recalls"] = _synced_recalls(red_team, red_fountain)

        # ── Feature 7: Map presence asymmetry index ───────────────────
        # 4x4 grid histogram per side per timestamp, then cosine distance
        # between blue hist and the MIRROR-ROTATED red hist.
        def _hist_4x4(snap: pd.DataFrame) -> np.ndarray:
            h = np.zeros((4, 4), dtype=float)
            if snap.empty:
                return h.flatten()
            xi = np.clip((snap["x"].values * 4).astype(int), 0, 3)
            yi = np.clip((snap["y"].values * 4).astype(int), 0, 3)
            for i, j in zip(yi, xi):
                h[i, j] += 1.0
            return h.flatten()

        def _mirror_rotate(flat_hist: np.ndarray) -> np.ndarray:
            """Mirror across the anti-diagonal: cell (i, j) -> (3-j, 3-i)."""
            h = flat_hist.reshape(4, 4)
            rotated = np.zeros_like(h)
            for i in range(4):
                for j in range(4):
                    rotated[3 - j, 3 - i] = h[i, j]
            return rotated.flatten()

        def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
            na = float(np.linalg.norm(a))
            nb = float(np.linalg.norm(b))
            if na == 0.0 or nb == 0.0:
                return float("nan")
            cos_sim = float(np.dot(a, b) / (na * nb))
            return 1.0 - cos_sim

        asym_window = df[
            (df["timestamp"] >= 240) & (df["timestamp"] <= 480)
        ]
        if asym_window.empty:
            features["map_asymmetry_index_mean"] = float("nan")
        else:
            distances: list[float] = []
            for ts, snap in asym_window.groupby("timestamp"):
                blue_snap = snap[snap["champion"].isin(blue_team)]
                red_snap = snap[snap["champion"].isin(red_team)]
                if blue_snap.empty or red_snap.empty:
                    continue
                blue_hist = _hist_4x4(blue_snap)
                red_hist = _hist_4x4(red_snap)
                red_mirrored = _mirror_rotate(red_hist)
                d = _cosine_distance(blue_hist, red_mirrored)
                if not np.isnan(d):
                    distances.append(d)
            features["map_asymmetry_index_mean"] = (
                float(np.mean(distances)) if distances else float("nan")
            )

        # ── Feature 8: Pre-3-min counter-jungle asymmetry ─────────────
        # Jungler seconds inside the OPPOSING team's jungle in [90, 180].
        def _in_any_zone(x: float, y: float, zone_names: tuple[str, ...]) -> bool:
            for zn in zone_names:
                x_min, y_min, x_max, y_max = ZONES[zn]
                if x_min <= x <= x_max and y_min <= y <= y_max:
                    return True
            return False

        blue_jungle_zones = ("bot_jungle_blue", "top_jungle_blue")
        red_jungle_zones = ("bot_jungle_red", "top_jungle_red")

        def _pre3min_invade(jungler: str | None, enemy_zones: tuple[str, ...]) -> float:
            if jungler is None:
                return float("nan")
            sub = df[
                (df["champion"] == jungler)
                & (df["timestamp"] >= 90)
                & (df["timestamp"] <= 180)
            ]
            if sub.empty:
                return float("nan")
            # Count unique timestamps where the jungler is inside the enemy jungle.
            in_enemy = sub.apply(
                lambda r: _in_any_zone(float(r["x"]), float(r["y"]), enemy_zones),
                axis=1,
            )
            if in_enemy.empty:
                return 0.0
            invade_rows = sub[in_enemy]
            if invade_rows.empty:
                return 0.0
            return float(invade_rows["timestamp"].nunique())

        blue_invade_secs = _pre3min_invade(blue_jungler, red_jungle_zones)
        red_invade_secs = _pre3min_invade(red_jungler, blue_jungle_zones)
        features["blue_pre3min_invade_secs"] = blue_invade_secs
        features["red_pre3min_invade_secs"] = red_invade_secs
        if np.isnan(blue_invade_secs) or np.isnan(red_invade_secs):
            features["pre3min_invade_diff"] = float("nan")
            features["pre3min_invade_min"] = float("nan")
        else:
            features["pre3min_invade_diff"] = blue_invade_secs - red_invade_secs
            features["pre3min_invade_min"] = float(min(blue_invade_secs, red_invade_secs))

        logger.info("Computed %d strategic spatial features", len(features))
        return features

    # ── Per-Objective Grouping Snapshots ─────────────────────────────

    def compute_objective_snapshot_features(
        self,
        df: pd.DataFrame,
        blue_team: list[str],
        red_team: list[str],
        snapshot_times: tuple[int, ...] = (180, 300, 420, 540, 660, 780, 900),
        snapshot_window: int = 30,
    ) -> dict:
        """Compute per-objective grouping snapshots at fixed time points.

        For each target time ``t`` in ``snapshot_times``, slice the positions
        DataFrame to a window ``[t - snapshot_window, t + snapshot_window)``
        and aggregate each champion's mean (x, y) over that window (centroid
        of detections). Then, for each side, compute:

            * ``grouping_dist`` — mean pairwise distance between teammates.
            * ``dragon_quadrant_count`` — number of champions inside the
              bottom-right quadrant (x in [0.5, 1.0], y in [0.5, 1.0]).
            * ``baron_quadrant_count`` — number of champions inside the
              top-left quadrant (x in [0.0, 0.5], y in [0.0, 0.5]).
            * ``enemy_half_count`` — blue: y < 0.5, red: y > 0.5.
            * ``centroid_x``, ``centroid_y`` — mean team position.
            * ``spread`` — std of pairwise distances.

        Edge cases:
            * Empty slice for a time ``t`` → all metrics NaN for that ``t``.
            * Only 1 champion available → ``grouping_dist`` and ``spread``
              are NaN, counts/centroids still computed.
            * All keys are present in the output dict even when NaN, so the
              feature matrix stays consistent across games.

        Returns:
            Flat dict with keys ``t{t}_{side}_{metric}`` (e.g.
            ``t300_blue_grouping_dist``). The caller is expected to prefix
            keys with ``sp_snap_`` when writing into the feature matrix.
        """
        metrics = (
            "grouping_dist",
            "dragon_quadrant_count",
            "baron_quadrant_count",
            "enemy_half_count",
            "centroid_x",
            "centroid_y",
            "spread",
        )

        features: dict = {}

        def _nan_side(t: int, side: str) -> None:
            for m in metrics:
                features[f"t{t}_{side}_{m}"] = float("nan")

        for t in snapshot_times:
            lo = t - snapshot_window
            hi = t + snapshot_window
            window = df[(df["timestamp"] >= lo) & (df["timestamp"] < hi)]

            if window.empty:
                _nan_side(t, "blue")
                _nan_side(t, "red")
                continue

            for side, team in (("blue", blue_team), ("red", red_team)):
                side_win = window[window["champion"].isin(team)]
                if side_win.empty:
                    _nan_side(t, side)
                    continue

                # Mean position per champion over the window.
                centroids = side_win.groupby("champion")[["x", "y"]].mean()
                if centroids.empty:
                    _nan_side(t, side)
                    continue

                coords = centroids[["x", "y"]].values
                n = coords.shape[0]

                # Grouping distance + spread (need >= 2 champions).
                if n >= 2:
                    dists = pdist(coords, metric="euclidean")
                    features[f"t{t}_{side}_grouping_dist"] = float(np.mean(dists))
                    features[f"t{t}_{side}_spread"] = float(np.std(dists))
                else:
                    features[f"t{t}_{side}_grouping_dist"] = float("nan")
                    features[f"t{t}_{side}_spread"] = float("nan")

                xs = coords[:, 0]
                ys = coords[:, 1]

                # Dragon quadrant: bottom-right (x in [0.5, 1], y in [0.5, 1]).
                dragon_mask = (xs >= 0.5) & (xs <= 1.0) & (ys >= 0.5) & (ys <= 1.0)
                features[f"t{t}_{side}_dragon_quadrant_count"] = float(
                    int(dragon_mask.sum())
                )

                # Baron quadrant: top-left (x in [0, 0.5], y in [0, 0.5]).
                baron_mask = (xs >= 0.0) & (xs <= 0.5) & (ys >= 0.0) & (ys <= 0.5)
                features[f"t{t}_{side}_baron_quadrant_count"] = float(
                    int(baron_mask.sum())
                )

                # Enemy-half count: blue moving into top half (y < 0.5),
                # red moving into bottom half (y > 0.5).
                if side == "blue":
                    enemy_mask = ys < 0.5
                else:
                    enemy_mask = ys > 0.5
                features[f"t{t}_{side}_enemy_half_count"] = float(
                    int(enemy_mask.sum())
                )

                features[f"t{t}_{side}_centroid_x"] = float(np.mean(xs))
                features[f"t{t}_{side}_centroid_y"] = float(np.mean(ys))

        logger.info(
            "Computed %d objective snapshot spatial features", len(features)
        )
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
