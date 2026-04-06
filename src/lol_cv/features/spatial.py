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
