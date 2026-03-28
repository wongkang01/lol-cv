"""
Spatial feature engineering from champion position data.

Converts raw (x, y, timestamp) tracking data into meaningful
features like movement speed, grouping distance, zone control,
heatmaps, and objective approach patterns.

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

    def __init__(self, grouping_threshold: float = 0.15, speed_window: int = 5):
        """
        Args:
            grouping_threshold: Max normalised distance to consider champions
                                as "grouped" together (default 0.15 ≈ 200px on 512 minimap).
            speed_window: Number of seconds for speed calculation sliding window.
        """
        self.grouping_threshold = grouping_threshold
        self.speed_window = speed_window

    # ── Per-Champion Features ────────────────────────────────────────

    def movement_speed(self, df: pd.DataFrame, champion: str) -> pd.DataFrame:
        """Calculate movement speed of a champion over time.

        Speed = euclidean distance between consecutive positions / time delta.

        Returns:
            DataFrame with columns [timestamp, speed].
        """
        champ_df = df[df["champion"] == champion].sort_values("timestamp").copy()
        if len(champ_df) < 2:
            return pd.DataFrame(columns=["timestamp", "speed"])

        dx = champ_df["x"].diff()
        dy = champ_df["y"].diff()
        dt = champ_df["timestamp"].diff()
        dist = np.sqrt(dx**2 + dy**2)
        speed = dist / dt.replace(0, np.nan)

        result = pd.DataFrame({
            "timestamp": champ_df["timestamp"],
            "speed": speed,
        }).dropna().reset_index(drop=True)
        return result

    def average_speed(self, df: pd.DataFrame, champion: str) -> float:
        """Average movement speed for a champion across the entire game."""
        speeds = self.movement_speed(df, champion)
        return speeds["speed"].mean() if not speeds.empty else 0.0

    def classify_zone(self, x: float, y: float) -> str:
        """Determine which map zone a position falls in.

        Returns:
            Zone name, or 'unknown' if no zone matches.
        """
        for zone_name, (x_min, y_min, x_max, y_max) in ZONES.items():
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

    # ── Team-Level Features ──────────────────────────────────────────

    def team_grouping_distance(self, df: pd.DataFrame, champions: list[str], timestamp: float) -> float:
        """Mean pairwise distance between team members at a given timestamp.

        Lower values indicate tighter grouping (e.g. during teamfights).

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

        # Bin x/y into grid cells
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
        dragon, baron, etc.

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

        Returns a flat dict suitable for use as a single row in an ML dataset.
        """
        features = {}

        for side, team in [("blue", blue_team), ("red", red_team)]:
            # Per-champion averages
            speeds = [self.average_speed(df, c) for c in team]
            features[f"{side}_avg_speed"] = np.mean(speeds)

            # Team grouping
            grouping = self.grouping_over_time(df, team)
            features[f"{side}_avg_grouping_dist"] = grouping["mean_distance"].mean()
            features[f"{side}_grouped_pct"] = grouping["is_grouped"].mean()

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
