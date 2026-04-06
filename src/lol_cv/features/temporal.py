"""
Temporal feature engineering from game event sequences.

Extracts timing-based patterns, game phase detection, event sequences,
and rotation frequency. Designed for both CV-extracted events and
Riot API timeline data.

Game phases (from config):
    - Early game: 0-15 min (laning, first objectives)
    - Mid game: 15-25 min (rotations, teamfight setup)
    - Late game: 25+ min (teamfights, baron, closing)
"""

import pandas as pd
import numpy as np

from lol_cv.features.spatial import ZONES, SpatialFeatures
from lol_cv.utils import setup_logger

logger = setup_logger("lol_cv.features.temporal")

# Default phase boundaries in seconds
PHASE_BOUNDARIES = {
    "early": (0, 900),       # 0-15 min
    "mid": (900, 1500),      # 15-25 min
    "late": (1500, float("inf")),
}

# Canonical event types for sequence mining
EVENT_TYPES = ["kill", "tower", "dragon", "baron", "inhibitor", "herald"]


class TemporalFeatures:
    """Compute temporal features from game event and position data."""

    def __init__(
        self,
        early_game_end: int = 900,
        mid_game_end: int = 1500,
        event_types: list[str] = None,
    ):
        """
        Args:
            early_game_end: Timestamp (seconds) marking end of early game.
            mid_game_end: Timestamp (seconds) marking end of mid game.
            event_types: List of event types to include in sequence analysis.
        """
        self.phases = {
            "early": (0, early_game_end),
            "mid": (early_game_end, mid_game_end),
            "late": (mid_game_end, float("inf")),
        }
        self.event_types = event_types or EVENT_TYPES
        self._spatial = SpatialFeatures()

    def detect_game_phase(self, timestamp: float) -> str:
        """Classify a game timestamp into early/mid/late phase.

        Args:
            timestamp: Game time in seconds.

        Returns:
            Phase name: 'early', 'mid', or 'late'.
        """
        for phase, (start, end) in self.phases.items():
            if start <= timestamp < end:
                return phase
        return "late"

    # ── Event Sequence Analysis ──────────────────────────────────────

    def event_sequence(self, events: pd.DataFrame) -> pd.DataFrame:
        """Extract an ordered sequence of key events for process mining.

        Args:
            events: DataFrame with columns [timestamp, event_type, team, ...].
                    Can come from Riot API timeline or CV-detected events.

        Returns:
            Filtered and sorted DataFrame with game_phase added.
        """
        filtered = events[events["event_type"].isin(self.event_types)].copy()
        filtered = filtered.sort_values("timestamp").reset_index(drop=True)
        filtered["game_phase"] = filtered["timestamp"].apply(self.detect_game_phase)
        filtered["event_order"] = range(len(filtered))
        return filtered

    def event_counts_by_phase(self, events: pd.DataFrame) -> dict:
        """Count events per type per game phase.

        Returns:
            Nested dict: {phase: {event_type: count}}.
        """
        seq = self.event_sequence(events)
        result = {}
        for phase in self.phases:
            phase_events = seq[seq["game_phase"] == phase]
            result[phase] = phase_events["event_type"].value_counts().to_dict()
        return result

    def first_event_timing(self, events: pd.DataFrame) -> dict:
        """Get timestamp of the first occurrence of each event type.

        Useful features: first_blood_time, first_tower_time, first_dragon_time.

        Returns:
            Dict mapping event type to first timestamp (or None).
        """
        seq = self.event_sequence(events)
        result = {}
        for et in self.event_types:
            matches = seq[seq["event_type"] == et]
            result[f"first_{et}_time"] = float(matches["timestamp"].iloc[0]) if not matches.empty else None
        return result

    def event_tempo(self, events: pd.DataFrame, window: int = 300) -> pd.DataFrame:
        """Compute rolling event frequency (events per 5-minute window).

        Higher tempo indicates more action-packed phases.

        Args:
            events: Event DataFrame.
            window: Window size in seconds (default 300 = 5 min).

        Returns:
            DataFrame with columns [window_start, event_count, tempo].
        """
        seq = self.event_sequence(events)
        if seq.empty:
            return pd.DataFrame(columns=["window_start", "event_count", "tempo"])

        max_time = seq["timestamp"].max()
        rows = []
        for start in range(0, int(max_time), window):
            end = start + window
            count = len(seq[(seq["timestamp"] >= start) & (seq["timestamp"] < end)])
            rows.append({
                "window_start": start,
                "event_count": count,
                "tempo": count / (window / 60),  # events per minute
            })
        return pd.DataFrame(rows)

    # ── Rotation / Zone Transition Analysis ──────────────────────────

    def rotation_frequency(
        self, positions: pd.DataFrame, champion: str, window_seconds: int = 60
    ) -> pd.DataFrame:
        """Count zone transitions per time window for a champion.

        A rotation is defined as moving from one named zone to another.
        High rotation frequency indicates roaming / macro play.

        Args:
            positions: Position DataFrame from MinimapTracker.
            champion: Champion name to analyze.
            window_seconds: Rolling window size.

        Returns:
            DataFrame with columns [window_start, transitions, zones_visited].
        """
        champ_df = positions[positions["champion"] == champion].sort_values("timestamp").copy()
        if champ_df.empty:
            return pd.DataFrame(columns=["window_start", "transitions", "zones_visited"])

        champ_df["zone"] = champ_df.apply(
            lambda r: self._spatial.classify_zone(r["x"], r["y"]), axis=1
        )

        max_time = champ_df["timestamp"].max()
        rows = []
        for start in range(0, int(max_time), window_seconds):
            end = start + window_seconds
            window_data = champ_df[
                (champ_df["timestamp"] >= start) & (champ_df["timestamp"] < end)
            ]
            if len(window_data) < 2:
                rows.append({"window_start": start, "transitions": 0, "zones_visited": 0})
                continue

            zones = window_data["zone"].tolist()
            transitions = sum(1 for i in range(1, len(zones)) if zones[i] != zones[i - 1])
            unique_zones = len(set(zones))
            rows.append({
                "window_start": start,
                "transitions": transitions,
                "zones_visited": unique_zones,
            })

        return pd.DataFrame(rows)

    # ── Gold & Lane Stability ───────────────────────────────────────

    def gold_diff_slope(self, ocr_df: pd.DataFrame, phase: str = None) -> float:
        """Linear regression slope of gold difference over time.

        Positive = blue team gaining gold advantage.
        If phase specified, only compute for that phase's time window.

        Args:
            ocr_df: DataFrame with columns [timestamp, gold_diff].
            phase: Optional game phase name ('early', 'mid', 'late').

        Returns:
            Slope of gold difference trend line, or NaN if insufficient data.
        """
        data = ocr_df.copy()
        if phase is not None:
            if phase not in self.phases:
                raise ValueError(f"Unknown phase: {phase}. Choose from {list(self.phases.keys())}")
            start, end = self.phases[phase]
            data = data[(data["timestamp"] >= start) & (data["timestamp"] < end)]

        data = data.dropna(subset=["timestamp", "gold_diff"])
        if len(data) < 2:
            return float("nan")

        timestamps = data["timestamp"].values.astype(float)
        gold_diffs = data["gold_diff"].values.astype(float)
        slope = np.polyfit(timestamps, gold_diffs, 1)[0]
        return float(slope)

    def lane_assignment_stability(
        self, positions_df: pd.DataFrame, champion: str, expected_zone: str
    ) -> float:
        """Fraction of time a champion spends in their expected zone/lane.

        Higher values indicate more stable laning; lower values indicate roaming.
        Uses SpatialFeatures.classify_zone internally.

        Args:
            positions_df: Position DataFrame with columns [timestamp, champion, x, y].
            champion: Champion name to analyze.
            expected_zone: Zone name the champion is expected to occupy (e.g. 'mid_lane').

        Returns:
            Fraction (0-1) of timestamps the champion is in the expected zone,
            or NaN if no data for the champion.
        """
        champ_df = positions_df[positions_df["champion"] == champion]
        if champ_df.empty:
            return float("nan")

        zones = champ_df.apply(
            lambda r: self._spatial.classify_zone(r["x"], r["y"]), axis=1
        )
        return float((zones == expected_zone).mean())

    # ── Aggregated Feature Vector ────────────────────────────────────

    def compute_all(
        self,
        events: pd.DataFrame,
        positions: pd.DataFrame = None,
        blue_team: list[str] = None,
        red_team: list[str] = None,
        ocr_df: pd.DataFrame = None,
    ) -> dict:
        """Compute a full temporal feature vector for a game.

        Returns a flat dict suitable for use as a single row in an ML dataset.
        """
        features = {}

        # First-event timings
        features.update(self.first_event_timing(events))

        # Event counts per phase
        phase_counts = self.event_counts_by_phase(events)
        for phase, counts in phase_counts.items():
            for et in self.event_types:
                features[f"{phase}_{et}_count"] = counts.get(et, 0)

        # Event tempo
        tempo = self.event_tempo(events)
        if not tempo.empty:
            features["avg_tempo"] = tempo["tempo"].mean()
            features["max_tempo"] = tempo["tempo"].max()
            features["tempo_variance"] = tempo["tempo"].var()

        # Rotation features (if position data available)
        if positions is not None and blue_team and red_team:
            for side, team in [("blue", blue_team), ("red", red_team)]:
                side_transitions = []
                for champ in team:
                    rot = self.rotation_frequency(positions, champ)
                    if not rot.empty:
                        side_transitions.append(rot["transitions"].mean())
                features[f"{side}_avg_rotations"] = np.mean(side_transitions) if side_transitions else 0.0

        # Gold difference slope per phase (if OCR data available)
        if ocr_df is not None and "gold_diff" in ocr_df.columns:
            features["gold_diff_slope_overall"] = self.gold_diff_slope(ocr_df)
            for phase in self.phases:
                features[f"gold_diff_slope_{phase}"] = self.gold_diff_slope(ocr_df, phase=phase)

        logger.info("Computed %d temporal features", len(features))
        return features
