"""
Stage 4 — Compute spatial + temporal features for every processed game.

For each game with a positions.csv (and optional ocr.csv):
    1. Load champion picks → blue_team, red_team lists
    2. Compute spatial features via SpatialFeatures.compute_all
    3. Compute temporal features via TemporalFeatures.compute_all
       (gold-derived features only if ocr.csv present)
    4. Append to a single feature matrix

Output:
    data/processed/features.csv  — one row per game (n_games × n_features)
    data/processed/features_meta.csv — match_id, winner, blue/red teams
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from lol_cv.features.spatial import SpatialFeatures
from lol_cv.features.temporal import TemporalFeatures
from lol_cv.utils import setup_logger

logger = setup_logger("scripts.run_features")

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
PICKS_PATH = REPO_ROOT / "data" / "champion_picks.json"
META_PATH = REPO_ROOT / "data" / "match_metadata.json"
WINNERS_PATH = REPO_ROOT / "data" / "game_winners.json"


def smooth_ocr(ocr_df: pd.DataFrame) -> pd.DataFrame:
    """Clean noisy OCR readings before computing features.

    - Drop rows with no game_time_seconds (no valid timer reading)
    - Sort by game_time_seconds
    - Drop rows where game_time_seconds is non-monotonic (clearly wrong)
    - For kills: enforce non-decreasing (kills only go up)
    - For gold: drop values outside (200, 100000)
    - Compute gold_diff for downstream features
    """
    if ocr_df.empty:
        return ocr_df

    df = ocr_df.dropna(subset=["game_time_seconds"]).copy()
    if df.empty:
        return df

    df = df.sort_values("game_time_seconds").reset_index(drop=True)

    # Drop non-monotonic timer outliers (kept value must be ≥ previous)
    keep_mask = [True]
    last_t = df["game_time_seconds"].iloc[0]
    for i in range(1, len(df)):
        t = df["game_time_seconds"].iloc[i]
        if t >= last_t and t - last_t < 600:  # at most 10 min jump
            keep_mask.append(True)
            last_t = t
        else:
            keep_mask.append(False)
    df = df[keep_mask].reset_index(drop=True)

    # Smooth kills: enforce non-decreasing
    for col in ("blue_kills", "red_kills"):
        if col in df.columns:
            vals = df[col].astype("float")
            # Reject impossible values (>50 kills total per team is rare)
            vals = vals.where((vals >= 0) & (vals <= 50), np.nan)
            vals = vals.cummax()  # only allow non-decreasing
            df[col] = vals

    # Filter gold to plausible range and drop NaN
    for col in ("blue_gold", "red_gold"):
        if col in df.columns:
            vals = df[col].astype("float")
            vals = vals.where((vals >= 200) & (vals <= 100_000), np.nan)
            df[col] = vals

    # Gold diff (NaN where either side missing)
    if "blue_gold" in df.columns and "red_gold" in df.columns:
        df["gold_diff"] = df["blue_gold"] - df["red_gold"]

    return df


def ocr_aggregate_features(ocr_df: pd.DataFrame) -> dict:
    """Compact aggregate features from cleaned OCR time series."""
    feats: dict = {}
    if ocr_df.empty:
        return feats

    # Game duration
    if "game_time_seconds" in ocr_df.columns:
        valid = ocr_df["game_time_seconds"].dropna()
        if not valid.empty:
            feats["game_duration_seconds"] = float(valid.max())

    # Final kill counts
    for col in ("blue_kills", "red_kills"):
        if col in ocr_df.columns:
            v = ocr_df[col].dropna()
            if not v.empty:
                feats[f"final_{col}"] = float(v.iloc[-1])

    # Gold diff trajectory
    if "gold_diff" in ocr_df.columns:
        gd = ocr_df["gold_diff"].dropna()
        if not gd.empty:
            feats["final_gold_diff"] = float(gd.iloc[-1])
            feats["max_gold_lead_blue"] = float(gd.max())
            feats["max_gold_lead_red"] = float((-gd).max())
            feats["mean_gold_diff"] = float(gd.mean())

    return feats


def kill_events_from_ocr(ocr_df: pd.DataFrame) -> pd.DataFrame:
    """Convert smoothed OCR kill counts into kill events."""
    if ocr_df.empty:
        return pd.DataFrame(columns=["timestamp", "event_type", "team"])

    rows = []
    prev_b = 0
    prev_r = 0
    for _, r in ocr_df.iterrows():
        ts = r.get("game_time_seconds")
        bk = r.get("blue_kills")
        rk = r.get("red_kills")
        if pd.isna(ts):
            continue
        if not pd.isna(bk) and bk > prev_b:
            for _ in range(int(bk - prev_b)):
                rows.append({"timestamp": float(ts), "event_type": "kill", "team": "blue"})
            prev_b = int(bk)
        if not pd.isna(rk) and rk > prev_r:
            for _ in range(int(rk - prev_r)):
                rows.append({"timestamp": float(ts), "event_type": "kill", "team": "red"})
            prev_r = int(rk)
    return pd.DataFrame(rows)


def main() -> None:
    if not PICKS_PATH.exists():
        sys.exit(f"Missing {PICKS_PATH}")
    if not WINNERS_PATH.exists():
        sys.exit(f"Missing {WINNERS_PATH} — run scripts/fetch_game_winners.py first")

    with open(PICKS_PATH) as f:
        picks = json.load(f)
    with open(WINNERS_PATH) as f:
        winners = json.load(f)

    spatial = SpatialFeatures()
    temporal = TemporalFeatures()

    rows = []
    meta_rows = []

    game_pattern = re.compile(r"_g\d+$")
    game_dirs = sorted(
        d for d in PROCESSED_DIR.iterdir()
        if d.is_dir() and game_pattern.search(d.name)
    )
    logger.info("Found %d processed game directories", len(game_dirs))

    for game_dir in game_dirs:
        match_id = game_dir.name
        positions_path = game_dir / "positions.csv"
        ocr_path = game_dir / "ocr.csv"

        if not positions_path.exists():
            logger.warning("[skip] %s — no positions.csv", match_id)
            continue
        if match_id not in picks:
            logger.warning("[skip] %s — no picks", match_id)
            continue

        positions_df = pd.read_csv(positions_path)
        if positions_df.empty:
            logger.warning("[skip] %s — empty positions", match_id)
            continue

        blue_team = picks[match_id]["blue_champions"]
        red_team = picks[match_id]["red_champions"]

        feats: dict = {"match_id": match_id}

        # ── Spatial features ──
        try:
            spatial_feats = spatial.compute_all(positions_df, blue_team, red_team)
            for k, v in spatial_feats.items():
                feats[f"sp_{k}"] = v
        except Exception:
            logger.exception("Spatial failed for %s", match_id)
            continue

        # ── OCR-derived features (best effort) ──
        ocr_smoothed = pd.DataFrame()
        if ocr_path.exists():
            ocr_df = pd.read_csv(ocr_path)
            ocr_smoothed = smooth_ocr(ocr_df)
            agg = ocr_aggregate_features(ocr_smoothed)
            for k, v in agg.items():
                feats[f"ocr_{k}"] = v

        # ── Temporal features ──
        try:
            events = kill_events_from_ocr(ocr_smoothed)
            temporal_feats = temporal.compute_all(
                events,
                positions=positions_df,
                blue_team=blue_team,
                red_team=red_team,
                ocr_df=ocr_smoothed if "gold_diff" in ocr_smoothed.columns else None,
            )
            for k, v in temporal_feats.items():
                feats[f"tp_{k}"] = v
        except Exception:
            logger.exception("Temporal failed for %s", match_id)

        rows.append(feats)

        win_info = winners.get(match_id, {})
        meta_rows.append({
            "match_id": match_id,
            "winner_side": win_info.get("winner_side"),
            "winner_team_code": win_info.get("winner_team_code"),
            "blue_team_code": win_info.get("blue_team_code"),
            "red_team_code": win_info.get("red_team_code"),
            "blue_team": ",".join(blue_team),
            "red_team": ",".join(red_team),
            "blue_kills_final": win_info.get("blue_kills"),
            "red_kills_final": win_info.get("red_kills"),
            "blue_gold_final": win_info.get("blue_gold"),
            "red_gold_final": win_info.get("red_gold"),
            "duration_seconds": win_info.get("duration_seconds"),
            "block": picks[match_id].get("match_info", {}).get("block"),
            "date": picks[match_id].get("match_info", {}).get("date"),
        })

        logger.info("[done] %s — %d features", match_id, len(feats) - 1)

    if not rows:
        sys.exit("No features computed.")

    feat_df = pd.DataFrame(rows).set_index("match_id")
    meta_df = pd.DataFrame(meta_rows).set_index("match_id")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    feat_df.to_csv(PROCESSED_DIR / "features.csv")
    meta_df.to_csv(PROCESSED_DIR / "features_meta.csv")

    logger.info(
        "Wrote features.csv: %d games × %d features", feat_df.shape[0], feat_df.shape[1]
    )
    logger.info("Wrote features_meta.csv: %d rows", meta_df.shape[0])


if __name__ == "__main__":
    main()
