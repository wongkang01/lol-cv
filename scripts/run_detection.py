"""
Stage 1 — Run filtered champion detection on every processed game.

For each game directory under ``data/raw/<match_id>/``:
    1. Load champion picks from ``data/champion_picks.json``
    2. Run YOLO + champion filter on every minimap frame in ``minimap/``
    3. Save per-frame positions DataFrame to
       ``data/processed/<match_id>/positions.csv``

Skips games where positions.csv already exists.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import pandas as pd

from lol_cv.extraction.minimap import MinimapTracker
from lol_cv.utils import setup_logger, get_data_dir, ensure_dir

logger = setup_logger("scripts.run_detection")

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
PICKS_PATH = REPO_ROOT / "data" / "champion_picks.json"
MODEL_PATH = REPO_ROOT / "data" / "models" / "yolov11m.pt"


def parse_frame_index(frame_filename: str) -> int:
    """Extract integer frame index from 'frame_NNNNNN.png'."""
    return int(frame_filename.replace("frame_", "").replace(".png", ""))


def detect_one_game(
    game_dir: Path,
    valid_champions: set[str],
    tracker: MinimapTracker,
    fps: float = 1.0,
) -> pd.DataFrame:
    """Run filtered detection on every minimap frame in a game directory.

    Args:
        game_dir: Path to ``data/raw/<match_id>/``.
        valid_champions: Set of champion names actually in this game.
        tracker: Pre-loaded MinimapTracker.
        fps: Sampling rate used during ingestion (1 fps default).

    Returns:
        DataFrame with [frame_idx, timestamp, champion, x, y, confidence].
    """
    minimap_dir = game_dir / "minimap"
    if not minimap_dir.exists():
        logger.warning("No minimap directory in %s — skipping", game_dir.name)
        return pd.DataFrame()

    frames = sorted(minimap_dir.glob("frame_*.png"))
    logger.info("%s — %d minimap frames, %d valid champions",
                game_dir.name, len(frames), len(valid_champions))

    rows = []
    for i, frame_path in enumerate(frames):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue
        detections = tracker.detect_frame_filtered(frame, valid_champions)
        frame_idx = parse_frame_index(frame_path.name)
        timestamp = frame_idx / fps  # frame_idx is in seconds at 1 fps
        for det in detections:
            rows.append({
                "frame_idx": frame_idx,
                "timestamp": timestamp,
                "champion": det["champion"],
                "x": det["x"],
                "y": det["y"],
                "confidence": det["confidence"],
            })
        if (i + 1) % 200 == 0:
            logger.info("  ... %d/%d frames done", i + 1, len(frames))

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["timestamp", "champion"]).reset_index(drop=True)
    return df


def main() -> None:
    if not PICKS_PATH.exists():
        sys.exit(f"Missing champion picks file at {PICKS_PATH}")
    if not MODEL_PATH.exists():
        sys.exit(f"Missing YOLO weights at {MODEL_PATH}")

    with open(PICKS_PATH) as f:
        all_picks = json.load(f)

    tracker = MinimapTracker(model_path=str(MODEL_PATH), confidence=0.4)
    tracker.load_model()

    import re
    game_pattern = re.compile(r"_g\d+$")
    game_dirs = sorted(
        d for d in RAW_DIR.iterdir()
        if d.is_dir() and game_pattern.search(d.name)
    )
    logger.info("Found %d game directories", len(game_dirs))

    total_done = 0
    skipped = 0
    failed = 0

    for game_dir in game_dirs:
        match_id = game_dir.name
        out_path = PROCESSED_DIR / match_id / "positions.csv"
        if out_path.exists():
            logger.info("[skip] %s already has positions.csv", match_id)
            skipped += 1
            continue

        if match_id not in all_picks:
            logger.warning("[fail] %s — no entry in champion_picks.json", match_id)
            failed += 1
            continue

        valid_champions = set(all_picks[match_id]["all_champions"])

        try:
            t0 = time.time()
            df = detect_one_game(game_dir, valid_champions, tracker)
            elapsed = time.time() - t0

            ensure_dir(out_path.parent)
            df.to_csv(out_path, index=False)
            logger.info(
                "[done] %s — %d detections in %.1fs (%.0f frames/s)",
                match_id, len(df), elapsed,
                (df["frame_idx"].nunique() if not df.empty else 0) / max(elapsed, 1e-6),
            )
            total_done += 1
        except Exception as exc:
            logger.exception("[fail] %s — %s", match_id, exc)
            failed += 1

    logger.info(
        "Detection finished: %d done, %d skipped, %d failed",
        total_done, skipped, failed,
    )


if __name__ == "__main__":
    main()
