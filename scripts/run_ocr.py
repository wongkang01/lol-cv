"""
Stage 2 — Run OCR on every cropped HUD region for every game.

Each game directory under ``data/raw/<match_id>/hud/`` contains
pre-cropped subfolders for ``timer``, ``kill_score``, ``blue_gold``,
``red_gold``, etc. The crops are tight but include adjacent icons,
so we OCR them and then post-process with regex.

Uses easyocr (much better than tesseract for the stylised LoL HUD font).
Reader is shared across all games to amortise model load time (~3s).

Output: ``data/processed/<match_id>/ocr.csv`` with columns
``[frame_idx, timestamp, blue_kills, red_kills, blue_gold, red_gold,
   game_time, game_time_seconds]``.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import easyocr

from lol_cv.utils import setup_logger, ensure_dir

logger = setup_logger("scripts.run_ocr")
logging.getLogger("easyocr").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

# Single global reader — heavy to construct
_reader: easyocr.Reader | None = None


def get_reader() -> easyocr.Reader:
    global _reader
    if _reader is None:
        logger.info("Loading easyocr reader (en, CPU)")
        _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _reader


def parse_frame_index(filename: str) -> int:
    return int(filename.replace("frame_", "").replace(".png", ""))


def upscale(img: np.ndarray, factor: int = 3) -> np.ndarray:
    return cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)


def ocr_text(img_path: Path, allowlist: str | None = None) -> str:
    """Run easyocr on a single tight HUD crop and concatenate the result."""
    img = cv2.imread(str(img_path))
    if img is None:
        return ""
    big = upscale(img, 3)
    reader = get_reader()
    kwargs = {"detail": 0, "paragraph": False}
    if allowlist:
        kwargs["allowlist"] = allowlist
    try:
        res = reader.readtext(big, **kwargs)
    except Exception:
        return ""
    return " ".join(res).strip()


def ocr_left_right(img_path: Path, allowlist: str = "0123456789") -> tuple[int | None, int | None]:
    """Crop image into left + right halves and OCR each separately.

    Used for the kill_score crop which has '<blue_kills> <icon> <red_kills>'.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return None, None
    h, w = img.shape[:2]
    left = upscale(img[:, : w // 2], 4)
    right = upscale(img[:, w // 2 :], 4)
    reader = get_reader()
    try:
        ltexts = reader.readtext(left, detail=0, paragraph=False, allowlist=allowlist)
        rtexts = reader.readtext(right, detail=0, paragraph=False, allowlist=allowlist)
    except Exception:
        return None, None
    return parse_int(" ".join(ltexts)), parse_int(" ".join(rtexts))


# ── Smart parsers ────────────────────────────────────────────────────


def parse_int(text: str) -> int | None:
    """Extract first integer (max 3 digits — kills, turrets) from text."""
    m = re.search(r"\d{1,3}", text)
    if m:
        return int(m.group(0))
    return None


def parse_timer(text: str) -> tuple[str | None, int | None]:
    """Find a 'M:SS' or 'MM:SS' substring anywhere in the OCR output."""
    # Search for digit:digit2 pattern, allowing 1-3 leading digits
    m = re.search(r"(\d{1,3}):(\d{2})", text)
    if not m:
        return None, None
    mins = int(m.group(1))
    secs = int(m.group(2))
    if secs >= 60 or mins > 99:
        return None, None  # Junk
    return f"{mins}:{secs:02d}", mins * 60 + secs


def parse_gold(text: str) -> float | None:
    """Find a 'X.Xk' or 'Xk' style gold value in the OCR output.

    Real gold values during a game are 0.5k - 80k. Anything outside this
    range is rejected.
    """
    # Look for digit(.digit) followed by k/K
    matches = re.findall(r"(\d+(?:\.\d+)?)\s*[kK]", text)
    candidates = []
    for m in matches:
        try:
            val = float(m) * 1000
            if 200 <= val <= 100000:
                candidates.append(val)
        except ValueError:
            continue
    if candidates:
        # Return the largest plausible value (gold is the biggest number on screen)
        return max(candidates)

    # Fallback: a bare integer 200-100000 (some broadcasts skip the k)
    bare = re.findall(r"\d{3,5}", text)
    for m in bare:
        try:
            val = int(m)
            if 200 <= val <= 100000:
                return float(val)
        except ValueError:
            continue
    return None


# ── Per-game OCR ─────────────────────────────────────────────────────


def ocr_one_game(game_dir: Path, sample_interval: int = 10) -> pd.DataFrame:
    """OCR HUD crops sampled every ``sample_interval`` seconds.

    Sampling every 10s gives ~150-220 OCR points per game which is
    plenty to compute trends (gold slope, kill tempo). Avoids running
    11k OCR calls per game which would take too long on CPU.
    """
    hud_dir = game_dir / "hud"
    if not hud_dir.exists():
        return pd.DataFrame()

    timer_dir = hud_dir / "timer"
    timer_frames = sorted(timer_dir.glob("frame_*.png"))
    if not timer_frames:
        return pd.DataFrame()

    # Subsample by frame index (frames are stored at 1 fps so 1 frame = 1 sec)
    timer_frames = timer_frames[::sample_interval]

    rows = []
    for i, timer_path in enumerate(timer_frames):
        frame_idx = parse_frame_index(timer_path.name)
        row = {"frame_idx": frame_idx, "timestamp": float(frame_idx)}

        timer_text = ocr_text(timer_path, "0123456789:")
        gt, gts = parse_timer(timer_text)
        row["game_time"] = gt
        row["game_time_seconds"] = gts

        # Kill score: split left/right
        ks_path = hud_dir / "kill_score" / timer_path.name
        if ks_path.exists():
            bk, rk = ocr_left_right(ks_path)
            row["blue_kills"] = bk
            row["red_kills"] = rk
        else:
            row["blue_kills"] = None
            row["red_kills"] = None

        # Gold (must contain k/K)
        for side in ("blue", "red"):
            p = hud_dir / f"{side}_gold" / timer_path.name
            row[f"{side}_gold"] = parse_gold(ocr_text(p, "0123456789.kK")) if p.exists() else None

        rows.append(row)

        if (i + 1) % 200 == 0:
            logger.info("  ... %d/%d frames OCRed", i + 1, len(timer_frames))

    return pd.DataFrame(rows)


def main() -> None:
    game_pattern = re.compile(r"_g\d+$")
    game_dirs = sorted(
        d for d in RAW_DIR.iterdir()
        if d.is_dir() and game_pattern.search(d.name)
    )
    logger.info("Found %d game directories", len(game_dirs))

    get_reader()  # warm up once

    done = skipped = failed = 0
    for game_dir in game_dirs:
        out_path = PROCESSED_DIR / game_dir.name / "ocr.csv"
        if out_path.exists():
            logger.info("[skip] %s", game_dir.name)
            skipped += 1
            continue
        logger.info("[run]  %s", game_dir.name)
        try:
            df = ocr_one_game(game_dir)
            ensure_dir(out_path.parent)
            df.to_csv(out_path, index=False)
            timer_pct = df["game_time_seconds"].notna().mean() * 100 if not df.empty else 0
            kill_pct = df["blue_kills"].notna().mean() * 100 if not df.empty else 0
            gold_pct = df["blue_gold"].notna().mean() * 100 if not df.empty else 0
            logger.info(
                "[done] %s — %d frames — timer %.0f%% kills %.0f%% gold %.0f%%",
                game_dir.name, len(df), timer_pct, kill_pct, gold_pct,
            )
            done += 1
        except Exception as exc:
            logger.exception("[fail] %s", game_dir.name)
            failed += 1

    logger.info("OCR finished: %d done, %d skipped, %d failed", done, skipped, failed)


if __name__ == "__main__":
    main()
