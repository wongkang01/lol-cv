"""
Stage 4b — Build regression targets CSV from API + OCR time series.

Each row in the resulting ``data/processed/targets.csv`` is a single game
with two layers of targets:

1. **API-anchored targets** (authoritative, available for all 45 games in
   ``data/game_winners.json``): ``gold_diff_final_api``, ``kill_diff_final_api``,
   ``gold_per_min_blue/red``, ``duration_seconds``. These are pulled directly
   from the lolesports livestats end-of-game frame and are not OCR-derived,
   so they have no measurement noise.

2. **OCR-derived mid-game targets** (best effort, often NaN due to OCR
   noise): ``gold_diff_t{300,600,900,1200}``, ``gold_diff_slope_0_600``,
   ``gold_diff_median_0_600``. These are computed from a heavily-cleaned
   per-frame OCR time series. The cleaner is much more aggressive than the
   one in ``scripts/run_features.py`` because the broadcast HUD OCR has a
   high rate of digit-hallucination errors (e.g. reading ``5200`` as
   ``52000``) that survive the wider-range filter used for feature
   computation.

   Cleaning rules (in order):
     a) ``smooth_ocr`` from ``lol_cv.features.temporal`` (range filter +
        non-decreasing kills + non-monotonic timer rejection).
     b) Wide centred rolling median (window=11) computes an "expected"
        gold value at each row. Any individual reading whose relative
        deviation from the rolling median exceeds 0.4 (40 %) is set to
        NaN.
     c) Tight centred rolling median (window=5) on the cleaned series.
     d) Slope is computed via Theil-Sen regression (robust to remaining
        outliers) over [0, 600] s.

   ``kill_diff_t*`` checkpoints are NOT emitted: even after cleaning, the
   OCR'd kill counts are unreliable at fixed checkpoints because each
   misread integer cannot be distinguished from a true integer. The API
   ``kill_diff_final_api`` is the only kill-based target.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import theilslopes

# Make `src/` importable so `lol_cv` resolves when this script is invoked
# directly from the repo root without PYTHONPATH=src being set.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from lol_cv.features.temporal import smooth_ocr  # noqa: E402
from lol_cv.utils import setup_logger  # noqa: E402

logger = setup_logger("scripts.build_targets")

REPO_ROOT = _REPO_ROOT
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
WINNERS_PATH = REPO_ROOT / "data" / "game_winners.json"
OUTPUT_PATH = PROCESSED_DIR / "targets.csv"

CHECKPOINTS = (300, 600, 900, 1200)
CHECKPOINT_TOLERANCE = 15        # seconds
SLOPE_WINDOW = (0, 600)
SLOPE_MIN_POINTS = 8
WIDE_MEDIAN_WINDOW = 11          # rows; ~50 s at 1 Hz HUD sample rate
TIGHT_MEDIAN_WINDOW = 5          # rows; final smoothing
RELATIVE_OUTLIER_THRESHOLD = 0.4  # 40 % deviation from local median


# ── Aggressive OCR cleaning ──────────────────────────────────────────────


def _reject_outliers(series: pd.Series, window: int) -> pd.Series:
    """Drop readings that diverge >RELATIVE_OUTLIER_THRESHOLD from a centred
    rolling median.

    Returns a copy with outliers replaced by NaN. Any row whose rolling
    median is itself NaN (insufficient context at the edges) is left as-is.
    """
    s = series.astype(float).copy()
    rolling = s.rolling(window=window, center=True, min_periods=3).median()
    # Avoid div-by-zero: only flag where rolling > 200 (well above any real
    # gold value's lower bound and far from the smooth_ocr range filter).
    safe = rolling.where(rolling.abs() > 200, other=np.nan)
    rel_dev = (s - rolling).abs() / safe.abs()
    bad = rel_dev > RELATIVE_OUTLIER_THRESHOLD
    s.loc[bad] = np.nan
    return s


def _aggressive_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Two-pass clean: outlier rejection then a tight median refit.

    Operates on the gold columns only — kills are integers and don't have
    a meaningful "rolling median". The smoothed gold series gives us
    ``gold_diff`` which is what the regression targets care about.
    """
    out = df.copy()
    for col in ("blue_gold", "red_gold"):
        if col not in out.columns:
            continue
        # Pass 1: reject relative outliers against a wide rolling median
        out[col] = _reject_outliers(out[col], WIDE_MEDIAN_WINDOW)
        # Pass 2: tight median to fill the small holes left by pass 1
        out[col] = (
            out[col]
            .rolling(window=TIGHT_MEDIAN_WINDOW, center=True, min_periods=2)
            .median()
        )
    if "blue_gold" in out.columns and "red_gold" in out.columns:
        out["gold_diff"] = out["blue_gold"] - out["red_gold"]
    return out


# ── Checkpoint extraction ────────────────────────────────────────────────


def _value_at_checkpoint(
    ocr_df: pd.DataFrame, t: float, tolerance: float = CHECKPOINT_TOLERANCE
) -> pd.Series | None:
    """Return the OCR row closest to ``t`` within ±tolerance seconds.

    Requires ``gold_diff`` to be non-NaN at the matched row — otherwise
    returns None (the cleaner may have rejected that frame).
    """
    if ocr_df.empty or "game_time_seconds" not in ocr_df.columns:
        return None
    sub = ocr_df.dropna(subset=["gold_diff"])
    if sub.empty:
        return None
    times = sub["game_time_seconds"].astype(float)
    deltas = (times - t).abs()
    idx = deltas.idxmin()
    if pd.isna(idx) or deltas.loc[idx] > tolerance:
        return None
    return sub.loc[idx]


def _gold_diff_slope_robust(ocr_df: pd.DataFrame) -> float:
    """Robust Theil-Sen slope of gold_diff over [0, 600] s.

    Theil-Sen is the median slope over all pairs of points, which is
    breakdown-robust up to ~29 % contamination — useful here because
    even after cleaning, a few OCR misreads can survive.
    """
    if ocr_df.empty or "gold_diff" not in ocr_df.columns:
        return float("nan")
    start, end = SLOPE_WINDOW
    window = ocr_df[
        (ocr_df["game_time_seconds"] >= start)
        & (ocr_df["game_time_seconds"] <= end)
    ].dropna(subset=["game_time_seconds", "gold_diff"])
    if len(window) < SLOPE_MIN_POINTS:
        return float("nan")
    xs = window["game_time_seconds"].to_numpy(dtype=float)
    ys = window["gold_diff"].to_numpy(dtype=float)
    slope, _intercept, _lo, _hi = theilslopes(ys, xs)
    return float(slope)


def _gold_diff_median(ocr_df: pd.DataFrame, t_lo: float, t_hi: float) -> float:
    """Median gold_diff over [t_lo, t_hi]. Robust to remaining outliers."""
    if ocr_df.empty or "gold_diff" not in ocr_df.columns:
        return float("nan")
    window = ocr_df[
        (ocr_df["game_time_seconds"] >= t_lo)
        & (ocr_df["game_time_seconds"] <= t_hi)
    ]["gold_diff"].dropna()
    if window.empty:
        return float("nan")
    return float(window.median())


# ── Per-game target computation ──────────────────────────────────────────


def _compute_targets_for_game(
    match_id: str, ocr_path: Path | None, winner_info: dict | None
) -> dict:
    """Compute API-anchored + OCR-derived targets for one game."""
    row: dict = {"match_id": match_id}

    # ── API-anchored targets (always present if winner_info exists) ──
    if winner_info is not None:
        bg = winner_info.get("blue_gold")
        rg = winner_info.get("red_gold")
        bk = winner_info.get("blue_kills")
        rk = winner_info.get("red_kills")
        dur = winner_info.get("duration_seconds")
        row["gold_diff_final_api"] = (
            float(bg - rg) if bg is not None and rg is not None else float("nan")
        )
        row["kill_diff_final_api"] = (
            float(bk - rk) if bk is not None and rk is not None else float("nan")
        )
        row["duration_seconds"] = (
            float(dur) if dur is not None else float("nan")
        )
        if dur and dur > 0 and bg is not None and rg is not None:
            row["gold_per_min_blue"] = float(bg) / (float(dur) / 60.0)
            row["gold_per_min_red"] = float(rg) / (float(dur) / 60.0)
            row["gold_per_min_diff"] = (
                row["gold_per_min_blue"] - row["gold_per_min_red"]
            )
        else:
            row["gold_per_min_blue"] = float("nan")
            row["gold_per_min_red"] = float("nan")
            row["gold_per_min_diff"] = float("nan")
    else:
        for k in (
            "gold_diff_final_api",
            "kill_diff_final_api",
            "duration_seconds",
            "gold_per_min_blue",
            "gold_per_min_red",
            "gold_per_min_diff",
        ):
            row[k] = float("nan")

    # ── OCR-derived mid-game targets ──
    if ocr_path is None or not ocr_path.exists():
        for t in CHECKPOINTS:
            row[f"gold_diff_t{t}"] = float("nan")
        row["gold_diff_slope_0_600"] = float("nan")
        row["gold_diff_median_0_300"] = float("nan")
        row["gold_diff_median_0_600"] = float("nan")
        row["n_ocr_rows_clean"] = 0
        return row

    raw = pd.read_csv(ocr_path)
    smoothed = smooth_ocr(raw)
    cleaned = _aggressive_clean(smoothed)
    n_clean = int(cleaned["gold_diff"].notna().sum()) if "gold_diff" in cleaned.columns else 0
    row["n_ocr_rows_clean"] = n_clean

    for t in CHECKPOINTS:
        ocr_row = _value_at_checkpoint(cleaned, t)
        if ocr_row is None:
            row[f"gold_diff_t{t}"] = float("nan")
        else:
            row[f"gold_diff_t{t}"] = float(ocr_row["gold_diff"])

    row["gold_diff_slope_0_600"] = _gold_diff_slope_robust(cleaned)
    row["gold_diff_median_0_300"] = _gold_diff_median(cleaned, 0, 300)
    row["gold_diff_median_0_600"] = _gold_diff_median(cleaned, 0, 600)

    return row


def main() -> None:
    if not PROCESSED_DIR.exists():
        sys.exit(f"Missing processed dir: {PROCESSED_DIR}")
    if not WINNERS_PATH.exists():
        sys.exit(
            f"Missing {WINNERS_PATH} — run scripts/fetch_game_winners.py first"
        )

    with open(WINNERS_PATH) as f:
        winners: dict = json.load(f)
    logger.info("Loaded %d winner entries from API", len(winners))

    game_pattern = re.compile(r"_g\d+$")
    game_dirs = sorted(
        d for d in PROCESSED_DIR.iterdir()
        if d.is_dir() and game_pattern.search(d.name)
    )
    logger.info("Found %d processed game directories", len(game_dirs))

    rows: list[dict] = []
    for game_dir in game_dirs:
        match_id = game_dir.name
        ocr_path = game_dir / "ocr.csv"
        winner_info = winners.get(match_id)
        if winner_info is None:
            logger.warning("[no-api] %s — not in game_winners.json", match_id)
        try:
            row = _compute_targets_for_game(match_id, ocr_path, winner_info)
        except Exception:
            logger.exception("Failed to build targets for %s", match_id)
            continue
        rows.append(row)
        logger.info(
            "[done] %s — gd_final_api=%s gd_t300=%s gd_t600=%s n_clean=%d",
            match_id,
            row.get("gold_diff_final_api"),
            row.get("gold_diff_t300"),
            row.get("gold_diff_t600"),
            row.get("n_ocr_rows_clean", 0),
        )

    if not rows:
        sys.exit("No targets computed.")

    col_order = [
        "match_id",
        # API-anchored (reliable)
        "gold_diff_final_api",
        "kill_diff_final_api",
        "duration_seconds",
        "gold_per_min_blue",
        "gold_per_min_red",
        "gold_per_min_diff",
        # OCR-derived (best effort)
        "gold_diff_t300",
        "gold_diff_t600",
        "gold_diff_t900",
        "gold_diff_t1200",
        "gold_diff_slope_0_600",
        "gold_diff_median_0_300",
        "gold_diff_median_0_600",
        "n_ocr_rows_clean",
    ]
    df = pd.DataFrame(rows)
    for col in col_order:
        if col not in df.columns:
            df[col] = float("nan")
    df = df[col_order].sort_values("match_id").set_index("match_id")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH)
    logger.info("Wrote %s: %d rows × %d cols", OUTPUT_PATH, *df.shape)

    # ── Reports ─────────────────────────────────────────────────────────
    print("\n=== targets.describe() ===")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(df.describe())

    print("\n=== NaN counts ===")
    print(df.isna().sum().to_string())

    print("\n=== Coverage ===")
    n = len(df)
    for col in col_order[1:]:
        non_nan = int(df[col].notna().sum())
        print(f"  {col:30s} {non_nan}/{n}")

    api = df["gold_diff_final_api"].dropna()
    if not api.empty:
        print("\n=== gold_diff_final_api distribution (API, authoritative) ===")
        print(
            f"  mean={api.mean():.0f}  median={api.median():.0f}  "
            f"min={api.min():.0f}  max={api.max():.0f}"
        )

    gd600 = df["gold_diff_t600"].dropna()
    if not gd600.empty:
        print("\n=== gold_diff_t600 distribution (OCR, mid-game) ===")
        print(
            f"  mean={gd600.mean():.0f}  median={gd600.median():.0f}  "
            f"min={gd600.min():.0f}  max={gd600.max():.0f}"
        )

    gm600 = df["gold_diff_median_0_600"].dropna()
    if not gm600.empty:
        print("\n=== gold_diff_median_0_600 distribution (OCR, robust) ===")
        print(
            f"  mean={gm600.mean():.0f}  median={gm600.median():.0f}  "
            f"min={gm600.min():.0f}  max={gm600.max():.0f}"
        )

    print(f"\nWrote {OUTPUT_PATH} ({n} rows)")


if __name__ == "__main__":
    main()
