"""Extract 5 minimap PNG frames for the thesis presentation.

For each scenario we:
  1. Load the candidate game's positions.csv
  2. Normalise timestamps so the per-game minimum is 0 (in-game seconds at 1 fps)
  3. Filter to the target time window
  4. Group by frame_idx and compute the scenario criterion
  5. Pick the best matching frame_idx (falling back to the nearest available PNG)
  6. Copy the PNG into charts/<output_filename>

The script prints, for every scenario, the chosen game, frame_idx, in-game time
and criterion value, then copies the file with shutil.copy2.

Run with:
    uv run python scripts/extract_slide_frames.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
CHARTS_DIR = REPO_ROOT / "charts"
PICKS_PATH = REPO_ROOT / "data" / "champion_picks.json"
WINNERS_PATH = REPO_ROOT / "data" / "game_winners.json"


# ───────────────────────── helpers ─────────────────────────


def load_picks() -> dict:
    return json.loads(PICKS_PATH.read_text())


def load_winners() -> dict:
    return json.loads(WINNERS_PATH.read_text())


def load_positions(game: str) -> pd.DataFrame:
    """Load positions.csv and normalise timestamps to in-game seconds."""
    df = pd.read_csv(PROCESSED_DIR / game / "positions.csv")
    if df.empty:
        return df
    df["t"] = df["timestamp"] - df["timestamp"].min()
    return df


def annotate_side(df: pd.DataFrame, picks: dict, game: str) -> pd.DataFrame:
    blue = set(picks[game]["blue_champions"])
    red = set(picks[game]["red_champions"])
    df = df.copy()
    df["side"] = df["champion"].map(
        lambda c: "blue" if c in blue else ("red" if c in red else "unknown")
    )
    return df[df["side"] != "unknown"].reset_index(drop=True)


def available_frames(game: str) -> set[int]:
    minimap_dir = RAW_DIR / game / "minimap"
    return {
        int(p.stem.split("_")[1])
        for p in minimap_dir.glob("frame_*.png")
    }


def nearest_available_frame(target: int, available: set[int]) -> int | None:
    if not available:
        return None
    if target in available:
        return target
    return min(available, key=lambda f: abs(f - target))


def quadrant_count(side_df: pd.DataFrame) -> int:
    quads = set()
    for _, row in side_df.iterrows():
        x, y = row["x"], row["y"]
        if x < 0.5 and y < 0.5:
            quads.add("TL")
        elif x >= 0.5 and y < 0.5:
            quads.add("TR")
        elif x < 0.5 and y >= 0.5:
            quads.add("BL")
        else:
            quads.add("BR")
    return len(quads)


def zone_count(frame_df: pd.DataFrame) -> int:
    """Coarse 4-zone count for frame 1 (no side split)."""
    return quadrant_count(frame_df)


# ───────────────────────── scenario solvers ─────────────────────────


def solve_frame1(games: Iterable[str], picks: dict) -> tuple:
    """Hero shot: 8+ champions visible across 4+ zones, minute 12-16."""
    best = None  # (score, game, frame_idx, value, t)
    for game in games:
        df = load_positions(game)
        if df.empty:
            continue
        df = annotate_side(df, picks, game)
        window = df[(df["t"] >= 720) & (df["t"] <= 960)]
        if window.empty:
            continue
        for frame_idx, grp in window.groupby("frame_idx"):
            n_champs = grp["champion"].nunique()
            zones = zone_count(grp)
            if n_champs < 8 or zones < 4:
                continue
            # Score: prefer more champions, then more zones, then frame closest to 14:00
            t = grp["t"].iloc[0]
            score = (n_champs, zones, -abs(t - 840))
            cand = (score, game, int(frame_idx), f"{n_champs} champs / {zones} zones", float(t))
            if best is None or cand[0] > best[0]:
                best = cand
        if best is not None and best[1] == game:
            # Got a match for the highest priority game already; keep searching
            # within this game only (don't downgrade to lower priority).
            return best
    return best


def solve_frame2(games: Iterable[str], picks: dict, winners: dict) -> tuple:
    """Blue champs in red bot jungle, blue won, minute 12-18."""
    best = None
    for game in games:
        if winners.get(game, {}).get("winner_side") != "blue":
            continue
        df = load_positions(game)
        if df.empty:
            continue
        df = annotate_side(df, picks, game)
        window = df[(df["t"] >= 720) & (df["t"] <= 1080)]
        if window.empty:
            continue
        for frame_idx, grp in window.groupby("frame_idx"):
            blue = grp[grp["side"] == "blue"]
            in_box = blue[
                (blue["x"] >= 0.65) & (blue["x"] <= 0.9)
                & (blue["y"] >= 0.35) & (blue["y"] <= 0.65)
            ]
            n = in_box["champion"].nunique()
            if n < 2:
                continue
            t = grp["t"].iloc[0]
            score = (n, -abs(t - 900))
            cand = (score, game, int(frame_idx), f"{n} blue in red bot-jungle", float(t))
            if best is None or cand[0] > best[0]:
                best = cand
        if best is not None and best[1] == game:
            return best
    return best


def solve_frame3(games: Iterable[str], picks: dict) -> tuple:
    """Dragon contest around minute 3."""
    best = None
    for game in games:
        df = load_positions(game)
        if df.empty:
            continue
        df = annotate_side(df, picks, game)
        window = df[(df["t"] >= 150) & (df["t"] <= 210)]
        if window.empty:
            continue
        for frame_idx, grp in window.groupby("frame_idx"):
            in_box = grp[
                (grp["x"] >= 0.5) & (grp["x"] <= 0.85)
                & (grp["y"] >= 0.6) & (grp["y"] <= 0.95)
            ]
            blue_n = in_box[in_box["side"] == "blue"]["champion"].nunique()
            red_n = in_box[in_box["side"] == "red"]["champion"].nunique()
            if max(blue_n, red_n) < 2:
                continue
            both_present = int(blue_n >= 1 and red_n >= 1)
            total = blue_n + red_n
            t = grp["t"].iloc[0]
            score = (both_present, total, -abs(t - 180))
            cand = (
                score,
                game,
                int(frame_idx),
                f"blue={blue_n}, red={red_n} in dragon area",
                float(t),
            )
            if best is None or cand[0] > best[0]:
                best = cand
        if best is not None and best[1] == game:
            return best
    return best


def solve_frame4(games: Iterable[str], picks: dict) -> tuple:
    """3+ blue champs at the blue base, minute 5-9."""
    best = None
    for game in games:
        df = load_positions(game)
        if df.empty:
            continue
        df = annotate_side(df, picks, game)
        window = df[(df["t"] >= 300) & (df["t"] <= 540)]
        if window.empty:
            continue
        for frame_idx, grp in window.groupby("frame_idx"):
            blue = grp[grp["side"] == "blue"]
            in_base = blue[
                (blue["x"] >= 0.0) & (blue["x"] <= 0.25)
                & (blue["y"] >= 0.75) & (blue["y"] <= 1.0)
            ]
            n = in_base["champion"].nunique()
            if n < 3:
                continue
            t = grp["t"].iloc[0]
            score = (n, -abs(t - 420))
            cand = (score, game, int(frame_idx), f"{n} blue at blue base", float(t))
            if best is None or cand[0] > best[0]:
                best = cand
        if best is not None and best[1] == game:
            return best
    return best


def solve_frame5(games: Iterable[str], picks: dict) -> tuple:
    """Map asymmetry: one side covers many quadrants, other is bunched."""
    best = None
    for game in games:
        df = load_positions(game)
        if df.empty:
            continue
        df = annotate_side(df, picks, game)
        window = df[(df["t"] >= 360) & (df["t"] <= 720)]
        if window.empty:
            continue
        for frame_idx, grp in window.groupby("frame_idx"):
            blue_q = quadrant_count(grp[grp["side"] == "blue"])
            red_q = quadrant_count(grp[grp["side"] == "red"])
            blue_n = grp[grp["side"] == "blue"]["champion"].nunique()
            red_n = grp[grp["side"] == "red"]["champion"].nunique()
            # Need both sides actually present
            if blue_n < 3 or red_n < 3:
                continue
            spread = max(blue_q, red_q)
            bunched = min(blue_q, red_q)
            asym = spread - bunched
            # Tier the criterion: 4 vs <=2 best, then 4 vs <=3, then >=3 vs <=2
            if spread == 4 and bunched <= 2:
                tier = 3
            elif spread == 4 and bunched <= 3:
                tier = 2
            elif spread >= 3 and bunched <= 2:
                tier = 1
            else:
                continue
            t = grp["t"].iloc[0]
            score = (tier, asym, -abs(t - 540))
            cand = (
                score,
                game,
                int(frame_idx),
                f"blue_q={blue_q}, red_q={red_q}",
                float(t),
            )
            if best is None or cand[0] > best[0]:
                best = cand
        if best is not None and best[1] == game and best[0][0] == 3:
            return best  # already optimal
    return best


# ───────────────────────── main ─────────────────────────


SCENARIOS: list[dict] = [
    {
        "name": "Frame 1 (slide1 hero)",
        "filename": "slide1_minimap_hero.png",
        "candidates": [
            "fs_G2_vs_BLG_finals_2026-03-22_g2",
            "fs_JDG_vs_BLG_semifinals_2026-03-21_g1",
            "fs_G2_vs_GEN_semifinals_2026-03-21_g1",
        ],
        "solver": "frame1",
    },
    {
        "name": "Frame 2 (slide11 q1 enemy jungle pressure)",
        "filename": "slide11_q1_enemy_jungle_pressure.png",
        "candidates": [
            # Listed candidates first (per instructions): try priority order
            "fs_G2_vs_BLG_finals_2026-03-22_g2",
            "fs_LYON_vs_GEN_groups_2026-03-19_g3",
            "fs_BFX_vs_BLG_groups_2026-03-16_g2",
            # Fallbacks discovered by scanning all blue-won games for the
            # criterion "blue champs deep in red bot-jungle" — the listed
            # priority candidates either weren't blue-won or had at most
            # 1 blue in the box. Extra candidates ranked by max blue count.
            "fs_JDG_vs_LOUD_groups_2026-03-19_g1",
            "fs_TSW_vs_G2_groups_2026-03-16_g2",
            "fs_G2_vs_GEN_semifinals_2026-03-21_g3",
        ],
        "solver": "frame2",
    },
    {
        "name": "Frame 3 (slide11 q2 objective priority)",
        "filename": "slide11_q2_objective_priority.png",
        "candidates": [
            "fs_BFX_vs_BLG_groups_2026-03-16_g1",
            "fs_G2_vs_BLG_finals_2026-03-22_g1",
            "fs_TSW_vs_BFX_groups_2026-03-18_g1",
        ],
        "solver": "frame3",
    },
    {
        "name": "Frame 4 (slide11 q3 coordinated tempo)",
        "filename": "slide11_q3_coordinated_tempo.png",
        "candidates": [
            # Listed candidates first (per instructions): try priority order
            "fs_LYON_vs_LOUD_groups_2026-03-17_g1",
            "fs_G2_vs_BLG_finals_2026-03-22_g3",
            "fs_JDG_vs_GEN_groups_2026-03-17_g2",
            # Fallbacks discovered by scanning all games for the criterion
            # "3+ blue champs at the blue base in minute 5-9". The listed
            # candidates yielded at most 2 blue in the strict box.
            "fs_LYON_vs_LOUD_groups_2026-03-17_g2",
            "fs_JDG_vs_GEN_groups_2026-03-17_g3",
            "fs_G2_vs_GEN_semifinals_2026-03-21_g2",
            "fs_TSW_vs_G2_groups_2026-03-16_g2",
        ],
        "solver": "frame4",
    },
    {
        "name": "Frame 5 (slide11 q4 map asymmetry)",
        "filename": "slide11_q4_map_asymmetry.png",
        "candidates": [
            "fs_G2_vs_GEN_semifinals_2026-03-21_g1",
            "fs_JDG_vs_BLG_semifinals_2026-03-21_g2",
            "fs_LYON_vs_LOUD_groups_2026-03-17_g3",
        ],
        "solver": "frame5",
    },
]


def fmt_t(seconds: float) -> str:
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m:02d}:{s:02d}"


def main() -> None:
    picks = load_picks()
    winners = load_winners()
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    solvers: dict[str, Callable] = {
        "frame1": lambda games: solve_frame1(games, picks),
        "frame2": lambda games: solve_frame2(games, picks, winners),
        "frame3": lambda games: solve_frame3(games, picks),
        "frame4": lambda games: solve_frame4(games, picks),
        "frame5": lambda games: solve_frame5(games, picks),
    }

    for scen in SCENARIOS:
        print(f"\n=== {scen['name']} ===")
        result = solvers[scen["solver"]](scen["candidates"])
        if result is None:
            print(f"  [WARN] no candidate satisfied criterion across {scen['candidates']}")
            continue
        score, game, frame_idx, value, t = result
        print(f"  game     : {game}")
        print(f"  frame_idx: {frame_idx}")
        print(f"  in-game t: {fmt_t(t)} ({t:.0f}s)")
        print(f"  criterion: {value}")

        # Verify PNG existence; fall back to nearest available frame.
        avail = available_frames(game)
        chosen = nearest_available_frame(frame_idx, avail)
        if chosen is None:
            print(f"  [ERROR] no PNGs available for {game}")
            continue
        if chosen != frame_idx:
            print(f"  -> nearest available PNG frame: {chosen}")
        src = RAW_DIR / game / "minimap" / f"frame_{chosen:06d}.png"
        dst = CHARTS_DIR / scen["filename"]
        shutil.copy2(src, dst)
        print(f"  copied   : {src.relative_to(REPO_ROOT)} -> {dst.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
