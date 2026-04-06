"""Quick OCR sanity check on 10 evenly-sampled frames from one knockout game."""

from __future__ import annotations

from pathlib import Path

from scripts.run_ocr import (
    get_reader, ocr_text, ocr_left_right,
    parse_gold, parse_timer,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def validate(game_dir: Path, n_samples: int = 10) -> None:
    hud = game_dir / "hud"
    timer_frames = sorted((hud / "timer").glob("frame_*.png"))
    if not timer_frames:
        print("no timer frames")
        return

    step = max(1, len(timer_frames) // n_samples)
    sample = timer_frames[::step][:n_samples]
    print(f"\n{game_dir.name} ({len(timer_frames)} frames, sampling {len(sample)})")
    print("=" * 80)
    print(f"{'frame':>20} {'timer':>10} {'kill':>10} {'b_gold':>10} {'r_gold':>10}")
    print("-" * 80)

    for tp in sample:
        timer_text = ocr_text(tp, "0123456789:")
        gt, _ = parse_timer(timer_text)

        bk, rk = ocr_left_right(hud / "kill_score" / tp.name)

        bg = parse_gold(ocr_text(hud / "blue_gold" / tp.name, "0123456789.kK"))
        rg = parse_gold(ocr_text(hud / "red_gold" / tp.name, "0123456789.kK"))

        kill_str = f"{bk}-{rk}"
        print(f"{tp.name:>20} {str(gt):>10} {kill_str:>10} {str(bg):>10} {str(rg):>10}")


if __name__ == "__main__":
    get_reader()
    for name in [
        "fs_G2_vs_BLG_finals_2026-03-22_g1",
        "fs_G2_vs_GEN_semifinals_2026-03-21_g1",
        "fs_JDG_vs_BLG_semifinals_2026-03-21_g1",
    ]:
        d = REPO_ROOT / "data" / "raw" / name
        if d.exists():
            validate(d)
