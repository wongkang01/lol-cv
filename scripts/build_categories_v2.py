"""
One-shot builder for feature_categories_v2.csv.

Reads features_corrected.csv header, assigns every non-label column to one
of the 8 new strategic categories, writes data/processed/feature_categories_v2.csv
with columns (feature, category_v2, rationale).

NOTE: this is a mapping builder, NOT the ablation runner. The ablation lives
in scripts/rerun_categories_v2.py.
"""

from __future__ import annotations

import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FEATURES_CSV = REPO_ROOT / "data" / "processed" / "features_corrected.csv"
OUT_CSV = REPO_ROOT / "data" / "processed" / "feature_categories_v2.csv"


def _is_snapshot(col: str) -> bool:
    return col.startswith("sp_snap_")


def _snap_time(col: str) -> int | None:
    """Return the tXXX seconds for a snapshot feature, or None."""
    if not col.startswith("sp_snap_t"):
        return None
    try:
        tok = col.split("_")[2]  # e.g. 't300'
        return int(tok.lstrip("t"))
    except Exception:
        return None


def categorize(col: str) -> tuple[str, str]:
    """Return (category_v2, rationale) for a single feature column."""

    # ── 1. JUNGLE PATHING & INVASION ──────────────────────────────────
    # Early jungle commit, level-1 invade, pre-3min invade, scuttle contest,
    # jungle-quadrant / enemy-jungle occupancy, map-asymmetry (pathing mirror).
    if col.startswith("sp_early_") and (
        "jgl_commit" in col
        or "lvl1_invade" in col
        or "scuttle" in col
        or "vertical_jungling" in col
    ):
        return (
            "jungle_pathing_invasion",
            "Early jungle decision (commit side, lvl1 invade, scuttle contest) — core jungle pathing signal",
        )
    if col.startswith("sp_strat_") and ("pre3min_invade" in col or "map_asymmetry" in col):
        return (
            "jungle_pathing_invasion",
            "Pre-3min jungle invade / mirror-asymmetry — jungle pathing decisions in the first clear",
        )
    if col.startswith("sp_") and "jungle" in col and not col.startswith("sp_early_") and not col.startswith("sp_strat_"):
        return (
            "jungle_pathing_invasion",
            "Full-game jungle-quadrant occupancy — which jungle quadrants a team lived in",
        )

    # ── 2. OBJECTIVE CONTESTATION & CONTROL ───────────────────────────
    # Dragon/baron/herald grouping, convergence, pit occupancy, snapshot
    # dragon/baron quadrant counts (team is physically near the pit), and
    # objective-timer / count temporal features.
    if col.startswith("sp_") and any(k in col for k in ("dragon_grouped", "baron_grouped", "dragon_avg_near", "baron_avg_near")):
        return (
            "objective_contestation",
            "Team-grouping density near dragon/baron — direct objective setup",
        )
    if col.startswith("sp_") and ("dragon_convergence" in col or "baron_convergence" in col):
        return (
            "objective_contestation",
            "Speed of collapsing onto dragon/baron — objective execution tempo",
        )
    if col.startswith("sp_") and ("dragon_pit" in col or "baron_pit" in col):
        return (
            "objective_contestation",
            "Time spent inside dragon/baron pit — objective presence",
        )
    if _is_snapshot(col) and ("dragon_quadrant_count" in col or "baron_quadrant_count" in col):
        return (
            "objective_contestation",
            "Snapshot of how many champs are in the dragon/baron quadrant — objective posture",
        )
    # Temporal (tp_) objective timers and count buckets.
    if col.startswith("tp_first_") and any(o in col for o in ("dragon", "baron", "herald")):
        return (
            "objective_contestation",
            "First objective timing (dragon/baron/herald) — direct objective control signal",
        )
    if col.startswith("tp_") and any(
        col.endswith(f"_{period}_{obj}_count")
        for period in ("early", "mid", "late")
        for obj in ("dragon", "baron", "herald")
    ):
        return (
            "objective_contestation",
            "Objective counts per game phase — cumulative objective control",
        )

    # ── 3. LANE PRIORITY & PRESSURE ───────────────────────────────────
    # Lane occupancy, mid roam (lane pressure via roams), bot zoning depth
    # (lane pressure), first tower / tower counts (lane priority cashout).
    if col.startswith("sp_") and ("_top_lane" in col or "_mid_lane" in col or "_bot_lane" in col):
        return (
            "lane_priority_pressure",
            "Fraction of time spent in a lane — who held lane priority",
        )
    if col.startswith("sp_early_") and ("mid_first_roam" in col or "mid_roam_target" in col):
        return (
            "lane_priority_pressure",
            "Mid laner first roam — lane-pressure conversion into side-lane influence",
        )
    if col.startswith("sp_strat_") and "bot_zoning" in col:
        return (
            "lane_priority_pressure",
            "Bot lane zoning depth — lane-pressure proxy",
        )
    if col.startswith("tp_first_tower_time") or (col.startswith("tp_") and col.endswith("_tower_count")):
        return (
            "lane_priority_pressure",
            "Tower timings / counts — cashout of lane priority",
        )
    if col.startswith("tp_first_inhibitor_time") or (col.startswith("tp_") and col.endswith("_inhibitor_count")):
        return (
            "lane_priority_pressure",
            "Inhibitor timing / counts — lane-pressure endgame conversion",
        )

    # ── 4. MAP CONTROL / TERRITORIAL DOMINANCE ────────────────────────
    # Team centroid positions, enemy-half counts, red_base / blue_base
    # occupancy, zone transitions / variety (territory covered), avg rotations.
    if _is_snapshot(col) and ("centroid_x" in col or "centroid_y" in col or "enemy_half_count" in col):
        return (
            "map_control_territory",
            "Snapshot of team centroid / enemy-half presence — territorial control at tempo moment",
        )
    if col.startswith("sp_") and ("zone_blue_base" in col or "zone_red_base" in col):
        return (
            "map_control_territory",
            "Time spent in each team's base — forced-defence / siege territory",
        )
    if col.startswith("sp_") and ("total_transitions" in col or "avg_transitions" in col or "avg_zones_visited" in col):
        return (
            "map_control_territory",
            "Map coverage breadth — how much territory the team touched",
        )
    if col.startswith("sp_strat_") and "synced_recalls" in col:
        return (
            "map_control_territory",
            "Synced team recalls — coordinated repositioning to reset map control",
        )
    if col.startswith("tp_") and "avg_rotations" in col:
        return (
            "map_control_territory",
            "Per-team average rotations — macro movement across the map",
        )

    # ── 5. TEAM COORDINATION / GROUPING ───────────────────────────────
    # Generic (non-objective) team spread / pairwise-distance / grouped_pct.
    if col.startswith("sp_") and ("avg_grouping_dist" in col or "grouped_pct" in col):
        return (
            "team_coordination_grouping",
            "Full-game team spread / grouped-time — overall coordination",
        )
    if _is_snapshot(col) and ("grouping_dist" in col or "spread" in col):
        return (
            "team_coordination_grouping",
            "Snapshot team spread — instantaneous coordination at tempo moment",
        )

    # ── 6. TEMPO / EARLY–MID TRANSITION ───────────────────────────────
    # Kill-based tempo stats (first kill, kill counts per phase, tempo stats).
    if col == "tp_first_kill_time" or (col.startswith("tp_") and col.endswith("_kill_count")):
        return (
            "tempo_early_mid_transition",
            "Kill tempo — first-blood timing and skirmish counts per phase",
        )
    if col.startswith("tp_") and ("avg_tempo" in col or "max_tempo" in col or "tempo_variance" in col):
        return (
            "tempo_early_mid_transition",
            "Overall action tempo — rate / spikiness of events",
        )

    # ── 7. RIVER / VISION PRESENCE ────────────────────────────────────
    if col.startswith("sp_") and ("river_top" in col or "river_bot" in col):
        return (
            "river_vision_presence",
            "Time in top/bot river — river vision / scuttle lane presence",
        )

    # ── 8. GAME STATE (OCR + GOLD-DIFF DERIVATIVES) ───────────────────
    # OCR final scores and gold-diff slope features are outcome-adjacent
    # state signals, not strategy directly. Grouping them separately makes
    # the leakage pathway visible.
    if col.startswith("ocr_"):
        return (
            "game_state_ocr",
            "OCR-read game state (gold diff, kills, duration) — outcome-adjacent state signal",
        )
    if col.startswith("tp_gold_diff_slope"):
        return (
            "game_state_ocr",
            "Gold-diff slope per phase — derivative of OCR gold diff",
        )

    # Fallback — should not hit for the 213 real features, but keep it safe.
    return ("uncategorized", f"Fallback: no rule matched for {col}")


def main() -> None:
    with FEATURES_CSV.open() as f:
        header = next(csv.reader(f))

    cols = [c for c in header if c != "match_id"]

    rows = []
    for col in cols:
        cat, rationale = categorize(col)
        rows.append({"feature": col, "category_v2": cat, "rationale": rationale})

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["feature", "category_v2", "rationale"])
        writer.writeheader()
        writer.writerows(rows)

    # Summary printout.
    from collections import Counter
    counts = Counter(r["category_v2"] for r in rows)
    print(f"Wrote {len(rows)} feature mappings to {OUT_CSV}")
    for cat, n in counts.most_common():
        print(f"  {cat:35s} {n:3d}")

    unc = [r["feature"] for r in rows if r["category_v2"] == "uncategorized"]
    if unc:
        print("\nUNCATEGORIZED (BUG):")
        for f in unc:
            print("  ", f)


if __name__ == "__main__":
    main()
