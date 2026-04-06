"""
Ingest First Stand 2026 tournament VODs (Groups → Finals, Mar 16-22).

Only the 8 teams that actually attended First Stand are included, and only
matches dated Mar 16-22 (the tournament window). This excludes regional
finals (LCK, CBLOL, LLA, LCS etc.) that happen to share block names.

Usage:
    python scripts/ingest_vods.py                    # Download all 45 games
    python scripts/ingest_vods.py --test             # Download 2 test games only
    python scripts/ingest_vods.py --teams G2 BLG     # Filter to specific teams
    python scripts/ingest_vods.py --dry-run           # Print batch list without downloading
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lol_cv.extraction.vod_discovery import VodDiscovery
from lol_cv.extraction.vod_processor import VodProcessor

# The 8 teams that actually attended the 2026 First Stand tournament.
FIRST_STAND_TEAMS = {
    "G2", "BLG", "BFX", "TSW", "GEN", "JDG", "LYON", "LOUD",
}

# First Stand 2026 tournament dates: Groups (Mar 16-20), Semis (Mar 21), Finals (Mar 22).
FIRST_STAND_START = "2026-03-16"
FIRST_STAND_END = "2026-03-22"

OUTPUT_DIR = Path("data/raw")
METADATA_PATH = Path("data/match_metadata.json")


def discover_matches(teams: set[str] | None = None) -> list[dict]:
    """Discover First Stand 2026 matches (date-filtered, team-filtered)."""
    discovery = VodDiscovery()
    # Include all blocks — we filter by date instead, so regional finals
    # happening on other dates with the same block name are excluded.
    matches = discovery.get_matches(stage="all")

    filter_teams = teams or FIRST_STAND_TEAMS
    first_stand = [
        m for m in matches
        if FIRST_STAND_START <= m["date"] <= FIRST_STAND_END
        and m["team1"]["code"] in filter_teams
        and m["team2"]["code"] in filter_teams
    ]

    print(discovery.summary(first_stand))
    return first_stand


def save_metadata(batch: list[dict], matches: list[dict]) -> None:
    """Save match metadata alongside the VODs for later use in the pipeline."""
    metadata = {
        "tournament": "First Stand 2026",
        "stage": "Knockouts → Finals",
        "matches": [],
    }

    for match in matches:
        match_entry = {
            "match_id": match["match_id"],
            "team1": match["team1"],
            "team2": match["team2"],
            "winner": match["winner"],
            "block": match["block"],
            "date": match["date"],
            "format": match["format"],
            "games": [],
        }
        for game in match["games"]:
            vod = game["vod"]
            game_entry = {
                "game_number": game["game_number"],
                "game_id": game["game_id"],
                "match_id": f"{match['match_id']}_g{game['game_number']}",
                "video_url": vod["url"],
                "start_time": vod["start_time"],
                "end_time": vod["end_time"],
                "duration_seconds": vod["duration"],
            }
            match_entry["games"].append(game_entry)
        metadata["matches"].append(match_entry)

    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved to {METADATA_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Ingest First Stand 2026 VODs")
    parser.add_argument("--test", action="store_true", help="Download only 2 test games")
    parser.add_argument("--teams", nargs="+", help="Filter to specific team codes")
    parser.add_argument("--dry-run", action="store_true", help="Print batch list only")
    args = parser.parse_args()

    teams = set(args.teams) if args.teams else None
    matches = discover_matches(teams=teams)

    discovery = VodDiscovery()
    batch = discovery.to_batch_list(matches)

    if args.test:
        # Pick 2 games: one group, one final
        test_ids = {
            "fs_G2_vs_BLG_finals_2026-03-22_g1",
            "fs_G2_vs_GEN_semifinals_2026-03-21_g1",
        }
        batch = [e for e in batch if e["match_id"] in test_ids]
        print(f"\n[TEST MODE] Processing {len(batch)} games only")

    if args.dry_run:
        print(f"\n{'='*70}")
        print(f"DRY RUN — {len(batch)} games would be downloaded:")
        print(f"{'='*70}")
        total_dur = 0
        for e in batch:
            meta = e["metadata"]
            dur = (e["end_time"] - e["start_time"]) if e["start_time"] and e["end_time"] else 0
            total_dur += dur
            print(
                f"  {e['match_id']:55s} | {meta['team1']:5s} vs {meta['team2']:5s} | "
                f"{dur//60}m | {meta['block']}"
            )
        print(f"\nTotal gameplay: {total_dur//3600}h {(total_dur%3600)//60}m")
        print(f"Estimated storage: ~{len(batch) * 250}MB video + ~{total_dur * 3 // 1024}MB frames")
        save_metadata(batch, matches)
        return

    # Save metadata before starting downloads.
    save_metadata(batch, matches)

    # Download and process.
    print(f"\n{'='*70}")
    print(f"Starting download of {len(batch)} games...")
    print(f"{'='*70}\n")

    processor = VodProcessor()
    results = processor.process_batch(batch, output_base_dir=str(OUTPUT_DIR))

    print(f"\nDone! {len(results)}/{len(batch)} games processed successfully.")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
