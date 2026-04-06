"""
Fetch per-game winners from the lolesports livestats API.

For each game in ``data/match_metadata.json``:
    1. Look up the game id
    2. Query getEventDetails to find which team_id was on blue/red side
    3. Query livestats window endpoint with a future startingTime to get
       the end-of-game frame
    4. Determine winner from inhibitors / kills / gold

Output: ``data/game_winners.json``::

    {
      "fs_BFX_vs_BLG_groups_2026-03-16_g1": {
          "winner_side": "red",            # 'blue' or 'red'
          "winner_team_code": "BLG",
          "blue_team_code": "BFX",
          "red_team_code": "BLG",
          "duration_seconds": 1760,
          "blue_kills": 9, "red_kills": 29,
          "blue_gold": 54706, "red_gold": 66623,
          "blue_inhibs": 0, "red_inhibs": 1
      },
      ...
    }
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("fetch_game_winners")

REPO_ROOT = Path(__file__).resolve().parents[1]
META_PATH = REPO_ROOT / "data" / "match_metadata.json"
OUT_PATH = REPO_ROOT / "data" / "game_winners.json"

# NOTE on authoritative winner source:
# The lolesports `getEventDetails` / `getGames` endpoints expose per-game
# `state: "completed"` but NO per-game winner field. The match-level
# `teams[].result.gameWins` only aggregates the series score. Similarly the
# livestats `window/{gameId}` feed's last frame only contains team stats
# (inhibitors, kills, gold) — there is no `winningTeam` flag.
#
# Because no authoritative field exists, we derive the winner heuristically
# from end-of-game team stats, ordered by reliability:
#   1. Inhibitors destroyed — effectively authoritative in LoL: a team can
#      only win by breaking the enemy nexus, which requires ≥1 inhibitor
#      down. If inhibitor counts differ, confidence = HIGH.
#   2. Total kills — strong proxy, used only when inhibs are tied (rare,
#      e.g. early surrender). confidence = MEDIUM.
#   3. Total gold — weakest tiebreaker, confidence = LOW. Logged as a
#      warning because it should essentially never fire in finished games.

API_KEY = "0TvQnueqKa5mxJntVWt0w4LpLfEkrV1Ta8rQBb9Z"
HEADERS = {"x-api-key": API_KEY}
BASE = "https://esports-api.lolesports.com/persisted/gw"
LIVESTATS = "https://feed.lolesports.com/livestats/v1/window"


def get_match_event_details(match_id: str) -> dict:
    r = requests.get(
        f"{BASE}/getEventDetails",
        headers=HEADERS,
        params={"hl": "en-US", "id": match_id},
        timeout=15,
    )
    r.raise_for_status()
    return r.json()


def fetch_game_end_state(game_id: str, date: str) -> dict | None:
    """Query livestats with a startingTime 1 day after `date` to get final frames."""
    # date is "2026-03-16" — pick the next day at 00:00 UTC
    next_day = (
        datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        + timedelta(days=1)
    ).strftime("%Y-%m-%dT00:00:00.000Z")

    r = requests.get(
        f"{LIVESTATS}/{game_id}",
        headers=HEADERS,
        params={"startingTime": next_day},
        timeout=15,
    )
    if r.status_code != 200:
        return None
    data = r.json()
    frames = data.get("frames", [])
    if not frames:
        return None
    last = frames[-1]
    return {
        "gameState": last.get("gameState"),
        "blue": last.get("blueTeam", {}),
        "red": last.get("redTeam", {}),
    }


def determine_winner(end_state: dict) -> str | None:
    """Decide winner from end-of-game stats. Returns 'blue', 'red', or None.

    Uses a cascade of tiebreakers: inhibitors > kills > gold. See
    ``determine_winner_with_confidence`` for the same result annotated
    with a confidence label (HIGH / MEDIUM / LOW).
    """
    winner, _ = determine_winner_with_confidence(end_state)
    return winner


def determine_winner_with_confidence(
    end_state: dict, game_match_id: str | None = None
) -> tuple[str | None, str | None]:
    """Decide winner from end-of-game stats and attach a confidence label.

    Returns:
        Tuple ``(winner_side, confidence)`` where ``winner_side`` is
        ``'blue'``, ``'red'``, or ``None`` and ``confidence`` is ``'HIGH'``,
        ``'MEDIUM'``, ``'LOW'``, or ``None``.

        - HIGH   — inhibitor counts differ (the loser had at least one
                   inhibitor destroyed, which is effectively required to
                   break the nexus)
        - MEDIUM — inhibs tied but kill diff > 5
        - LOW    — all other ties, falling back to gold (should be rare)
    """
    if not end_state:
        return None, None
    bi = end_state["blue"].get("inhibitors") or 0
    ri = end_state["red"].get("inhibitors") or 0
    bk = end_state["blue"].get("totalKills") or 0
    rk = end_state["red"].get("totalKills") or 0
    bg = end_state["blue"].get("totalGold") or 0
    rg = end_state["red"].get("totalGold") or 0

    if bi != ri:
        return ("blue" if bi > ri else "red"), "HIGH"

    # Tiebreaker 1: kills — medium confidence if the diff is > 5
    if bk != rk:
        confidence = "MEDIUM" if abs(bk - rk) > 5 else "LOW"
        if confidence == "LOW":
            logger.warning(
                "[low-conf] %s — inhibs tied (%d=%d), small kill diff %d-%d",
                game_match_id or "?", bi, ri, bk, rk,
            )
        return ("blue" if bk > rk else "red"), confidence

    # Tiebreaker 2: gold-only (should almost never fire for finished games)
    if bg != rg:
        logger.warning(
            "[low-conf] %s — falling back to gold-only tiebreaker "
            "(inhibs=%d/%d kills=%d/%d gold=%d/%d)",
            game_match_id or "?", bi, ri, bk, rk, bg, rg,
        )
        return ("blue" if bg > rg else "red"), "LOW"

    logger.warning(
        "[no-winner] %s — all tiebreakers tied", game_match_id or "?"
    )
    return None, None


def main() -> None:
    with open(META_PATH) as f:
        meta = json.load(f)

    out = {}
    failed = []

    # Group games by parent match for efficient API queries
    for match in meta["matches"]:
        # Parent match id is on the games' game_id attributes — but we need
        # the lolesports match event id, not our derived match_id. Get it
        # from the first game (game_ids share the parent prefix... actually no).
        # Just iterate games and query event details once per match.
        team1_code = match["team1"]["code"]
        team2_code = match["team2"]["code"]

        # Use the FIRST game id minus 1 to derive parent? Cheaper: iterate
        # games and query getEventDetails for each match.
        # The match parent id can be obtained from the livestats response
        # under esportsMatchId. Get one game first to find it.
        games = match.get("games", [])
        if not games:
            continue

        first_game_id = games[0]["game_id"]
        try:
            r = requests.get(
                f"{LIVESTATS}/{first_game_id}", headers=HEADERS, timeout=15
            )
            esports_match_id = r.json().get("esportsMatchId")
        except Exception as e:
            print(f"[skip match] {match['match_id']} — {e}")
            continue

        if not esports_match_id:
            print(f"[skip match] {match['match_id']} — no esportsMatchId")
            continue

        try:
            details = get_match_event_details(esports_match_id)
        except Exception as e:
            print(f"[skip match] {match['match_id']} — getEventDetails {e}")
            continue

        # team_id → code
        team_lookup: dict[str, str] = {}
        for t in details["data"]["event"]["match"]["teams"]:
            team_lookup[t["id"]] = t["code"]

        api_games = {str(g["number"]): g for g in details["data"]["event"]["match"]["games"]}

        for game in games:
            game_num = str(game["game_number"])
            game_match_id = game["match_id"]
            api_game = api_games.get(game_num)
            if not api_game:
                print(f"[fail] {game_match_id} — no api game {game_num}")
                failed.append(game_match_id)
                continue

            # blue/red team code for this game
            sides = {t["side"]: team_lookup.get(t["id"], "?") for t in api_game["teams"]}
            blue_code = sides.get("blue", "?")
            red_code = sides.get("red", "?")

            end_state = fetch_game_end_state(game["game_id"], match["date"])
            if not end_state:
                print(f"[fail] {game_match_id} — no end state")
                failed.append(game_match_id)
                continue
            if end_state["gameState"] != "finished":
                print(f"[warn] {game_match_id} — gameState={end_state['gameState']}")

            winner_side, confidence = determine_winner_with_confidence(
                end_state, game_match_id
            )
            winner_code = blue_code if winner_side == "blue" else red_code if winner_side == "red" else "?"

            out[game_match_id] = {
                "winner_side": winner_side,
                "winner_team_code": winner_code,
                "winner_confidence": confidence,
                "blue_team_code": blue_code,
                "red_team_code": red_code,
                "blue_kills": end_state["blue"].get("totalKills"),
                "red_kills": end_state["red"].get("totalKills"),
                "blue_gold": end_state["blue"].get("totalGold"),
                "red_gold": end_state["red"].get("totalGold"),
                "blue_inhibs": end_state["blue"].get("inhibitors"),
                "red_inhibs": end_state["red"].get("inhibitors"),
                "duration_seconds": game.get("duration_seconds"),
                "game_state": end_state["gameState"],
            }
            print(
                f"[ok]   {game_match_id}  {blue_code} (blue) vs {red_code} (red) "
                f"→ {winner_code} [{confidence}]  "
                f"({end_state['blue'].get('totalKills')}-{end_state['red'].get('totalKills')})"
            )

    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {len(out)} game winners to {OUT_PATH}")
    if failed:
        print(f"FAILED ({len(failed)}): {failed}")


if __name__ == "__main__":
    main()
