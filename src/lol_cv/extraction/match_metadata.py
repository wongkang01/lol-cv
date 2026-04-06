"""
Fetch per-game match metadata (champion picks, roles, sides) from the
lolesports livestats API.

This metadata is critical for narrowing YOLO minimap detection from
170 possible champions down to the exact 10 played in each game,
eliminating misidentifications.

Requires game IDs from the lolesports VOD discovery API (stored in
``data/match_metadata.json`` by the ingestion script).
"""

from __future__ import annotations

import json
from pathlib import Path

import requests

from lol_cv.utils import setup_logger

logger = setup_logger("lol_cv.extraction.match_metadata")

LIVESTATS_URL = "https://feed.lolesports.com/livestats/v1/window"
API_KEY = "0TvQnueqKa5mxJntVWt0w4LpLfEkrV1Ta8rQBb9Z"
HEADERS = {"x-api-key": API_KEY}


class MatchMetadataFetcher:
    """Fetch champion picks and player info for tournament games.

    Example::

        fetcher = MatchMetadataFetcher()
        picks = fetcher.get_champion_picks("115570888977308436")
        # picks = {
        #     "blue_team": [
        #         {"champion": "Yorick", "role": "top", "player": "G2 BrokenBlade"},
        #         ...
        #     ],
        #     "red_team": [...],
        #     "blue_champions": ["Yorick", "JarvanIV", "Aurora", "Yunara", "Nami"],
        #     "red_champions": ["Gnar", "Poppy", "Annie", "Sivir", "Lulu"],
        #     "all_champions": ["Yorick", "JarvanIV", ..., "Lulu"],
        # }
    """

    def get_champion_picks(self, game_id: str) -> dict:
        """Fetch champion picks for a single game.

        Args:
            game_id: The lolesports game ID (e.g. ``"115570888977308436"``).

        Returns:
            Dict with ``blue_team``, ``red_team`` (lists of player dicts),
            ``blue_champions``, ``red_champions``, ``all_champions`` (lists
            of champion ID strings matching YOLO class names).
        """
        url = f"{LIVESTATS_URL}/{game_id}"
        logger.info("Fetching champion picks for game %s", game_id)

        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()

        data = resp.json()
        meta = data["gameMetadata"]

        blue_team = self._parse_team(meta["blueTeamMetadata"])
        red_team = self._parse_team(meta["redTeamMetadata"])

        blue_champs = [p["champion"] for p in blue_team]
        red_champs = [p["champion"] for p in red_team]

        return {
            "blue_team": blue_team,
            "red_team": red_team,
            "blue_champions": blue_champs,
            "red_champions": red_champs,
            "all_champions": blue_champs + red_champs,
        }

    @staticmethod
    def _parse_team(team_meta: dict) -> list[dict]:
        """Parse participant metadata into a clean list."""
        players = []
        for p in team_meta.get("participantMetadata", []):
            players.append({
                "champion": p["championId"],
                "role": p.get("role", "unknown"),
                "player": p.get("summonerName", ""),
                "participant_id": p.get("participantId"),
            })
        return players

    def get_all_picks(self, metadata_path: str | Path) -> dict[str, dict]:
        """Fetch champion picks for all games in match_metadata.json.

        Args:
            metadata_path: Path to the metadata JSON file created by
                the ingestion script (``data/match_metadata.json``).

        Returns:
            Dict mapping ``match_id`` (e.g. ``"fs_G2_vs_BLG_finals_g1"``)
            to champion picks dict.
        """
        with open(metadata_path) as f:
            metadata = json.load(f)

        all_picks: dict[str, dict] = {}

        for match in metadata.get("matches", []):
            for game in match.get("games", []):
                game_id = game.get("game_id")
                match_id = game.get("match_id")

                if not game_id or not match_id:
                    continue

                try:
                    picks = self.get_champion_picks(game_id)
                    picks["match_info"] = {
                        "team1": match["team1"]["code"],
                        "team2": match["team2"]["code"],
                        "winner": match["winner"],
                        "block": match["block"],
                        "date": match["date"],
                        "game_number": game["game_number"],
                    }
                    all_picks[match_id] = picks
                    logger.info(
                        "%s: %s vs %s",
                        match_id,
                        picks["blue_champions"],
                        picks["red_champions"],
                    )
                except Exception:
                    logger.exception("Failed to fetch picks for %s (game_id=%s)", match_id, game_id)

        logger.info("Fetched picks for %d/%d games",
                     len(all_picks),
                     sum(len(m.get("games", [])) for m in metadata.get("matches", [])))
        return all_picks

    @staticmethod
    def save_picks(picks: dict[str, dict], output_path: str | Path) -> None:
        """Save champion picks to a JSON file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(picks, f, indent=2)
        logger.info("Saved champion picks to %s", output_path)

    @staticmethod
    def load_picks(picks_path: str | Path) -> dict[str, dict]:
        """Load previously saved champion picks."""
        with open(picks_path) as f:
            return json.load(f)
