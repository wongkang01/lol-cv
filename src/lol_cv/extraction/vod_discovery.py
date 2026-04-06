"""
Automated VOD discovery via the lolesports.com API.

Queries the public esports API to retrieve YouTube VOD URLs and exact
gameplay timestamps for every game in a tournament.  Generates batch
lists ready for :class:`VodProcessor.process_batch`.

The API key is public — it is hardcoded in the lolesports.com web
client and requires no authentication.
"""

from __future__ import annotations

import requests

from lol_cv.utils import setup_logger

logger = setup_logger("lol_cv.extraction.vod_discovery")

BASE_URL = "https://esports-api.lolesports.com/persisted/gw"
API_KEY = "0TvQnueqKa5mxJntVWt0w4LpLfEkrV1Ta8rQBb9Z"
HEADERS = {"x-api-key": API_KEY}

# First Stand 2026 league ID.
FIRST_STAND_LEAGUE_ID = "113464388705111224"

# International stage block names (excludes regional qualifiers).
INTERNATIONAL_BLOCKS = {"Groups", "Knockouts", "Semifinals", "Finals"}


class VodDiscovery:
    """Discover tournament VODs and generate processing batch lists.

    Example usage::

        discovery = VodDiscovery()
        matches = discovery.get_matches(stage="international")
        batch = discovery.to_batch_list(matches)
        # Feed directly into VodProcessor.process_batch(batch)
    """

    def __init__(self, league_id: str = FIRST_STAND_LEAGUE_ID):
        self.league_id = league_id
        self._raw_events: list[dict] | None = None

    # ── API queries ────────────────────────────────────────────────

    def _fetch_vods(self) -> list[dict]:
        """Fetch all VOD events from the lolesports API.

        Returns:
            List of raw event dicts from the API response.
        """
        if self._raw_events is not None:
            return self._raw_events

        url = f"{BASE_URL}/getVods"
        params = {"hl": "en-US", "id": self.league_id}

        logger.info("Fetching VODs for league %s", self.league_id)
        resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
        resp.raise_for_status()

        data = resp.json()
        self._raw_events = data["data"]["schedule"]["events"]
        logger.info("Fetched %d events", len(self._raw_events))
        return self._raw_events

    # ── Match parsing ──────────────────────────────────────────────

    def get_matches(
        self,
        stage: str | None = "international",
        blocks: set[str] | None = None,
        teams: set[str] | None = None,
    ) -> list[dict]:
        """Parse API events into structured match/game dicts.

        Args:
            stage: Shorthand filter. ``"international"`` selects only
                Groups, Knockouts, Semifinals, Finals.  ``"all"`` or
                ``None`` returns everything.
            blocks: Explicit set of block names to include.  Overrides
                *stage* if provided.
            teams: If provided, only include matches where at least one
                team code is in this set (case-sensitive).

        Returns:
            List of match dicts, each containing::

                {
                    "match_id": "fs_G2_vs_BLG_finals",
                    "block": "Finals",
                    "date": "2026-03-22",
                    "team1": {"code": "G2", "name": "G2 Esports", "wins": 1},
                    "team2": {"code": "BLG", "name": "Bilibili Gaming", "wins": 3},
                    "winner": "team2",
                    "format": "bestOf5",
                    "games": [
                        {
                            "game_number": 1,
                            "game_id": "115570888977308436",
                            "vod": {
                                "video_id": "I4uLe1RegxQ",
                                "url": "https://www.youtube.com/watch?v=I4uLe1RegxQ",
                                "start_time": 5,
                                "end_time": 2755,
                                "platform": "youtube",
                            },
                        },
                        ...
                    ],
                }
        """
        events = self._fetch_vods()

        allowed_blocks = blocks
        if allowed_blocks is None and stage == "international":
            allowed_blocks = INTERNATIONAL_BLOCKS

        matches = []
        for event in events:
            # Filter by block.
            block_name = event.get("blockName", "")
            if allowed_blocks and block_name not in allowed_blocks:
                continue

            # Must have games with VODs.
            games_raw = event.get("games", [])
            if not any(g.get("vods") for g in games_raw):
                continue

            # Parse teams.
            teams_raw = event["match"]["teams"]
            t1 = teams_raw[0]
            t2 = teams_raw[1]

            # Filter by team.
            if teams:
                t1_code = t1.get("code", "")
                t2_code = t2.get("code", "")
                if t1_code not in teams and t2_code not in teams:
                    continue

            match = self._parse_match(event, t1, t2, block_name)
            if match["games"]:  # Only include if at least one game has a usable VOD
                matches.append(match)

        logger.info(
            "Found %d matches (%d total games)",
            len(matches),
            sum(len(m["games"]) for m in matches),
        )
        return matches

    def _parse_match(
        self, event: dict, t1: dict, t2: dict, block_name: str
    ) -> dict:
        """Parse a single API event into a structured match dict."""
        t1_code = t1.get("code", "UNK")
        t2_code = t2.get("code", "UNK")
        t1_wins = t1.get("result", {}).get("gameWins", 0)
        t2_wins = t2.get("result", {}).get("gameWins", 0)

        if t1_wins > t2_wins:
            winner = "team1"
        elif t2_wins > t1_wins:
            winner = "team2"
        else:
            winner = None

        strategy = event["match"].get("strategy", {})
        fmt = f"{strategy.get('type', 'bestOf')}{strategy.get('count', '?')}"

        date_str = event.get("startTime", "")[:10]

        match_id = f"fs_{t1_code}_vs_{t2_code}_{block_name.lower().replace(' ', '_')}_{date_str}"

        # Prefer a single broadcast video ID that covers every game in the
        # match. This way a best-of-5 is downloaded once, not 5 times.
        shared_video_id = self._find_shared_video_id(event.get("games", []))

        games = []
        for gi, game_raw in enumerate(event.get("games", [])):
            vod = self._pick_best_vod(
                game_raw.get("vods", []),
                preferred_video_id=shared_video_id,
            )
            if vod:
                games.append({
                    "game_number": gi + 1,
                    "game_id": game_raw.get("id", ""),
                    "vod": vod,
                })

        return {
            "match_id": match_id,
            "block": block_name,
            "date": date_str,
            "team1": {"code": t1_code, "name": t1.get("name", t1_code), "wins": t1_wins},
            "team2": {"code": t2_code, "name": t2.get("name", t2_code), "wins": t2_wins},
            "winner": winner,
            "format": fmt,
            "games": games,
        }

    @staticmethod
    def _find_shared_video_id(games: list[dict]) -> str | None:
        """Find a YouTube video ID that has timestamps in every game of a match.

        Best-of-N broadcasts are usually uploaded as one long video covering
        all games. When the same video ID appears with timestamps across every
        game of a match, we should use it for all of them — downloading one
        file instead of many.

        When multiple video IDs cover every game, pick the one whose first
        game starts earliest (smallest ``startMillis`` in game 1).  This
        favours per-match trimmed clips (G1 at ~0s) over full-day broadcasts
        (G1 at ~30 min into the stream), minimising download size and skipping
        pre-match content.

        Args:
            games: List of raw game dicts from the API (each with ``vods``).

        Returns:
            Shared video ID, or ``None`` if no single video covers all games.
        """
        per_game_ids: list[set[str]] = []
        for game_raw in games:
            ids = set()
            for v in game_raw.get("vods", []):
                param = v.get("parameter", "")
                if param and not param.isdigit() and v.get("startMillis") is not None:
                    ids.add(param)
            if ids:
                per_game_ids.append(ids)

        if not per_game_ids:
            return None

        # Intersection of all per-game sets = video IDs covering every game.
        shared = set.intersection(*per_game_ids)
        if not shared:
            return None

        # Among shared IDs, pick the one with the earliest G1 start time.
        game1_vods = games[0].get("vods", []) if games else []
        first_starts: dict[str, int] = {
            v["parameter"]: v["startMillis"]
            for v in game1_vods
            if v.get("parameter") in shared and v.get("startMillis") is not None
        }
        if first_starts:
            return min(first_starts, key=first_starts.get)

        return sorted(shared)[0]

    @staticmethod
    def _pick_best_vod(
        vods: list[dict], preferred_video_id: str | None = None
    ) -> dict | None:
        """Select the best VOD for a game from the available options.

        Priority:
            1. The *preferred_video_id* if present (shared broadcast ID).
            2. YouTube with timestamps (prefer earliest start = per-game clip).
            3. YouTube without timestamps (standalone per-game uploads).
            4. Twitch (less accessible for automated download).

        Returns:
            VOD info dict, or ``None`` if no usable VOD found.
        """
        yt_with_ts = []
        yt_without_ts = []
        preferred_match: dict | None = None

        for v in vods:
            param = v.get("parameter", "")
            if not param:
                continue

            is_youtube = not param.isdigit()
            has_timestamps = v.get("startMillis") is not None

            # Short-circuit to the preferred broadcast if it's present.
            if (
                preferred_video_id is not None
                and param == preferred_video_id
                and has_timestamps
            ):
                start_s = int(v["startMillis"] / 1000)
                end_s = int(v["endMillis"] / 1000)
                preferred_match = {
                    "video_id": param,
                    "url": f"https://www.youtube.com/watch?v={param}",
                    "start_time": start_s,
                    "end_time": end_s,
                    "duration": end_s - start_s,
                    "platform": "youtube",
                }
                continue

            if is_youtube and has_timestamps:
                start_s = int(v["startMillis"] / 1000)
                end_s = int(v["endMillis"] / 1000)
                duration = end_s - start_s
                yt_with_ts.append({
                    "video_id": param,
                    "url": f"https://www.youtube.com/watch?v={param}",
                    "start_time": start_s,
                    "end_time": end_s,
                    "duration": duration,
                    "platform": "youtube",
                })
            elif is_youtube:
                yt_without_ts.append({
                    "video_id": param,
                    "url": f"https://www.youtube.com/watch?v={param}",
                    "start_time": None,
                    "end_time": None,
                    "duration": None,
                    "platform": "youtube",
                })

        if preferred_match is not None:
            return preferred_match

        if yt_with_ts:
            # Prefer the VOD with the earliest start time — likely a
            # per-game clip rather than a full broadcast.
            yt_with_ts.sort(key=lambda v: v["start_time"])
            return yt_with_ts[0]

        if yt_without_ts:
            return yt_without_ts[0]

        return None

    # ── Batch list generation ──────────────────────────────────────

    def to_batch_list(self, matches: list[dict]) -> list[dict]:
        """Convert parsed matches into a list for VodProcessor.process_batch.

        Each game becomes one entry in the batch list.

        Returns:
            List of dicts ready for ``VodProcessor.process_batch``::

                [
                    {
                        "url": "https://www.youtube.com/watch?v=...",
                        "match_id": "fs_G2_vs_BLG_finals_2026-03-22_g1",
                        "start_time": 5,
                        "end_time": 2755,
                        "metadata": { ... },
                    },
                    ...
                ]
        """
        batch = []
        for match in matches:
            for game in match["games"]:
                vod = game["vod"]
                game_id = f"{match['match_id']}_g{game['game_number']}"

                entry = {
                    "url": vod["url"],
                    "match_id": game_id,
                    "start_time": vod["start_time"],
                    "end_time": vod["end_time"],
                    "metadata": {
                        "team1": match["team1"]["code"],
                        "team2": match["team2"]["code"],
                        "winner": match["winner"],
                        "block": match["block"],
                        "date": match["date"],
                        "game_number": game["game_number"],
                        "format": match["format"],
                    },
                }
                batch.append(entry)

        logger.info("Generated batch list with %d games", len(batch))
        return batch

    # ── Convenience methods ────────────────────────────────────────

    def discover_and_prepare(
        self,
        stage: str | None = "international",
        teams: set[str] | None = None,
    ) -> list[dict]:
        """One-call method: fetch, parse, and return a batch list.

        Args:
            stage: ``"international"`` or ``"all"``.
            teams: Optional team filter.

        Returns:
            Batch list ready for ``VodProcessor.process_batch``.
        """
        matches = self.get_matches(stage=stage, teams=teams)
        return self.to_batch_list(matches)

    def summary(self, matches: list[dict] | None = None) -> str:
        """Print a human-readable summary of discovered matches.

        Args:
            matches: Pre-fetched matches, or ``None`` to fetch fresh.

        Returns:
            Formatted summary string.
        """
        if matches is None:
            matches = self.get_matches()

        lines = []
        total_games = 0
        total_duration = 0

        for m in matches:
            t1 = m["team1"]["code"]
            t2 = m["team2"]["code"]
            w = m["winner"]
            winner_str = t1 if w == "team1" else t2 if w == "team2" else "?"
            n_games = len(m["games"])
            total_games += n_games

            game_strs = []
            for g in m["games"]:
                d = g["vod"].get("duration")
                if d:
                    total_duration += d
                    game_strs.append(f"G{g['game_number']}({d//60}m)")
                else:
                    game_strs.append(f"G{g['game_number']}(?)")

            lines.append(
                f"  {m['block']:15s} {m['date']} | {t1:6s} vs {t2:6s} "
                f"({m['format']}) → {winner_str:6s} | {' '.join(game_strs)}"
            )

        header = (
            f"Tournament VOD Summary\n"
            f"{'='*70}\n"
            f"Matches: {len(matches)}  |  Games: {total_games}  |  "
            f"Total gameplay: {total_duration//3600}h {(total_duration%3600)//60}m\n"
            f"{'='*70}"
        )
        return header + "\n" + "\n".join(lines)
