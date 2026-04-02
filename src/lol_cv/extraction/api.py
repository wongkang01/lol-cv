"""
Riot API client — OPTIONAL utility, not part of the core CV pipeline.

The primary data source is tournament VODs (2026 First Stand), which are
played on the tournament realm and have NO API data available. This client
is retained for potential future use (e.g. validating CV accuracy against
ranked solo queue matches where both data sources exist).

Access tiers:
    - Development key: 20 req/s, 100 req/2min — expires every 24h.
    - Personal key: same limits, no expiry — ideal for university projects.
    - Production key: ~300 req/s — requires hosted app with ToS.

Match-v5 endpoints use regional routing (americas, europe, asia, sea).
Other endpoints (summoner, league) use platform routing (na1, euw1, kr, etc.).
"""

import os
import time

import requests

from lol_cv.utils import setup_logger

logger = setup_logger("lol_cv.extraction.api")

# Regional routing for match-v5 endpoints
REGIONAL_HOSTS = {
    "na1": "americas", "br1": "americas", "la1": "americas", "la2": "americas",
    "euw1": "europe", "eun1": "europe", "tr1": "europe", "ru": "europe",
    "kr": "asia", "jp1": "asia",
    "oc1": "sea", "ph2": "sea", "sg2": "sea", "th2": "sea", "tw2": "sea", "vn2": "sea",
}


class RiotApiClient:
    """Lightweight wrapper around Riot Games API for match data."""

    def __init__(self, api_key: str = None, platform: str = "euw1"):
        """
        Args:
            api_key: Riot API key. Falls back to RIOT_API_KEY env var.
            platform: Platform routing value (e.g. 'euw1', 'na1', 'kr').
        """
        self.api_key = api_key or os.getenv("RIOT_API_KEY")
        self.platform = platform
        self.region = REGIONAL_HOSTS.get(platform, "europe")
        self._session = requests.Session()
        self._session.headers["X-Riot-Token"] = self.api_key or ""

    @property
    def _regional_url(self) -> str:
        return f"https://{self.region}.api.riotgames.com"

    @property
    def _platform_url(self) -> str:
        return f"https://{self.platform}.api.riotgames.com"

    def _get(self, url: str, params: dict = None) -> dict:
        """Make a GET request with rate-limit retry."""
        for attempt in range(4):
            resp = self._session.get(url, params=params)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 2 ** attempt))
                logger.warning("Rate limited, retrying in %ds", retry_after)
                time.sleep(retry_after)
                continue
            resp.raise_for_status()
        raise requests.HTTPError(f"Failed after 4 retries: {url}")

    # ── Match-v5 (regional routing) ──────────────────────────────────

    def get_match(self, match_id: str) -> dict:
        """Fetch full match data by match ID.

        Returns dict with keys: metadata, info (participants, teams, outcome).
        """
        url = f"{self._regional_url}/lol/match/v5/matches/{match_id}"
        logger.info("Fetching match %s", match_id)
        return self._get(url)

    def get_match_timeline(self, match_id: str) -> dict:
        """Fetch match timeline — per-minute frames with gold/XP/position + events.

        The timeline provides 1-minute interval snapshots of each participant's
        position, gold, XP, and level, plus discrete events (kills, objectives).
        CV-extracted data at per-second resolution complements this.
        """
        url = f"{self._regional_url}/lol/match/v5/matches/{match_id}/timeline"
        logger.info("Fetching timeline for %s", match_id)
        return self._get(url)

    def get_match_ids(
        self, puuid: str, count: int = 20, queue: int = None, start: int = 0
    ) -> list[str]:
        """Fetch recent match IDs for a player.

        Args:
            puuid: Player's PUUID (globally unique).
            count: Number of match IDs to return (max 100).
            queue: Queue type filter (420 = Ranked Solo, 440 = Ranked Flex).
            start: Pagination offset.

        Returns:
            List of match ID strings (e.g. ['EUW1_1234567890', ...]).
        """
        url = f"{self._regional_url}/lol/match/v5/matches/by-puuid/{puuid}/ids"
        params = {"count": min(count, 100), "start": start}
        if queue is not None:
            params["queue"] = queue
        logger.info("Fetching %d match IDs for puuid=%s...", count, puuid[:8])
        return self._get(url, params=params)

    # ── Summoner / Account (platform routing) ────────────────────────

    def get_account_by_riot_id(self, game_name: str, tag_line: str) -> dict:
        """Look up a player's PUUID by Riot ID (e.g. 'Player#EUW').

        Uses the account-v1 endpoint on regional routing.
        """
        url = (
            f"{self._regional_url}/riot/account/v1"
            f"/accounts/by-riot-id/{game_name}/{tag_line}"
        )
        logger.info("Looking up %s#%s", game_name, tag_line)
        return self._get(url)

    def get_summoner_by_puuid(self, puuid: str) -> dict:
        """Fetch summoner profile (level, icon) by PUUID."""
        url = f"{self._platform_url}/lol/summoner/v4/summoners/by-puuid/{puuid}"
        return self._get(url)
