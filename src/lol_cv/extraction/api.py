"""
Riot API client for supplementary match data.

Used to enrich CV-extracted data with gold/XP timelines,
match outcomes, and event logs that the API provides.
"""

import os


class RiotApiClient:
    """Lightweight wrapper around Riot Games API for match data."""

    BASE_URL = "https://americas.api.riotgames.com"

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("RIOT_API_KEY")

    def get_match(self, match_id: str) -> dict:
        """Fetch full match data by match ID."""
        # TODO: Implement API call
        raise NotImplementedError

    def get_match_timeline(self, match_id: str) -> dict:
        """Fetch match timeline (events with timestamps)."""
        # TODO: Implement API call
        raise NotImplementedError

    def get_match_ids(self, puuid: str, count: int = 20) -> list:
        """Fetch recent match IDs for a player."""
        # TODO: Implement API call
        raise NotImplementedError
