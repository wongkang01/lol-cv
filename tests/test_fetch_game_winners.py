"""Unit tests for the ``determine_winner`` helper in ``scripts/fetch_game_winners``.

The helper decides the winning side of a League of Legends game from an
``end_state`` dictionary with the shape::

    {
        "blue": {"inhibitors": int, "totalKills": int, "totalGold": int, ...},
        "red":  {"inhibitors": int, "totalKills": int, "totalGold": int, ...},
    }

It returns ``"blue"``, ``"red"`` or ``None``. The tiebreak order is
inhibitors -> kills -> gold, and ``None``/missing fields are treated as
zero.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.fetch_game_winners import determine_winner  # noqa: E402


def _state(
    bi: int | None = 0,
    ri: int | None = 0,
    bk: int | None = 0,
    rk: int | None = 0,
    bg: int | None = 0,
    rg: int | None = 0,
) -> dict:
    """Build a minimal ``end_state`` dict for tests."""
    return {
        "blue": {"inhibitors": bi, "totalKills": bk, "totalGold": bg},
        "red": {"inhibitors": ri, "totalKills": rk, "totalGold": rg},
    }


class TestDetermineWinner:
    # ── Inhibitor rule (primary) ────────────────────────────────────
    def test_blue_more_inhibs(self):
        assert determine_winner(_state(bi=2, ri=0)) == "blue"

    def test_red_more_inhibs(self):
        assert determine_winner(_state(bi=0, ri=3)) == "red"

    def test_inhibs_dominate_over_kills_and_gold(self):
        """Blue has more inhibs but fewer kills/gold — still wins."""
        assert (
            determine_winner(_state(bi=1, ri=0, bk=5, rk=25, bg=40000, rg=70000))
            == "blue"
        )

    # ── Kill tiebreaker ─────────────────────────────────────────────
    def test_equal_inhibs_blue_more_kills(self):
        assert determine_winner(_state(bi=0, ri=0, bk=20, rk=10)) == "blue"

    def test_equal_inhibs_red_more_kills(self):
        assert determine_winner(_state(bi=1, ri=1, bk=8, rk=15)) == "red"

    # ── Gold tiebreaker ─────────────────────────────────────────────
    def test_equal_inhibs_equal_kills_blue_more_gold(self):
        assert (
            determine_winner(_state(bi=0, ri=0, bk=10, rk=10, bg=60000, rg=55000))
            == "blue"
        )

    def test_equal_inhibs_equal_kills_red_more_gold(self):
        assert (
            determine_winner(_state(bi=2, ri=2, bk=15, rk=15, bg=50000, rg=58000))
            == "red"
        )

    # ── Dead tie ────────────────────────────────────────────────────
    def test_identical_end_state_returns_none(self):
        assert (
            determine_winner(_state(bi=0, ri=0, bk=10, rk=10, bg=50000, rg=50000))
            is None
        )

    # ── None / missing fields ──────────────────────────────────────
    def test_none_inhibitors_treated_as_zero(self):
        """None inhibitors on both sides should be treated as 0 without crashing."""
        # Inhibs tie at 0, blue has 3 more kills — tiebreak to kills.
        assert (
            determine_winner(_state(bi=None, ri=None, bk=7, rk=4, bg=40000, rg=40000))
            == "blue"
        )

    def test_none_kills_and_gold_treated_as_zero(self):
        state = {
            "blue": {"inhibitors": 1, "totalKills": None, "totalGold": None},
            "red": {"inhibitors": 0, "totalKills": None, "totalGold": None},
        }
        # Inhibs differ -> blue wins without crashing on None fields.
        assert determine_winner(state) == "blue"

    def test_missing_fields_treated_as_zero(self):
        """Completely missing keys should be treated as 0 (via .get)."""
        state = {"blue": {}, "red": {}}
        # Nothing to break a tie on — dead tie should return None.
        assert determine_winner(state) is None

    def test_partially_missing_fields(self):
        """Blue has inhibs, red has kills+gold — inhibs rule fires first."""
        state = {
            "blue": {"inhibitors": 2},
            "red": {"totalKills": 5, "totalGold": 30000},
        }
        assert determine_winner(state) == "blue"

    # ── Defensive: empty / None input ───────────────────────────────
    def test_empty_end_state_returns_none(self):
        assert determine_winner({}) is None

    def test_none_end_state_returns_none(self):
        assert determine_winner(None) is None


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
