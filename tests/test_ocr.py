"""Unit tests for OCR/API/VLM — pure functions only.

Tests that depend on heavy dependencies (paddleocr, google-genai, ultralytics)
are skipped when those packages aren't installed.
"""

import importlib
import os
import re

import pytest


def _try_import(dotted_path):
    """Try to import a module, return (module, None) or (None, error)."""
    try:
        return importlib.import_module(dotted_path), None
    except (ImportError, ModuleNotFoundError) as e:
        return None, str(e)


# ── Timer Parsing (pure Python, no deps) ────────────────────────────
# Extracted directly from ocr.py to test without paddleocr import.


def timer_to_seconds(timer_str):
    """Pure reimplementation for testing without paddleocr."""
    if not timer_str:
        return None
    match = re.match(r"(\d+):(\d{2})", timer_str)
    if match:
        return int(match.group(1)) * 60 + int(match.group(2))
    return None


class TestTimerToSeconds:
    def test_standard_format(self):
        assert timer_to_seconds("15:30") == 930

    def test_single_digit_minutes(self):
        assert timer_to_seconds("5:00") == 300

    def test_zero(self):
        assert timer_to_seconds("0:00") == 0

    def test_long_game(self):
        assert timer_to_seconds("45:59") == 2759

    def test_none_input(self):
        assert timer_to_seconds(None) is None

    def test_empty_string(self):
        assert timer_to_seconds("") is None

    def test_invalid_format(self):
        assert timer_to_seconds("abc") is None

    def test_missing_leading_zero(self):
        assert timer_to_seconds("1:05") == 65


# ── Riot API Client Validation ──────────────────────────────────────


class TestRiotApiClientValidation:
    def test_missing_api_key_raises(self):
        """RiotApiClient should raise ValueError when no key is provided."""
        original = os.environ.pop("RIOT_API_KEY", None)
        try:
            mod, err = _try_import("lol_cv.extraction.api")
            if mod is None:
                pytest.skip(f"Cannot import api module: {err}")
            with pytest.raises(ValueError, match="Riot API key not configured"):
                mod.RiotApiClient(api_key=None)
        finally:
            if original is not None:
                os.environ["RIOT_API_KEY"] = original


# ── VLM Lazy Init Validation ────────────────────────────────────────


class TestVlmAnalyzerValidation:
    def test_missing_api_key_raises_on_use(self):
        """VlmAnalyzer should raise ValueError on first client access, not init."""
        mod, err = _try_import("lol_cv.extraction.vlm")
        if mod is None:
            pytest.skip(f"Cannot import vlm module: {err}")

        original = os.environ.pop("GEMINI_API_KEY", None)
        try:
            analyzer = mod.VlmAnalyzer(api_key=None)
            with pytest.raises(ValueError, match="Gemini API key not configured"):
                _ = analyzer.client
        finally:
            if original is not None:
                os.environ["GEMINI_API_KEY"] = original
