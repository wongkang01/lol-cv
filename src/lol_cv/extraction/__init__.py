"""
Data extraction modules.

- minimap: Champion position tracking via YOLO (pyLoL weights) on minimap frames
- ocr: HUD data extraction (gold, KDA, items, timers) via PaddleOCR
- api: Riot API client for supplementary match data
- vlm: Vision-Language Model analysis via Gemini (Flash + Embedding)
"""

from lol_cv.extraction.api import RiotApiClient
from lol_cv.extraction.minimap import MinimapTracker
from lol_cv.extraction.ocr import HudExtractor
from lol_cv.extraction.vlm import VlmAnalyzer

__all__ = ["RiotApiClient", "MinimapTracker", "HudExtractor", "VlmAnalyzer"]
