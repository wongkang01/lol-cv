"""
Data extraction modules.

- minimap: Champion position tracking via YOLO (pyLoL weights) on minimap frames
- ocr: HUD data extraction (gold, KDA, items, timers) via PaddleOCR
- api: Riot API client for supplementary match data
- vlm: Vision-Language Model analysis via Gemini (Flash + Embedding)
"""

def __getattr__(name):
    """Lazy imports to avoid pulling in heavy deps (ultralytics, paddleocr, genai)
    at package-import time."""
    if name == "RiotApiClient":
        from lol_cv.extraction.api import RiotApiClient
        return RiotApiClient
    if name == "MinimapTracker":
        from lol_cv.extraction.minimap import MinimapTracker
        return MinimapTracker
    if name == "HudExtractor":
        from lol_cv.extraction.ocr import HudExtractor
        return HudExtractor
    if name == "VlmAnalyzer":
        from lol_cv.extraction.vlm import VlmAnalyzer
        return VlmAnalyzer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["RiotApiClient", "MinimapTracker", "HudExtractor", "VlmAnalyzer"]
