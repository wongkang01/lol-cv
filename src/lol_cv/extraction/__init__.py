"""
Data extraction modules.

- minimap: Champion position tracking via YOLO (pyLoL weights) on minimap frames
- ocr: HUD data extraction (gold, KDA, items, timers) via PaddleOCR
- api: Riot API client for supplementary match data
- vlm: Vision-Language Model analysis via Gemini (Flash + Embedding)
- vod_processor: VOD download, frame extraction, and region cropping pipeline
- benchmark: Detection model benchmarking (YOLOv8 vs YOLOv11 comparison)
- vod_discovery: Automated VOD URL + timestamp discovery via lolesports API
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
    if name == "VodProcessor":
        from lol_cv.extraction.vod_processor import VodProcessor
        return VodProcessor
    if name == "DetectionBenchmark":
        from lol_cv.extraction.benchmark import DetectionBenchmark
        return DetectionBenchmark
    if name == "VodDiscovery":
        from lol_cv.extraction.vod_discovery import VodDiscovery
        return VodDiscovery
    if name == "MatchMetadataFetcher":
        from lol_cv.extraction.match_metadata import MatchMetadataFetcher
        return MatchMetadataFetcher
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["RiotApiClient", "MinimapTracker", "HudExtractor", "VlmAnalyzer", "VodProcessor", "DetectionBenchmark", "VodDiscovery"]
