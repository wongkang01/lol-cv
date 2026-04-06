"""
OCR-based HUD data extraction from game screenshots.

Extracts gold counts, KDA, CS, levels, items, and timers from the
League of Legends spectator mode HUD.

NOTE: This module contains a legacy PaddleOCR-based ``HudExtractor``.
The active OCR pipeline is ``scripts/run_ocr.py``, which uses easyocr.
PaddleOCR is no longer a project dependency — the import is loaded
lazily inside ``HudExtractor._get_ocr`` so that this module can still
be imported (for tests and downstream imports) without paddleocr
installed. Instantiating ``HudExtractor(engine='paddleocr')`` will
raise ``ImportError`` with a helpful message directing users to
``scripts/run_ocr.py`` or to install paddleocr explicitly.

Spectator mode HUD layout (1920x1080, 2026 First Stand broadcast):
    - Top bar: team scores, gold, turrets (y: 0-80, full width)
    - Game timer: center below scoreboard (x: 925-995, y: 55-78)
    - Kill score: center top (x: 895-1025, y: 0-28)
    - Gold: blue (x: 750-850), red (x: 1070-1170), y: 0-28
    - Turrets: blue (x: 700-740), red (x: 1180-1220), y: 0-28
    - Player scorecards: blue (x: 195-730), red (x: 1190-1640), y: 818-1015
    - Minimap: bottom-right corner (x: 1650-1920, y: 790-1080)

ROI coordinates must be adjusted for different resolutions.
"""

import re
from typing import TYPE_CHECKING

import cv2
import numpy as np

from lol_cv.utils import setup_logger

if TYPE_CHECKING:
    from paddleocr import PaddleOCR

logger = setup_logger("lol_cv.extraction.ocr")

# Default ROI regions for 1920x1080 spectator mode.
# Format: (x1, y1, x2, y2) — top-left and bottom-right corners.
DEFAULT_REGIONS = {
    "timer": (892, 72, 1035, 112),
    "scoreboard_top": (0, 0, 1920, 80),
    # Kill score at top center (blue kills left, red kills right)
    "kill_score": (877, 7, 1050, 52),
    # Individual gold displays
    "blue_gold": (712, 7, 862, 52),
    "red_gold": (1065, 7, 1215, 52),
    # Turret counts
    "blue_turrets": (645, 7, 712, 52),
    "red_turrets": (1215, 7, 1282, 52),
    # Player scorecards (bottom panel)
    "blue_scorecards": (195, 818, 730, 1015),
    "red_scorecards": (1190, 818, 1640, 1015),
}


class HudExtractor:
    """Extract structured data from LoL spectator HUD using OCR."""

    def __init__(self, engine: str = "paddleocr", lang: str = "en"):
        """
        Args:
            engine: OCR engine to use ('paddleocr' or 'tesseract').
            lang: Language for OCR recognition.
        """
        self.engine = engine
        self._ocr = None
        self._lang = lang

    def _get_ocr(self) -> "PaddleOCR":
        """Lazy-load PaddleOCR engine.

        Importing paddleocr is deferred until the first OCR call so the
        rest of this module can be imported in environments without
        paddleocr installed (e.g. the test suite).
        """
        if self._ocr is None:
            try:
                from paddleocr import PaddleOCR  # noqa: WPS433 (late import)
            except ImportError as exc:
                raise ImportError(
                    "paddleocr is not installed. The active OCR pipeline uses "
                    "easyocr via scripts/run_ocr.py. Install paddleocr "
                    "explicitly (`pip install paddleocr paddlepaddle`) if you "
                    "need this legacy HudExtractor."
                ) from exc
            logger.info("Initializing PaddleOCR engine")
            self._ocr = PaddleOCR(use_angle_cls=True, lang=self._lang, show_log=False)
        return self._ocr

    def _crop_region(self, frame: np.ndarray, region: tuple[int, int, int, int]) -> np.ndarray:
        """Crop a region from a frame."""
        x1, y1, x2, y2 = region
        return frame[y1:y2, x1:x2]

    def _ocr_region(self, frame: np.ndarray, region: tuple[int, int, int, int]) -> list[str]:
        """Run OCR on a cropped region of a frame.

        Args:
            frame: Full BGR frame (numpy array).
            region: (x1, y1, x2, y2) crop coordinates.

        Returns:
            List of detected text strings.
        """
        crop = self._crop_region(frame, region)
        ocr = self._get_ocr()
        result = ocr.ocr(crop, cls=True)

        texts = []
        if result and result[0]:
            for line in result[0]:
                text = line[1][0]  # (text, confidence)
                texts.append(text)
        return texts

    def extract_game_timer(self, frame: np.ndarray, region: tuple = None) -> str | None:
        """Extract the in-game timestamp from a frame.

        Args:
            frame: Full BGR frame.
            region: Timer ROI override. Defaults to center-top.

        Returns:
            Time string in "MM:SS" format, or None if not detected.
        """
        region = region or DEFAULT_REGIONS["timer"]
        texts = self._ocr_region(frame, region)

        for text in texts:
            # Match MM:SS or M:SS pattern
            match = re.search(r"(\d{1,2}:\d{2})", text)
            if match:
                return match.group(1)
        return None

    def timer_to_seconds(self, timer_str: str) -> int | None:
        """Convert "MM:SS" timer string to total seconds."""
        if not timer_str:
            return None
        match = re.match(r"(\d+):(\d{2})", timer_str)
        if match:
            return int(match.group(1)) * 60 + int(match.group(2))
        return None

    def extract_scoreboard(self, frame: np.ndarray, region: tuple = None) -> dict:
        """Extract top-bar scoreboard data (kills, gold, turrets).

        Args:
            frame: Full BGR frame.
            region: Scoreboard ROI override.

        Returns:
            Dict with extracted values: {blue_kills, red_kills, blue_gold, red_gold, ...}.
            Values are None where OCR fails to extract.
        """
        region = region or DEFAULT_REGIONS["scoreboard_top"]
        texts = self._ocr_region(frame, region)

        result = {
            "blue_kills": None,
            "red_kills": None,
            "blue_gold": None,
            "red_gold": None,
            "raw_text": texts,
        }

        # Try to parse kill score (typically "X - Y" in center)
        for text in texts:
            kill_match = re.search(r"(\d+)\s*[-–]\s*(\d+)", text)
            if kill_match:
                result["blue_kills"] = int(kill_match.group(1))
                result["red_kills"] = int(kill_match.group(2))
                break

        # Try to parse gold values (e.g. "12.5k" or "12500")
        for text in texts:
            gold_match = re.findall(r"(\d+\.?\d*)\s*k", text, re.IGNORECASE)
            if len(gold_match) >= 2:
                result["blue_gold"] = float(gold_match[0]) * 1000
                result["red_gold"] = float(gold_match[1]) * 1000

        return result

    def extract_kill_score(self, frame: np.ndarray) -> tuple[int | None, int | None]:
        """Extract just the kill score from center-top of the HUD.

        Returns:
            (blue_kills, red_kills) tuple.
        """
        texts = self._ocr_region(frame, DEFAULT_REGIONS["kill_score"])
        for text in texts:
            match = re.search(r"(\d+)\s*[-–]\s*(\d+)", text)
            if match:
                return int(match.group(1)), int(match.group(2))
        return None, None

    def extract_all(self, frame: np.ndarray) -> dict:
        """Extract all available HUD data from a single frame.

        Returns:
            Dict with keys: game_time, game_time_seconds, scoreboard.
        """
        timer = self.extract_game_timer(frame)
        scoreboard = self.extract_scoreboard(frame)

        return {
            "game_time": timer,
            "game_time_seconds": self.timer_to_seconds(timer),
            "scoreboard": scoreboard,
        }

    def extract_from_video(
        self, video_path: str, fps: int = 1
    ) -> list[dict]:
        """Extract HUD data from every sampled frame in a video.

        Args:
            video_path: Path to spectator-mode video.
            fps: Frames per second to sample.

        Returns:
            List of extraction results, one per sampled frame.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(video_fps / fps))

        logger.info("Extracting HUD data from %s (sampling every %d frames)", video_path, frame_interval)

        results = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                data = self.extract_all(frame)
                data["video_time"] = frame_idx / video_fps
                results.append(data)

            frame_idx += 1

        cap.release()
        logger.info("Extracted HUD data for %d frames", len(results))
        return results
