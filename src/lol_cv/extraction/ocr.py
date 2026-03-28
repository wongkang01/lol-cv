"""
OCR-based HUD data extraction from game screenshots.

Extracts gold counts, KDA, CS, items, levels, and timers
from the League of Legends spectator mode HUD.
"""


class HudExtractor:
    """Extract structured data from LoL HUD using OCR."""

    def __init__(self, engine: str = "paddleocr"):
        """
        Args:
            engine: OCR engine to use ('paddleocr', 'tesseract', or 'vlm').
        """
        self.engine = engine

    def extract_scoreboard(self, frame) -> dict:
        """Extract scoreboard data (gold, KDA, CS) from a game frame."""
        # TODO: Define ROI regions for scoreboard elements, run OCR
        raise NotImplementedError

    def extract_game_timer(self, frame) -> float:
        """Extract the in-game timestamp from a frame."""
        # TODO: OCR on timer region
        raise NotImplementedError

    def extract_items(self, frame) -> dict:
        """Extract item builds for all players from a frame."""
        # TODO: Item slot detection and classification
        raise NotImplementedError
