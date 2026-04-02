"""
Vision-Language Model analysis of game screenshots.

Uses Gemini Flash for tactical analysis of game frames and
Gemini Embedding for multimodal embeddings (similarity search / clustering).

Recommended models (March 2026):
    - Gemini 3 Flash: $0.30/M tokens, 86.9% Video-MMMU, native video support.
    - Gemini 3.1 Flash-Lite: $0.075/M tokens for bulk processing.
    - Gemini Embedding 2: $0.25/M tokens, 3072-dim multimodal embeddings.
"""

import base64
import mimetypes
import os
from pathlib import Path

from google import genai

from lol_cv.utils import setup_logger

logger = setup_logger("lol_cv.extraction.vlm")

# Supported image MIME types for Gemini API
_IMAGE_MIMETYPES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}


def _detect_mime(path: Path) -> str:
    """Detect MIME type for an image file."""
    mime, _ = mimetypes.guess_type(str(path))
    if mime and mime.startswith("image/"):
        return mime
    return "image/jpeg"  # safe fallback

# Default structured prompt for tactical analysis of a spectator-mode frame.
DEFAULT_ANALYSIS_PROMPT = """Analyze this League of Legends spectator-mode screenshot.
Extract the following in JSON format:
{
    "game_time": "estimated in-game time (MM:SS)",
    "game_phase": "early / mid / late",
    "team_gold_diff": "approximate gold difference (positive = blue side ahead)",
    "kill_score": "blue_kills - red_kills if visible",
    "map_state": {
        "towers_remaining": {"blue": null, "red": null},
        "dragon_soul_status": null,
        "baron_status": null
    },
    "tactical_assessment": {
        "positioning_quality": "brief assessment of team positioning",
        "vision_control": "which team has better vision control and why",
        "objective_priority": "what objective should each team be playing for",
        "win_probability": "estimated win probability for blue side (0-100)"
    },
    "key_observations": ["up to 3 notable observations about the game state"]
}
Only include fields you can confidently determine from the screenshot."""


class VlmAnalyzer:
    """Analyze game screenshots using Gemini Vision-Language Models."""

    def __init__(
        self,
        model: str = "gemini-3-flash",
        embed_model: str = "gemini-embedding-002",
        api_key: str = None,
    ):
        """
        Args:
            model: Gemini model name for analysis.
            embed_model: Gemini model name for embeddings.
            api_key: Gemini API key. Falls back to GEMINI_API_KEY env var.
        """
        self.model = model
        self.embed_model = embed_model
        self._api_key = api_key or os.getenv("GEMINI_API_KEY")
        self._client = None  # Lazy-initialized on first use

    @property
    def client(self) -> genai.Client:
        """Lazy-initialize the Gemini client on first use."""
        if self._client is None:
            if not self._api_key:
                raise ValueError(
                    "Gemini API key not configured. Set GEMINI_API_KEY env var "
                    "or pass api_key= to VlmAnalyzer."
                )
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def analyze_frame(self, frame_path: str, prompt: str = None) -> str:
        """Send a game frame to Gemini with a structured prompt.

        Args:
            frame_path: Path to the screenshot (PNG/JPG).
            prompt: Analysis prompt. Defaults to tactical analysis prompt.

        Returns:
            Model's text response (typically JSON-formatted analysis).
        """
        prompt = prompt or DEFAULT_ANALYSIS_PROMPT
        image_path = Path(frame_path)

        # Read and encode image
        image_bytes = image_path.read_bytes()
        mime = _detect_mime(image_path)

        logger.info("Analyzing frame %s with %s", image_path.name, self.model)
        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                {
                    "parts": [
                        {"inline_data": {"mime_type": mime, "data": base64.b64encode(image_bytes).decode()}},
                        {"text": prompt},
                    ]
                }
            ],
        )
        return response.text

    def analyze_frames_batch(self, frame_paths: list[str], prompt: str = None) -> list[str]:
        """Analyze multiple frames sequentially.

        Args:
            frame_paths: List of paths to screenshots.
            prompt: Shared analysis prompt for all frames.

        Returns:
            List of model responses, one per frame.
        """
        results = []
        for path in frame_paths:
            try:
                results.append(self.analyze_frame(path, prompt))
            except Exception as e:
                logger.error("Failed to analyze %s: %s", path, e)
                results.append(None)
        return results

    def embed_frame(self, frame_path: str) -> list[float]:
        """Generate a multimodal embedding for a game frame.

        Uses Gemini Embedding model to produce a 3072-dimensional vector
        that can be used for similarity search and clustering across
        game states.

        Args:
            frame_path: Path to the screenshot.

        Returns:
            3072-dimensional embedding vector.
        """
        image_path = Path(frame_path)
        image_bytes = image_path.read_bytes()
        mime = _detect_mime(image_path)

        logger.info("Embedding frame %s", image_path.name)
        response = self.client.models.embed_content(
            model=self.embed_model,
            contents=[
                {
                    "parts": [
                        {"inline_data": {"mime_type": mime, "data": base64.b64encode(image_bytes).decode()}},
                    ]
                }
            ],
        )
        return response.embeddings[0].values

    def embed_text(self, text: str) -> list[float]:
        """Generate a text embedding for cross-modal similarity search.

        Allows querying game frame embeddings with text descriptions
        (e.g. "teamfight at baron pit").

        Args:
            text: Text query to embed.

        Returns:
            3072-dimensional embedding vector in the same space as frame embeddings.
        """
        response = self.client.models.embed_content(
            model=self.embed_model,
            contents=text,
        )
        return response.embeddings[0].values
