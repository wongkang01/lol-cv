"""
Minimap champion tracking using pyLoL's YOLOv12 model.

Extracts per-second champion positions from minimap frames.
Builds on the autoLeague library from the pyLoL replay extractor.
"""


class MinimapTracker:
    """Track champion positions on the minimap using YOLOv12."""

    def __init__(self, model_path: str = None, confidence: float = 0.4):
        """
        Args:
            model_path: Path to YOLOv12 weights (.pt file).
                        If None, uses pyLoL's default Roboflow model.
            confidence: Detection confidence threshold.
        """
        self.model_path = model_path
        self.confidence = confidence
        self.model = None

    def load_model(self):
        """Load the YOLOv12 detection model."""
        # TODO: Load ultralytics YOLO model or Roboflow model
        raise NotImplementedError

    def extract_positions(self, video_path: str, fps: int = 1) -> dict:
        """
        Extract champion positions from a minimap video.

        Args:
            video_path: Path to minimap capture video.
            fps: Frames per second to sample (default: 1 = every second).

        Returns:
            Dict with timestamps as keys, list of (champion, x, y) as values.
        """
        # TODO: Implement frame-by-frame detection
        raise NotImplementedError

    def positions_to_dataframe(self, positions: dict):
        """Convert raw positions dict to a pandas DataFrame."""
        # TODO: Structure data as DataFrame with columns:
        # [timestamp, champion, team, x, y, zone]
        raise NotImplementedError
