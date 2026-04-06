"""
Minimap champion tracking using YOLO object detection.

Extracts per-second champion positions from minimap frames.
Uses pyLoL's pretrained weights (YOLOv8 architecture, branded as "yolov12.pt")
which supports 167/170 champions at mAP 92.2%.

The model weights (227MB) are hosted on Google Drive and must be downloaded
to data/models/yolov12.pt before use. Alternatively, the model can be loaded
from the Roboflow project 'lolpago-multi-tracking-service' (version 18).

Weights download: https://drive.google.com/uc?export=download&id=1ymd7Thcz1XdejEW94LjSFDl3zDqYH0qq

Known weight limitations (yolov11m.pt, 170 classes)
---------------------------------------------------
- ``XinZhao`` is a class in the model but its visual detector is broken:
  across the 34 games in our dataset the sprite fires 0-7 times per game,
  with *zero* firings in the first ~5 minutes at any confidence threshold
  down to 0.01. The class name itself matches what the lolesports picks API
  returns exactly, so this is NOT a name-normalisation issue — it is a pure
  weight limitation. Features that depend on early-game jungler tracking
  (e.g. ``sp_early_blue_jgl_commit_side``) will be NaN whenever either
  jungler is XinZhao. Fixing this requires new / retrained weights.

- ``Zaahen`` and ``Yunara`` (post-training release champions) are not in the
  model's class vocabulary at all. Any game featuring these champions will
  have no detections for them. No normalisation can recover this; new weights
  are the only fix.

The ``detect_frame_filtered`` method uses a normalised-name comparison
(case-insensitive, whitespace/punctuation-stripped) so future name-casing
mismatches between picks data and model classes are tolerated automatically.
A warning is logged at detect time when a game's valid-champion set contains
names the model does not know, making this class of failure noisy instead of
silent.
"""

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from lol_cv.utils import setup_logger, get_model_path

logger = setup_logger("lol_cv.extraction.minimap")

# Champion class names are embedded in the YOLO model's metadata.
# Blue-side champions are listed first (indices 0-4), red-side next (5-9)
# in a typical match, but the model detects all 167 champions by class name.

# Minimap coordinate bounds (512x512 input after resize).
MINIMAP_SIZE = 512


def _normalise_champion_name(name: str) -> str:
    """Normalise a champion name for robust cross-source matching.

    Strips whitespace, underscores, apostrophes and hyphens and lower-cases
    the result so that ``"Xin Zhao"``, ``"Xin_Zhao"``, ``"XinZhao"`` and
    ``"xinzhao"`` all collapse to the same key.
    """
    return (
        name.replace(" ", "")
        .replace("_", "")
        .replace("'", "")
        .replace("-", "")
        .lower()
    )


class MinimapTracker:
    """Track champion positions on the minimap using YOLO."""

    def __init__(self, model_path: str = None, confidence: float = 0.4, overlap: float = 0.3):
        """
        Args:
            model_path: Path to YOLO weights (.pt file).
                        Defaults to data/models/yolov12.pt.
            confidence: Detection confidence threshold.
            overlap: IoU overlap threshold for NMS.
        """
        self.model_path = model_path or str(get_model_path("yolov12.pt"))
        self.confidence = confidence
        self.overlap = overlap
        self.model = None
        # Map from normalised-name -> canonical class name (populated in load_model).
        self._normalised_class_names: dict[str, str] = {}
        # Per-instance set of valid-champion names already warned about, to
        # avoid spamming the log on every frame.
        self._unknown_warned: set[str] = set()

    def load_model(self) -> YOLO:
        """Load the YOLO detection model from weights file.

        Returns:
            The loaded YOLO model instance.
        """
        logger.info("Loading YOLO model from %s", self.model_path)
        self.model = YOLO(self.model_path)
        # Build a normalised lookup so callers can match champion names without
        # worrying about case / whitespace / punctuation differences between
        # data sources.
        self._normalised_class_names = {
            _normalise_champion_name(n): n for n in self.model.names.values()
        }
        logger.info(
            "Model loaded — %d classes: %s",
            len(self.model.names),
            list(self.model.names.values())[:5],
        )
        return self.model

    def detect_frame(self, frame: np.ndarray) -> list[dict]:
        """Run detection on a single minimap frame.

        Args:
            frame: BGR image (numpy array) of the minimap region.

        Returns:
            List of detections, each: {champion, x, y, confidence, bbox}.
            Coordinates are normalised to [0, 1] relative to minimap size.
        """
        if self.model is None:
            self.load_model()

        # Resize to model input size
        resized = cv2.resize(frame, (MINIMAP_SIZE, MINIMAP_SIZE))
        results = self.model.predict(
            resized, conf=self.confidence, iou=self.overlap, verbose=False
        )

        detections = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            x_center, y_center, w, h = box.xywhn[0].tolist()
            detections.append({
                "champion": self.model.names[cls_id],
                "x": x_center,
                "y": y_center,
                "confidence": float(box.conf[0]),
                "bbox": box.xyxy[0].tolist(),
            })
        return detections

    def detect_frame_filtered(
        self, frame: np.ndarray, valid_champions: set[str]
    ) -> list[dict]:
        """Run detection and keep only champions that are in the game.

        Uses match metadata to discard false positives from the detector.
        When multiple detections map to the same champion, the highest
        confidence one is kept.

        Names are compared after normalisation (case-insensitive, with
        whitespace/underscores/apostrophes/hyphens stripped). Matched
        detections are relabelled with the caller-supplied canonical name
        from ``valid_champions`` so that downstream feature engineering
        always sees the picks-API spelling.

        The first call on a given ``MinimapTracker`` instance also logs a
        warning listing any names in ``valid_champions`` that do not exist
        in the model's class vocabulary (e.g. post-training release champs
        like ``Zaahen``/``Yunara``) — those champions will never be
        detected no matter how many frames are processed.

        Args:
            frame: BGR image of the minimap region.
            valid_champions: Set of champion names actually in this game
                (e.g. from MatchMetadataFetcher).

        Returns:
            Filtered list of detections (max 10 — one per champion).
        """
        if self.model is None:
            self.load_model()

        # Build normalised-key -> canonical caller-side name lookup.
        canonical_by_norm: dict[str, str] = {
            _normalise_champion_name(c): c for c in valid_champions
        }

        # Warn once per instance about champions the model doesn't know.
        unknown = {
            canonical_by_norm[k]
            for k in canonical_by_norm
            if k not in self._normalised_class_names
        }
        new_unknown = unknown - self._unknown_warned
        if new_unknown:
            logger.warning(
                "Valid champions not present in model vocabulary "
                "(will never be detected): %s",
                sorted(new_unknown),
            )
            self._unknown_warned.update(new_unknown)

        raw = self.detect_frame(frame)

        # Keep only valid champions, relabelled to the caller's canonical name.
        filtered: list[dict] = []
        for d in raw:
            key = _normalise_champion_name(d["champion"])
            canonical = canonical_by_norm.get(key)
            if canonical is None:
                continue
            if canonical != d["champion"]:
                d = {**d, "champion": canonical}
            filtered.append(d)

        # Deduplicate: keep highest confidence per champion
        best: dict[str, dict] = {}
        for d in filtered:
            champ = d["champion"]
            if champ not in best or d["confidence"] > best[champ]["confidence"]:
                best[champ] = d

        return list(best.values())

    def extract_positions(self, video_path: str, fps: int = 1) -> dict[float, list[dict]]:
        """Extract champion positions from a minimap video.

        Args:
            video_path: Path to minimap video file (MP4/AVI).
            fps: Frames per second to sample (1 = every second).

        Returns:
            Dict mapping game-time seconds to list of detections.
        """
        if self.model is None:
            self.load_model()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(video_fps / fps))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(
            "Processing %s — %.0f fps, %d frames, sampling every %d frames",
            video_path, video_fps, total_frames, frame_interval,
        )

        positions = {}
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                game_time = frame_idx / video_fps
                detections = self.detect_frame(frame)
                positions[game_time] = detections

                if frame_idx % (frame_interval * 60) == 0:
                    logger.info(
                        "t=%.0fs — %d champions detected", game_time, len(detections)
                    )

            frame_idx += 1

        cap.release()
        logger.info("Extracted positions for %d timestamps", len(positions))
        return positions

    def positions_to_dataframe(self, positions: dict[float, list[dict]]) -> pd.DataFrame:
        """Convert raw positions dict to a structured DataFrame.

        Args:
            positions: Output from extract_positions().

        Returns:
            DataFrame with columns: [timestamp, champion, x, y, confidence].
        """
        rows = []
        for timestamp, detections in positions.items():
            for det in detections:
                rows.append({
                    "timestamp": timestamp,
                    "champion": det["champion"],
                    "x": det["x"],
                    "y": det["y"],
                    "confidence": det["confidence"],
                })
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("timestamp").reset_index(drop=True)
        logger.info("Created DataFrame with %d position records", len(df))
        return df

    def extract_from_full_frame(
        self, video_path: str, minimap_region: tuple[int, int, int, int], fps: int = 1
    ) -> dict[float, list[dict]]:
        """Extract champion positions from a full-screen game video.

        Crops the minimap region from each frame before running detection.
        Useful when working with full spectator-mode recordings or YouTube VODs
        instead of pre-cropped minimap videos.

        Args:
            video_path: Path to full-screen game video.
            minimap_region: (x1, y1, x2, y2) pixel coordinates of the minimap
                           in the full frame. Default spectator 1080p is
                           approximately (1650, 790, 1920, 1080).
            fps: Frames per second to sample.

        Returns:
            Dict mapping game-time seconds to list of detections.
        """
        if self.model is None:
            self.load_model()

        x1, y1, x2, y2 = minimap_region
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(video_fps / fps))

        logger.info("Processing full-frame video, cropping minimap at (%d,%d)-(%d,%d)", x1, y1, x2, y2)

        positions = {}
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                minimap_crop = frame[y1:y2, x1:x2]
                game_time = frame_idx / video_fps
                positions[game_time] = self.detect_frame(minimap_crop)

            frame_idx += 1

        cap.release()
        logger.info("Extracted positions for %d timestamps from full-frame video", len(positions))
        return positions
