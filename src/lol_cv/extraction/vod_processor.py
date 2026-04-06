"""
VOD (Video on Demand) processing pipeline for League of Legends broadcast analysis.

Downloads tournament VODs from YouTube, extracts frames at configurable FPS,
and crops minimap / HUD regions from full-screen 1080p spectator-mode frames.

Requires:
    - yt-dlp (CLI tool, installed separately)
    - OpenCV (cv2) for frame extraction and image cropping
    - tqdm for progress bars
"""

from __future__ import annotations

import subprocess
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from lol_cv.utils import setup_logger, load_config, ensure_dir

logger = setup_logger("lol_cv.extraction.vod_processor")

# Reference resolution these pixel coordinates were calibrated against.
REFERENCE_WIDTH = 1920
REFERENCE_HEIGHT = 1080

# Default HUD region coordinates at 1920x1080 (x1, y1, x2, y2).
# These are automatically scaled to the actual frame resolution at runtime
# by :meth:`_scale_region` so the same config works for 1080p, 720p, etc.
DEFAULT_HUD_REGIONS: dict[str, tuple[int, int, int, int]] = {
    "scoreboard": (0, 0, 1920, 80),
    "timer": (892, 72, 1035, 112),       # game clock (below scoreboard center)
    "kill_score": (877, 7, 1050, 52),    # "3 | 1" center
    "blue_gold": (712, 7, 862, 52),      # blue team gold "10.3K"
    "red_gold": (1065, 7, 1215, 52),     # red team gold "9.7K"
    "blue_turrets": (645, 7, 712, 52),   # small turret count
    "red_turrets": (1215, 7, 1282, 52),  # small turret count
}

# Default minimap region for 1080p spectator mode.
DEFAULT_MINIMAP_REGION: tuple[int, int, int, int] = (1650, 790, 1920, 1080)


def _scale_region(
    region: tuple[int, int, int, int],
    frame_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Scale a region from reference resolution to actual frame resolution.

    Args:
        region: ``(x1, y1, x2, y2)`` at reference 1920x1080.
        frame_shape: Actual ``(height, width)`` of the target frame.

    Returns:
        Scaled ``(x1, y1, x2, y2)`` for the target frame, clamped to bounds.
    """
    h, w = frame_shape[:2]
    sx = w / REFERENCE_WIDTH
    sy = h / REFERENCE_HEIGHT
    x1, y1, x2, y2 = region
    return (
        max(0, min(w, int(x1 * sx))),
        max(0, min(h, int(y1 * sy))),
        max(0, min(w, int(x2 * sx))),
        max(0, min(h, int(y2 * sy))),
    )


class VodProcessor:
    """Download tournament VODs, extract frames, and crop game-UI regions.

    Typical usage::

        processor = VodProcessor()
        processor.process_vod(
            url_or_path="https://www.youtube.com/watch?v=...",
            match_id="fs_2026_match01",
            output_base_dir="data/raw",
        )
    """

    def __init__(self, config: dict | None = None):
        """
        Args:
            config: Pipeline configuration dict. If *None*, loads
                    ``configs/default.yaml`` via :func:`load_config`.
        """
        self.config = config or load_config()
        ocr_regions = (
            self.config.get("extraction", {}).get("ocr", {}).get("regions", {})
        )

        # Build HUD regions from config, excluding the minimap entry (handled
        # separately by :meth:`crop_minimap`).
        self.hud_regions: dict[str, tuple[int, int, int, int]] = {}
        for name, coords in ocr_regions.items():
            if name == "minimap":
                continue
            self.hud_regions[name] = tuple(coords)  # type: ignore[arg-type]

        if not self.hud_regions:
            self.hud_regions = DEFAULT_HUD_REGIONS.copy()

    # ------------------------------------------------------------------
    # 1. Download
    # ------------------------------------------------------------------

    def download_vod(
        self,
        url: str,
        output_dir: str | Path,
        quality: str = "1080p",
    ) -> Path:
        """Download a YouTube VOD using *yt-dlp*.

        Args:
            url: YouTube video URL.
            output_dir: Directory to save the downloaded file.
            quality: Maximum video quality (e.g. ``"1080p"``, ``"720p"``).

        Returns:
            Path to the downloaded MP4 file.

        Raises:
            RuntimeError: If yt-dlp exits with a non-zero code.
            FileNotFoundError: If yt-dlp is not installed.
        """
        output_dir = ensure_dir(Path(output_dir))

        if shutil.which("yt-dlp") is None:
            raise FileNotFoundError(
                "yt-dlp is not installed or not on PATH. "
                "Install it with: pip install yt-dlp"
            )

        height = quality.rstrip("p")
        output_template = str(output_dir / "%(title)s.%(ext)s")

        cmd = [
            "yt-dlp",
            "-f", f"bestvideo[height<={height}]+bestaudio/best[height<={height}]/best",
            "--merge-output-format", "mp4",
            "-o", output_template,
            "--no-playlist",
            "--remote-components", "ejs:github",
            "--extractor-args", "youtube:player_client=web_creator",
        ]

        # Use exported cookies file if available (safer than --cookies-from-browser).
        cookies_path = Path("cookies.txt")
        if cookies_path.exists():
            cmd.extend(["--cookies", str(cookies_path.resolve())])

        cmd.append(url)

        logger.info("Downloading VOD: %s (quality=%s)", url, quality)
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error("yt-dlp stderr:\n%s", result.stderr)
            raise RuntimeError(f"yt-dlp failed (exit {result.returncode}): {result.stderr}")

        # Find the downloaded file (yt-dlp may sanitise the title).
        mp4_files = sorted(output_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
        if not mp4_files:
            raise RuntimeError("Download completed but no MP4 file found in output directory.")

        downloaded = mp4_files[-1]
        logger.info("Downloaded: %s", downloaded)
        return downloaded

    # ------------------------------------------------------------------
    # 2. Frame extraction
    # ------------------------------------------------------------------

    def extract_frames(
        self,
        video_path: str | Path,
        output_dir: str | Path,
        fps: int = 1,
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> list[Path]:
        """Extract frames from a video at the given FPS.

        Frames are saved as PNG files named ``frame_SSSSSS.png`` where
        *SSSSSS* is the timestamp in seconds, zero-padded to six digits.

        Use *start_time* and *end_time* to extract only the actual gameplay
        portion of a broadcast VOD, skipping pre-match interviews, draft
        phase, and post-match content.

        Args:
            video_path: Path to the source video file.
            output_dir: Directory to save extracted frames.
            fps: Frames per second to extract (default 1).
            start_time: VOD timestamp (seconds) where gameplay begins.
                        ``None`` starts from the beginning.
            end_time: VOD timestamp (seconds) where gameplay ends.
                      ``None`` processes until the end of the video.

        Returns:
            List of paths to the saved frame images.

        Raises:
            FileNotFoundError: If the video file cannot be opened.
        """
        video_path = Path(video_path)
        output_dir = ensure_dir(Path(output_dir))

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, int(video_fps / fps))

        # Compute frame boundaries from timestamps.
        start_frame = int(start_time * video_fps) if start_time else 0
        end_frame = int(end_time * video_fps) if end_time else total_frames

        # Seek to start position if needed.
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        usable_frames = end_frame - start_frame
        expected_outputs = usable_frames // frame_interval

        time_range = ""
        if start_time or end_time:
            time_range = f" [{start_time or 0}s – {end_time or '∞'}s]"

        logger.info(
            "Extracting frames from %s%s — %.0f fps, %d usable frames, "
            "sampling every %d frames (~%d output frames)",
            video_path.name, time_range, video_fps,
            usable_frames, frame_interval, expected_outputs,
        )

        saved_paths: list[Path] = []
        frame_idx = start_frame

        with tqdm(total=expected_outputs, desc="Extracting frames", unit="frame") as pbar:
            while frame_idx < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break

                if (frame_idx - start_frame) % frame_interval == 0:
                    timestamp_sec = int(frame_idx / video_fps)
                    filename = f"frame_{timestamp_sec:06d}.png"
                    out_path = output_dir / filename
                    cv2.imwrite(str(out_path), frame)
                    saved_paths.append(out_path)
                    pbar.update(1)

                frame_idx += 1

        cap.release()
        logger.info("Extracted %d frames to %s", len(saved_paths), output_dir)
        return saved_paths

    # ------------------------------------------------------------------
    # 3. Minimap cropping
    # ------------------------------------------------------------------

    def crop_minimap(
        self,
        frame_path: str | Path,
        region: tuple[int, int, int, int] = DEFAULT_MINIMAP_REGION,
    ) -> Path:
        """Crop the minimap region from a full-screen 1080p frame.

        The cropped image is saved into a ``minimap/`` directory that mirrors
        the source frame directory.

        Args:
            frame_path: Path to the full-screen frame image.
            region: ``(x1, y1, x2, y2)`` pixel coordinates of the minimap.

        Returns:
            Path to the saved cropped minimap image.

        Raises:
            FileNotFoundError: If the source frame does not exist.
        """
        frame_path = Path(frame_path)
        if not frame_path.exists():
            raise FileNotFoundError(f"Frame not found: {frame_path}")

        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise FileNotFoundError(f"Could not read frame: {frame_path}")

        x1, y1, x2, y2 = _scale_region(region, frame.shape)
        cropped = frame[y1:y2, x1:x2]

        # Save alongside the original in a parallel minimap/ directory.
        minimap_dir = ensure_dir(frame_path.parent.parent / "minimap")
        out_path = minimap_dir / frame_path.name
        cv2.imwrite(str(out_path), cropped)
        return out_path

    # ------------------------------------------------------------------
    # 4. HUD region cropping
    # ------------------------------------------------------------------

    def crop_hud_regions(
        self,
        frame_path: str | Path,
        regions: dict[str, tuple[int, int, int, int]] | None = None,
    ) -> dict[str, Path]:
        """Crop multiple HUD regions from a frame.

        Each region is saved under ``hud/<region_name>/`` in a directory
        parallel to the source frame directory.

        Args:
            frame_path: Path to the full-screen frame image.
            regions: Mapping of region names to ``(x1, y1, x2, y2)`` pixel
                     coordinates. If *None*, uses the regions loaded from the
                     pipeline config.

        Returns:
            Dict mapping region names to saved cropped-image paths.

        Raises:
            FileNotFoundError: If the source frame does not exist.
        """
        frame_path = Path(frame_path)
        if not frame_path.exists():
            raise FileNotFoundError(f"Frame not found: {frame_path}")

        if regions is None:
            regions = self.hud_regions

        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise FileNotFoundError(f"Could not read frame: {frame_path}")

        results: dict[str, Path] = {}

        for name, region in regions.items():
            x1, y1, x2, y2 = _scale_region(region, frame.shape)
            cropped = frame[y1:y2, x1:x2]
            region_dir = ensure_dir(frame_path.parent.parent / "hud" / name)
            out_path = region_dir / frame_path.name
            cv2.imwrite(str(out_path), cropped)
            results[name] = out_path

        return results

    # ------------------------------------------------------------------
    # 5. Gameplay phase detection
    # ------------------------------------------------------------------

    @staticmethod
    def detect_frame_phase(frame_or_path) -> str:
        """Classify a frame as 'gameplay', 'draft', 'loading', or 'other'.

        Uses lightweight pixel heuristics on key regions:
        - Minimap region (1650, 790, 1920, 1080): bright green/brown during
          gameplay (mean gray ~60-75), nearly black during draft/loading (~15-17)
        - Top scoreboard bar (0, 0, 1920, 80): consistent dark semi-transparent
          bar during gameplay (mean brightness ~30-50)
        - Bottom 40% of screen: dark draft overlay during champion select

        The checks are pure pixel-level (no OCR, no model loading) so the
        method can run on every frame without slowing down the pipeline.

        Args:
            frame_or_path: Either a file path (str/Path) or a numpy array
                           (BGR image as returned by ``cv2.imread``).

        Returns:
            One of ``'gameplay'``, ``'draft'``, ``'loading'``, or ``'other'``.
        """
        # ---- Load frame ------------------------------------------------
        if isinstance(frame_or_path, np.ndarray):
            frame = frame_or_path
        else:
            frame = cv2.imread(str(frame_or_path))
        if frame is None:
            return "other"

        h, w = frame.shape[:2]

        # ---- Region: minimap (bottom-right corner, ~14% of width/height) -----
        # Scale relative to actual frame size so the detector works at any
        # resolution (1080p, 720p, 480p, etc.). These ratios match the
        # standard LoL spectator layout.
        mm_x1 = int(w * 0.859)  # 1650 / 1920
        mm_y1 = int(h * 0.731)  # 790 / 1080
        mm_x2 = w
        mm_y2 = h
        minimap_roi = frame[mm_y1:mm_y2, mm_x1:mm_x2]
        if minimap_roi.size == 0:
            return "other"
        minimap_gray = cv2.cvtColor(minimap_roi, cv2.COLOR_BGR2GRAY)
        minimap_brightness = float(np.mean(minimap_gray))

        # Green channel in the minimap region — gameplay terrain is
        # distinctly green (mean ~60-65) vs. draft/loading (<20).
        minimap_green = float(np.mean(minimap_roi[:, :, 1]))  # BGR -> G

        # ---- Region: top scoreboard bar (~7.4% of height) --------------
        sb_height = max(1, int(h * 0.074))  # 80 / 1080
        sb_roi = frame[0:sb_height, 0:w]
        sb_gray = cv2.cvtColor(sb_roi, cv2.COLOR_BGR2GRAY)
        sb_brightness = float(np.mean(sb_gray))

        # During gameplay the scoreboard is a dark semi-transparent bar
        # (mean ~30-50) with some bright text/icons.  A completely dark
        # bar (< 10) or a very bright bar (> 80) indicates a non-gameplay
        # phase.
        scoreboard_present = 15 < sb_brightness < 80

        # ---- Region: bottom 40% of screen (draft overlay check) --------
        bottom_roi = frame[int(h * 0.6):h, 0:w]
        bottom_gray = cv2.cvtColor(bottom_roi, cv2.COLOR_BGR2GRAY)
        bottom_brightness = float(np.mean(bottom_gray))

        # ---- Region: full-screen brightness for loading detection ------
        full_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        full_brightness = float(np.mean(full_gray))

        # ---- Classification logic --------------------------------------

        # Gameplay: minimap region is noticeably bright (>40) with a green
        # tint AND the scoreboard bar is present.
        if minimap_brightness > 40 and minimap_green > 35 and scoreboard_present:
            return "gameplay"

        # Draft: minimap region is very dark (<20) and the bottom 40% has
        # the dark draft overlay (brightness < 50) while the upper portion
        # (stage cameras) is brighter.
        upper_roi = frame[0:int(h * 0.6), 0:w]
        upper_brightness = float(np.mean(cv2.cvtColor(upper_roi, cv2.COLOR_BGR2GRAY)))
        if (minimap_brightness < 20
                and bottom_brightness < 50
                and upper_brightness > bottom_brightness + 10):
            return "draft"

        # Loading: the entire screen is mostly dark / uniform.
        if full_brightness < 25:
            return "loading"

        # Everything else (post-game, interviews, ads, replays, etc.)
        return "other"

    def filter_non_gameplay_frames(self, frame_paths: list[Path]) -> list[Path]:
        """Remove non-gameplay frames (draft, loading, replays, post-game).

        Classifies every frame via :meth:`detect_frame_phase` and keeps only
        those labelled ``'gameplay'``.

        Args:
            frame_paths: List of frame image paths.

        Returns:
            Filtered list containing only gameplay frames.
        """
        gameplay_frames: list[Path] = []
        phase_counts: dict[str, int] = {}

        for fp in frame_paths:
            phase = self.detect_frame_phase(fp)
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
            if phase == "gameplay":
                gameplay_frames.append(fp)

        removed = len(frame_paths) - len(gameplay_frames)
        if removed > 0:
            breakdown = ", ".join(
                f"{phase}={count}" for phase, count in sorted(phase_counts.items())
            )
            logger.info(
                "Phase filter: kept %d gameplay frames, removed %d "
                "(%.1f%% of %d total). Breakdown: %s",
                len(gameplay_frames), removed,
                100 * removed / len(frame_paths), len(frame_paths),
                breakdown,
            )
        return gameplay_frames

    # ------------------------------------------------------------------
    # 6. Full pipeline
    # ------------------------------------------------------------------

    def process_vod(
        self,
        url_or_path: str | Path,
        match_id: str,
        output_base_dir: str | Path = "data/raw",
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> Path:
        """Run the full VOD processing pipeline.

        Workflow: download (if URL) -> extract frames (within time range)
        -> filter non-gameplay frames -> crop minimap + HUD for gameplay
        frames only.

        Use *start_time* / *end_time* to coarsely trim the VOD. The phase
        detector then removes remaining draft, loading, and post-game
        frames automatically.

        Output layout::

            {output_base_dir}/{match_id}/
                frames/
                minimap/
                hud/
                    scoreboard/
                    timer/

        Args:
            url_or_path: YouTube URL **or** local path to a video file.
            match_id: Unique identifier for the match (used as directory name).
            output_base_dir: Root directory for all match outputs.
            start_time: VOD timestamp (seconds) where gameplay begins.
            end_time: VOD timestamp (seconds) where gameplay ends.

        Returns:
            Path to the match output directory.
        """
        match_dir = ensure_dir(Path(output_base_dir) / match_id)
        frames_dir = ensure_dir(match_dir / "frames")

        # Step 1: obtain the video file.
        path = Path(url_or_path)
        if path.exists():
            video_path = path
            logger.info("Using local video: %s", video_path)
        else:
            video_path = self.download_vod(str(url_or_path), match_dir)

        # Step 2: extract frames (only the gameplay portion).
        frame_paths = self.extract_frames(
            video_path, frames_dir, start_time=start_time, end_time=end_time,
        )

        # Step 3: filter out non-gameplay frames (draft, loading, etc.).
        frame_paths = self.filter_non_gameplay_frames(frame_paths)

        # Step 4: crop minimap and HUD regions for each frame.
        logger.info("Cropping minimap and HUD regions for %d frames", len(frame_paths))
        for fp in tqdm(frame_paths, desc="Cropping regions", unit="frame"):
            self.crop_minimap(fp)
            self.crop_hud_regions(fp)

        logger.info("Pipeline complete for match %s — output at %s", match_id, match_dir)
        return match_dir

    # ------------------------------------------------------------------
    # 7. Batch processing
    # ------------------------------------------------------------------

    def process_batch(
        self,
        vod_list: list[dict[str, str]],
        output_base_dir: str | Path = "data/raw",
    ) -> list[Path]:
        """Process multiple VODs sequentially.

        Args:
            vod_list: List of dicts, each with keys ``"url"`` and
                      ``"match_id"``, plus optional ``"start_time"`` and
                      ``"end_time"`` (seconds) to trim broadcast content::

                          [
                              {"url": "https://...", "match_id": "fs_match01",
                               "start_time": 1234, "end_time": 3456},
                              {"url": "https://...", "match_id": "fs_match02"},
                          ]
            output_base_dir: Root directory for all match outputs.

        Returns:
            List of match output directory paths.
        """
        logger.info("Starting batch processing of %d VODs", len(vod_list))
        results: list[Path] = []

        for i, entry in enumerate(vod_list, 1):
            url = entry["url"]
            match_id = entry["match_id"]
            start_time = entry.get("start_time")
            end_time = entry.get("end_time")
            logger.info("Processing VOD %d/%d — %s (%s)", i, len(vod_list), match_id, url)

            # Skip already-completed games.
            match_dir = Path(output_base_dir) / match_id
            minimap_dir = match_dir / "minimap"
            if minimap_dir.exists() and len(list(minimap_dir.glob("*.png"))) > 100:
                logger.info("Skipping %s — already processed (%d minimap frames)",
                            match_id, len(list(minimap_dir.glob("*.png"))))
                results.append(match_dir)
                continue

            try:
                result_dir = self.process_vod(
                    url, match_id, output_base_dir,
                    start_time=start_time, end_time=end_time,
                )
                results.append(result_dir)
                # Brief pause between downloads to avoid rate limiting.
                time.sleep(5)
            except Exception:
                logger.exception("Failed to process %s (%s)", match_id, url)

        logger.info(
            "Batch complete — %d/%d VODs processed successfully",
            len(results), len(vod_list),
        )
        return results
