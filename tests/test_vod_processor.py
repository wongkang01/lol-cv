"""Unit tests for the VOD processing pipeline."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import cv2
import numpy as np
import pytest

from lol_cv.extraction.vod_processor import (
    DEFAULT_HUD_REGIONS,
    DEFAULT_MINIMAP_REGION,
    VodProcessor,
)


@pytest.fixture
def processor():
    """Create a VodProcessor with default HUD regions (no config file needed)."""
    config = {"extraction": {"ocr": {"regions": {}}}}
    return VodProcessor(config=config)


@pytest.fixture
def fake_frame(tmp_path):
    """Create a real 1920x1080 image file and return its path.

    The image is placed inside a ``frames/`` subdirectory so that the
    crop methods can write to sibling directories (``minimap/``, ``hud/``).
    """
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    # Paint a recognisable colour block in the minimap region so we can
    # verify the crop content later.
    x1, y1, x2, y2 = DEFAULT_MINIMAP_REGION
    img[y1:y2, x1:x2] = (0, 255, 0)  # green
    frame_path = frames_dir / "frame_000010.png"
    cv2.imwrite(str(frame_path), img)
    return frame_path


# ── download_vod ───────────────────────────────────────────────────


class TestDownloadVod:
    def test_raises_when_ytdlp_not_installed(self, processor, tmp_path):
        with patch("lol_cv.extraction.vod_processor.shutil.which", return_value=None):
            with pytest.raises(FileNotFoundError, match="yt-dlp is not installed"):
                processor.download_vod("https://youtube.com/watch?v=abc", tmp_path)

    def test_calls_subprocess_with_correct_args(self, processor, tmp_path):
        # Create a dummy mp4 so the post-download glob finds something.
        dummy_mp4 = tmp_path / "video.mp4"
        dummy_mp4.write_bytes(b"\x00")

        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch("lol_cv.extraction.vod_processor.shutil.which", return_value="/usr/bin/yt-dlp"), \
             patch("lol_cv.extraction.vod_processor.subprocess.run", return_value=mock_result) as mock_run:
            result = processor.download_vod("https://youtube.com/watch?v=abc", tmp_path, quality="720p")

            mock_run.assert_called_once()
            cmd = mock_run.call_args[0][0]
            assert cmd[0] == "yt-dlp"
            assert "720" in cmd[cmd.index("-f") + 1]
            assert "--no-playlist" in cmd
            assert "https://youtube.com/watch?v=abc" in cmd

        assert result == dummy_mp4

    def test_raises_runtime_error_on_nonzero_exit(self, processor, tmp_path):
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "some error"

        with patch("lol_cv.extraction.vod_processor.shutil.which", return_value="/usr/bin/yt-dlp"), \
             patch("lol_cv.extraction.vod_processor.subprocess.run", return_value=mock_result):
            with pytest.raises(RuntimeError, match="yt-dlp failed"):
                processor.download_vod("https://youtube.com/watch?v=abc", tmp_path)


# ── extract_frames ─────────────────────────────────────────────────


class TestExtractFrames:
    def test_raises_for_nonexistent_video(self, processor, tmp_path):
        fake_video = tmp_path / "nonexistent.mp4"
        with pytest.raises(FileNotFoundError, match="Cannot open video"):
            processor.extract_frames(fake_video, tmp_path / "out")

    def test_extracts_correct_number_of_frames(self, processor, tmp_path):
        """Mock cv2.VideoCapture to simulate a 30fps video with 90 frames.

        At fps=1, we expect frame_interval = 30, so 90 // 30 = 3 output
        frames (indices 0, 30, 60).
        """
        output_dir = tmp_path / "frames_out"

        fake_frame_img = np.zeros((100, 100, 3), dtype=np.uint8)
        total_frames = 90
        video_fps = 30.0
        call_count = {"n": 0}

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: video_fps,
            cv2.CAP_PROP_FRAME_COUNT: total_frames,
        }[prop]

        def fake_read():
            if call_count["n"] < total_frames:
                call_count["n"] += 1
                return True, fake_frame_img.copy()
            return False, None

        mock_cap.read.side_effect = fake_read

        with patch("lol_cv.extraction.vod_processor.cv2.VideoCapture", return_value=mock_cap), \
             patch("lol_cv.extraction.vod_processor.cv2.imwrite") as mock_imwrite:
            paths = processor.extract_frames(tmp_path / "fake.mp4", output_dir, fps=1)

        assert len(paths) == 3
        assert mock_imwrite.call_count == 3

    def test_frame_filename_pattern(self, processor, tmp_path):
        """Frame filenames should follow frame_SSSSSS.png pattern."""
        output_dir = tmp_path / "frames_out"

        fake_frame_img = np.zeros((100, 100, 3), dtype=np.uint8)
        total_frames = 60
        video_fps = 30.0
        call_count = {"n": 0}

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: video_fps,
            cv2.CAP_PROP_FRAME_COUNT: total_frames,
        }[prop]

        def fake_read():
            if call_count["n"] < total_frames:
                call_count["n"] += 1
                return True, fake_frame_img.copy()
            return False, None

        mock_cap.read.side_effect = fake_read

        with patch("lol_cv.extraction.vod_processor.cv2.VideoCapture", return_value=mock_cap), \
             patch("lol_cv.extraction.vod_processor.cv2.imwrite"):
            paths = processor.extract_frames(tmp_path / "fake.mp4", output_dir, fps=1)

        for p in paths:
            assert p.name.startswith("frame_")
            assert p.name.endswith(".png")
            # The numeric part between "frame_" and ".png" should be 6 digits.
            numeric_part = p.stem.split("_")[1]
            assert len(numeric_part) == 6
            assert numeric_part.isdigit()

    def test_fps_ratio_affects_output_count(self, processor, tmp_path):
        """Extracting at fps=10 from a 30fps video with 90 frames should
        produce 90 // 3 = 30 output frames."""
        output_dir = tmp_path / "frames_out"

        fake_frame_img = np.zeros((100, 100, 3), dtype=np.uint8)
        total_frames = 90
        video_fps = 30.0
        call_count = {"n": 0}

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: video_fps,
            cv2.CAP_PROP_FRAME_COUNT: total_frames,
        }[prop]

        def fake_read():
            if call_count["n"] < total_frames:
                call_count["n"] += 1
                return True, fake_frame_img.copy()
            return False, None

        mock_cap.read.side_effect = fake_read

        with patch("lol_cv.extraction.vod_processor.cv2.VideoCapture", return_value=mock_cap), \
             patch("lol_cv.extraction.vod_processor.cv2.imwrite"):
            paths = processor.extract_frames(tmp_path / "fake.mp4", output_dir, fps=10)

        # frame_interval = 30 / 10 = 3, so 90 / 3 = 30 frames
        assert len(paths) == 30


# ── crop_minimap ───────────────────────────────────────────────────


class TestCropMinimap:
    def test_raises_for_nonexistent_frame(self, processor, tmp_path):
        with pytest.raises(FileNotFoundError, match="Frame not found"):
            processor.crop_minimap(tmp_path / "nope.png")

    def test_creates_cropped_image_in_minimap_dir(self, processor, fake_frame):
        result = processor.crop_minimap(fake_frame)

        assert result.exists()
        assert result.parent.name == "minimap"
        assert result.name == fake_frame.name

    def test_cropped_dimensions(self, processor, fake_frame):
        x1, y1, x2, y2 = DEFAULT_MINIMAP_REGION
        expected_w = x2 - x1
        expected_h = y2 - y1

        result = processor.crop_minimap(fake_frame)
        cropped = cv2.imread(str(result))

        assert cropped.shape[1] == expected_w  # width
        assert cropped.shape[0] == expected_h  # height

    def test_cropped_content(self, processor, fake_frame):
        """The minimap region was painted green in the fixture; verify it."""
        result = processor.crop_minimap(fake_frame)
        cropped = cv2.imread(str(result))

        # Every pixel in the cropped minimap should be green.
        assert np.all(cropped[:, :, 0] == 0)    # B
        assert np.all(cropped[:, :, 1] == 255)  # G
        assert np.all(cropped[:, :, 2] == 0)    # R


# ── crop_hud_regions ───────────────────────────────────────────────


class TestCropHudRegions:
    def test_raises_for_nonexistent_frame(self, processor, tmp_path):
        with pytest.raises(FileNotFoundError, match="Frame not found"):
            processor.crop_hud_regions(tmp_path / "missing.png")

    def test_creates_subdirectories(self, processor, fake_frame):
        results = processor.crop_hud_regions(fake_frame)

        for region_name, path in results.items():
            assert path.exists()
            assert path.parent.name == region_name
            assert path.parent.parent.name == "hud"
            assert path.name == fake_frame.name

    def test_returns_correct_dict_mapping(self, processor, fake_frame):
        results = processor.crop_hud_regions(fake_frame)

        # Default regions are "scoreboard" and "timer".
        assert set(results.keys()) == set(DEFAULT_HUD_REGIONS.keys())
        for name in DEFAULT_HUD_REGIONS:
            assert name in results

    def test_cropped_dimensions(self, processor, fake_frame):
        results = processor.crop_hud_regions(fake_frame)

        for name, (x1, y1, x2, y2) in DEFAULT_HUD_REGIONS.items():
            expected_w = x2 - x1
            expected_h = y2 - y1
            cropped = cv2.imread(str(results[name]))
            assert cropped.shape[1] == expected_w, f"{name} width mismatch"
            assert cropped.shape[0] == expected_h, f"{name} height mismatch"

    def test_custom_regions(self, processor, fake_frame):
        """Passing explicit regions overrides the defaults."""
        custom = {"my_region": (0, 0, 100, 50)}
        results = processor.crop_hud_regions(fake_frame, regions=custom)

        assert set(results.keys()) == {"my_region"}
        cropped = cv2.imread(str(results["my_region"]))
        assert cropped.shape == (50, 100, 3)


# ── process_batch ──────────────────────────────────────────────────


class TestProcessBatch:
    def test_handles_exceptions_gracefully(self, processor, tmp_path):
        """If one VOD fails, the batch should continue processing the rest."""
        vod_list = [
            {"url": "https://fail.example/1", "match_id": "match_fail"},
            {"url": "https://fail.example/2", "match_id": "match_also_fail"},
        ]

        with patch.object(processor, "process_vod", side_effect=RuntimeError("boom")):
            results = processor.process_batch(vod_list, output_base_dir=tmp_path)

        # Both failed, so no results, but no exception propagated.
        assert results == []

    def test_successful_vods_returned(self, processor, tmp_path):
        """Successful VOD outputs should be collected in the results list."""
        vod_list = [
            {"url": "https://ok.example/1", "match_id": "match_ok_1"},
            {"url": "https://ok.example/2", "match_id": "match_ok_2"},
        ]

        def fake_process_vod(url, match_id, output_base_dir, **kwargs):
            return Path(output_base_dir) / match_id

        with patch.object(processor, "process_vod", side_effect=fake_process_vod):
            results = processor.process_batch(vod_list, output_base_dir=tmp_path)

        assert len(results) == 2
        assert results[0] == tmp_path / "match_ok_1"
        assert results[1] == tmp_path / "match_ok_2"

    def test_partial_failure(self, processor, tmp_path):
        """If one VOD fails and one succeeds, only the success is returned."""
        vod_list = [
            {"url": "https://fail.example/1", "match_id": "match_fail"},
            {"url": "https://ok.example/2", "match_id": "match_ok"},
        ]

        def fake_process_vod(url, match_id, output_base_dir, **kwargs):
            if match_id == "match_fail":
                raise RuntimeError("download failed")
            return Path(output_base_dir) / match_id

        with patch.object(processor, "process_vod", side_effect=fake_process_vod):
            results = processor.process_batch(vod_list, output_base_dir=tmp_path)

        assert len(results) == 1
        assert results[0] == tmp_path / "match_ok"
