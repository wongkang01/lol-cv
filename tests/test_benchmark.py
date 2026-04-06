"""Unit tests for detection benchmark pure functions and class scaffolding."""

import sys
from unittest.mock import MagicMock

import pandas as pd
import pytest

# Prevent the top-level `from ultralytics import YOLO` from failing
# in environments without ultralytics installed, and avoid loading a
# real model in any test.
sys.modules.setdefault("ultralytics", MagicMock())

from lol_cv.extraction.benchmark import (
    DetectionBenchmark,
    _compute_ap,
    _match_detections,
    compute_iou,
)


# ── Helper factories ───────────────────────────────────────────────


def _box(x_min: float, y_min: float, x_max: float, y_max: float) -> dict:
    """Create a normalised bounding-box dict."""
    return {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max}


def _det(
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    confidence: float = 0.9,
    champion: str = "Jinx",
) -> dict:
    """Create a detection dict."""
    return {
        "x_min": x_min,
        "y_min": y_min,
        "x_max": x_max,
        "y_max": y_max,
        "confidence": confidence,
        "champion": champion,
    }


def _gt(
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    champion: str = "Jinx",
) -> dict:
    """Create a ground-truth dict."""
    return {
        "x_min": x_min,
        "y_min": y_min,
        "x_max": x_max,
        "y_max": y_max,
        "champion": champion,
    }


# ── compute_iou ────────────────────────────────────────────────────


class TestComputeIoU:
    def test_perfect_overlap(self):
        """Identical boxes must give IoU = 1.0."""
        box = _box(0.1, 0.1, 0.5, 0.5)
        assert compute_iou(box, box) == pytest.approx(1.0)

    def test_no_overlap(self):
        """Disjoint boxes must give IoU = 0.0."""
        a = _box(0.0, 0.0, 0.2, 0.2)
        b = _box(0.5, 0.5, 0.8, 0.8)
        assert compute_iou(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self):
        """Partially overlapping boxes produce a known IoU value."""
        a = _box(0.0, 0.0, 0.4, 0.4)  # area = 0.16
        b = _box(0.2, 0.2, 0.6, 0.6)  # area = 0.16
        # intersection: [0.2,0.2]->[0.4,0.4] = 0.2*0.2 = 0.04
        # union: 0.16 + 0.16 - 0.04 = 0.28
        expected = 0.04 / 0.28
        assert compute_iou(a, b) == pytest.approx(expected)

    def test_one_box_inside_the_other(self):
        """A small box fully contained in a large one."""
        outer = _box(0.0, 0.0, 1.0, 1.0)  # area = 1.0
        inner = _box(0.2, 0.2, 0.4, 0.4)  # area = 0.04
        # intersection = inner area = 0.04, union = 1.0
        assert compute_iou(outer, inner) == pytest.approx(0.04)

    def test_edge_touching_boxes(self):
        """Boxes that share an edge but have no overlapping area -> IoU = 0."""
        a = _box(0.0, 0.0, 0.5, 0.5)
        b = _box(0.5, 0.0, 1.0, 0.5)
        assert compute_iou(a, b) == pytest.approx(0.0)

    def test_zero_area_box(self):
        """A zero-area (degenerate) box must give IoU = 0."""
        point = _box(0.3, 0.3, 0.3, 0.3)
        normal = _box(0.0, 0.0, 0.5, 0.5)
        assert compute_iou(point, normal) == pytest.approx(0.0)

    def test_zero_area_both_boxes(self):
        """Two identical zero-area boxes must still give IoU = 0 (no area)."""
        point = _box(0.5, 0.5, 0.5, 0.5)
        assert compute_iou(point, point) == pytest.approx(0.0)


# ── _match_detections ──────────────────────────────────────────────


class TestMatchDetections:
    def test_empty_detections_some_gts(self):
        """No detections means all GTs are false negatives."""
        tp, fp, fn = _match_detections([], [_gt(0, 0, 0.5, 0.5), _gt(0.5, 0.5, 1, 1)])
        assert (tp, fp, fn) == (0, 0, 2)

    def test_some_detections_empty_gts(self):
        """No GTs means all detections are false positives."""
        tp, fp, fn = _match_detections(
            [_det(0, 0, 0.5, 0.5), _det(0.5, 0.5, 1, 1)], []
        )
        assert (tp, fp, fn) == (0, 2, 0)

    def test_both_empty(self):
        """Empty inputs produce (0, 0, 0)."""
        assert _match_detections([], []) == (0, 0, 0)

    def test_perfect_matches(self):
        """Each detection exactly matches a GT -> all true positives."""
        gts = [_gt(0, 0, 0.5, 0.5), _gt(0.5, 0.5, 1, 1)]
        dets = [_det(0, 0, 0.5, 0.5, confidence=0.9), _det(0.5, 0.5, 1, 1, confidence=0.8)]
        tp, fp, fn = _match_detections(dets, gts)
        assert (tp, fp, fn) == (2, 0, 0)

    def test_all_detections_miss(self):
        """Detections that don't overlap any GT are all FP, GTs are all FN."""
        gts = [_gt(0.0, 0.0, 0.1, 0.1)]
        dets = [_det(0.8, 0.8, 1.0, 1.0)]
        tp, fp, fn = _match_detections(dets, gts)
        assert (tp, fp, fn) == (0, 1, 1)

    def test_greedy_matching_highest_confidence_first(self):
        """The higher-confidence detection should claim the GT first.

        Two detections overlap the same GT box.  The greedy algorithm should
        assign the higher-confidence detection to the GT, and the lower-
        confidence one becomes a false positive.
        """
        gt = _gt(0.0, 0.0, 0.5, 0.5)
        high = _det(0.0, 0.0, 0.5, 0.5, confidence=0.95)
        low = _det(0.0, 0.0, 0.5, 0.5, confidence=0.50)
        tp, fp, fn = _match_detections([low, high], [gt])
        # high claims gt -> tp=1; low unmatched -> fp=1; no remaining gt -> fn=0
        assert (tp, fp, fn) == (1, 1, 0)

    def test_class_mismatch_no_match(self):
        """Detection and GT with different champion names must not match."""
        gt = _gt(0.0, 0.0, 0.5, 0.5, champion="Thresh")
        det = _det(0.0, 0.0, 0.5, 0.5, confidence=0.9, champion="Jinx")
        tp, fp, fn = _match_detections([det], [gt])
        assert (tp, fp, fn) == (0, 1, 1)

    def test_iou_below_threshold(self):
        """Overlap exists but is below the iou_threshold -> no match."""
        gt = _gt(0.0, 0.0, 0.5, 0.5)
        # Slightly overlapping box
        det = _det(0.4, 0.4, 0.9, 0.9)
        tp, fp, fn = _match_detections([det], [gt], iou_threshold=0.5)
        assert tp == 0
        assert fp == 1
        assert fn == 1


# ── _compute_ap ────────────────────────────────────────────────────


class TestComputeAP:
    def test_perfect_classifier(self):
        """Constant precision=1.0 across all recall levels -> AP = 1.0."""
        precisions = [1.0, 1.0, 1.0, 1.0]
        recalls = [0.25, 0.50, 0.75, 1.0]
        assert _compute_ap(precisions, recalls) == pytest.approx(1.0)

    def test_empty_lists(self):
        """Empty inputs -> AP = 0.0."""
        assert _compute_ap([], []) == pytest.approx(0.0)

    def test_monotonically_decreasing_precision(self):
        """P-R curve with decreasing precision yields a known AP.

        Sentinels: mrec = [0, 0.5, 1.0, 1.0], mpre = [1.0, 0.8, 0.4, 0.0]
        After right-to-left max: mpre = [1.0, 0.8, 0.4, 0.0]
        Change-points at i=1 (0.0->0.5) and i=2 (0.5->1.0) and i=3 (1.0->1.0).
        AP = (0.5-0.0)*0.8 + (1.0-0.5)*0.4 + (1.0-1.0)*0.0 = 0.4+0.2 = 0.6
        """
        precisions = [0.8, 0.4]
        recalls = [0.5, 1.0]
        assert _compute_ap(precisions, recalls) == pytest.approx(0.6)

    def test_single_point(self):
        """A single (precision, recall) pair.

        Sentinels: mrec = [0, 0.5, 1.0], mpre = [1.0, 0.9, 0.0]
        After right-to-left max: mpre = [1.0, 0.9, 0.0]
        AP = (0.5-0.0)*0.9 + (1.0-0.5)*0.0 = 0.45
        """
        assert _compute_ap([0.9], [0.5]) == pytest.approx(0.45)


# ── DetectionBenchmark (no YOLO) ───────────────────────────────────


class TestDetectionBenchmarkInit:
    """Test the class scaffolding without loading any real model."""

    @pytest.fixture
    def bench(self):
        return DetectionBenchmark(
            models={"yolov8": "weights/yolov8.pt"},
            thresholds=[0.3, 0.5, 0.7],
            iou_threshold=0.5,
            input_size=640,
        )

    def test_stores_config(self, bench):
        assert bench.model_configs == {"yolov8": "weights/yolov8.pt"}
        assert bench.thresholds == [0.3, 0.5, 0.7]
        assert bench.iou_threshold == 0.5
        assert bench.input_size == 640

    def test_default_thresholds_when_none(self):
        bench = DetectionBenchmark(models={"m": "p"}, thresholds=None)
        assert len(bench.thresholds) > 0

    def test_pr_curve_data_empty_before_evaluate(self, bench):
        assert bench.get_pr_curve_data() == {}

    def test_optimal_thresholds_empty_before_evaluate(self, bench):
        assert bench.get_optimal_thresholds() == {}

    def test_summary_empty_before_evaluate(self, bench):
        df = bench.summary()
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_results_dataframe_empty_before_evaluate(self, bench):
        df = bench._results_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert df.empty
