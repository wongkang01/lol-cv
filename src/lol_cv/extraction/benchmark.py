"""
Detection benchmark for comparing YOLO model variants.

Supports side-by-side evaluation of multiple YOLO models (e.g. YOLOv8 vs YOLOv11)
on the same set of minimap frames, computing standard object-detection metrics
(mAP, precision, recall, F1) across a sweep of confidence thresholds.

Designed for RQ2: quantifying detection accuracy on tournament broadcast footage.
"""

import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from lol_cv.utils import setup_logger

logger = setup_logger("lol_cv.extraction.benchmark")

# Default confidence threshold sweep range.
DEFAULT_THRESHOLDS = np.arange(0.20, 0.85, 0.05).round(2).tolist()

# IoU thresholds used for mAP@50:95 (COCO-style).
MAP_IOU_THRESHOLDS = np.arange(0.50, 1.00, 0.05).round(2).tolist()


def compute_iou(box_a: dict, box_b: dict) -> float:
    """Compute Intersection-over-Union between two normalised bounding boxes.

    Each box is a dict with keys ``x_min``, ``y_min``, ``x_max``, ``y_max``
    (all in [0, 1] normalised coordinates).

    Args:
        box_a: First bounding box.
        box_b: Second bounding box.

    Returns:
        IoU value in [0, 1].
    """
    x_left = max(box_a["x_min"], box_b["x_min"])
    y_top = max(box_a["y_min"], box_b["y_min"])
    x_right = min(box_a["x_max"], box_b["x_max"])
    y_bottom = min(box_a["y_max"], box_b["y_max"])

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)
    area_a = (box_a["x_max"] - box_a["x_min"]) * (box_a["y_max"] - box_a["y_min"])
    area_b = (box_b["x_max"] - box_b["x_min"]) * (box_b["y_max"] - box_b["y_min"])
    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def _match_detections(
    detections: list[dict],
    ground_truths: list[dict],
    iou_threshold: float = 0.5,
) -> tuple[int, int, int]:
    """Match detections to ground-truth boxes using greedy IoU matching.

    Each ground-truth box is matched to at most one detection (the highest-IoU
    one that exceeds *iou_threshold*).  Unmatched detections count as false
    positives; unmatched ground truths count as false negatives.

    Args:
        detections: Model outputs, each with ``champion``, ``x_min``, ``y_min``,
            ``x_max``, ``y_max``, ``confidence``.
        ground_truths: Annotations, each with ``champion``, ``x_min``, ``y_min``,
            ``x_max``, ``y_max``.
        iou_threshold: Minimum IoU to consider a match.

    Returns:
        (true_positives, false_positives, false_negatives)
    """
    if not ground_truths:
        return 0, len(detections), 0
    if not detections:
        return 0, 0, len(ground_truths)

    # Sort detections by confidence (descending) for greedy assignment.
    sorted_dets = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    matched_gt: set[int] = set()
    tp = 0
    fp = 0

    for det in sorted_dets:
        best_iou = 0.0
        best_gt_idx = -1
        for gt_idx, gt in enumerate(ground_truths):
            if gt_idx in matched_gt:
                continue
            # Optionally require class match.
            if gt.get("champion") and det.get("champion") and gt["champion"] != det["champion"]:
                continue
            iou = compute_iou(det, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1

    fn = len(ground_truths) - len(matched_gt)
    return tp, fp, fn


def _compute_ap(precisions: list[float], recalls: list[float]) -> float:
    """Compute Average Precision using the 11-point interpolation method.

    Args:
        precisions: Precision values at successive confidence thresholds
            (ordered from high to low confidence).
        recalls: Corresponding recall values.

    Returns:
        Average Precision (area under the interpolated P-R curve).
    """
    if not precisions or not recalls:
        return 0.0

    # Append sentinel values.
    mrec = [0.0] + list(recalls) + [1.0]
    mpre = [1.0] + list(precisions) + [0.0]

    # Make precision monotonically decreasing (right-to-left max).
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # Find recall change-points and sum areas.
    ap = 0.0
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]

    return ap


class DetectionBenchmark:
    """Benchmark harness for comparing YOLO detection models.

    Example usage::

        bench = DetectionBenchmark(
            models={"yolov8": "weights/yolov8.pt", "yolov11": "weights/yolov11.pt"},
        )
        bench.evaluate(frames, annotations)
        df = bench.summary()
    """

    def __init__(
        self,
        models: dict[str, str],
        thresholds: list[float] | None = None,
        iou_threshold: float = 0.5,
        input_size: int = 512,
    ):
        """
        Args:
            models: Mapping of model name to weights path (e.g.
                ``{"yolov8": "data/models/yolov8.pt"}``).
            thresholds: Confidence thresholds to sweep.  Defaults to 0.20-0.80
                in steps of 0.05.
            iou_threshold: IoU threshold used for TP/FP matching at the
                per-threshold level (mAP uses its own sweep).
            input_size: Resize frames to this square size before inference.
        """
        self.model_configs = models
        self.thresholds = thresholds or DEFAULT_THRESHOLDS
        self.iou_threshold = iou_threshold
        self.input_size = input_size

        self._models: dict[str, YOLO] = {}
        self._results: list[dict] = []
        self._pr_curves: dict[str, dict] = {}

        logger.info(
            "Benchmark initialised — %d model(s), %d confidence thresholds",
            len(models),
            len(self.thresholds),
        )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_models(self) -> dict[str, YOLO]:
        """Load all YOLO models into memory.

        Returns:
            Dict mapping model name to loaded YOLO instance.
        """
        for name, path in self.model_configs.items():
            logger.info("Loading model '%s' from %s", name, path)
            model = YOLO(path)
            self._models[name] = model
            logger.info(
                "Model '%s' loaded — %d classes",
                name,
                len(model.names),
            )
        return self._models

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def _prepare_frame(self, frame: np.ndarray | str) -> np.ndarray:
        """Load and resize a frame to the expected input size.

        Args:
            frame: Either a numpy BGR array or a file path string.

        Returns:
            Resized BGR numpy array.

        Raises:
            FileNotFoundError: If *frame* is a path that cannot be read.
        """
        if isinstance(frame, (str, Path)):
            img = cv2.imread(str(frame))
            if img is None:
                raise FileNotFoundError(f"Cannot read image: {frame}")
        else:
            img = frame

        return cv2.resize(img, (self.input_size, self.input_size))

    def _run_model(
        self,
        model: YOLO,
        frame: np.ndarray,
        confidence: float,
    ) -> tuple[list[dict], float]:
        """Run a single model on one frame and return detections + elapsed time.

        Args:
            model: A loaded YOLO model.
            frame: Pre-processed BGR frame.
            confidence: Confidence threshold for this run.

        Returns:
            (detections, elapsed_seconds).  Each detection is a dict with keys
            ``champion``, ``x_min``, ``y_min``, ``x_max``, ``y_max``,
            ``confidence``.
        """
        t0 = time.perf_counter()
        results = model.predict(frame, conf=confidence, verbose=False)
        elapsed = time.perf_counter() - t0

        detections = []
        h, w = frame.shape[:2]
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                "champion": model.names[cls_id],
                "x_min": x1 / w,
                "y_min": y1 / h,
                "x_max": x2 / w,
                "y_max": y2 / h,
                "confidence": float(box.conf[0]),
            })

        return detections, elapsed

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        frames: list[np.ndarray | str],
        annotations: list[list[dict]],
    ) -> None:
        """Run all models on *frames* and compute metrics against *annotations*.

        This is the main entry point.  Results are stored internally and can be
        retrieved via :meth:`summary` or :meth:`get_pr_curve_data`.

        Args:
            frames: List of images (numpy arrays or file paths).
            annotations: Parallel list — one list of ground-truth dicts per
                frame.  Each dict has ``champion``, ``x_min``, ``y_min``,
                ``x_max``, ``y_max`` (normalised [0, 1]).

        Raises:
            ValueError: If *frames* and *annotations* have different lengths.
        """
        if len(frames) != len(annotations):
            raise ValueError(
                f"frames ({len(frames)}) and annotations ({len(annotations)}) "
                "must have the same length"
            )

        if not self._models:
            self.load_models()

        # Pre-load / resize all frames once.
        prepared: list[np.ndarray] = [self._prepare_frame(f) for f in frames]

        self._results.clear()
        self._pr_curves.clear()

        for model_name, model in self._models.items():
            logger.info("Evaluating model '%s' across %d frames", model_name, len(prepared))

            # Per-threshold metrics accumulators.
            threshold_metrics: list[dict] = []

            for conf_thresh in self.thresholds:
                total_tp = 0
                total_fp = 0
                total_fn = 0
                total_time = 0.0

                for frame_arr, gt_boxes in zip(prepared, annotations):
                    dets, elapsed = self._run_model(model, frame_arr, conf_thresh)
                    tp, fp, fn = _match_detections(dets, gt_boxes, self.iou_threshold)
                    total_tp += tp
                    total_fp += fp
                    total_fn += fn
                    total_time += elapsed

                precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
                recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
                f1 = (
                    2 * precision * recall / (precision + recall)
                    if (precision + recall) > 0
                    else 0.0
                )
                fps_value = len(prepared) / total_time if total_time > 0 else 0.0

                threshold_metrics.append({
                    "model": model_name,
                    "confidence_threshold": conf_thresh,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "fps": fps_value,
                    "tp": total_tp,
                    "fp": total_fp,
                    "fn": total_fn,
                })

            # --- Compute mAP ---
            # For mAP we run at a very low confidence so we get all possible
            # detections, then vary the IoU threshold.
            all_dets_per_frame: list[list[dict]] = []
            map_time = 0.0
            for frame_arr in prepared:
                dets, elapsed = self._run_model(model, frame_arr, confidence=0.01)
                all_dets_per_frame.append(dets)
                map_time += elapsed

            map50 = self._compute_map_at_iou(all_dets_per_frame, annotations, 0.50)
            map50_95 = float(np.mean([
                self._compute_map_at_iou(all_dets_per_frame, annotations, t)
                for t in MAP_IOU_THRESHOLDS
            ]))

            # Attach mAP to every threshold row (it is threshold-independent).
            for row in threshold_metrics:
                row["mAP@50"] = map50
                row["mAP@50:95"] = map50_95

            self._results.extend(threshold_metrics)

            # Store P-R curve data.
            self._pr_curves[model_name] = {
                "thresholds": [m["confidence_threshold"] for m in threshold_metrics],
                "precision": [m["precision"] for m in threshold_metrics],
                "recall": [m["recall"] for m in threshold_metrics],
                "f1": [m["f1"] for m in threshold_metrics],
            }

            # Log best F1.
            best = max(threshold_metrics, key=lambda m: m["f1"])
            logger.info(
                "Model '%s' — best F1=%.3f at conf=%.2f  |  mAP@50=%.3f  mAP@50:95=%.3f",
                model_name,
                best["f1"],
                best["confidence_threshold"],
                map50,
                map50_95,
            )

    # ------------------------------------------------------------------
    # mAP helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_map_at_iou(
        all_detections: list[list[dict]],
        all_annotations: list[list[dict]],
        iou_threshold: float,
    ) -> float:
        """Compute mean Average Precision at a single IoU threshold.

        Pools all detections across frames, ranks by confidence, and computes
        per-class AP using 11-point interpolation.

        Args:
            all_detections: Detections per frame (from a low-confidence run).
            all_annotations: Ground-truth annotations per frame.
            iou_threshold: IoU threshold for TP/FP assignment.

        Returns:
            mAP value (macro-averaged over classes present in ground truth).
        """
        # Collect per-class data.
        class_dets: dict[str, list[tuple[float, dict, int]]] = {}
        class_gt: dict[str, list[tuple[dict, int]]] = {}

        for frame_idx, (dets, gts) in enumerate(zip(all_detections, all_annotations)):
            for gt in gts:
                cls = gt.get("champion", "unknown")
                class_gt.setdefault(cls, []).append((gt, frame_idx))
            for det in dets:
                cls = det.get("champion", "unknown")
                class_dets.setdefault(cls, []).append(
                    (det["confidence"], det, frame_idx)
                )

        if not class_gt:
            return 0.0

        aps: list[float] = []

        for cls in class_gt:
            gt_items = class_gt[cls]
            n_gt = len(gt_items)
            if n_gt == 0:
                continue

            det_items = class_dets.get(cls, [])
            # Sort detections by confidence descending.
            det_items.sort(key=lambda x: x[0], reverse=True)

            # Build frame-indexed gt lookup.
            gt_by_frame: dict[int, list[tuple[dict, bool]]] = {}
            for gt_box, fidx in gt_items:
                gt_by_frame.setdefault(fidx, []).append([gt_box, False])

            precisions = []
            recalls = []
            tp_cum = 0
            fp_cum = 0

            for _, det_box, fidx in det_items:
                matched = False
                if fidx in gt_by_frame:
                    best_iou = 0.0
                    best_idx = -1
                    for gi, (gt_box, already_matched) in enumerate(gt_by_frame[fidx]):
                        if already_matched:
                            continue
                        iou = compute_iou(det_box, gt_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_idx = gi
                    if best_iou >= iou_threshold and best_idx >= 0:
                        gt_by_frame[fidx][best_idx][1] = True
                        tp_cum += 1
                        matched = True

                if not matched:
                    fp_cum += 1

                precisions.append(tp_cum / (tp_cum + fp_cum))
                recalls.append(tp_cum / n_gt)

            ap = _compute_ap(precisions, recalls)
            aps.append(ap)

        return float(np.mean(aps)) if aps else 0.0

    # ------------------------------------------------------------------
    # Results access
    # ------------------------------------------------------------------

    def get_pr_curve_data(self) -> dict[str, dict]:
        """Return precision-recall curve data for each model.

        Returns:
            Dict mapping model name to a dict with keys ``thresholds``,
            ``precision``, ``recall``, ``f1`` (each a list of floats).
        """
        return self._pr_curves

    def get_optimal_thresholds(self) -> dict[str, dict]:
        """Find the confidence threshold that maximises F1 for each model.

        Returns:
            Dict mapping model name to ``{"threshold": float, "f1": float,
            "precision": float, "recall": float}``.
        """
        df = self._results_dataframe()
        if df.empty:
            return {}

        result: dict[str, dict] = {}
        for model_name, group in df.groupby("model"):
            best_row = group.loc[group["f1"].idxmax()]
            result[model_name] = {
                "threshold": float(best_row["confidence_threshold"]),
                "f1": float(best_row["f1"]),
                "precision": float(best_row["precision"]),
                "recall": float(best_row["recall"]),
            }
        return result

    def _results_dataframe(self) -> pd.DataFrame:
        """Return the raw results as a DataFrame (internal helper)."""
        if not self._results:
            return pd.DataFrame()
        return pd.DataFrame(self._results)

    def summary(self) -> pd.DataFrame:
        """Return a comparison DataFrame summarising each model at its optimal threshold.

        Columns: ``model``, ``optimal_conf``, ``precision``, ``recall``, ``f1``,
        ``mAP@50``, ``mAP@50:95``, ``fps``.

        Returns:
            Summary DataFrame with one row per model, sorted by F1 descending.
        """
        df = self._results_dataframe()
        if df.empty:
            logger.warning("No results — call evaluate() first")
            return pd.DataFrame()

        rows: list[dict] = []
        for model_name, group in df.groupby("model"):
            best = group.loc[group["f1"].idxmax()]
            rows.append({
                "model": model_name,
                "optimal_conf": float(best["confidence_threshold"]),
                "precision": float(best["precision"]),
                "recall": float(best["recall"]),
                "f1": float(best["f1"]),
                "mAP@50": float(best["mAP@50"]),
                "mAP@50:95": float(best["mAP@50:95"]),
                "fps": float(best["fps"]),
            })

        summary_df = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
        logger.info("Summary:\n%s", summary_df.to_string(index=False))
        return summary_df

    def full_results(self) -> pd.DataFrame:
        """Return the full per-threshold results for all models.

        Returns:
            DataFrame with one row per (model, confidence_threshold) pair.
        """
        return self._results_dataframe()
