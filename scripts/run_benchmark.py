"""
RQ2 — Benchmark YOLOv11 size variants on tournament minimap frames.

Compares yolov11n / yolov11m / yolov11l / yolov11x on a sample of frames
drawn from across the dataset. Since we don't have hand-labeled ground
truth, we use the LARGEST model (yolov11x) as pseudo-ground-truth and
measure smaller models' agreement with it.

Metrics reported per model:
    - Inference speed (ms per frame, FPS)
    - Mean detection count (how many champions detected per frame)
    - Mean confidence
    - Agreement with yolov11x (precision / recall against pseudo-GT)

Output:
    data/processed/analysis/benchmark.csv
    data/processed/analysis/plots/benchmark.png
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lol_cv.extraction.minimap import MinimapTracker
from lol_cv.utils import setup_logger, ensure_dir

logger = setup_logger("scripts.run_benchmark")

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
MODELS_DIR = REPO_ROOT / "data" / "models"
ANALYSIS = REPO_ROOT / "data" / "processed" / "analysis"
PLOTS = ANALYSIS / "plots"
PICKS_PATH = REPO_ROOT / "data" / "champion_picks.json"

MODEL_VARIANTS = {
    "yolov11n": MODELS_DIR / "yolov11n.pt",
    "yolov11m": MODELS_DIR / "yolov11m.pt",
    "yolov11l": MODELS_DIR / "yolov11l.pt",
    "yolov11x": MODELS_DIR / "yolov11x.pt",
}

# Sample 200 frames total — distributed across all available games for diversity
N_FRAMES = 200


def sample_frames() -> list[tuple[Path, set[str]]]:
    """Return list of (frame_path, valid_champions) tuples sampled across games."""
    with open(PICKS_PATH) as f:
        picks = json.load(f)

    game_pattern = re.compile(r"_g\d+$")
    game_dirs = [
        d for d in RAW_DIR.iterdir()
        if d.is_dir() and game_pattern.search(d.name) and (d / "minimap").exists()
    ]
    if not game_dirs:
        return []

    frames_per_game = max(1, N_FRAMES // len(game_dirs))
    sampled = []
    for d in game_dirs:
        if d.name not in picks:
            continue
        valid = set(picks[d.name]["all_champions"])
        all_frames = sorted((d / "minimap").glob("frame_*.png"))
        if not all_frames:
            continue
        step = max(1, len(all_frames) // frames_per_game)
        sampled.extend((p, valid) for p in all_frames[::step][:frames_per_game])
        if len(sampled) >= N_FRAMES:
            break

    return sampled[:N_FRAMES]


def benchmark_model(
    model_path: Path, samples: list[tuple[Path, set[str]]]
) -> tuple[pd.DataFrame, list[list[dict]]]:
    """Run model on samples; return per-frame stats + raw detections."""
    tracker = MinimapTracker(model_path=str(model_path), confidence=0.4)
    tracker.load_model()

    rows = []
    raw_detections = []
    for frame_path, valid in samples:
        img = cv2.imread(str(frame_path))
        if img is None:
            raw_detections.append([])
            continue
        t0 = time.perf_counter()
        dets = tracker.detect_frame(img)
        elapsed = time.perf_counter() - t0
        filtered = [d for d in dets if d["champion"] in valid]
        raw_detections.append(filtered)
        rows.append({
            "frame": frame_path.name,
            "n_detections_raw": len(dets),
            "n_detections_filtered": len(filtered),
            "mean_conf_raw": float(np.mean([d["confidence"] for d in dets])) if dets else 0.0,
            "mean_conf_filt": float(np.mean([d["confidence"] for d in filtered])) if filtered else 0.0,
            "ms_per_frame": elapsed * 1000.0,
        })
    return pd.DataFrame(rows), raw_detections


def agreement_with_reference(
    candidate: list[list[dict]], reference: list[list[dict]], dist_threshold: float = 0.05
) -> dict:
    """Compute precision/recall of candidate vs reference using normalised position matching.

    A candidate detection matches a reference detection iff:
        - Same champion class
        - Euclidean (x,y) distance < dist_threshold (in [0,1] minimap coords)
    """
    tp, fp, fn = 0, 0, 0
    for c_dets, r_dets in zip(candidate, reference):
        matched_ref = set()
        for c in c_dets:
            best_idx = -1
            best_dist = dist_threshold
            for i, r in enumerate(r_dets):
                if i in matched_ref or r["champion"] != c["champion"]:
                    continue
                d = ((c["x"] - r["x"]) ** 2 + (c["y"] - r["y"]) ** 2) ** 0.5
                if d < best_dist:
                    best_dist = d
                    best_idx = i
            if best_idx >= 0:
                tp += 1
                matched_ref.add(best_idx)
            else:
                fp += 1
        fn += len(r_dets) - len(matched_ref)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def main() -> None:
    ensure_dir(ANALYSIS)
    ensure_dir(PLOTS)

    samples = sample_frames()
    logger.info("Sampled %d frames across games", len(samples))
    if not samples:
        logger.error("No frames available — run detection first?")
        return

    summary_rows = []
    raw_per_model: dict[str, list[list[dict]]] = {}

    for name, path in MODEL_VARIANTS.items():
        if not path.exists():
            logger.warning("Skip %s — weights missing at %s", name, path)
            continue
        logger.info("Benchmarking %s", name)
        df, raws = benchmark_model(path, samples)
        raw_per_model[name] = raws

        size_mb = path.stat().st_size / 1_000_000
        summary_rows.append({
            "model": name,
            "size_mb": round(size_mb, 1),
            "ms_per_frame_mean": float(df["ms_per_frame"].mean()),
            "ms_per_frame_p95": float(df["ms_per_frame"].quantile(0.95)),
            "fps": 1000.0 / float(df["ms_per_frame"].mean()),
            "mean_dets_raw": float(df["n_detections_raw"].mean()),
            "mean_dets_filt": float(df["n_detections_filtered"].mean()),
            "mean_conf_raw": float(df["mean_conf_raw"].mean()),
            "mean_conf_filt": float(df["mean_conf_filt"].mean()),
        })

    if not summary_rows:
        logger.error("No models ran")
        return

    summary = pd.DataFrame(summary_rows).set_index("model")

    # Agreement with largest model (yolov11x) as pseudo-GT
    if "yolov11x" in raw_per_model:
        ref = raw_per_model["yolov11x"]
        for name, raws in raw_per_model.items():
            if name == "yolov11x":
                summary.loc[name, "precision_vs_x"] = 1.0
                summary.loc[name, "recall_vs_x"] = 1.0
                summary.loc[name, "f1_vs_x"] = 1.0
            else:
                agree = agreement_with_reference(raws, ref)
                summary.loc[name, "precision_vs_x"] = agree["precision"]
                summary.loc[name, "recall_vs_x"] = agree["recall"]
                summary.loc[name, "f1_vs_x"] = agree["f1"]

    summary.to_csv(ANALYSIS / "benchmark.csv")
    logger.info("\n%s", summary.to_string())

    # Plot: speed vs agreement
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(summary.index, summary["fps"], color="#3B82F6")
    axes[0].set_ylabel("FPS (CPU)")
    axes[0].set_title("Inference speed (CPU, 512×512 input)")
    axes[0].grid(axis="y", alpha=0.3)

    if "f1_vs_x" in summary.columns:
        axes[1].bar(summary.index, summary["f1_vs_x"], color="#10B981")
        axes[1].set_ylabel("F1 vs YOLOv11x (pseudo-GT)")
        axes[1].set_ylim(0, 1.05)
        axes[1].set_title("Agreement with largest model")
        axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle("YOLOv11 size variant benchmark — RQ2")
    fig.tight_layout()
    fig.savefig(PLOTS / "benchmark.png", dpi=150)
    plt.close(fig)
    logger.info("Saved %s", PLOTS / "benchmark.png")


if __name__ == "__main__":
    main()
