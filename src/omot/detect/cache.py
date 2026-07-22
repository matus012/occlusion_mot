"""Run a detector once per sequence and cache person detections to .npz.

FIXED DETECTIONS invariant (context.md D1): trackers are always compared on identical
cached detections. This module is the only place detector inference happens.
ultralytics (AGPL-3.0, context.md D9) is imported lazily so the rest of the package
works without it (e.g. synthetic tests, CI without GPU).
"""
from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np

from omot.data.mot import MOTSequence

logger = logging.getLogger(__name__)

PERSON_CLASS = 0  # COCO


def select_device(requested: str | None = None) -> str:
    """Injected-device policy with CPU fallback (no hardcoded cuda:0 at call sites)."""
    import torch

    if requested is not None:
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_seeds(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def cache_path(cache_dir: Path, seq_name: str, model_name: str) -> Path:
    return cache_dir / f"{seq_name}__{model_name}.npz"


def cache_detections(
    seq: MOTSequence,
    cache_dir: Path,
    model_name: str = "yolo11x.pt",
    device: str | None = None,
    conf_floor: float = 0.05,
    seed: int = 0,
    overwrite: bool = False,
) -> Path:
    """Detect persons on every frame of `seq`; save (frames, boxes) npz; return path.

    boxes: (N, 5) [x, y, w, h, score] tlwh; frames: (N,) 1-based frame ids.
    """
    out = cache_path(cache_dir, seq.name, Path(model_name).stem)
    if out.exists() and not overwrite:
        logger.info("cache hit: %s", out)
        return out

    from ultralytics import YOLO

    set_seeds(seed)
    dev = select_device(device)
    model = YOLO(model_name)
    logger.info("caching detections: %s -> %s (device=%s)", seq.name, out, dev)

    frames_col: list[np.ndarray] = []
    boxes_col: list[np.ndarray] = []
    for frame in range(1, seq.seq_length + 1):
        img = seq.frame_path(frame)
        if not img.exists():
            raise FileNotFoundError(f"frame image missing: {img}")
        result = model.predict(
            source=str(img), device=dev, classes=[PERSON_CLASS], conf=conf_floor,
            verbose=False, imgsz=1280,
        )[0]
        xywh = result.boxes.xywh.cpu().numpy()  # center-based
        conf = result.boxes.conf.cpu().numpy()
        tlwh = xywh.copy()
        tlwh[:, 0] -= tlwh[:, 2] / 2
        tlwh[:, 1] -= tlwh[:, 3] / 2
        boxes = np.hstack([tlwh, conf[:, None]]).astype(np.float32)
        frames_col.append(np.full(len(boxes), frame, dtype=np.int32))
        boxes_col.append(boxes)

    frames = np.concatenate(frames_col) if frames_col else np.zeros(0, dtype=np.int32)
    boxes = np.concatenate(boxes_col) if boxes_col else np.zeros((0, 5), dtype=np.float32)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, frames=frames, boxes=boxes, model=Path(model_name).stem, seed=seed)
    logger.info("cached %d detections over %d frames", len(boxes), seq.seq_length)
    return out


def load_cached_detections(path: Path) -> dict[int, np.ndarray]:
    """npz -> {frame: (M, 5) [x, y, w, h, score]}; frames with zero detections are absent
    (consumers use .get(frame, empty))."""
    blob = np.load(path, allow_pickle=False)
    frames: np.ndarray = blob["frames"]
    boxes: np.ndarray = blob["boxes"]
    out: dict[int, np.ndarray] = {}
    for f in np.unique(frames):
        out[int(f)] = boxes[frames == f].astype(np.float64)
    return out
