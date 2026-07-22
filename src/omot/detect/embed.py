"""Appearance embeddings for cached detections (D20).

One GPU pass per sequence: crop every cached detection box from its frame, embed with a
torchvision ResNet18 (ImageNet) trunk, L2-normalize, store alongside the detection cache.
The tracker consumes embeddings by (frame, det_index) — image-free at association time,
same invariant as fixed detections (D1). Embedder is swappable (phase 6: CARLA/PERUN-trained).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from omot.data.mot import MOTSequence
from omot.detect.cache import load_cached_detections, select_device, set_seeds

logger = logging.getLogger(__name__)

EMBED_DIM = 512
CROP_HW = (128, 64)  # h, w — person aspect


def embed_cache_path(cache_dir: Path, seq_name: str, model_name: str) -> Path:
    return cache_dir / f"{seq_name}__{model_name}__emb.npz"


def _build_embedder(device: str):  # -> tuple[torch.nn.Module, callable]
    import torch
    from torchvision.models import ResNet18_Weights, resnet18

    weights = ResNet18_Weights.IMAGENET1K_V1
    net = resnet18(weights=weights)
    net.fc = torch.nn.Identity()
    net.eval().to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def normalize(batch: torch.Tensor) -> torch.Tensor:
        return (batch / 255.0 - mean) / std

    return net, normalize


def cache_embeddings(
    seq: MOTSequence,
    det_cache: Path,
    cache_dir: Path,
    device: str | None = None,
    batch_size: int = 256,
    seed: int = 0,
    overwrite: bool = False,
) -> Path:
    """Embed every cached detection of `seq`; save (frames, embeddings) npz; return path."""
    import cv2
    import torch

    model_stem = det_cache.stem.split("__")[1]
    out = embed_cache_path(cache_dir, seq.name, model_stem)
    if out.exists() and not overwrite:
        logger.info("embedding cache hit: %s", out)
        return out

    set_seeds(seed)
    dev = select_device(device)
    net, normalize = _build_embedder(dev)
    dets = load_cached_detections(det_cache)

    frames_col: list[np.ndarray] = []
    embs_col: list[np.ndarray] = []
    batch: list[np.ndarray] = []
    batch_frames: list[int] = []

    def flush() -> None:
        if not batch:
            return
        arr = np.stack(batch)  # (B, H, W, 3) uint8
        with torch.no_grad():
            t = torch.from_numpy(arr).to(dev).permute(0, 3, 1, 2).float()
            emb = net(normalize(t)).cpu().numpy().astype(np.float32)
        emb /= np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8, None)
        embs_col.append(emb)
        frames_col.append(np.array(batch_frames, dtype=np.int32))
        batch.clear()
        batch_frames.clear()

    for frame in range(1, seq.seq_length + 1):
        boxes = dets.get(frame)
        if boxes is None or not len(boxes):
            continue
        img = cv2.imread(str(seq.frame_path(frame)))
        assert img is not None, f"frame missing: {seq.frame_path(frame)}"
        ih, iw = img.shape[:2]
        for box in boxes:
            x, y, w, h = box[:4]
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2, y2 = min(iw, int(x + w)), min(ih, int(y + h))
            if x2 - x1 < 4 or y2 - y1 < 8:
                crop = np.zeros((*CROP_HW, 3), dtype=np.uint8)
            else:
                crop = cv2.resize(img[y1:y2, x1:x2], (CROP_HW[1], CROP_HW[0]))
            batch.append(crop[:, :, ::-1].copy())  # BGR -> RGB
            batch_frames.append(frame)
            if len(batch) >= batch_size:
                flush()
    flush()

    frames = np.concatenate(frames_col) if frames_col else np.zeros(0, dtype=np.int32)
    embs = (
        np.concatenate(embs_col) if embs_col else np.zeros((0, EMBED_DIM), dtype=np.float32)
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, frames=frames, embeddings=embs, embedder="resnet18_in1k")
    logger.info("%s: embedded %d detections -> %s", seq.name, len(embs), out)
    return out


def load_cached_embeddings(path: Path) -> dict[int, np.ndarray]:
    """npz -> {frame: (M, D) L2-normalized embeddings, row-aligned with the detection
    cache's per-frame box order}."""
    blob = np.load(path, allow_pickle=False)
    frames: np.ndarray = blob["frames"]
    embs: np.ndarray = blob["embeddings"]
    return {int(f): embs[frames == f] for f in np.unique(frames)}
