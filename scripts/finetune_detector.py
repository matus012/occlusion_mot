"""L5: MOT-finetune a person detector on MOT17 DEV-HALF frames only (prototype).

D18 invariant: dataset built strictly from dev-half frames (1..seq_length//2 of each
FRCNN train sequence) — asserted on frame indices, never the val half. Within the dev
half, the last ~10% of each sequence's frames are held out as a monitoring val split
(NOT the eval val-half — purely for train-loop early-stopping signal); the rest is
train. Detector output feeds a NEW cache tag via cache_detections.py --weights/--model
(FIXED DETECTIONS invariant, context.md D1) — never overwrites the yolo11x cache.

Usage:
  .venv/Scripts/python.exe scripts/finetune_detector.py --prep-only
  .venv/Scripts/python.exe scripts/finetune_detector.py --epochs 10 --tag y11s_proto
"""
from __future__ import annotations

import argparse
import logging
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omot.data.mot import MOTSequence, half_split_frames, load_split  # noqa: E402
from omot.detect.cache import select_device  # noqa: E402
from omot.io.mot_format import COL  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
                    stream=sys.stdout)
logger = logging.getLogger("finetune_detector")

ROOT = Path(__file__).resolve().parents[1]


def set_seeds(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _split_dev_frames(seq: MOTSequence, val_frac: float) -> tuple[list[int], list[int]]:
    """(train_frames, monitor_val_frames) within the dev half only. D18 hard requirement:
    every returned frame index is <= seq_length // 2 (asserted)."""
    dev, _val = half_split_frames(seq.seq_length)
    dev_frames = list(dev)
    mid = seq.seq_length // 2
    assert all(f <= mid for f in dev_frames), (
        f"{seq.name}: dev-half frame leaked past mid={mid} (D18 violation)"
    )
    n_dev = len(dev_frames)
    n_val = max(1, round(n_dev * val_frac)) if n_dev > 1 else 0
    train_frames = dev_frames[: n_dev - n_val]
    monitor_frames = dev_frames[n_dev - n_val :]
    assert set(train_frames).isdisjoint(monitor_frames)
    assert set(train_frames) | set(monitor_frames) == set(dev_frames)
    return train_frames, monitor_frames


def _yolo_label_lines(
    gt: np.ndarray, frame: int, min_vis: float, img_w: int, img_h: int
) -> list[str]:
    """GT rows for `frame` with class==pedestrian(1), conf flag==1, vis>=min_vis ->
    YOLO txt lines (class 0, normalized cx cy w h, clipped to [0, 1])."""
    mask = (
        (gt[:, COL.FRAME].astype(np.int64) == frame)
        & (gt[:, COL.CLS].astype(np.int64) == 1)
        & (gt[:, COL.CONF].astype(np.int64) == 1)
        & (gt[:, COL.VIS] >= min_vis)
    )
    lines: list[str] = []
    for row in gt[mask]:
        x, y, w, h = row[COL.X], row[COL.Y], row[COL.W], row[COL.H]
        cx = np.clip((x + w / 2) / img_w, 0.0, 1.0)
        cy = np.clip((y + h / 2) / img_h, 0.0, 1.0)
        nw = np.clip(w / img_w, 0.0, 1.0)
        nh = np.clip(h / img_h, 0.0, 1.0)
        lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    return lines


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _expected_counts(seqs: list[MOTSequence], val_frac: float) -> tuple[int, int]:
    n_train = n_val = 0
    for seq in seqs:
        train_frames, monitor_frames = _split_dev_frames(seq, val_frac)
        n_train += len(train_frames)
        n_val += len(monitor_frames)
    return n_train, n_val


def prepare_dataset(
    data_root: Path, out_dir: Path, min_vis: float, val_frac: float = 0.1
) -> tuple[Path, dict[str, int]]:
    """Build the YOLO-format dataset from MOT17 dev-half GT. Idempotent: if `out_dir`
    already holds the expected image counts, skip rebuilding and reuse it."""
    seqs = load_split(data_root, "train", detector="FRCNN")
    assert seqs, f"no FRCNN train sequences found under {data_root}"
    n_train_expected, n_val_expected = _expected_counts(seqs, val_frac)

    dataset_yaml = out_dir / "dataset.yaml"
    img_train, img_val = out_dir / "images" / "train", out_dir / "images" / "val"
    lbl_train, lbl_val = out_dir / "labels" / "train", out_dir / "labels" / "val"

    if dataset_yaml.exists():
        n_train_actual = len(list(img_train.glob("*.jpg"))) if img_train.is_dir() else 0
        n_val_actual = len(list(img_val.glob("*.jpg"))) if img_val.is_dir() else 0
        if n_train_actual == n_train_expected and n_val_actual == n_val_expected:
            logger.info(
                "dataset exists with matching frame counts (train=%d val=%d) -> skip prep: %s",
                n_train_actual, n_val_actual, out_dir,
            )
            return dataset_yaml, {"train": n_train_actual, "val": n_val_actual}
        logger.info(
            "dataset dir stale (found train=%d val=%d, expected train=%d val=%d) -> rebuilding",
            n_train_actual, n_val_actual, n_train_expected, n_val_expected,
        )
        shutil.rmtree(out_dir)

    for d in (img_train, img_val, lbl_train, lbl_val):
        d.mkdir(parents=True, exist_ok=True)

    n_train = n_val = 0
    for seq in seqs:
        assert seq.gt is not None, f"{seq.name}: GT required for detector finetuning"
        train_frames, monitor_frames = _split_dev_frames(seq, val_frac)
        for frames, img_dir, lbl_dir in (
            (train_frames, img_train, lbl_train),
            (monitor_frames, img_val, lbl_val),
        ):
            for frame in frames:
                assert frame <= seq.seq_length // 2, (
                    f"{seq.name}: frame {frame} beyond dev half (D18 violation)"
                )
                stem = f"{seq.name}_{frame:06d}"
                _link_or_copy(seq.frame_path(frame), img_dir / f"{stem}.jpg")
                lines = _yolo_label_lines(seq.gt, frame, min_vis, seq.img_width, seq.img_height)
                (lbl_dir / f"{stem}.txt").write_text(
                    "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
                )
        n_train += len(train_frames)
        n_val += len(monitor_frames)

    dataset_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(out_dir.resolve()),
                "train": "images/train",
                "val": "images/val",
                "names": {0: "person"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    logger.info("built dataset: train=%d val=%d -> %s", n_train, n_val, out_dir)
    return dataset_yaml, {"train": n_train, "val": n_val}


def train(
    dataset_yaml: Path,
    project: Path,
    tag: str,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
    seed: int,
    base_weights: str,
    workers: int = 2,
) -> Path:
    from ultralytics import YOLO

    model = YOLO(base_weights)
    results = model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        seed=seed,
        single_cls=True,
        project=str(project),
        name=tag,
        exist_ok=True,
        workers=workers,
    )
    metrics: Any = getattr(results, "results_dict", results)
    logger.info("final metrics: %s", metrics)
    best_path = project / tag / "weights" / "best.pt"
    logger.info("best checkpoint: %s", best_path)
    return best_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=ROOT / "data" / "MOT17")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data" / "cache" / "det_finetune")
    ap.add_argument("--min-vis", type=float, default=0.1)
    ap.add_argument("--val-frac", type=float, default=0.1,
                    help="fraction of each sequence's dev-half frames held out for monitoring")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None, help="cuda|cpu (default: auto)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="y11s_proto", help="run name under --project")
    ap.add_argument("--project", type=Path, default=ROOT / "data" / "models" / "det_finetune")
    ap.add_argument("--base-weights", default="yolo11s.pt")
    ap.add_argument("--prep-only", action="store_true",
                    help="build the dataset and exit before training")
    args = ap.parse_args()

    set_seeds(args.seed)
    device = select_device(args.device)

    dataset_yaml, counts = prepare_dataset(
        args.data_root, args.out_dir, args.min_vis, args.val_frac
    )
    logger.info("dataset ready (train=%d val=%d): %s", counts["train"], counts["val"],
                dataset_yaml)

    if args.prep_only:
        logger.info("prep-only: stopping before training")
        return 0

    best_path = train(
        dataset_yaml=dataset_yaml,
        project=args.project,
        tag=args.tag,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        seed=args.seed,
        base_weights=args.base_weights,
    )
    logger.info("done -> %s", best_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
