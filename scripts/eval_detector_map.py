"""Detector mAP50-95 measured ON MOT17 dev-half, from the cached detections (D60).

WHY THIS EXISTS. perun_detector_v1.md section 3 fixes the dose-response x-axis as
"detector `mAP50-95` measured on MOT17 dev-half". The first Stage-1 implementation read
mAP straight out of ultralytics' `results.csv`, which is validation on the TRAINING mix's
own held-out split (MOT20 frames) -- a different dataset from the y-axis. Symptom on the
first four units: mAP spanned 0.6011-0.6060 (range 0.005) while oracle_ceiling spanned
0.25-0.33. Regressing a MOT17 quantity on a MOT20 quantity with no x-variance would have
produced a meaningless slope and a garbage CI.

This module computes the x-axis where the doc says it lives, from the SAME cached
detections the tracker consumes -- so the two axes describe one detector on one dataset.

No re-training is needed: detections are already cached per detector tag.

  python scripts/eval_detector_map.py --model <cache_tag>
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omot.data.mot import half_split_frames, load_split  # noqa: E402
from omot.detect.cache import cache_path, load_cached_detections  # noqa: E402
from omot.io.mot_format import COL  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
logger = logging.getLogger("eval_detector_map")

ROOT = Path(__file__).resolve().parents[1]
IOU_THRESHOLDS = np.arange(0.5, 1.0, 0.05)  # COCO 0.50:0.05:0.95


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between two sets of tlwh boxes -> (len(a), len(b))."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    ax1, ay1 = a[:, 0, None], a[:, 1, None]
    ax2, ay2 = ax1 + a[:, 2, None], ay1 + a[:, 3, None]
    bx1, by1 = b[None, :, 0], b[None, :, 1]
    bx2, by2 = bx1 + b[None, :, 2], by1 + b[None, :, 3]
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = iw * ih
    union = (a[:, 2, None] * a[:, 3, None]) + (b[None, :, 2] * b[None, :, 3]) - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def average_precision(tp: np.ndarray, conf: np.ndarray, n_gt: int) -> float:
    """101-point interpolated AP (COCO convention) over score-sorted detections."""
    if n_gt == 0:
        return float("nan")
    if len(tp) == 0:
        return 0.0
    order = np.argsort(-conf)
    tp = tp[order]
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(1.0 - tp)
    recall = cum_tp / n_gt
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
    # make precision monotonically decreasing, then sample at 101 recall points
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    grid = np.linspace(0, 1, 101)
    idx = np.searchsorted(recall, grid, side="left")
    sampled = np.where(idx < len(precision), precision[np.clip(idx, 0, len(precision) - 1)], 0.0)
    return float(sampled.mean())


def evaluate(model_tag: str, data_root: Path, cache_dir: Path,
             min_vis: float = 0.0) -> dict[str, float]:
    """mAP50, mAP50-95 for `model_tag` over MOT17 dev-half.

    GT is consider-flagged pedestrians (CLS==1, CONF==1) with visibility > min_vis --
    the same population the detection stream is expected to cover.
    """
    seqs = load_split(data_root, "train", detector="FRCNN")
    per_iou_tp: dict[float, list[np.ndarray]] = {t: [] for t in IOU_THRESHOLDS}
    confs: list[np.ndarray] = []
    n_gt = 0

    for seq in seqs:
        assert seq.gt is not None, f"{seq.name}: GT required"
        dev_frames, _ = half_split_frames(seq.seq_length)
        dets_by_frame = load_cached_detections(cache_path(cache_dir, seq.name, model_tag))
        gt = seq.gt
        keep = (gt[:, COL.CLS] == 1.0) & (gt[:, COL.CONF] == 1.0) & (gt[:, COL.VIS] > min_vis)
        gt = gt[keep]

        for frame in dev_frames:
            g = gt[gt[:, COL.FRAME] == frame]
            gboxes = g[:, [COL.X, COL.Y, COL.W, COL.H]].astype(np.float64)
            n_gt += len(gboxes)
            d = dets_by_frame.get(frame)
            if d is None or len(d) == 0:
                continue
            dboxes = np.asarray(d[:, :4], dtype=np.float64)
            scores = np.asarray(d[:, 4], dtype=np.float64)
            order = np.argsort(-scores)
            dboxes, scores = dboxes[order], scores[order]
            confs.append(scores)
            ious = iou_matrix(dboxes, gboxes)
            for thr in IOU_THRESHOLDS:
                tp = np.zeros(len(dboxes))
                taken = np.zeros(len(gboxes), dtype=bool)
                for di in range(len(dboxes)):
                    if len(gboxes) == 0:
                        break
                    cand = np.where((ious[di] >= thr) & ~taken)[0]
                    if len(cand):
                        best = cand[np.argmax(ious[di, cand])]
                        taken[best] = True
                        tp[di] = 1.0
                per_iou_tp[thr].append(tp)

    conf_all = np.concatenate(confs) if confs else np.zeros(0)
    aps = {}
    for thr in IOU_THRESHOLDS:
        tp_all = np.concatenate(per_iou_tp[thr]) if per_iou_tp[thr] else np.zeros(0)
        aps[round(float(thr), 2)] = average_precision(tp_all, conf_all, n_gt)

    valid = [v for v in aps.values() if not np.isnan(v)]
    out = {
        "model": model_tag,
        "n_gt": n_gt,
        "n_det": int(len(conf_all)),
        "mAP50_mot17dev": aps[0.5],
        "mAP50_95_mot17dev": float(np.mean(valid)) if valid else float("nan"),
        "per_iou": aps,
        "note": "measured on MOT17 dev-half from the cached detections (D60)",
    }
    logger.info("%s: mAP50=%.4f mAP50-95=%.4f over %d GT / %d dets",
                model_tag, out["mAP50_mot17dev"], out["mAP50_95_mot17dev"],
                n_gt, out["n_det"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="detection cache tag")
    ap.add_argument("--data-root", type=Path, default=ROOT / "data" / "MOT17")
    ap.add_argument("--cache-dir", type=Path, default=ROOT / "data" / "cache" / "detections")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    res = evaluate(args.model, args.data_root, args.cache_dir)
    dest = args.out or (ROOT / "results" / f"detmap_{args.model}.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(res, indent=2), encoding="utf-8")
    logger.info("-> %s", dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
