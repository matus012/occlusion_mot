"""Phase 2: run a tracker over MOT17 val-half on cached detections -> TrackEval -> results JSON.

Trackers:
  ours      — our ByteTracker (src/omot/track/bytetrack.py)
  reference — supervision's ByteTrack (installed in phase 2) on the SAME cached detections

FIXED DETECTIONS invariant (context.md D1): both trackers read the same npz caches created by
scripts/cache_detections.py. Usage:
  .venv/Scripts/python.exe scripts/run_baseline.py --tracker ours
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omot.data.mot import export_half_gt, half_split_frames, load_split  # noqa: E402
from omot.detect.cache import cache_path, load_cached_detections  # noqa: E402
from omot.eval.trackeval_runner import run_trackeval  # noqa: E402
from omot.io.mot_format import write_mot  # noqa: E402
from omot.track.bytetrack import ByteTracker  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
logger = logging.getLogger("run_baseline")

ROOT = Path(__file__).resolve().parents[1]


def track_ours(
    dets_by_frame: dict[int, np.ndarray], frames: range
) -> tuple[np.ndarray, np.ndarray]:
    """Run our ByteTracker over `frames` (rebased to 1).

    Returns (active_rows, coasting_rows) as MOT rows [frame, id, x, y, w, h, conf];
    coasting rows carry conf 0 (Kalman predictions of lost tracks, consumed by G2 eval).
    """
    tracker = ByteTracker()
    rows: list[np.ndarray] = []
    coast: list[np.ndarray] = []
    for i, f in enumerate(frames, start=1):
        dets = dets_by_frame.get(f, np.zeros((0, 5)))
        for x, y, w, h, score, tid in tracker.update(dets, frame_id=i):
            rows.append(np.array([i, tid, x, y, w, h, score]))
        for x, y, w, h, score, tid in tracker.coasting:
            coast.append(np.array([i, tid, x, y, w, h, score]))
    to_arr = lambda r: np.stack(r) if r else np.zeros((0, 7))  # noqa: E731
    return to_arr(rows), to_arr(coast)


def track_reference(dets_by_frame: dict[int, np.ndarray], frames: range) -> np.ndarray:
    """Reference ByteTrack (supervision) on identical detections; returns MOT rows."""
    import supervision as sv

    tracker = sv.ByteTrack()
    rows: list[np.ndarray] = []
    for i, f in enumerate(frames, start=1):
        dets = dets_by_frame.get(f, np.zeros((0, 5)))
        xyxy = dets[:, :4].copy()
        xyxy[:, 2:] += xyxy[:, :2]
        detections = sv.Detections(
            xyxy=xyxy.astype(np.float32),
            confidence=dets[:, 4].astype(np.float32),
            class_id=np.zeros(len(dets), dtype=int),
        )
        tracked = tracker.update_with_detections(detections)
        for (x1, y1, x2, y2), score, tid in zip(
            tracked.xyxy, tracked.confidence, tracked.tracker_id, strict=True
        ):
            rows.append(np.array([i, tid, x1, y1, x2 - x1, y2 - y1, score]))
    return np.stack(rows) if rows else np.zeros((0, 7))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracker", choices=["ours", "reference"], default="ours")
    ap.add_argument("--data-root", type=Path, default=ROOT / "data" / "MOT17")
    ap.add_argument("--cache-dir", type=Path, default=ROOT / "data" / "cache" / "detections")
    ap.add_argument("--model", default="yolo11x", help="detector cache name stem")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    seqs = load_split(args.data_root, "train", detector="FRCNN")
    work = ROOT / "results" / "raw" / "val_half"
    gt_root = work / "gt"
    seq_info = export_half_gt(seqs, gt_root, half="val")

    tracker_name = "bytetrack_ours" if args.tracker == "ours" else "bytetrack_reference"
    trk_dir = work / "trackers" / tracker_name
    trk_dir.mkdir(parents=True, exist_ok=True)

    for seq in seqs:
        npz = cache_path(args.cache_dir, seq.name, args.model)
        if not npz.exists():
            raise FileNotFoundError(
                f"detection cache missing: {npz} — run scripts/cache_detections.py first"
            )
        dets = load_cached_detections(npz)
        _, val = half_split_frames(seq.seq_length)
        if args.tracker == "ours":
            rows, coast = track_ours(dets, val)
            write_mot(work / "coasting" / f"{seq.name}.txt", coast)
        else:
            rows = track_reference(dets, val)
        write_mot(trk_dir / f"{seq.name}.txt", rows)
        logger.info("%s: %d output rows", seq.name, len(rows))

    out_json = ROOT / "results" / (
        "baseline_ours.json" if args.tracker == "ours" else "baseline_reference.json"
    )
    metrics = run_trackeval(
        gt_root, work / "trackers", tracker_name, seq_info, output_json=out_json
    )
    print(f"\n=== {tracker_name} on MOT17 val-half ===")
    for k, v in metrics.items():
        print(f"  {k:>8}: {v:.2f}" if isinstance(v, float) else f"  {k:>8}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
