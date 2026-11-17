"""Stage 0 (perun_detector_v1.md): synthesise a PERFECT-DETECTOR detection cache from GT.

Writes caches byte-compatible with `omot.detect.cache.cache_detections`, so every
downstream consumer (cache_embeddings.py, run_hidden.py --model <tag>) treats them as an
ordinary detector. Nothing about the tracker or the eval changes; only the detection
source does. That is the whole point of the Stage-0 kill gate: it bounds what ANY
detector improvement could possibly buy.

WHY TWO VARIANTS (specification gap resolved, D54)
--------------------------------------------------
perun_detector_v1.md section 2 says "MOT17 dev-half GT boxes as the detection stream" and
justifies the kill criterion with "GT boxes are the supremum of any detector's output".
Those two statements are not the same set, and the difference changes the verdict:

  gtvis  -- GT rows with visibility > 0. A detector can only fire on pixels it can see,
            so this is the honest supremum of a DETECTOR and it drives the kill criterion.
  gtall  -- every consider-flagged pedestrian row regardless of visibility, including
            fully occluded targets. No detector can produce this (it sees through
            occluders); reported as an informative upper-upper bound only.

Using gtall for the kill would be too permissive: it could clear 0.55 in a world where no
real detector ever could, and the workstream would spend ~21 h chasing an unreachable
ceiling. Using gtvis is faithful to the document's own stated rationale. Both are emitted
and both are reported.

Usage:
  .venv/Scripts/python.exe scripts/cache_gt_detections.py --variant gtvis
  .venv/Scripts/python.exe scripts/cache_gt_detections.py --variant gtall
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omot.data.mot import load_split  # noqa: E402
from omot.detect.cache import cache_path  # noqa: E402
from omot.io.mot_format import COL  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
logger = logging.getLogger("cache_gt_detections")

ROOT = Path(__file__).resolve().parents[1]

VARIANTS = {
    # tag -> minimum GT visibility for a row to become a detection
    "gtvis": 0.0,   # strictly greater than 0 (see _select)
    "gtall": None,  # no visibility filter at all
}


def _select(gt: np.ndarray, min_vis: float | None) -> np.ndarray:
    """Consider-flagged pedestrians, optionally filtered by visibility.

    Class/consider convention matches omot.eval.occlusion.extract_segments exactly
    (CLS == 1, CONF == 1) so the detection stream and the segment definition agree on
    what a 'person' is.
    """
    keep = (gt[:, COL.CLS] == 1.0) & (gt[:, COL.CONF] == 1.0)
    if min_vis is not None:
        keep &= gt[:, COL.VIS] > min_vis
    return gt[keep]


def build_gt_cache(
    seq_dir_name: str, gt: np.ndarray, out: Path, min_vis: float | None, tag: str
) -> tuple[int, int]:
    """Write one npz in cache_detections' exact format. Returns (n_frames, n_boxes)."""
    rows = _select(gt, min_vis)
    order = np.argsort(rows[:, COL.FRAME], kind="stable")
    rows = rows[order]

    frames = rows[:, COL.FRAME].astype(np.int64)
    boxes = np.empty((len(rows), 5), dtype=np.float32)
    boxes[:, 0] = rows[:, COL.X]
    boxes[:, 1] = rows[:, COL.Y]
    boxes[:, 2] = rows[:, COL.W]
    boxes[:, 3] = rows[:, COL.H]
    boxes[:, 4] = 1.0  # a perfect detector is perfectly confident

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, frames=frames, boxes=boxes, model=tag, seed=0)
    return len(np.unique(frames)), len(frames)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=ROOT / "data" / "MOT17")
    ap.add_argument("--cache-dir", type=Path, default=ROOT / "data" / "cache" / "detections")
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="gtvis")
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    min_vis = VARIANTS[args.variant]
    seqs = load_split(args.data_root, "train", detector="FRCNN")
    if args.only is not None:
        seqs = [s for s in seqs if s.name == args.only]
        if not seqs:
            raise SystemExit(f"sequence {args.only!r} not found")

    total_boxes = 0
    for seq in seqs:
        assert seq.gt is not None, f"{seq.name}: no GT (Stage 0 needs the train split)"
        out = cache_path(args.cache_dir, seq.name, args.variant)
        n_frames, n_boxes = build_gt_cache(seq.name, seq.gt, out, min_vis, args.variant)
        total_boxes += n_boxes
        logger.info(
            "%s: %d boxes over %d/%d frames -> %s",
            seq.name, n_boxes, n_frames, seq.seq_length, out.name,
        )
    logger.info("variant %s: %d sequences, %d boxes total", args.variant, len(seqs), total_boxes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
