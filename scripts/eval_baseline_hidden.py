"""Phase 3: measure BASELINE hidden-state quality (G2 metrics) on MOT17 val-half.

Consumes:
  results/occlusion_segments.json      (extract_occlusion_segments.py, full run)
  results/raw/val_half/trackers/bytetrack_ours/<seq>.txt   (run_baseline.py --tracker ours)
  results/raw/val_half/coasting/<seq>.txt                  (Kalman coasting, conf 0)

Segments carry ORIGINAL frame numbers; tracker output is rebased to 1 within the val
half — segment frames are shifted by -(val_start_frame - 1) before evaluation. Only
segments fully inside the val half participate (the extractor already ran on val GT).

Writes results/baseline_hidden.json — the empirical floor used to re-freeze G2
thresholds (the hidden-state module must beat these numbers).
"""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omot.eval.hidden_eval import SegmentResult, aggregate, evaluate_segments  # noqa: E402
from omot.eval.occlusion import OcclusionSegment  # noqa: E402
from omot.io.mot_format import read_mot  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
log = logging.getLogger("eval_baseline_hidden")

ROOT = Path(__file__).resolve().parents[1]


def seg_from_dict(d: dict) -> OcclusionSegment:
    return OcclusionSegment(
        track_id=int(d["track_id"]),
        last_visible_frame=int(d["last_visible_frame"]),
        reemergence_frame=int(d["reemergence_frame"]),
        last_visible_box=np.array(d["last_visible_box"], dtype=np.float64),
        reemergence_box=np.array(d["reemergence_box"], dtype=np.float64),
    )


def to_eval_rows(mot_rows: np.ndarray) -> np.ndarray:
    """MOT rows (frame, id, x, y, w, h, ...) -> evaluator layout (same, first 6 cols)."""
    return mot_rows[:, :6]


def main() -> int:
    seg_index = json.loads(
        (ROOT / "results" / "occlusion_segments.json").read_text(encoding="utf-8")
    )
    if seg_index.get("partial"):
        raise SystemExit("occlusion_segments.json is a partial preview — rerun extractor first")

    trk_dir = ROOT / "results" / "raw" / "val_half" / "trackers" / "bytetrack_ours"
    coast_dir = ROOT / "results" / "raw" / "val_half" / "coasting"
    all_results: list[SegmentResult] = []
    per_seq: dict[str, dict] = {}

    for seq_name, entry in seg_index["sequences"].items():
        segs_raw = entry["val_half"]
        if not segs_raw:
            per_seq[seq_name] = {"n_segments": 0}
            continue
        shift = int(entry["val_start_frame"]) - 1
        segments = [
            replace(
                seg_from_dict(d),
                last_visible_frame=int(d["last_visible_frame"]) - shift,
                reemergence_frame=int(d["reemergence_frame"]) - shift,
            )
            for d in segs_raw
        ]
        active = to_eval_rows(read_mot(trk_dir / f"{seq_name}.txt"))
        coast_path = coast_dir / f"{seq_name}.txt"
        coasting = (
            to_eval_rows(read_mot(coast_path)) if coast_path.exists() else np.zeros((0, 6))
        )
        results = evaluate_segments(segments, active, coasting, float(entry["diagonal"]))
        all_results.extend(results)
        per_seq[seq_name] = aggregate(results)
        log.info("%s: %s", seq_name, per_seq[seq_name])

    overall = aggregate(all_results)
    out = {"overall": overall, "per_sequence": per_seq}
    dest = ROOT / "results" / "baseline_hidden.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log.info("OVERALL baseline G2 floor: %s", overall)
    log.info("-> %s", dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
