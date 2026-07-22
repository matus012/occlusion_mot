"""Phase 3: extract occlusion segments from MOT17 train GT (D7 definition, gates.yaml G2).

Per sequence, segments are extracted from the FULL GT (for threshold design stats) and from
the VAL half only (frames > seq_length//2 — the G2 evaluation set; dev half stays reserved
for hidden-state tuning). Writes results/occlusion_segments.json.

Usage: .venv/Scripts/python.exe scripts/extract_occlusion_segments.py [--partial]
  --partial: allow running on however many sequences are downloaded (progress preview);
             the final committed artifact must be produced without --partial (all 7).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omot.data.mot import load_split  # noqa: E402
from omot.eval.occlusion import OcclusionSegment, extract_segments, segment_stats  # noqa: E402
from omot.io.mot_format import COL  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
log = logging.getLogger("extract_occlusion_segments")

ROOT = Path(__file__).resolve().parents[1]
N_TRAIN_SEQS = 7
VIS_LO, VIS_HI, MIN_LEN = 0.25, 0.5, 5


def seg_to_dict(seg: OcclusionSegment) -> dict:
    d = asdict(seg)
    d["last_visible_box"] = [float(v) for v in seg.last_visible_box]
    d["reemergence_box"] = [float(v) for v in seg.reemergence_box]
    d["gap_length"] = seg.gap_length
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=ROOT / "data" / "MOT17")
    ap.add_argument("--partial", action="store_true")
    args = ap.parse_args()

    seqs = load_split(args.data_root, "train", detector="FRCNN")
    if len(seqs) < N_TRAIN_SEQS and not args.partial:
        raise SystemExit(
            f"only {len(seqs)}/{N_TRAIN_SEQS} sequences present; use --partial for a preview"
        )

    out: dict[str, dict] = {
        "params": {"vis_lo": VIS_LO, "vis_hi": VIS_HI, "min_len": MIN_LEN},
        "partial": bool(args.partial or len(seqs) < N_TRAIN_SEQS),
        "sequences": {},
    }
    all_full: list[OcclusionSegment] = []
    all_val: list[OcclusionSegment] = []
    diag_sum = 0.0
    for seq in seqs:
        assert seq.gt is not None, f"{seq.name}: GT missing"
        full = extract_segments(seq.gt, VIS_LO, VIS_HI, MIN_LEN)
        mid = seq.seq_length // 2
        val_gt = seq.gt[seq.gt[:, COL.FRAME] > mid]
        val = extract_segments(val_gt, VIS_LO, VIS_HI, MIN_LEN)
        out["sequences"][seq.name] = {
            "seq_length": seq.seq_length,
            "val_start_frame": mid + 1,
            "diagonal": seq.diagonal,
            "full": [seg_to_dict(s) for s in full],
            "val_half": [seg_to_dict(s) for s in val],
        }
        all_full.extend(full)
        all_val.extend(val)
        diag_sum += seq.diagonal
        log.info("%s: %d full / %d val-half segments", seq.name, len(full), len(val))

    mean_diag = diag_sum / max(len(seqs), 1)
    out["stats_full"] = segment_stats(all_full, mean_diag)
    out["stats_val_half"] = segment_stats(all_val, mean_diag)
    dest = ROOT / "results" / "occlusion_segments.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log.info("full: %s", out["stats_full"])
    log.info("val_half: %s", out["stats_val_half"])
    log.info("-> %s", dest)
    n_val = int(out["stats_val_half"].get("n_segments", 0))
    if not out["partial"] and n_val < 100:
        log.info("WARNING: only %d val-half segments (< G2 min_segments=100)", n_val)
    return 0


if __name__ == "__main__":
    sys.exit(main())
