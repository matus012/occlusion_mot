"""G2a-paired runner (D28-final): trained embedder vs ImageNet-R18 null, paired McNemar.

READ-ONLY over existing tracker outputs in results/raw/<half>_half/ — no re-tracking.
Both tags are evaluated over the same segment index, then compared on the intersection
of their association scopes (fixed denominator). Emits per-segment outcome vectors,
discordant-pair counts, and the exact one-sided p (omot.eval.paired).

VAL-INVARIANT from the D28-final freeze: this exact code path runs at val time;
--canonical (val only) merges the paired fields into results/hidden_state.json.

Usage:
  .venv/Scripts/python.exe scripts/paired_test.py --tag-a hidden_d26best \
      --tag-b hidden_audit_in45 --half dev
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omot.eval.hidden_eval import SegmentResult, evaluate_segments  # noqa: E402
from omot.eval.occlusion import OcclusionSegment  # noqa: E402
from omot.eval.paired import paired_g2a  # noqa: E402
from omot.io.mot_format import read_mot  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger("paired_test")

ROOT = Path(__file__).resolve().parents[1]


def eval_tag(
    tag: str, half: str, seg_index: dict, keys_out: list[str] | None
) -> tuple[list[str], list[SegmentResult]]:
    """Evaluate one tag's stored outputs over the full segment index of the half."""
    work = ROOT / "results" / "raw" / f"{half}_half"
    keys: list[str] = []
    results: list[SegmentResult] = []
    for seq_name, entry in seg_index["sequences"].items():
        seg_dicts = entry[f"{half}_half"]
        if not seg_dicts:
            continue
        trk = read_mot(work / "trackers" / tag / f"{seq_name}.txt")[:, :6]
        cpath = work / "coasting" / tag / f"{seq_name}.txt"
        coast = read_mot(cpath)[:, :6] if cpath.exists() else np.zeros((0, 6))
        shift = 0 if half == "dev" else int(entry["val_start_frame"]) - 1
        segments = [
            OcclusionSegment(
                track_id=int(d["track_id"]),
                last_visible_frame=int(d["last_visible_frame"]) - shift,
                reemergence_frame=int(d["reemergence_frame"]) - shift,
                last_visible_box=np.array(d["last_visible_box"], dtype=np.float64),
                reemergence_box=np.array(d["reemergence_box"], dtype=np.float64),
            )
            for d in seg_dicts
        ]
        keys.extend(
            f"{seq_name}:t{d['track_id']}:f{d['last_visible_frame']}-{d['reemergence_frame']}"
            for d in seg_dicts
        )
        results.extend(
            evaluate_segments(segments, trk, coast, float(entry["diagonal"]))
        )
    if keys_out is not None:
        assert keys == keys_out, "segment index mismatch between tags"
    return keys, results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag-a", default="hidden_d26best",
                    help="config A = the module claim (trained embedder)")
    ap.add_argument("--tag-b", default="hidden_audit_in45",
                    help="config B = the null (ImageNet-R18 appearance)")
    ap.add_argument("--half", choices=["dev", "val"], default="dev")
    ap.add_argument("--canonical", action="store_true",
                    help="val only: merge paired fields into results/hidden_state.json")
    args = ap.parse_args()
    assert not (args.canonical and args.half != "val"), "--canonical requires --half val"

    seg_index = json.loads(
        (ROOT / "results" / "occlusion_segments.json").read_text(encoding="utf-8")
    )
    assert not seg_index.get("partial"), "need the full occlusion_segments.json"

    keys, res_a = eval_tag(args.tag_a, args.half, seg_index, None)
    _, res_b = eval_tag(args.tag_b, args.half, seg_index, keys)
    paired = paired_g2a(keys, res_a, res_b)
    paired["tag_a"] = args.tag_a
    paired["tag_b"] = args.tag_b
    paired["half"] = args.half

    dest = ROOT / "results" / f"paired_g2a_{args.half}_{args.tag_a}_vs_{args.tag_b}.json"
    dest.write_text(json.dumps(paired, indent=2), encoding="utf-8")
    logger.info(
        "n_intersection=%d retention a=%.3f b=%.3f discordant %d/%d p=%.4f -> %s",
        paired["n_intersection"], paired["retention_a"], paired["retention_b"],
        paired["discordant_a_only"], paired["discordant_b_only"],
        paired["p_value"], dest,
    )

    if args.canonical:
        hs_path = ROOT / "results" / "hidden_state.json"
        hs = json.loads(hs_path.read_text(encoding="utf-8"))
        hs["g2a_paired_p"] = paired["p_value"]
        hs["g2a_paired_discordant_a_only"] = paired["discordant_a_only"]
        hs["g2a_paired_discordant_b_only"] = paired["discordant_b_only"]
        hs["g2a_paired_n_intersection"] = paired["n_intersection"]
        hs["g2a_paired_tags"] = f"{args.tag_a} vs {args.tag_b}"
        hs_path.write_text(json.dumps(hs, indent=2), encoding="utf-8")
        logger.info("canonical merge -> %s", hs_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
