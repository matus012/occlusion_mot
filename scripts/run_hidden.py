"""Phase 4: run OcclusionAwareTracker on a MOT17 half-split -> G1 (TrackEval) + G2 metrics.

Dev half is for tuning (--half dev, default); val half is touched only for final numbers
(--half val --canonical writes the gates-facing results/tracker_ours.json +
results/hidden_state.json).

Usage:
  .venv/Scripts/python.exe scripts/run_hidden.py --half dev --occl-buffer 90 \
      --damping 0.95 --recover-gate 1.5 [--skip-trackeval] [--tag grid01]
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omot.data.mot import export_half_gt, half_split_frames, load_split  # noqa: E402
from omot.detect.cache import cache_path, load_cached_detections  # noqa: E402
from omot.eval.hidden_eval import SegmentResult, aggregate, evaluate_segments  # noqa: E402
from omot.eval.occlusion import OcclusionSegment  # noqa: E402
from omot.hidden.occlusion_tracker import HiddenConfig, OcclusionAwareTracker  # noqa: E402
from omot.io.mot_format import write_mot  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger("run_hidden")

ROOT = Path(__file__).resolve().parents[1]


def seg_from_dict(d: dict, shift: int) -> OcclusionSegment:
    return OcclusionSegment(
        track_id=int(d["track_id"]),
        last_visible_frame=int(d["last_visible_frame"]) - shift,
        reemergence_frame=int(d["reemergence_frame"]) - shift,
        last_visible_box=np.array(d["last_visible_box"], dtype=np.float64),
        reemergence_box=np.array(d["reemergence_box"], dtype=np.float64),
    )


def track_hidden(
    dets_by_frame: dict[int, np.ndarray],
    frames: range,
    cfg: HiddenConfig,
    embs_by_frame: dict[int, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    tracker = OcclusionAwareTracker(cfg)
    rows: list[np.ndarray] = []
    coast: list[np.ndarray] = []
    for i, f in enumerate(frames, start=1):
        dets = dets_by_frame.get(f, np.zeros((0, 5)))
        embs = embs_by_frame.get(f) if embs_by_frame is not None else None
        if embs is not None and len(embs) != len(dets):
            raise ValueError(f"frame {f}: {len(embs)} embeddings vs {len(dets)} detections")
        for x, y, w, h, score, tid in tracker.update(dets, frame_id=i, embeddings=embs):
            rows.append(np.array([i, tid, x, y, w, h, score]))
        for x, y, w, h, score, tid in tracker.coasting:
            coast.append(np.array([i, tid, x, y, w, h, score]))
    to_arr = lambda r: np.stack(r) if r else np.zeros((0, 7))  # noqa: E731
    return to_arr(rows), to_arr(coast)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--half", choices=["dev", "val"], default="dev")
    ap.add_argument("--occl-buffer", type=int, default=90)
    ap.add_argument("--damping", type=float, default=0.95)
    ap.add_argument("--recover-gate", type=float, default=1.5)
    ap.add_argument("--overlap-thresh", type=float, default=0.25)
    ap.add_argument("--noise-scale", type=float, default=3.0,
                    help="measurement-noise inflation for low-confidence matches (kf mode)")
    ap.add_argument("--lowconf-mode", choices=["coast", "kf"], default="coast")
    ap.add_argument("--app-gate-lost", type=float, default=-1.0,
                    help="cos-dist veto for lost-track stage-1 matches; negative disables")
    ap.add_argument("--app-gate-recover", type=float, default=-1.0,
                    help="cos-dist veto for recovery matches; negative disables")
    ap.add_argument("--tag", default="hidden")
    ap.add_argument("--skip-trackeval", action="store_true")
    ap.add_argument("--canonical", action="store_true",
                    help="val only: write gates-facing tracker_ours/hidden_state JSONs")
    ap.add_argument("--data-root", type=Path, default=ROOT / "data" / "MOT17")
    ap.add_argument("--cache-dir", type=Path, default=ROOT / "data" / "cache" / "detections")
    ap.add_argument("--model", default="yolo11x")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    assert not (args.canonical and args.half != "val"), "--canonical requires --half val"

    random.seed(args.seed)
    np.random.seed(args.seed)

    cfg = HiddenConfig(
        occl_buffer=args.occl_buffer,
        vel_damping=args.damping,
        recover_gate=args.recover_gate,
        occl_overlap_thresh=args.overlap_thresh,
        lowconf_noise_scale=args.noise_scale,
        lowconf_mode=args.lowconf_mode,
        app_gate_lost=None if args.app_gate_lost < 0 else args.app_gate_lost,
        app_gate_recover=None if args.app_gate_recover < 0 else args.app_gate_recover,
    )
    seg_index = json.loads(
        (ROOT / "results" / "occlusion_segments.json").read_text(encoding="utf-8")
    )
    assert not seg_index.get("partial"), "need the full occlusion_segments.json"

    seqs = load_split(args.data_root, "train", detector="FRCNN")
    work = ROOT / "results" / "raw" / f"{args.half}_half"
    trk_name = f"hidden_{args.tag}"
    trk_dir = work / "trackers" / trk_name
    trk_dir.mkdir(parents=True, exist_ok=True)
    coast_dir = work / "coasting" / trk_name

    from omot.detect.embed import embed_cache_path, load_cached_embeddings

    use_app = cfg.app_gate_lost is not None or cfg.app_gate_recover is not None
    all_results: list[SegmentResult] = []
    for seq in seqs:
        dets = load_cached_detections(cache_path(args.cache_dir, seq.name, args.model))
        embs = None
        if use_app:
            epath = embed_cache_path(args.cache_dir, seq.name, args.model)
            if not epath.exists():
                raise FileNotFoundError(f"embedding cache missing: {epath}")
            embs = load_cached_embeddings(epath)
        dev, val = half_split_frames(seq.seq_length)
        frames = dev if args.half == "dev" else val
        rows, coast = track_hidden(dets, frames, cfg, embs)
        write_mot(trk_dir / f"{seq.name}.txt", rows)
        write_mot(coast_dir / f"{seq.name}.txt", coast)

        entry = seg_index["sequences"][seq.name]
        shift = 0 if args.half == "dev" else int(entry["val_start_frame"]) - 1
        segments = [seg_from_dict(d, shift) for d in entry[f"{args.half}_half"]]
        if segments:
            results = evaluate_segments(
                segments, rows[:, :6], coast[:, :6], float(entry["diagonal"])
            )
            all_results.extend(results)

    g2 = aggregate(all_results)
    out: dict[str, dict] = {"config": {
        "occl_buffer": cfg.occl_buffer, "vel_damping": cfg.vel_damping,
        "recover_gate": cfg.recover_gate, "occl_overlap_thresh": cfg.occl_overlap_thresh,
        "lowconf_noise_scale": cfg.lowconf_noise_scale, "lowconf_mode": cfg.lowconf_mode,
        "app_gate_lost": cfg.app_gate_lost, "app_gate_recover": cfg.app_gate_recover,
    }, "g2": g2}

    if not args.skip_trackeval:
        from omot.eval.trackeval_runner import run_trackeval

        gt_root = work / "gt"
        seq_info = export_half_gt(seqs, gt_root, half=args.half)
        out["g1"] = run_trackeval(gt_root, work / "trackers", trk_name, seq_info)

    dest = ROOT / "results" / f"hidden_{args.half}_{args.tag}.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    logger.info("g2: %s", g2)
    if "g1" in out:
        logger.info("g1: %s", out["g1"])
    logger.info("-> %s", dest)

    if args.canonical:
        (ROOT / "results" / "tracker_ours.json").write_text(
            json.dumps(out["g1"], indent=2), encoding="utf-8"
        )
        (ROOT / "results" / "hidden_state.json").write_text(
            json.dumps(g2, indent=2), encoding="utf-8"
        )
        logger.info("canonical gates artifacts written (tracker_ours.json, hidden_state.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
