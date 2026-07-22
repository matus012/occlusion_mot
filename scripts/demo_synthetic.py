"""Demo (phase 1, dataset-free): synthetic occlusion scene -> ByteTracker -> TrackEval.

Runs the entire pipeline end-to-end: scene generation, tracking, MOT-format export,
HOTA/IDF1/MOTA evaluation. Optional --render writes an annotated mp4.

Usage: .venv/Scripts/python.exe scripts/demo_synthetic.py [--render out.mp4]
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omot.io.mot_format import write_mot  # noqa: E402
from omot.synth import SynthConfig, SynthScene, generate  # noqa: E402
from omot.track.bytetrack import ByteTracker  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
logger = logging.getLogger("demo_synthetic")

SEQINFO = """[Sequence]
name=SYN-DEMO
imDir=img1
frameRate=30
seqLength={length}
imWidth={width}
imHeight={height}
imExt=.jpg
"""


def track_scene(scene: SynthScene) -> np.ndarray:
    """Run ByteTracker over the scene; return MOT rows [frame, id, x, y, w, h, conf]."""
    tracker = ByteTracker()
    rows: list[np.ndarray] = []
    for f in sorted(scene.detections):
        for x, y, w, h, score, tid in tracker.update(scene.detections[f], frame_id=f):
            rows.append(np.array([f, tid, x, y, w, h, score]))
    return np.stack(rows) if rows else np.zeros((0, 7))


def evaluate(scene: SynthScene, pred: np.ndarray) -> dict[str, float | int]:
    from omot.eval.trackeval_runner import run_trackeval

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        gt_dir = tmp / "gt" / "SYN-DEMO" / "gt"
        gt_dir.mkdir(parents=True)
        cfg = scene.config
        (tmp / "gt" / "SYN-DEMO" / "seqinfo.ini").write_text(
            SEQINFO.format(length=cfg.n_frames, width=cfg.width, height=cfg.height),
            encoding="utf-8",
        )
        write_mot(gt_dir / "gt.txt", scene.gt)
        trk = tmp / "trackers" / "bytetrack_ours"
        trk.mkdir(parents=True)
        write_mot(trk / "SYN-DEMO.txt", pred)
        return run_trackeval(
            tmp / "gt", tmp / "trackers", "bytetrack_ours", {"SYN-DEMO": cfg.n_frames}
        )


def render(scene: SynthScene, pred: np.ndarray, out_path: Path) -> None:
    import cv2

    cfg = scene.config
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (cfg.width, cfg.height)
    )
    rng = np.random.default_rng(0)
    colors = {int(t): tuple(int(c) for c in rng.integers(64, 255, 3)) for t in set(pred[:, 1])}
    for f in range(1, cfg.n_frames + 1):
        img = np.full((cfg.height, cfg.width, 3), 245, dtype=np.uint8)
        if cfg.occluder is not None:
            x1, y1, x2, y2 = (int(v) for v in cfg.occluder)
            cv2.rectangle(img, (x1, y1), (x2, y2), (180, 180, 180), -1)
        for row in pred[pred[:, 0] == f]:
            _, tid, x, y, w, h = row[:6]
            c = colors[int(tid)]
            cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), c, 2)
            cv2.putText(img, f"id{int(tid)}", (int(x), int(y) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
        writer.write(img)
    writer.release()
    logger.info("rendered %s", out_path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", type=Path, default=None, help="write annotated mp4 here")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    cfg = SynthConfig(n_frames=200, n_agents=8, seed=args.seed,
                      occluder=(560.0, 0.0, 720.0, 720.0))
    scene = generate(cfg)
    n_occ = int((scene.gt[:, 8] == 0.0).sum())
    logger.info("scene: %d agents, %d frames, %d occluded gt boxes",
                cfg.n_agents, cfg.n_frames, n_occ)

    pred = track_scene(scene)
    logger.info("tracked: %d output boxes, %d identities",
                len(pred), len(np.unique(pred[:, 1])))

    metrics = evaluate(scene, pred)
    print("\n=== synthetic demo metrics (occluded scene, ByteTracker baseline) ===")
    for k, v in metrics.items():
        print(f"  {k:>8}: {v:.2f}" if isinstance(v, float) else f"  {k:>8}: {v}")

    if args.render is not None:
        render(scene, pred, args.render)
    return 0


if __name__ == "__main__":
    sys.exit(main())
