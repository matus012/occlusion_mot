"""Phase 6: identity-labeled person crops from rendered sim sequences (re-ID training set).

For every GT row with visibility >= --min-vis and a usable box, crops the person patch,
resizes to 128x64, and stores it under data/sim/reid/<seq>/<walker_id>/ with the
visibility encoded in the filename (occlusion-robust training needs partially-occluded
positives — the trainer chooses its own visibility mix). Writes an index.json with
per-identity counts and split hints (identity-disjoint train/val by hash).

Usage: .venv/Scripts/python.exe scripts/extract_reid_dataset.py
         [--render-root data/sim/carla_render] [--min-vis 0.2]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omot.data.mot import load_sequence  # noqa: E402
from omot.io.mot_format import COL  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s", stream=sys.stdout)
logger = logging.getLogger("extract_reid_dataset")

ROOT = Path(__file__).resolve().parents[1]
CROP_HW = (128, 64)
MIN_BOX_H = 24  # px


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--render-root", type=Path, default=ROOT / "data" / "sim" / "carla_render")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "sim" / "reid")
    ap.add_argument("--min-vis", type=float, default=0.2)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    args = ap.parse_args()

    seq_dirs = sorted(
        p for p in args.render_root.iterdir() if (p / "gt" / "gt.txt").exists()
    )
    if not seq_dirs:
        raise SystemExit(f"no rendered sequences under {args.render_root}")

    index: dict[str, dict] = {}
    n_crops = 0
    for seq_dir in seq_dirs:
        seq = load_sequence(seq_dir)
        assert seq.gt is not None
        gt = seq.gt[seq.gt[:, COL.VIS] >= args.min_vis]
        by_frame: dict[int, np.ndarray] = {
            int(f): gt[gt[:, COL.FRAME] == f] for f in np.unique(gt[:, COL.FRAME])
        }
        for frame, rows in by_frame.items():
            img = cv2.imread(str(seq.frame_path(frame)))
            if img is None:
                continue
            ih, iw = img.shape[:2]
            for r in rows:
                x, y, w, h = r[COL.X : COL.H + 1]
                if h < MIN_BOX_H:
                    continue
                x1, y1 = max(0, int(x)), max(0, int(y))
                x2, y2 = min(iw, int(x + w)), min(ih, int(y + h))
                if x2 - x1 < 8 or y2 - y1 < MIN_BOX_H:
                    continue
                wid = int(r[COL.ID])
                identity = f"{seq.name}_w{wid:03d}"
                dest_dir = args.out / seq.name / f"w{wid:03d}"
                dest_dir.mkdir(parents=True, exist_ok=True)
                vis_tag = int(round(float(r[COL.VIS]) * 100))
                crop = cv2.resize(img[y1:y2, x1:x2], (CROP_HW[1], CROP_HW[0]))
                cv2.imwrite(str(dest_dir / f"f{frame:06d}_v{vis_tag:03d}.jpg"), crop)
                entry = index.setdefault(
                    identity,
                    {"seq": seq.name, "walker": wid, "n": 0, "n_occluded": 0,
                     "split": "val" if (
                         int(hashlib.sha1(identity.encode()).hexdigest(), 16) % 100
                         < args.val_fraction * 100
                     ) else "train"},
                )
                entry["n"] += 1
                if float(r[COL.VIS]) < 0.5:
                    entry["n_occluded"] += 1
                n_crops += 1
        logger.info("%s: done", seq.name)

    splits = {s: sum(1 for e in index.values() if e["split"] == s) for s in ("train", "val")}
    meta = {
        "n_identities": len(index),
        "n_crops": n_crops,
        "identity_split_counts": splits,
        "min_vis": args.min_vis,
        "render_root": str(args.render_root),
        "identities": index,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "index.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("identities=%d crops=%d split=%s -> %s",
                len(index), n_crops, splits, args.out / "index.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
