"""Phase 6: identity-labeled person crops for re-ID training (D24 two-source design).

Sources:
  --source sim        rendered CARLA scenarios. Identity = BLUEPRINT id (walkers_meta.json)
                      — per-scenario walker labels collapse onto shared appearances
                      (D24 audit: 161 labels / 10 blueprints), so blueprint IS the
                      appearance-correct identity. 45 classes max (CARLA ceiling).
  --source mot17-dev  MOT17 train sequences, DEV HALF ONLY (frames <= seq_length//2,
                      D18-compliant). Identity = per-sequence GT track id; hundreds of
                      real identities with GT visibility tags.

Both write crops (128x64, visibility in filename) + index.json with identity-disjoint
train/val split hints. The trainer consumes one or both (sim2real ablation arms).

Usage:
  .venv/Scripts/python.exe scripts/extract_reid_dataset.py --source sim
  .venv/Scripts/python.exe scripts/extract_reid_dataset.py --source mot17-dev
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

from omot.data.mot import load_sequence, load_split  # noqa: E402
from omot.io.mot_format import COL  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s", stream=sys.stdout)
logger = logging.getLogger("extract_reid_dataset")

ROOT = Path(__file__).resolve().parents[1]
CROP_HW = (128, 64)


def split_of(identity: str, val_fraction: float) -> str:
    h = int(hashlib.sha1(identity.encode()).hexdigest(), 16) % 100
    return "val" if h < val_fraction * 100 else "train"


def extract_rows(
    seq, rows: np.ndarray, identity_of_row, out_dir: Path, index: dict, val_fraction: float,
    min_box_h: float,
) -> int:
    """Crop all GT rows of one sequence; identity_of_row maps a row -> label or None."""
    n = 0
    by_frame = {int(f): rows[rows[:, COL.FRAME] == f] for f in np.unique(rows[:, COL.FRAME])}
    for frame, frame_rows in by_frame.items():
        img = cv2.imread(str(seq.frame_path(frame)))
        if img is None:
            continue
        ih, iw = img.shape[:2]
        for r in frame_rows:
            identity = identity_of_row(r)
            if identity is None:
                continue
            x, y, w, h = r[COL.X : COL.H + 1]
            if h < min_box_h:
                continue
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2, y2 = min(iw, int(x + w)), min(ih, int(y + h))
            if x2 - x1 < 8 or y2 - y1 < min_box_h * 0.75:
                continue
            safe_ident = identity.replace(".", "_")
            dest_dir = out_dir / safe_ident
            dest_dir.mkdir(parents=True, exist_ok=True)
            vis_tag = int(round(float(r[COL.VIS]) * 100))
            crop = cv2.resize(img[y1:y2, x1:x2], (CROP_HW[1], CROP_HW[0]))
            cv2.imwrite(str(dest_dir / f"{seq.name}_f{frame:06d}_v{vis_tag:03d}.jpg"), crop)
            entry = index.setdefault(
                safe_ident,
                {"n": 0, "n_occluded": 0, "split": split_of(safe_ident, val_fraction),
                 "sources": []},
            )
            entry["n"] += 1
            if float(r[COL.VIS]) < 0.5:
                entry["n_occluded"] += 1
            if seq.name not in entry["sources"]:
                entry["sources"].append(seq.name)
            n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["sim", "mot17-dev"], required=True)
    ap.add_argument("--render-root", type=Path, default=ROOT / "data" / "sim" / "carla_render")
    ap.add_argument("--mot-root", type=Path, default=ROOT / "data" / "MOT17")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--min-vis", type=float, default=0.2)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    args = ap.parse_args()

    out = args.out or (ROOT / "data" / "reid" / args.source.replace("-", "_"))
    index: dict[str, dict] = {}
    n_crops = 0

    if args.source == "sim":
        seq_dirs = sorted(
            p for p in args.render_root.iterdir() if (p / "gt" / "gt.txt").exists()
        )
        if not seq_dirs:
            raise SystemExit(f"no rendered sequences under {args.render_root}")
        for seq_dir in seq_dirs:
            meta_path = seq_dir / "walkers_meta.json"
            if not meta_path.exists():
                raise SystemExit(f"{seq_dir.name}: walkers_meta.json missing — re-render "
                                 f"with the D24 driver (blueprint labels required)")
            bp_of = json.loads(meta_path.read_text(encoding="utf-8"))["blueprint_of_walker"]
            seq = load_sequence(seq_dir)
            assert seq.gt is not None
            rows = seq.gt[seq.gt[:, COL.VIS] >= args.min_vis]

            def ident(r, _bp_of=bp_of):  # noqa: ANN001
                return _bp_of.get(str(int(r[COL.ID])))

            n_crops += extract_rows(seq, rows, ident, out, index, args.val_fraction,
                                    min_box_h=24)
            logger.info("%s: done", seq.name)
    else:
        seqs = load_split(args.mot_root, "train", detector="FRCNN")
        for seq in seqs:
            assert seq.gt is not None
            mid = seq.seq_length // 2
            gt = seq.gt
            mask = (
                (gt[:, COL.FRAME] <= mid)
                & (gt[:, COL.CLS] == 1.0)
                & (gt[:, COL.CONF] == 1.0)
                & (gt[:, COL.VIS] >= args.min_vis)
            )
            rows = gt[mask]

            def ident(r, _seq=seq.name):  # noqa: ANN001
                return f"{_seq}_gt{int(r[COL.ID]):04d}"

            n_crops += extract_rows(seq, rows, ident, out, index, args.val_fraction,
                                    min_box_h=40)
            logger.info("%s: done (dev half, %d rows eligible)", seq.name, len(rows))

    splits = {s: sum(1 for e in index.values() if e["split"] == s) for s in ("train", "val")}
    meta = {
        "source": args.source,
        "n_identities": len(index),
        "n_crops": n_crops,
        "identity_split_counts": splits,
        "min_vis": args.min_vis,
        "identities": index,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("source=%s identities=%d crops=%d split=%s -> %s",
                args.source, len(index), n_crops, splits, out / "index.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
