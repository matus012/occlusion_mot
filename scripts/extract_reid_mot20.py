"""Phase 6 / D37 (2c): identity-labeled person crops from MOT20 train.

MOT20 is not part of the MOT17 half-split eval protocol (D4/D18), so FULL sequences are
used (no dev/val frame split). Instead this extends the OCCLUDED-QUERY re-ID protocol
(D26): MOT20's dense-crowd sequences contribute val identities carrying real GT
visibility tags, directly enlarging the identity pool that D36's local curve showed as
the binding constraint on assoc quality.

Output contract mirrors extract_reid_dataset.py exactly: crops written to
data/reid/mot20/<identity>/<seq>_f<frame>_v<pct>.jpg, index.json with
{source, n_identities, n_crops, identity_split_counts, identities: {id: {n, n_occluded,
split, sources}}}. `identities` here are per-sequence GT track ids ("<seq>_gt<id>"),
namespaced the same way as the mot17-dev source so train_reid.py's --sources mot20 needs
no special-casing.

GT filtering: class == 1 (pedestrian), conf flag == 1 (per the MOT20 gt.txt spec, conf==1
marks in-evaluation rows). Visibility tag comes straight from gt col 8, uncapped by a
min-vis floor by default (min-vis=0.0) — unlike mot17-dev's extractor, the low-visibility
rows are exactly what the occluded-query protocol needs in val.

Per-identity sampling: identities with fewer than --min-crops eligible rows are dropped
(too little signal for triplet mining); dense identities are capped at
--max-crops-per-id, sampled evenly across their frame span (np.linspace over the
frame-sorted row list) rather than truncated, so early/mid/late appearance is preserved.

Split hints: identity-disjoint, seeded 85/15 train/val PER SEQUENCE (--val-fraction,
--seed) — deterministic via random.Random(f"{seed}:{seq_name}") so results/replays are
reproducible independent of the mot17-dev/sim split_of() hash convention.

Usage:
  .venv/Scripts/python.exe scripts/extract_reid_mot20.py [--mot-root data/MOT20]
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omot.data.mot import MOTSequence, load_split  # noqa: E402
from omot.io.mot_format import COL  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s", stream=sys.stdout)
logger = logging.getLogger("extract_reid_mot20")

ROOT = Path(__file__).resolve().parents[1]
CROP_HW = (128, 64)


def split_hints_for_sequence(
    identities: list[str], seq_name: str, val_fraction: float, seed: int
) -> dict[str, str]:
    """Deterministic identity-disjoint train/val split for one sequence's identities.

    Seeded on (seed, seq_name) so re-runs with the same --seed reproduce the same split;
    different sequences/seeds get independent shuffles. Each identity is assigned exactly
    one label -> disjoint by construction.
    """
    rng = random.Random(f"{seed}:{seq_name}")
    shuffled = sorted(identities)
    rng.shuffle(shuffled)
    n_val = round(val_fraction * len(shuffled))
    val_set = set(shuffled[:n_val])
    return {ident: ("val" if ident in val_set else "train") for ident in shuffled}


def select_rows_for_identities(
    rows: np.ndarray, min_crops: int, max_crops_per_id: int
) -> dict[int, list[np.ndarray]]:
    """Group class/conf-filtered GT rows by track id; drop sparse ids; evenly cap dense
    ids at max_crops_per_id (frame-sorted, linspace-sampled indices)."""
    by_id: dict[int, list[np.ndarray]] = defaultdict(list)
    for r in rows:
        by_id[int(r[COL.ID])].append(r)
    selected: dict[int, list[np.ndarray]] = {}
    for tid, id_rows in by_id.items():
        if len(id_rows) < min_crops:
            continue
        id_rows = sorted(id_rows, key=lambda r: r[COL.FRAME])
        if len(id_rows) > max_crops_per_id:
            idxs = sorted(set(
                np.linspace(0, len(id_rows) - 1, max_crops_per_id).round().astype(int).tolist()
            ))
            id_rows = [id_rows[i] for i in idxs]
        selected[tid] = id_rows
    return selected


def extract_sequence(
    seq: MOTSequence, out_dir: Path, index: dict, val_fraction: float, seed: int,
    min_vis: float, min_box_h: float, max_crops_per_id: int, min_crops: int,
) -> int:
    assert seq.gt is not None, f"{seq.name}: gt.txt required"
    gt = seq.gt
    mask = (gt[:, COL.CLS] == 1.0) & (gt[:, COL.CONF] == 1.0) & (gt[:, COL.VIS] >= min_vis)
    rows = gt[mask]
    selected = select_rows_for_identities(rows, min_crops, max_crops_per_id)
    if not selected:
        logger.info("%s: no identities meet min-crops=%d threshold", seq.name, min_crops)
        return 0

    identities = {tid: f"{seq.name}_gt{tid:04d}" for tid in selected}
    split_hints = split_hints_for_sequence(
        list(identities.values()), seq.name, val_fraction, seed
    )

    by_frame: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
    for tid, id_rows in selected.items():
        for r in id_rows:
            by_frame[int(r[COL.FRAME])].append((tid, r))

    n = 0
    for frame in sorted(by_frame):
        img = cv2.imread(str(seq.frame_path(frame)))
        if img is None:
            continue
        ih, iw = img.shape[:2]
        for tid, r in by_frame[frame]:
            x, y, w, h = r[COL.X : COL.H + 1]
            if h < min_box_h:
                continue
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2, y2 = min(iw, int(x + w)), min(ih, int(y + h))
            if x2 - x1 < 8 or y2 - y1 < min_box_h * 0.75:
                continue
            identity = identities[tid]
            dest_dir = out_dir / identity
            dest_dir.mkdir(parents=True, exist_ok=True)
            vis_tag = int(round(float(r[COL.VIS]) * 100))
            crop = cv2.resize(img[y1:y2, x1:x2], (CROP_HW[1], CROP_HW[0]))
            cv2.imwrite(str(dest_dir / f"{seq.name}_f{frame:06d}_v{vis_tag:03d}.jpg"), crop)
            entry = index.setdefault(
                identity,
                {"n": 0, "n_occluded": 0, "split": split_hints[identity], "sources": []},
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
    ap.add_argument("--mot-root", type=Path, default=ROOT / "data" / "MOT20")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--min-vis", type=float, default=0.0)
    ap.add_argument("--min-box-h", type=float, default=25.0)
    ap.add_argument("--max-crops-per-id", type=int, default=200)
    ap.add_argument("--min-crops", type=int, default=10)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = args.out or (ROOT / "data" / "reid" / "mot20")
    seqs = load_split(args.mot_root, "train")  # MOT20 has no detector-variant suffix
    if not seqs:
        raise SystemExit(f"no MOT20 train sequences under {args.mot_root}")

    index: dict[str, dict] = {}
    n_crops = 0
    for seq in seqs:
        n = extract_sequence(
            seq, out, index, args.val_fraction, args.seed, args.min_vis, args.min_box_h,
            args.max_crops_per_id, args.min_crops,
        )
        n_crops += n
        logger.info("%s: %d crops extracted", seq.name, n)

    splits = {s: sum(1 for e in index.values() if e["split"] == s) for s in ("train", "val")}
    meta = {
        "source": "mot20",
        "n_identities": len(index),
        "n_crops": n_crops,
        "identity_split_counts": splits,
        "min_vis": args.min_vis,
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "identities": index,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("source=mot20 identities=%d crops=%d split=%s -> %s",
                len(index), n_crops, splits, out / "index.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
