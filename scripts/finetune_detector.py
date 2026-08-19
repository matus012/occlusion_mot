"""L5: MOT-finetune a person detector on MOT17 DEV-HALF frames only (prototype).

D18 invariant: dataset built strictly from dev-half frames (1..seq_length//2 of each
FRCNN train sequence) — asserted on frame indices, never the val half. Within the dev
half, the last ~10% of each sequence's frames are held out as a monitoring val split
(NOT the eval val-half — purely for train-loop early-stopping signal); the rest is
train. Detector output feeds a NEW cache tag via cache_detections.py --weights/--model
(FIXED DETECTIONS invariant, context.md D1) — never overwrites the yolo11x cache.

D43-delta(b): --mix selects the TRAIN-split data source. mot17dev (default) is the
original dev-half-only behavior. mot17dev_carla additionally folds every rendered
CARLA scenario under --carla-root (data/sim/carla_render/<scenario>/{seqinfo.ini,
gt/gt.txt,img1/}) into the TRAIN split ONLY -- the monitoring val split stays
MOT17-only so the two mixes remain comparable on identical validation frames. The
CARLA scenario layout mirrors MOTSequence exactly (verified against
crowd_merge_0017: gt row = frame,id,x,y,w,h,conf,cls=1,vis), so `_yolo_label_lines`
is reused verbatim (conf flag==1, class==1, vis>=min_vis -- no extra filtering).
Dataset dir encodes the mix (data/cache/det_finetune_<mix>/) so the two mixes never
collide on disk and idempotency stays correct per mix.

Usage:
  .venv/Scripts/python.exe scripts/finetune_detector.py --prep-only
  .venv/Scripts/python.exe scripts/finetune_detector.py --prep-only --mix mot17dev_carla
  .venv/Scripts/python.exe scripts/finetune_detector.py --epochs 10 --tag y11s_proto
"""
from __future__ import annotations

import argparse
import logging
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omot.data.mot import MOTSequence, half_split_frames, load_sequence, load_split  # noqa: E402
from omot.detect.cache import select_device  # noqa: E402
from omot.io.mot_format import COL  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
                    stream=sys.stdout)
logger = logging.getLogger("finetune_detector")

ROOT = Path(__file__).resolve().parents[1]
VALID_MIXES = ("mot17dev", "mot17dev_carla", "mot20", "mot20_carla")

# perun_detector_v1.md section 1 (D53): the mot20* mixes exist so the detector can be
# trained on sources DISJOINT from the MOT17 evaluation half. That disjointness is the
# entire honesty argument of the G2b workstream -- if a single MOT17 frame reaches a
# mot20* train split, every downstream number silently reverts to dev-optimistic. It is
# asserted at prep time (_assert_no_mot17), not merely documented.
MOT17_FAMILY = ("mot17dev", "mot17dev_carla")
MOT20_FAMILY = ("mot20", "mot20_carla")
CARLA_MIXES = ("mot17dev_carla", "mot20_carla")


def mix_family(mix: str) -> str:
    assert mix in VALID_MIXES, f"unknown mix {mix!r}"
    return "mot17dev" if mix in MOT17_FAMILY else "mot20"


def _assert_no_mot17(paths: list[Path], mix: str) -> None:
    """Hard guard: no MOT17-derived frame may enter a mot20* split (perun_detector_v1.md).

    Checks the resolved source path, not the stem, so a renamed copy cannot slip past.
    """
    if mix_family(mix) != "mot20":
        return
    bad = [str(q) for q in paths if "MOT17" in str(q.resolve())]
    assert not bad, (
        f"mix={mix} is MOT17-disjoint by construction, but {len(bad)} source frame(s) "
        f"resolve under a MOT17 path -- this would make every G2b number dev-optimistic. "
        f"First offenders: {bad[:3]}"
    )


def set_seeds(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _split_all_frames(seq: MOTSequence, val_frac: float) -> tuple[list[int], list[int]]:
    """(train, monitor) over the WHOLE sequence -- for MOT17-disjoint sources only.

    MOT20 is not an evaluation half, so there is no D18 half-split to respect: every
    frame is legitimately trainable. The tail val_frac is held out purely as an
    ultralytics monitoring split.
    """
    frames = list(range(1, seq.seq_length + 1))
    n = len(frames)
    n_val = max(1, round(n * val_frac)) if n > 1 else 0
    train_frames, monitor_frames = frames[: n - n_val], frames[n - n_val :]
    assert set(train_frames).isdisjoint(monitor_frames)
    assert set(train_frames) | set(monitor_frames) == set(frames)
    return train_frames, monitor_frames


def _split_dev_frames(seq: MOTSequence, val_frac: float) -> tuple[list[int], list[int]]:
    """(train_frames, monitor_val_frames) within the dev half only. D18 hard requirement:
    every returned frame index is <= seq_length // 2 (asserted)."""
    dev, _val = half_split_frames(seq.seq_length)
    dev_frames = list(dev)
    mid = seq.seq_length // 2
    assert all(f <= mid for f in dev_frames), (
        f"{seq.name}: dev-half frame leaked past mid={mid} (D18 violation)"
    )
    n_dev = len(dev_frames)
    n_val = max(1, round(n_dev * val_frac)) if n_dev > 1 else 0
    train_frames = dev_frames[: n_dev - n_val]
    monitor_frames = dev_frames[n_dev - n_val :]
    assert set(train_frames).isdisjoint(monitor_frames)
    assert set(train_frames) | set(monitor_frames) == set(dev_frames)
    return train_frames, monitor_frames


def _yolo_label_lines(
    gt: np.ndarray, frame: int, min_vis: float, img_w: int, img_h: int
) -> list[str]:
    """GT rows for `frame` with class==pedestrian(1), conf flag==1, vis>=min_vis ->
    YOLO txt lines (class 0, normalized cx cy w h, clipped to [0, 1])."""
    mask = (
        (gt[:, COL.FRAME].astype(np.int64) == frame)
        & (gt[:, COL.CLS].astype(np.int64) == 1)
        & (gt[:, COL.CONF].astype(np.int64) == 1)
        & (gt[:, COL.VIS] >= min_vis)
    )
    lines: list[str] = []
    for row in gt[mask]:
        x, y, w, h = row[COL.X], row[COL.Y], row[COL.W], row[COL.H]
        cx = np.clip((x + w / 2) / img_w, 0.0, 1.0)
        cy = np.clip((y + h / 2) / img_h, 0.0, 1.0)
        nw = np.clip(w / img_w, 0.0, 1.0)
        nh = np.clip(h / img_h, 0.0, 1.0)
        lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    return lines


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _expected_counts(
    seqs: list[MOTSequence], val_frac: float,
    mix: str = "mot17dev", carla_seqs: list[MOTSequence] | None = None,
) -> tuple[int, int]:
    splitter = _split_all_frames if mix_family(mix) == "mot20" else _split_dev_frames
    n_train = n_val = 0
    for seq in seqs:
        train_frames, monitor_frames = splitter(seq, val_frac)
        n_train += len(train_frames)
        n_val += len(monitor_frames)
    if mix in CARLA_MIXES:
        assert carla_seqs is not None, f"mix={mix} requires carla_seqs"
        n_train += sum(cseq.seq_length for cseq in carla_seqs)  # ALL frames -> TRAIN only
    return n_train, n_val


def _carla_scenario_dirs(carla_root: Path) -> list[Path]:
    """CARLA scenario dirs with a valid MOT-style layout (seqinfo.ini + gt/gt.txt).
    Empty list (not an error) if `carla_root` doesn't exist -- mix=mot17dev never
    needs it."""
    if not carla_root.is_dir():
        return []
    return sorted(
        p for p in carla_root.iterdir()
        if p.is_dir() and (p / "seqinfo.ini").exists() and (p / "gt" / "gt.txt").exists()
    )


def _carla_train_sequences(carla_root: Path) -> list[MOTSequence]:
    """Every rendered CARLA scenario, loaded via the MOTSequence reader (D22/D23: a
    scenario dir mirrors the MOT gt/gt.txt + img1/ layout by construction -- verified
    against crowd_merge_0017: 9-col gt rows, class==1, conf==1.0)."""
    seqs = [load_sequence(p) for p in _carla_scenario_dirs(carla_root)]
    for seq in seqs:
        assert seq.gt is not None, f"{seq.name}: CARLA scenario missing GT"
    return seqs


def _default_out_dir(mix: str) -> Path:
    """D43-delta(b): dataset dir name encodes the mix so mot17dev / mot17dev_carla
    never collide on disk (and each stays independently idempotent)."""
    return ROOT / "data" / "cache" / f"det_finetune_{mix}"


def prepare_dataset(
    data_root: Path, out_dir: Path, min_vis: float, val_frac: float = 0.1,
    mix: str = "mot17dev", carla_root: Path | None = None,
    mot20_root: Path | None = None,
) -> tuple[Path, dict[str, int]]:
    """Build the YOLO-format dataset from MOT17 dev-half GT (+ CARLA scenario frames
    in TRAIN only when mix=mot17dev_carla). Idempotent: if `out_dir` already holds
    the expected image counts, skip rebuilding and reuse it."""
    assert mix in VALID_MIXES, f"unknown --mix {mix!r} (valid: {VALID_MIXES})"
    family = mix_family(mix)
    if family == "mot20":
        assert mot20_root is not None, f"mix={mix} requires --mot20-root"
        seqs = load_split(mot20_root, "train")
        assert seqs, f"no MOT20 train sequences found under {mot20_root}"
        _assert_no_mot17([seq.root for seq in seqs], mix)
    else:
        seqs = load_split(data_root, "train", detector="FRCNN")
        assert seqs, f"no FRCNN train sequences found under {data_root}"

    carla_seqs: list[MOTSequence] = []
    if mix in CARLA_MIXES:
        assert carla_root is not None, f"mix={mix} requires --carla-root"
        carla_seqs = _carla_train_sequences(carla_root)
        assert carla_seqs, f"mix={mix} but no CARLA scenarios found under {carla_root}"
        _assert_no_mot17([c.root for c in carla_seqs], mix)

    n_train_expected, n_val_expected = _expected_counts(seqs, val_frac, mix, carla_seqs)

    dataset_yaml = out_dir / "dataset.yaml"
    img_train, img_val = out_dir / "images" / "train", out_dir / "images" / "val"
    lbl_train, lbl_val = out_dir / "labels" / "train", out_dir / "labels" / "val"

    if dataset_yaml.exists():
        n_train_actual = len(list(img_train.glob("*.jpg"))) if img_train.is_dir() else 0
        n_val_actual = len(list(img_val.glob("*.jpg"))) if img_val.is_dir() else 0
        if n_train_actual == n_train_expected and n_val_actual == n_val_expected:
            logger.info(
                "dataset exists with matching frame counts (train=%d val=%d) -> skip prep: %s",
                n_train_actual, n_val_actual, out_dir,
            )
            return dataset_yaml, {"train": n_train_actual, "val": n_val_actual}
        logger.info(
            "dataset dir stale (found train=%d val=%d, expected train=%d val=%d) -> rebuilding",
            n_train_actual, n_val_actual, n_train_expected, n_val_expected,
        )
        shutil.rmtree(out_dir)

    for d in (img_train, img_val, lbl_train, lbl_val):
        d.mkdir(parents=True, exist_ok=True)

    splitter = _split_all_frames if family == "mot20" else _split_dev_frames
    n_train = n_val = 0
    for seq in seqs:
        assert seq.gt is not None, f"{seq.name}: GT required for detector finetuning"
        train_frames, monitor_frames = splitter(seq, val_frac)
        for frames, img_dir, lbl_dir in (
            (train_frames, img_train, lbl_train),
            (monitor_frames, img_val, lbl_val),
        ):
            for frame in frames:
                # D18 half-split applies to the EVAL dataset only. MOT20 is not an eval
                # half, so every frame is trainable there; MOT17 keeps the hard guard.
                assert family == "mot20" or frame <= seq.seq_length // 2, (
                    f"{seq.name}: frame {frame} beyond dev half (D18 violation)"
                )
                stem = f"{seq.name}_{frame:06d}"
                _link_or_copy(seq.frame_path(frame), img_dir / f"{stem}.jpg")
                lines = _yolo_label_lines(seq.gt, frame, min_vis, seq.img_width, seq.img_height)
                (lbl_dir / f"{stem}.txt").write_text(
                    "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
                )
        n_train += len(train_frames)
        n_val += len(monitor_frames)

    if mix in CARLA_MIXES:
        for cseq in carla_seqs:
            assert cseq.gt is not None, f"{cseq.name}: CARLA scenario missing GT"
            for frame in range(1, cseq.seq_length + 1):  # ALL frames -> TRAIN only, never val
                stem = f"carla_{cseq.name}_{frame:06d}"
                _link_or_copy(cseq.frame_path(frame), img_train / f"{stem}.jpg")
                lines = _yolo_label_lines(
                    cseq.gt, frame, min_vis, cseq.img_width, cseq.img_height
                )
                (lbl_train / f"{stem}.txt").write_text(
                    "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
                )
            n_train += cseq.seq_length
        logger.info(
            "mix=%s: added %d CARLA scenario(s) / %d frames to TRAIN only (monitoring val "
            "split stays MOT17-only so the two mixes remain comparable)",
            mix, len(carla_seqs), sum(cseq.seq_length for cseq in carla_seqs),
        )

    if family == "mot20":
        # Belt and braces: re-assert on the MATERIALISED dataset, not just the sources.
        staged = list(img_train.glob("*.jpg")) + list(img_val.glob("*.jpg"))
        leaked = [q for q in staged if q.name.startswith("MOT17")]
        assert not leaked, (
            f"mix={mix}: {len(leaked)} MOT17 frame(s) materialised into the dataset "
            f"(first: {leaked[:3]}) -- the MOT17-disjointness contract is broken"
        )

    dataset_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(out_dir.resolve()),
                "train": "images/train",
                "val": "images/val",
                "names": {0: "person"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    logger.info("built dataset (mix=%s): train=%d val=%d -> %s", mix, n_train, n_val, out_dir)
    return dataset_yaml, {"train": n_train, "val": n_val}


def train(
    dataset_yaml: Path,
    project: Path,
    tag: str,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
    seed: int,
    base_weights: str,
    workers: int = 2,
) -> Path:
    from ultralytics import YOLO

    model = YOLO(base_weights)
    results = model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        seed=seed,
        single_cls=True,
        project=str(project),
        name=tag,
        exist_ok=True,
        workers=workers,
    )
    metrics: Any = getattr(results, "results_dict", results)
    logger.info("final metrics: %s", metrics)
    best_path = project / tag / "weights" / "best.pt"
    logger.info("best checkpoint: %s", best_path)
    return best_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=ROOT / "data" / "MOT17")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: data/cache/det_finetune_<mix> (D43-delta(b): mix-encoded "
                         "so mot17dev / mot17dev_carla never collide on disk)")
    ap.add_argument("--carla-root", type=Path, default=ROOT / "data" / "sim" / "carla_render",
                    help="CARLA scenario dirs (seqinfo.ini + gt/gt.txt + img1/), used only "
                         "when --mix mot17dev_carla")
    ap.add_argument("--mot20-root", type=Path, default=ROOT / "data" / "MOT20",
                    help="MOT20 root (train/<seq>/{seqinfo.ini,gt/gt.txt,img1/}), used only "
                         "when --mix mot20 or mot20_carla")
    ap.add_argument("--mix", choices=VALID_MIXES, default="mot17dev",
                    help="mot17dev (default): MOT17 dev-half GT only, unchanged behavior. "
                         "mot17dev_carla: additionally folds CARLA scenario frames into the "
                         "TRAIN split only (monitoring val stays MOT17-only). "
                         "mot20 / mot20_carla (perun_detector_v1.md, D53): MOT17-DISJOINT "
                         "training sources, so a detector trained on them and evaluated on "
                         "MOT17 dev-half is honest by construction -- no dev-optimistic "
                         "label, no val consumption")
    ap.add_argument("--min-vis", type=float, default=0.1)
    ap.add_argument("--val-frac", type=float, default=0.1,
                    help="fraction of each sequence's dev-half frames held out for monitoring")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None, help="cuda|cpu (default: auto)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="y11s_proto", help="run name under --project")
    ap.add_argument("--project", type=Path, default=ROOT / "data" / "models" / "det_finetune")
    ap.add_argument("--base-weights", default="yolo11s.pt")
    ap.add_argument("--prep-only", action="store_true",
                    help="build the dataset and exit before training")
    args = ap.parse_args()

    set_seeds(args.seed)
    device = select_device(args.device)
    out_dir = args.out_dir or _default_out_dir(args.mix)

    dataset_yaml, counts = prepare_dataset(
        args.data_root, out_dir, args.min_vis, args.val_frac, args.mix, args.carla_root,
        args.mot20_root,
    )
    logger.info("dataset ready (mix=%s, train=%d val=%d): %s", args.mix, counts["train"],
                counts["val"], dataset_yaml)

    if args.prep_only:
        logger.info("prep-only: stopping before training")
        return 0

    best_path = train(
        dataset_yaml=dataset_yaml,
        project=args.project,
        tag=args.tag,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        seed=args.seed,
        base_weights=args.base_weights,
    )
    logger.info("done -> %s", best_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
