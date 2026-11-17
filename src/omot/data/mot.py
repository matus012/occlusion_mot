"""MOTChallenge sequence loading with the train half-split protocol (context.md D4).

Half-split: first half of each train sequence = dev, second half = val (ByteTrack ablation
protocol). GT-only operations work without images on disk, so tests need no dataset.
"""
from __future__ import annotations

import configparser
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from omot.io.mot_format import COL, read_mot

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MOTSequence:
    name: str
    root: Path
    frame_rate: float
    seq_length: int
    img_width: int
    img_height: int
    gt: np.ndarray | None  # (N, 9) MOT rows, or None for test split

    @property
    def img_dir(self) -> Path:
        return self.root / "img1"

    def frame_path(self, frame: int) -> Path:
        return self.img_dir / f"{frame:06d}.jpg"

    @property
    def diagonal(self) -> float:
        return float(np.hypot(self.img_width, self.img_height))


def parse_seqinfo(path: Path) -> dict[str, str]:
    parser = configparser.ConfigParser()
    parser.read(path)
    return dict(parser["Sequence"])


def load_sequence(seq_dir: Path) -> MOTSequence:
    """Load one sequence dir (seqinfo.ini required, gt/gt.txt optional)."""
    info = parse_seqinfo(seq_dir / "seqinfo.ini")
    gt_path = seq_dir / "gt" / "gt.txt"
    gt = read_mot(gt_path) if gt_path.exists() else None
    return MOTSequence(
        name=info.get("name", seq_dir.name),
        root=seq_dir,
        frame_rate=float(info.get("framerate", 30)),
        seq_length=int(info["seqlength"]),
        img_width=int(info["imwidth"]),
        img_height=int(info["imheight"]),
        gt=gt,
    )


def load_split(root: Path, split: str = "train", detector: str = "FRCNN") -> list[MOTSequence]:
    """Load all sequences of a split. For MOT17 the three detector variants share images/GT,
    so only `detector`-suffixed dirs are kept (all dirs if no suffix matches, e.g. MOT20)."""
    split_dir = root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"split dir not found: {split_dir}")
    seq_dirs = sorted(p for p in split_dir.iterdir() if (p / "seqinfo.ini").exists())
    filtered = [p for p in seq_dirs if p.name.endswith(f"-{detector}")]
    if not filtered:
        filtered = seq_dirs
    seqs = [load_sequence(p) for p in filtered]
    logger.info("loaded %d sequences from %s", len(seqs), split_dir)
    return seqs


def half_split_frames(seq_length: int) -> tuple[range, range]:
    """(dev_frames, val_frames), 1-based inclusive ranges as Python ranges."""
    mid = seq_length // 2
    return range(1, mid + 1), range(mid + 1, seq_length + 1)


def filter_gt_frames(gt: np.ndarray, frames: range, rebase: bool = True) -> np.ndarray:
    """Rows whose frame is in `frames`; frame numbers rebased to start at 1 if `rebase`."""
    mask = np.isin(gt[:, COL.FRAME].astype(np.int64), np.asarray(frames, dtype=np.int64))
    out = gt[mask].copy()
    if rebase and out.size:
        out[:, COL.FRAME] -= frames.start - 1
    return out


_SEQINFO_TEMPLATE = """[Sequence]
name={name}
imDir=img1
frameRate={fps}
seqLength={length}
imWidth={width}
imHeight={height}
imExt=.jpg
"""


def export_half_gt(seqs: list[MOTSequence], out_root: Path, half: str = "val") -> dict[str, int]:
    """Write a TrackEval-ready GT folder for the dev/val half of each sequence
    (frames rebased to 1). Returns {seq_name: half_length} for SEQ_INFO."""
    from omot.io.mot_format import write_mot

    assert half in ("dev", "val"), f"half must be dev|val, got {half!r}"
    seq_info: dict[str, int] = {}
    for seq in seqs:
        assert seq.gt is not None, f"{seq.name}: GT required for half-split export"
        dev, val = half_split_frames(seq.seq_length)
        frames = dev if half == "dev" else val
        gt_half = filter_gt_frames(seq.gt, frames, rebase=True)
        seq_dir = out_root / seq.name
        (seq_dir / "gt").mkdir(parents=True, exist_ok=True)
        (seq_dir / "seqinfo.ini").write_text(
            _SEQINFO_TEMPLATE.format(
                name=seq.name, fps=seq.frame_rate, length=len(frames),
                width=seq.img_width, height=seq.img_height,
            ),
            encoding="utf-8",
        )
        write_mot(seq_dir / "gt" / "gt.txt", gt_half)
        seq_info[seq.name] = len(frames)
    logger.info("exported %s-half GT for %d sequences -> %s", half, len(seqs), out_root)
    return seq_info
