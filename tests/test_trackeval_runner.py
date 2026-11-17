"""TrackEval wrapper on a tiny synthetic GT/prediction pair — validates the whole
HOTA/IDF1 pipeline without the real dataset."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from omot.io.mot_format import COL, write_mot
from omot.synth import SynthConfig, generate

trackeval = pytest.importorskip("trackeval")

from omot.eval.trackeval_runner import run_trackeval  # noqa: E402

SEQINFO = """[Sequence]
name={name}
imDir=img1
frameRate=30
seqLength={length}
imWidth=1280
imHeight=720
imExt=.jpg
"""


def _setup(tmp_path: Path, pred: np.ndarray, gt: np.ndarray, length: int) -> tuple[Path, Path]:
    seq = "SYN-01"
    gt_dir = tmp_path / "gt" / seq / "gt"
    gt_dir.mkdir(parents=True)
    (tmp_path / "gt" / seq / "seqinfo.ini").write_text(
        SEQINFO.format(name=seq, length=length), encoding="utf-8"
    )
    write_mot(gt_dir / "gt.txt", gt)
    trk_dir = tmp_path / "trackers" / "test_tracker"
    trk_dir.mkdir(parents=True)
    write_mot(trk_dir / f"{seq}.txt", pred)
    return tmp_path / "gt", tmp_path / "trackers"


def test_perfect_predictions_score_100(tmp_path: Path) -> None:
    scene = generate(SynthConfig(n_frames=40, n_agents=3, seed=0))
    gt = scene.gt
    pred = gt[:, :7].copy()  # frame, id, box, conf — identical to GT
    gt_folder, trk_folder = _setup(tmp_path, pred, gt, 40)
    res = run_trackeval(gt_folder, trk_folder, "test_tracker", {"SYN-01": 40},
                        output_json=tmp_path / "out.json")
    assert res["hota"] == pytest.approx(100.0, abs=0.5)
    assert res["idf1"] == pytest.approx(100.0, abs=0.5)
    assert res["mota"] == pytest.approx(100.0, abs=0.5)
    assert res["idsw"] == 0
    assert (tmp_path / "out.json").exists()


def test_id_switch_detected(tmp_path: Path) -> None:
    scene = generate(SynthConfig(n_frames=40, n_agents=2, seed=1))
    gt = scene.gt
    pred = gt[:, :7].copy()
    # swap the two ids from frame 21 on -> 2 id switches, big IDF1 hit
    late = pred[:, COL.FRAME] > 20
    ones = late & (pred[:, COL.ID] == 1)
    twos = late & (pred[:, COL.ID] == 2)
    pred[ones, COL.ID] = 2
    pred[twos, COL.ID] = 1
    gt_folder, trk_folder = _setup(tmp_path, pred, gt, 40)
    res = run_trackeval(gt_folder, trk_folder, "test_tracker", {"SYN-01": 40})
    assert res["idsw"] >= 2
    assert res["idf1"] < 80.0
    assert res["mota"] > 80.0  # detection quality still perfect
