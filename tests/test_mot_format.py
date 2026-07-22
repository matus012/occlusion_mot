"""MOT-format IO roundtrip and column semantics."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from omot.io.mot_format import COL, N_COLS, read_mot, write_mot


def test_roundtrip_9col(tmp_path: Path) -> None:
    rows = np.array(
        [
            [2, 7, 10.5, 20.25, 50.0, 100.0, 0.9, 1, 0.75],
            [1, 3, 5.0, 6.0, 30.0, 60.0, 1.0, 1, 1.0],
        ]
    )
    p = tmp_path / "x.txt"
    write_mot(p, rows)
    back = read_mot(p)
    assert back.shape == (2, N_COLS)
    # sorted by frame
    assert back[0, COL.FRAME] == 1 and back[1, COL.FRAME] == 2
    np.testing.assert_allclose(back[1, :6], rows[0, :6], atol=0.01)
    np.testing.assert_allclose(back[1, COL.VIS], 0.75, atol=1e-4)


def test_write_6col_pads_conf_and_placeholders(tmp_path: Path) -> None:
    rows = np.array([[1, 1, 0.0, 0.0, 10.0, 10.0]])
    p = tmp_path / "y.txt"
    write_mot(p, rows)
    back = read_mot(p)
    assert back[0, COL.CONF] == 1.0
    assert back[0, COL.CLS] == -1
    assert back[0, COL.VIS] == -1


def test_read_pads_and_truncates(tmp_path: Path) -> None:
    p = tmp_path / "z.txt"
    p.write_text("1,2,3,4,5,6,0.5\n1,1,1,1,1,1,1,1,1,999\n", encoding="utf-8")
    back = read_mot(p)
    assert back.shape == (2, N_COLS)
    assert back[1, COL.CONF] == 0.5  # 7-col row padded with -1 (sorted: id 1 first)
    assert back[1, COL.VIS] == -1


def test_empty_file(tmp_path: Path) -> None:
    p = tmp_path / "empty.txt"
    p.write_text("", encoding="utf-8")
    assert read_mot(p).shape == (0, N_COLS)
