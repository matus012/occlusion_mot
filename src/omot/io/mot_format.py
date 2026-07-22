"""MOTChallenge text-format IO.

Rows: frame(1-based), id, x, y, w, h, conf, class, visibility  (9 columns, float64).
GT files carry class/visibility; detection/result files use -1 placeholders.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

COL = SimpleNamespace(FRAME=0, ID=1, X=2, Y=3, W=4, H=5, CONF=6, CLS=7, VIS=8)
N_COLS = 9


def read_mot(path: Path | str) -> np.ndarray:
    """Read a MOT-format file into an (N, 9) float64 array, sorted by (frame, id).

    Files with 10 columns (raw MOT17 det/gt with z or extra col) are truncated to 9;
    files with fewer than 9 are right-padded with -1.
    """
    path = Path(path)
    rows: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        vals = [float(v) for v in line.split(",")]
        vals = vals[:N_COLS] + [-1.0] * max(0, N_COLS - len(vals))
        rows.append(vals)
    if not rows:
        return np.zeros((0, N_COLS))
    raw = np.array(rows, dtype=np.float64)
    order = np.lexsort((raw[:, COL.ID], raw[:, COL.FRAME]))
    return raw[order]


def write_mot(path: Path | str, rows: np.ndarray) -> None:
    """Write an (N, >=6) array as MOT format; missing trailing columns filled with -1
    (conf defaults to 1 when absent so GT written from 6-col data stays valid)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = np.asarray(rows, dtype=np.float64).reshape(-1, rows.shape[-1] if rows.size else 6)
    n = rows.shape[0]
    out = -np.ones((n, N_COLS))
    out[:, : rows.shape[1]] = rows[:, :N_COLS]
    if rows.shape[1] <= COL.CONF:
        out[:, COL.CONF] = 1.0
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in out:
            f.write(
                f"{int(r[0])},{int(r[1])},{r[2]:.2f},{r[3]:.2f},{r[4]:.2f},{r[5]:.2f},"
                f"{r[6]:.4f},{int(r[7])},{r[8]:.4f}\n"
            )
