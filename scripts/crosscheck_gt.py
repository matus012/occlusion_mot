"""Cross-mirror GT integrity check (D13): compare our downloaded MOT17 GT (ling1016/MOT17)
against the independent Lekim89/MOT17 upload (ByteTrack-style val-half ablation split).

For each train sequence present in both: take our full gt.txt, slice the val half with a
small start-offset search (ablation splits differ by one frame across the ecosystem),
rebase frames to 1, and compare (frame, x, y, w, h, conf, cls, vis) row multisets, plus an
id-bijection check. Agreement across two unrelated uploaders => GT content is authentic.

Writes results/mot17_gt_crosscheck.json. Exit 0 if every compared sequence matches >= 99.9%.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omot.io.mot_format import COL, read_mot  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
log = logging.getLogger("crosscheck_gt")

ROOT = Path(__file__).resolve().parents[1]
REF_REPO = "Lekim89/MOT17"
SEQ_LENGTHS = {
    "MOT17-02": 600, "MOT17-04": 1050, "MOT17-05": 837, "MOT17-09": 525,
    "MOT17-10": 654, "MOT17-11": 900, "MOT17-13": 750,
}


def row_keyset(gt: np.ndarray) -> set[tuple[float, ...]]:
    """Id-agnostic row keys (frame, box, conf, cls, vis) rounded to 2 decimals."""
    cols = [COL.FRAME, COL.X, COL.Y, COL.W, COL.H, COL.CONF, COL.CLS, COL.VIS]
    return {tuple(np.round(r[cols], 2)) for r in gt}


def _unique_box_index(gt: np.ndarray) -> dict[tuple[float, ...], float]:
    """(frame, x, y, w, h) -> id, keeping only keys that occur exactly once."""
    counts: dict[tuple[float, ...], int] = {}
    ids: dict[tuple[float, ...], float] = {}
    for r in gt:
        k = (round(r[COL.FRAME]), *(round(r[c], 1) for c in (COL.X, COL.Y, COL.W, COL.H)))
        counts[k] = counts.get(k, 0) + 1
        ids[k] = r[COL.ID]
    return {k: ids[k] for k, n in counts.items() if n == 1}


def id_bijection_ok(ours: np.ndarray, theirs: np.ndarray) -> bool:
    """Consistent 1:1 id mapping over unambiguous full-box joins."""
    ours_idx = _unique_box_index(ours)
    theirs_idx = _unique_box_index(theirs)
    fwd: dict[float, float] = {}
    rev: dict[float, float] = {}
    for k, a in ours_idx.items():
        b = theirs_idx.get(k)
        if b is None:
            continue
        if fwd.setdefault(a, b) != b or rev.setdefault(b, a) != a:
            return False
    return True


def compare_seq(our_gt: np.ndarray, ref_gt: np.ndarray, seq_len: int) -> dict[str, float | int]:
    ref_keys = row_keyset(ref_gt)
    best = {"offset": -1, "match": 0.0, "id_bijection": False}
    for offset in (0, 1, 2):
        start = seq_len // 2 + offset  # candidate first val frame (1-based)
        mask = our_gt[:, COL.FRAME] >= start
        sliced = our_gt[mask].copy()
        sliced[:, COL.FRAME] -= start - 1
        keys = row_keyset(sliced)
        denom = max(len(ref_keys), 1)
        match = len(keys & ref_keys) / denom
        if match > best["match"]:
            best = {
                "offset": offset,
                "match": round(match, 6),
                "id_bijection": id_bijection_ok(sliced, ref_gt),
                "our_rows": int(len(sliced)),
                "ref_rows": int(len(ref_gt)),
            }
    return best


def main() -> int:
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    all_files = api.list_repo_files(REF_REPO, repo_type="dataset")
    report: dict[str, dict] = {}
    failures: list[str] = []

    for base, seq_len in SEQ_LENGTHS.items():
        our_path = ROOT / "data" / "MOT17" / "train" / f"{base}-FRCNN" / "gt" / "gt.txt"
        if not our_path.exists():
            report[base] = {"status": "ours-missing"}
            continue
        ref_rel = next(
            (f for f in all_files if f.endswith(f"{base}-FRCNN/gt/gt.txt")), None
        )
        if ref_rel is None:
            report[base] = {"status": "ref-missing"}
            continue
        ref_local = hf_hub_download(REF_REPO, ref_rel, repo_type="dataset")
        result = compare_seq(read_mot(our_path), read_mot(ref_local), seq_len)
        ok = result["match"] >= 0.999 and result["id_bijection"]
        result["status"] = "ok" if ok else "MISMATCH"
        report[base] = result
        if not ok:
            failures.append(f"{base}: match={result['match']}, bijection={result['id_bijection']}")
        log.info("%s: %s", base, result)

    out = ROOT / "results" / "mot17_gt_crosscheck.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    compared = [b for b, r in report.items() if r.get("status") in ("ok", "MISMATCH")]
    log.info("compared %d sequences, %d failures -> %s", len(compared), len(failures), out)
    return 0 if compared and not failures else 1


if __name__ == "__main__":
    sys.exit(main())
