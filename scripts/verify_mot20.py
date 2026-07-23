"""Verify a MOT20 train copy against the official MOT20 spec (structure + internal
consistency) — D37 2c data-source expansion, verify_mot17.py's pattern extended to MOT20.

MOT20 has no DPM/FRCNN/SDP detector-variant split (unlike MOT17) and is not part of the
MOT17 half-split eval protocol — this script checks the four train sequences directly:
  MOT20-01 (429, 1920x1080) MOT20-02 (2782, 1920x1080)
  MOT20-03 (2405, 1173x880) MOT20-05 (3315, 1654x1080)

Checks per sequence:
  - seqinfo.ini seqLength/imWidth/imHeight match the official spec table
  - img1/ frame count == seqLength (+ first/last frame spot-probe)
  - gt/gt.txt exists, parses; frame ids in [1, seqLength]; class column values in the
    MOTChallenge annotation-class range [1, 12]; visibility column in [0, 1]
  - SHA256 of gt.txt + seqinfo.ini only (hashing every jpg is too slow); jpg count recorded

Exit 0 = verified; exit 1 = failures (report printed and written to
results/mot20_verification.json).

Usage: .venv/Scripts/python.exe scripts/verify_mot20.py [--data-root data/MOT20]
       [--source URL] [--revision REV]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omot.data.mot import parse_seqinfo  # noqa: E402
from omot.io.mot_format import COL, read_mot  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
log = logging.getLogger("verify_mot20")

ROOT = Path(__file__).resolve().parents[1]

# Official MOT20 train spec: sequence -> (seqLength, imWidth, imHeight)
MOT20_SPEC: dict[str, tuple[int, int, int]] = {
    "MOT20-01": (429, 1920, 1080),
    "MOT20-02": (2782, 1920, 1080),
    "MOT20-03": (2405, 1173, 880),
    "MOT20-05": (3315, 1654, 1080),
}
# MOTChallenge annotation classes 1-12 (1=pedestrian ... 12=reflection); anything outside
# this range cannot come from a genuine MOTChallenge gt.txt.
ALLOWED_CLASSES = set(range(1, 13))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(
    data_root: Path, spec: dict[str, tuple[int, int, int]] | None = None
) -> tuple[list[str], dict[str, dict], dict[str, str]]:
    """Returns (failures, per_seq_details, file_sha256_hashes)."""
    spec = MOT20_SPEC if spec is None else spec
    failures: list[str] = []
    per_seq: dict[str, dict] = {}
    hashes: dict[str, str] = {}
    train = data_root / "train"
    if not train.is_dir():
        return [f"missing train dir: {train}"], per_seq, hashes

    for name, (length, width, height) in spec.items():
        detail = {
            "seqinfo_ok": False, "img_count_ok": False, "gt_exists": False,
            "frame_range_ok": False, "class_ok": False, "vis_ok": False,
            "n_imgs": 0, "n_gt_rows": 0, "passed": False,
        }
        seq_failures: list[str] = []
        seq_dir = train / name
        if not seq_dir.is_dir():
            seq_failures.append(f"{name}: sequence dir missing")
            failures.extend(seq_failures)
            per_seq[name] = detail
            continue

        ini = seq_dir / "seqinfo.ini"
        if not ini.exists():
            seq_failures.append(f"{name}: seqinfo.ini missing")
        else:
            info = parse_seqinfo(ini)
            got = (int(info["seqlength"]), int(info["imwidth"]), int(info["imheight"]))
            detail["seqinfo_ok"] = got == (length, width, height)
            if not detail["seqinfo_ok"]:
                seq_failures.append(f"{name}: seqinfo {got} != official {(length, width, height)}")
            hashes[f"{name}/seqinfo.ini"] = sha256_file(ini)

        img1 = seq_dir / "img1"
        n_imgs = len(list(img1.glob("*.jpg"))) if img1.is_dir() else 0
        detail["n_imgs"] = n_imgs
        detail["img_count_ok"] = n_imgs == length
        if not detail["img_count_ok"]:
            seq_failures.append(f"{name}: img1 has {n_imgs} jpgs, expected {length}")
        for probe in (1, length):
            if not (img1 / f"{probe:06d}.jpg").exists():
                seq_failures.append(f"{name}: frame {probe:06d}.jpg missing")

        gt_path = seq_dir / "gt" / "gt.txt"
        if not gt_path.exists():
            seq_failures.append(f"{name}: gt/gt.txt missing")
        else:
            detail["gt_exists"] = True
            gt = read_mot(gt_path)
            detail["n_gt_rows"] = int(len(gt))
            hashes[f"{name}/gt.txt"] = sha256_file(gt_path)
            if len(gt) == 0:
                seq_failures.append(f"{name}: gt.txt empty")
            else:
                frames = gt[:, COL.FRAME]
                frame_range_ok = bool(frames.min() >= 1 and frames.max() <= length)
                detail["frame_range_ok"] = frame_range_ok
                if not frame_range_ok:
                    seq_failures.append(
                        f"{name}: gt frames out of spec [{frames.min()},{frames.max()}] "
                        f"expected [1,{length}]"
                    )
                classes = {int(c) for c in gt[:, COL.CLS]}
                bad_classes = sorted(classes - ALLOWED_CLASSES)
                detail["class_ok"] = not bad_classes
                if bad_classes:
                    seq_failures.append(
                        f"{name}: gt class column has out-of-spec values {bad_classes} "
                        f"(allowed 1-12)"
                    )
                vis = gt[:, COL.VIS]
                vis_ok = bool(vis.min() >= 0.0 and vis.max() <= 1.0)
                detail["vis_ok"] = vis_ok
                if not vis_ok:
                    seq_failures.append(
                        f"{name}: gt visibility out of [0,1] range [{vis.min()},{vis.max()}]"
                    )

        detail["passed"] = not seq_failures
        per_seq[name] = detail
        failures.extend(seq_failures)

    return failures, per_seq, hashes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=ROOT / "data" / "MOT20")
    ap.add_argument("--source", default="unknown", help="provenance URL/mirror to record")
    ap.add_argument("--revision", default="unknown", help="pinned mirror revision/commit")
    args = ap.parse_args()

    failures, per_seq, hashes = verify(args.data_root)
    manifest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode("utf-8")).hexdigest()
    report = {
        "source": args.source,
        "revision": args.revision,
        "data_root": str(args.data_root),
        "passed": not failures,
        "n_failures": len(failures),
        "per_seq": per_seq,
        "failures": failures,
        "manifest_sha256": manifest,
        "file_sha256": hashes,
    }
    out = ROOT / "results" / "mot20_verification.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    log.info("verification %s — %d failures, manifest sha256=%s",
              "PASSED" if not failures else "FAILED", len(failures), manifest)
    for f in failures:
        log.info("  FAIL: %s", f)
    log.info("report -> %s", out)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
