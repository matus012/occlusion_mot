"""Verify a MOT17 train copy against the official spec (structure + internal consistency).

Used for ANY source (official zip or mirror — user directive 2026-07-22): the official site
publishes no checksums, so verification is structural against the published spec plus
internal-consistency checks, with SHA256 hashes recorded for provenance:
  - all 21 train sequence dirs present (7 sequences x DPM/FRCNN/SDP)
  - seqinfo.ini matches official seqLength / imWidth / imHeight
  - img1/ contains exactly seqLength frames named 000001.jpg..0#seqLength.jpg
  - gt/gt.txt parses; frames in [1, seqLength]; ids >= 1; visibility in [0, 1]
  - gt.txt byte-identical across the three detector variants of each sequence
  - SHA256 of every gt.txt + seqinfo.ini -> results/mot17_verification.json

Exit 0 = verified; exit 1 = failures (report printed and written).
Usage: .venv/Scripts/python.exe scripts/verify_mot17.py [--data-root data/MOT17] [--source URL]
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
log = logging.getLogger("verify_mot17")

ROOT = Path(__file__).resolve().parents[1]

# Official MOT17 train spec: sequence -> (seqLength, imWidth, imHeight)
OFFICIAL_TRAIN_SPEC: dict[str, tuple[int, int, int]] = {
    "MOT17-02": (600, 1920, 1080),
    "MOT17-04": (1050, 1920, 1080),
    "MOT17-05": (837, 640, 480),
    "MOT17-09": (525, 1920, 1080),
    "MOT17-10": (654, 1920, 1080),
    "MOT17-11": (900, 1920, 1080),
    "MOT17-13": (750, 1920, 1080),
}
DETECTORS = ("DPM", "FRCNN", "SDP")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(data_root: Path, layout: str = "official") -> tuple[list[str], dict[str, str]]:
    """Returns (failures, hashes).

    layout="official": every variant dir carries seqinfo/img1/gt/det (the official zip).
    layout="hf-dedup": images+gt+seqinfo live under FRCNN only; DPM/SDP carry det/det.txt
    only (ling1016/MOT17 mirror layout, D13). Cross-variant gt equality is then untestable;
    GT trust rests on spec/stat checks + independent cross-mirror comparison.
    """
    failures: list[str] = []
    hashes: dict[str, str] = {}
    train = data_root / "train"
    if not train.is_dir():
        return [f"missing train dir: {train}"], hashes

    for base, (length, width, height) in OFFICIAL_TRAIN_SPEC.items():
        gt_hashes: set[str] = set()
        for det in DETECTORS:
            name = f"{base}-{det}"
            seq_dir = train / name
            if not seq_dir.is_dir():
                failures.append(f"{name}: sequence dir missing")
                continue

            if layout == "hf-dedup" and det != "FRCNN":
                det_file = seq_dir / "det" / "det.txt"
                if not det_file.exists():
                    failures.append(f"{name}: det/det.txt missing (hf-dedup layout)")
                else:
                    hashes[f"{name}/det.txt"] = sha256_file(det_file)
                continue

            ini = seq_dir / "seqinfo.ini"
            if not ini.exists():
                failures.append(f"{name}: seqinfo.ini missing")
                continue
            info = parse_seqinfo(ini)
            got = (int(info["seqlength"]), int(info["imwidth"]), int(info["imheight"]))
            if got != (length, width, height):
                failures.append(f"{name}: seqinfo {got} != official {(length, width, height)}")
            hashes[f"{name}/seqinfo.ini"] = sha256_file(ini)

            img1 = seq_dir / "img1"
            n_imgs = len(list(img1.glob("*.jpg"))) if img1.is_dir() else 0
            if n_imgs != length:
                failures.append(f"{name}: img1 has {n_imgs} jpgs, expected {length}")
            for probe in (1, length):
                if not (img1 / f"{probe:06d}.jpg").exists():
                    failures.append(f"{name}: frame {probe:06d}.jpg missing")

            gt_path = seq_dir / "gt" / "gt.txt"
            if not gt_path.exists():
                failures.append(f"{name}: gt/gt.txt missing")
                continue
            gt = read_mot(gt_path)
            frames = gt[:, COL.FRAME]
            if len(gt) == 0:
                failures.append(f"{name}: gt.txt empty")
            elif not (
                frames.min() >= 1
                and frames.max() <= length
                and gt[:, COL.ID].min() >= 1
                and 0.0 <= gt[:, COL.VIS].min()
                and gt[:, COL.VIS].max() <= 1.0
            ):
                failures.append(
                    f"{name}: gt out of spec (frames [{frames.min()},{frames.max()}], "
                    f"ids>= {gt[:, COL.ID].min()}, vis [{gt[:, COL.VIS].min()},"
                    f"{gt[:, COL.VIS].max()}])"
                )
            digest = sha256_file(gt_path)
            hashes[f"{name}/gt.txt"] = digest
            gt_hashes.add(digest)

        if len(gt_hashes) > 1:
            failures.append(f"{base}: gt.txt differs across detector variants {sorted(gt_hashes)}")
    return failures, hashes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=ROOT / "data" / "MOT17")
    ap.add_argument("--source", default="unknown", help="provenance URL/revision to record")
    ap.add_argument("--layout", choices=["official", "hf-dedup"], default="official")
    args = ap.parse_args()

    failures, hashes = verify(args.data_root, layout=args.layout)
    manifest = hashlib.sha256(
        json.dumps(hashes, sort_keys=True).encode("utf-8")
    ).hexdigest()
    report = {
        "passed": not failures,
        "source": args.source,
        "data_root": str(args.data_root),
        "failures": failures,
        "manifest_sha256": manifest,
        "file_sha256": hashes,
    }
    out = ROOT / "results" / "mot17_verification.json"
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
