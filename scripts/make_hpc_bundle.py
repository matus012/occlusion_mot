"""PERUN transfer bundle: build / verify / unpack (HPC-readiness, D47).

One command on the dev box produces a single self-contained tar.gz; three copy-paste
lines on the cluster turn it back into a runnable tree (see HPC_RUNBOOK.md).

    build   -- stage every input the sweep config declares, hash it, archive it
    verify  -- re-check every manifest entry's sha256 (runs cluster-side, stdlib only)
    unpack  -- expand the staged tars into <dest>/repo (+ <dest>/wheelhouse)

WHAT GETS HASHED. A bundle entry is either a LOOSE FILE (hashed individually) or a
TREE PACKED INTO A TAR (the tar hashed individually, plus member/byte counts). The
per-file guarantee is preserved for everything a human would inspect by hand, while
data/reid's 509k crops ride inside one tar: a tar's sha256 covers every byte of every
member, so integrity is strictly stronger than a per-file listing -- and the manifest
stays a few KB instead of ~50 MB, with verification a few sequential reads instead of
half a million stats. The member/byte counts in the manifest make a truncated or
partially-staged tree impossible to miss.

WHY NESTED TARS AT ALL. Cluster home directories carry inode quotas (commonly 100k-500k);
data/reid alone is 509,122 files. Shipping it as one tar member lets the operator choose
where those inodes land (`unpack --dest $SCRATCH/...`) instead of discovering the quota
the hard way mid-extraction.

Usage (dev box):
  .venv/Scripts/python.exe scripts/make_hpc_bundle.py build
  .venv/Scripts/python.exe scripts/make_hpc_bundle.py build --refresh-lock --refresh-wheelhouse
  .venv/Scripts/python.exe scripts/make_hpc_bundle.py build --no-archive   # stage only, for rsync

Usage (cluster, system python3, no venv needed -- stdlib only on these paths):
  python3 make_hpc_bundle.py verify
  python3 make_hpc_bundle.py unpack --dest .
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tarfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
                    stream=sys.stdout)
logger = logging.getLogger("make_hpc_bundle")

ROOT = Path(__file__).resolve().parents[1]

BUNDLE_VERSION = 2
BUNDLE_DIRNAME = "omot_hpc"
MANIFEST_NAME = "MANIFEST.json"
DEFAULT_CONFIG = "configs/sweep/perun_full.yaml"

# Pinned re-ID checkpoints (val_manifest.md D35/D45 freeze). reid_conv.pt is the
# val-pinned embedder; reid_proto.pt is the R3 report-only arm from the same freeze.
PINNED_CHECKPOINTS = ("data/models/reid_conv.pt", "data/models/reid_proto.pt")

# torchvision downloads this at `resnet18(weights=IMAGENET1K_V1)` -- the trunk
# initialiser for EVERY train_reid.py run AND arm A's entire embedder. Without it
# pre-staged, an offline compute node fails at the first unit.
RESNET18_ASSET = "resnet18-f37072fd.pth"

# gzip level 1: the payload is ~90% already-compressed media (JPEG crops, PNG renders)
# and zip-compressed wheels. Higher levels burn minutes to save low single-digit percent.
GZIP_LEVEL = 1


# --------------------------------------------------------------------------- model


@dataclass(frozen=True)
class Requirement:
    """One input the sweep config declares it needs on the cluster."""

    key: str
    kind: str
    path: str  # repo-relative, POSIX
    reason: str


@dataclass
class EntrySpec:
    """One member of the bundle: a loose file, or a tree packed into a tar."""

    dest: str  # path inside the bundle
    kind: str  # "file" | "tar"
    sources: list[str] = field(default_factory=list)  # repo-relative, POSIX
    covers: list[str] = field(default_factory=list)  # Requirement.key values
    unpack_to: str = "repo"  # "repo" | "bundle" | "" (no unpack)
    note: str = ""
    # Arcnames to store members under, positionally matched to `sources`. Empty means
    # "same as sources" (the usual case: repo-relative in, repo-relative out). Only
    # the wheelhouse differs -- it is staged from data/wheelhouse but unpacks to
    # <dest>/wheelhouse, beside repo/ rather than inside it.
    arcnames: list[str] = field(default_factory=list)


def _posix(p: Path, root: Path) -> str:
    return p.resolve().relative_to(root.resolve()).as_posix()


def mot17_sequences(root: Path = ROOT) -> list[str]:
    """FRCNN train sequences, exactly as omot.data.mot.load_split enumerates them.

    Globbed rather than hardcoded so the completeness guard fails loudly if the
    dev box and the sweep config disagree about what data exists.
    """
    train = root / "data" / "MOT17" / "train"
    if not train.is_dir():
        return []
    return sorted(d.name for d in train.iterdir() if d.is_dir() and d.name.endswith("-FRCNN"))


def arm_eval_sources(cfg: dict[str, Any], arm: str) -> list[str]:
    """Sources an arm touches. Arm A trains on nothing but EVALUATES the ImageNet
    null over every source, so it pulls the full set in (sweep_common.arm_sources)."""
    all_sources = ["mot17_dev", "sim", "mot20", "market1501"]
    sources = cfg["arms"][arm]["sources"]
    return list(sources) if sources is not None else all_sources


def required_inputs(cfg: dict[str, Any], root: Path = ROOT) -> list[Requirement]:
    """Every non-repo input the config's unit list will read on the cluster.

    Derived from the config, never hardcoded -- this is the function the pytest
    completeness guard compares the bundle plan against.
    """
    reqs: list[Requirement] = []

    sources: set[str] = set()
    for arm in cfg["arms"]:
        sources.update(arm_eval_sources(cfg, arm))
    for src in sorted(sources):
        reqs.append(Requirement(
            f"reid:{src}", "reid_source", f"data/reid/{src}",
            f"train_reid.py --sources {src} (pre-resized 64x128 crops + index.json)",
        ))

    reqs.append(Requirement(
        "mot17", "mot17", "data/MOT17",
        "cache_embeddings.py crops detections from frames; run_hidden.py needs GT; "
        "finetune_detector.py builds its dev-half dataset from these frames",
    ))

    for seq in mot17_sequences(root):
        reqs.append(Requirement(
            f"detcache:{seq}", "detection_cache",
            f"data/cache/detections/{seq}__yolo11x.npz",
            "FIXED DETECTIONS invariant (CLAUDE.md): every arm consumes the identical "
            "yolo11x cache; only the embedder tag varies",
        ))

    reqs.append(Requirement(
        "segments", "repo_file", "results/occlusion_segments.json",
        "run_hidden.py asserts the full (non-partial) occlusion segment index",
    ))

    det = cfg.get("detector")
    if det and det.get("enabled"):
        for model in det["models"]:
            weights = det.get("base_weights") or f"{model}.pt"
            reqs.append(Requirement(
                f"weights:{weights}", "base_weights", weights,
                f"finetune_detector.py --base-weights {weights}; ultralytics would "
                "otherwise fetch it from GitHub at runtime",
            ))
        reqs.append(Requirement(
            "weights:yolo26n.pt", "base_weights", "yolo26n.pt",
            "ultralytics.utils.checks.check_amp() loads yolo26n.pt before every "
            "detector train; offline it degrades to a warning, but AMP then silently "
            "goes unverified",
        ))
        if any("carla" in mix for mix in det["mixes"]):
            reqs.append(Requirement(
                "carla_render", "carla_render", "data/sim/carla_render",
                "the mot17dev_carla mix folds rendered CARLA scenario frames into TRAIN",
            ))

    reqs.append(Requirement(
        "asset:resnet18", "asset", f"assets/torch/hub/checkpoints/{RESNET18_ASSET}",
        "torchvision ResNet18_Weights.IMAGENET1K_V1 -- trunk init for every train_reid.py "
        "run and the whole of arm A; staged so TORCH_HOME resolves it offline",
    ))
    reqs.append(Requirement(
        "asset:ultralytics_font", "asset", "assets/ultralytics/Arial.ttf",
        "ultralytics plotting font; fetched from GitHub on first use unless "
        "YOLO_CONFIG_DIR already holds it",
    ))

    for ckpt in PINNED_CHECKPOINTS:
        reqs.append(Requirement(
            f"checkpoint:{Path(ckpt).name}", "checkpoint", ckpt,
            "val_manifest.md-pinned embedder checkpoint (sha256-frozen, D35/D45)",
        ))

    reqs.append(Requirement(
        "wheelhouse", "wheelhouse", "data/wheelhouse",
        "offline pip install source: linux_x86_64 / cp311 wheels for requirements-lock.txt",
    ))
    reqs.append(Requirement(
        "lock", "repo_file", "requirements-lock.txt",
        "exact cluster environment, resolved for x86_64-manylinux_2_28 + cu126",
    ))
    return reqs


def plan_entries(cfg: dict[str, Any], root: Path = ROOT) -> list[EntrySpec]:
    """The bundle layout. Every Requirement must be covered by exactly one entry."""
    sources = sorted({s for arm in cfg["arms"] for s in arm_eval_sources(cfg, arm)})
    seqs = mot17_sequences(root)

    entries: list[EntrySpec] = [
        EntrySpec(
            dest="repo.tar", kind="tar", sources=["@git-archive"],
            covers=["segments", "lock"], unpack_to="repo",
            note="git archive HEAD -- tracked files only, no .git, no gitignored junk",
        ),
        EntrySpec(
            dest="wheelhouse.tar", kind="tar", sources=["data/wheelhouse"],
            arcnames=["wheelhouse"], covers=["wheelhouse"], unpack_to="bundle",
            note="unpacks beside repo/ as wheelhouse/, not into the repo tree",
        ),
        EntrySpec(
            dest="payload/data_reid.tar", kind="tar",
            sources=[f"data/reid/{s}" for s in sources],
            covers=[f"reid:{s}" for s in sources], unpack_to="repo",
            note="pre-resized 64x128 crops -- the dominant inode cost of the bundle",
        ),
        EntrySpec(
            dest="payload/data_MOT17.tar", kind="tar", sources=["data/MOT17"],
            covers=["mot17"], unpack_to="repo",
        ),
        EntrySpec(
            dest="payload/data_cache_detections.tar", kind="tar",
            sources=[f"data/cache/detections/{s}__yolo11x.npz" for s in seqs],
            covers=[f"detcache:{s}" for s in seqs], unpack_to="repo",
            note="yolo11x caches ONLY: per-unit emb_* caches are regenerated cluster-side",
        ),
        EntrySpec(
            dest="payload/models.tar", kind="tar", sources=list(PINNED_CHECKPOINTS),
            covers=[f"checkpoint:{Path(c).name}" for c in PINNED_CHECKPOINTS],
            unpack_to="repo",
        ),
        EntrySpec(
            dest="payload/assets.tar", kind="tar",
            sources=[f"assets/torch/hub/checkpoints/{RESNET18_ASSET}",
                     "assets/ultralytics/Arial.ttf"],
            covers=["asset:resnet18", "asset:ultralytics_font"], unpack_to="repo",
            note="TORCH_HOME / YOLO_CONFIG_DIR targets; see scripts/hpc_bootstrap.sh",
        ),
    ]

    det = cfg.get("detector")
    if det and det.get("enabled"):
        weights = [det.get("base_weights") or f"{m}.pt" for m in det["models"]]
        weights.append("yolo26n.pt")
        entries.append(EntrySpec(
            dest="payload/base_weights.tar", kind="tar", sources=weights,
            covers=[f"weights:{w}" for w in weights], unpack_to="repo",
            note="*.pt is gitignored, so these never ride along in repo.tar",
        ))
        if any("carla" in mix for mix in det["mixes"]):
            entries.append(EntrySpec(
                dest="payload/data_sim_carla_render.tar", kind="tar",
                sources=["data/sim/carla_render"], covers=["carla_render"],
                unpack_to="repo",
                note="raw renders, not the prepped YOLO dataset: finetune_detector.py "
                     "writes absolute paths into its data yaml, so prep runs cluster-side",
            ))

    entries.append(EntrySpec(
        dest=MANIFEST_NAME, kind="file", sources=[], covers=[], unpack_to="",
    ))
    return entries


def coverage_gaps(cfg: dict[str, Any], root: Path = ROOT) -> list[str]:
    """Requirement keys the bundle plan does NOT ship. Empty list == complete."""
    covered = {k for e in plan_entries(cfg, root) for k in e.covers}
    return sorted(r.key for r in required_inputs(cfg, root) if r.key not in covered)


# --------------------------------------------------------------------------- hashing


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------- build


def git(*args: str, root: Path = ROOT) -> str:
    out = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True)
    return out.stdout.strip()


def git_state(root: Path = ROOT) -> dict[str, Any]:
    try:
        dirty = git("status", "--porcelain", "--untracked-files=no", root=root)
        return {
            "commit": git("rev-parse", "HEAD", root=root),
            "short": git("rev-parse", "--short", "HEAD", root=root),
            "branch": git("rev-parse", "--abbrev-ref", "HEAD", root=root),
            "dirty": bool(dirty),
            "dirty_paths": dirty.splitlines(),
        }
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:  # not a git repo
        logger.warning("git state unavailable: %s", exc)
        return {"commit": None, "short": "nogit", "branch": None, "dirty": False,
                "dirty_paths": []}


def _add_tree(tar: tarfile.TarFile, src: Path, arcname: str) -> tuple[int, int]:
    """Add a file or directory tree with POSIX arcnames. Returns (members, bytes)."""
    members = 0
    total = 0
    if src.is_file():
        info = tar.gettarinfo(str(src), arcname=arcname)
        info.mode = 0o755 if src.suffix in {".sh", ".sbatch"} else 0o644
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        with src.open("rb") as f:
            tar.addfile(info, f)
        return 1, info.size
    for path in sorted(src.rglob("*")):
        if path.is_dir():
            continue
        rel = path.relative_to(src).as_posix()
        info = tar.gettarinfo(str(path), arcname=f"{arcname}/{rel}")
        info.mode = 0o644
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        with path.open("rb") as f:
            tar.addfile(info, f)
        members += 1
        total += info.size
    return members, total


def build_entry(spec: EntrySpec, stage: Path, root: Path) -> dict[str, Any]:
    dest = stage / spec.dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    members = 0
    total = 0

    if spec.sources == ["@git-archive"]:
        logger.info("staging %s <- git archive HEAD", spec.dest)
        subprocess.run(["git", "archive", "--format=tar", "-o", str(dest), "HEAD"],
                       cwd=root, check=True)
        with tarfile.open(dest, "r:") as tar:
            infos = tar.getmembers()
        members = sum(1 for i in infos if i.isfile())
        total = sum(i.size for i in infos)
    else:
        missing = [s for s in spec.sources if not (root / s).exists()]
        if missing:
            raise FileNotFoundError(
                f"{spec.dest}: missing bundle input(s): {missing}\n"
                f"  (nothing is silently dropped -- stage the input or fix the sweep config)"
            )
        logger.info("staging %s <- %d source(s)", spec.dest, len(spec.sources))
        arcnames = spec.arcnames or spec.sources
        assert len(arcnames) == len(spec.sources), f"{spec.dest}: arcname/source mismatch"
        with tarfile.open(dest, "w:") as tar:
            for src, arc in zip(spec.sources, arcnames, strict=True):
                m, b = _add_tree(tar, root / src, arc)
                members += m
                total += b

    digest = sha256_file(dest)
    logger.info("  %s  %.3f GB  %d members  sha256=%s...",
                spec.dest, dest.stat().st_size / 1e9, members, digest[:16])
    return {
        "path": spec.dest,
        "kind": spec.kind,
        "sha256": digest,
        "bytes": dest.stat().st_size,
        "members": members,
        "member_bytes": total,
        "covers": spec.covers,
        "unpack_to": spec.unpack_to,
        "note": spec.note,
    }


def stage_assets(root: Path) -> None:
    """Copy the two runtime assets torch/ultralytics would otherwise fetch at runtime
    out of their per-user caches and into assets/ (bundle-only, gitignored)."""
    torch_dest = root / "assets" / "torch" / "hub" / "checkpoints" / RESNET18_ASSET
    if not torch_dest.exists():
        home = Path(os.environ.get("TORCH_HOME") or (Path.home() / ".cache" / "torch"))
        src = home / "hub" / "checkpoints" / RESNET18_ASSET
        if not src.exists():
            raise FileNotFoundError(
                f"{RESNET18_ASSET} not in the local torch hub cache ({src}).\n"
                f"  Populate it once with network access:\n"
                f"    python -c \"from torchvision.models import resnet18, ResNet18_Weights; "
                f"resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)\""
            )
        torch_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, torch_dest)
        logger.info("staged torch hub asset <- %s", src)

    font_dest = root / "assets" / "ultralytics" / "Arial.ttf"
    if not font_dest.exists():
        candidates = [
            Path(os.environ["YOLO_CONFIG_DIR"]) / "Arial.ttf"
            if os.environ.get("YOLO_CONFIG_DIR") else None,
            Path(os.environ.get("APPDATA", "")) / "Ultralytics" / "Arial.ttf",
            Path.home() / ".config" / "Ultralytics" / "Arial.ttf",
        ]
        src = next((c for c in candidates if c and c.exists()), None)
        if src is None:
            raise FileNotFoundError(
                "Arial.ttf not found in any ultralytics config dir; run any ultralytics "
                "command once with network access, or drop the file at "
                f"{font_dest} manually."
            )
        font_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, font_dest)
        logger.info("staged ultralytics font <- %s", src)


def refresh_lock(root: Path) -> None:
    cmd = [
        "uv", "pip", "compile", "requirements-hpc.in",
        "--python-platform", "x86_64-manylinux_2_28", "--python-version", "3.11",
        "--extra-index-url", "https://download.pytorch.org/whl/cu126",
        "--index-strategy", "unsafe-best-match", "-o", "requirements-lock.txt",
    ]
    logger.info("$ %s", " ".join(cmd))
    subprocess.run(cmd, cwd=root, check=True)


def linux_platform_tags() -> list[str]:
    """Every x86_64 manylinux platform tag a glibc-2.28 node can run.

    pip's --platform is LITERAL: it does not expand manylinux_2_28 downward to the
    older tags that a 2.28 system is perfectly capable of running. Passing only the
    floor tag silently drops real wheels (kiwisolver ships manylinux_2_17, cuDNN 9.10
    ships manylinux_2_27) and the download dies with a bogus "from versions: none".
    musllinux is deliberately absent -- these are glibc nodes.
    """
    tags = [f"manylinux_2_{minor}_x86_64" for minor in range(28, 4, -1)]
    tags += ["manylinux2014_x86_64", "manylinux2010_x86_64", "manylinux1_x86_64"]
    return tags


def refresh_wheelhouse(root: Path, python_exe: str) -> None:
    """Download linux_x86_64 / cp311 wheels for the lock into data/wheelhouse.

    glibc >= 2.28 (RHEL/Rocky 8+) is the hard floor: torch 2.13 publishes no wheel
    below manylinux_2_28, so an older cluster could not run this stack at all.
    hpc_bootstrap.sh asserts the node's glibc up front rather than letting the install
    succeed and then fail at `import torch`.

    --abi accepts abi3 and none alongside cp311: a growing number of projects ship
    cp3x-abi3 wheels, and pure-Python deps are py3-none-any.
    """
    dest = root / "data" / "wheelhouse"
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [
        python_exe, "-m", "pip", "download", "-r", "requirements-lock.txt",
        "-d", str(dest), "--only-binary=:all:",
    ]
    for tag in linux_platform_tags():
        cmd += ["--platform", tag]
    cmd += [
        "--python-version", "3.11", "--implementation", "cp",
        "--abi", "cp311", "--abi", "abi3", "--abi", "none",
        "--extra-index-url", "https://download.pytorch.org/whl/cu126",
    ]
    logger.info("$ %s", " ".join(cmd))
    subprocess.run(cmd, cwd=root, check=True)


def check_wheelhouse(root: Path, python_exe: str) -> int:
    """Resolve requirements-lock.txt offline against data/wheelhouse, for LINUX.

    This is as far as a Windows dev box can validate the cluster environment. pip
    runs its real resolver with --no-index (wheelhouse only) and the cluster's target
    tags, so a missing or wrong-platform wheel fails HERE. What it cannot prove is
    that the wheels then IMPORT on a real Linux node -- that needs the cluster.
    """
    wheelhouse = root / "data" / "wheelhouse"
    target = root / "dist" / "_resolve_probe"
    if target.exists():
        shutil.rmtree(target)
    cmd = [
        python_exe, "-m", "pip", "install", "--dry-run", "--ignore-installed",
        "--no-index", "--find-links", str(wheelhouse), "--only-binary=:all:",
        "--target", str(target), "--python-version", "3.11",
        "--implementation", "cp", "--abi", "cp311", "--abi", "abi3", "--abi", "none",
        "-r", "requirements-lock.txt",
    ]
    for tag in linux_platform_tags():
        cmd += ["--platform", tag]
    logger.info("resolving the lock offline against %s (linux/cp311 target)", wheelhouse)
    proc = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
    if proc.returncode != 0:
        logger.error("WHEELHOUSE INCOMPLETE:\n%s", (proc.stderr or proc.stdout)[-2000:])
        return 1
    n = len(list(wheelhouse.glob("*.whl")))
    size = sum(p.stat().st_size for p in wheelhouse.glob("*.whl"))
    logger.info("WHEELHOUSE OK: %d wheels, %.2f GB, every lock entry resolves offline",
                n, size / 1e9)
    return 0


def build(
    config: str = DEFAULT_CONFIG,
    root: Path = ROOT,
    out_dir: Path | None = None,
    archive: bool = True,
    allow_dirty: bool = False,
) -> Path:
    import yaml  # local: verify/unpack must stay stdlib-only for the cluster side

    config_path = root / config
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    out_dir = out_dir or (root / "dist")

    state = git_state(root)
    if state["dirty"] and not allow_dirty:
        raise SystemExit(
            "refusing to build from a dirty worktree: repo.tar comes from `git archive "
            "HEAD`, so uncommitted changes would be SILENTLY ABSENT from the bundle.\n"
            "  commit them, or pass --allow-dirty if you truly mean to ship HEAD.\n"
            "  dirty: " + ", ".join(state["dirty_paths"][:10])
        )

    gaps = coverage_gaps(cfg, root)
    if gaps:
        raise SystemExit(f"bundle plan does not cover required input(s): {gaps}")

    stage_assets(root)

    stage = out_dir / BUNDLE_DIRNAME
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    specs = [s for s in plan_entries(cfg, root) if s.kind == "tar"]
    entries = [build_entry(s, stage, root) for s in specs]

    units = _enumerate_units(cfg)
    manifest = {
        "bundle_version": BUNDLE_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": state,
        "config": config,
        "config_sha256": sha256_file(config_path),
        "sweep": {"name": cfg["name"], "n_units": len(units), "units": units},
        "requirements": [r.__dict__ for r in required_inputs(cfg, root)],
        "entries": entries,
        "totals": {
            "bytes": sum(e["bytes"] for e in entries),
            "members": sum(e["members"] for e in entries),
        },
    }
    (stage / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # standalone copies so the cluster can verify/unpack before anything is expanded
    shutil.copy2(Path(__file__), stage / "make_hpc_bundle.py")
    runbook = root / "HPC_RUNBOOK.md"
    if runbook.exists():
        shutil.copy2(runbook, stage / "HPC_RUNBOOK.md")

    logger.info("staged bundle: %s (%.2f GB, %d members)",
                stage, manifest["totals"]["bytes"] / 1e9, manifest["totals"]["members"])

    if not archive:
        return stage

    tgz = out_dir / f"{BUNDLE_DIRNAME}_{state['short']}.tar.gz"
    logger.info("compressing -> %s (gzip level %d)", tgz, GZIP_LEVEL)
    with tarfile.open(tgz, "w:gz", compresslevel=GZIP_LEVEL) as tar:
        tar.add(stage, arcname=BUNDLE_DIRNAME)
    digest = sha256_file(tgz)
    (tgz.parent / f"{tgz.name}.sha256").write_text(f"{digest}  {tgz.name}\n",
                                                   encoding="utf-8", newline="\n")
    shutil.copy2(stage / MANIFEST_NAME, tgz.parent / f"{tgz.name}.manifest.json")
    logger.info("bundle: %s  %.2f GB  sha256=%s", tgz, tgz.stat().st_size / 1e9, digest)
    return tgz


def _enumerate_units(cfg: dict[str, Any]) -> list[str]:
    """Mirror of sweep_common.enumerate_units, inlined so `verify` stays stdlib-only."""
    units: list[str] = []
    for arm in sorted(cfg["arms"]):
        pools = cfg["arms"][arm].get("pools", cfg["pools"])
        for pool in pools:
            for seed in cfg["seeds"]:
                units.append(f"{arm}:{'full' if pool is None else pool}:{seed}")
    det = cfg.get("detector")
    if det and det.get("enabled"):
        for model in det["models"]:
            for mix in det["mixes"]:
                units.append(f"detector:{model}:{mix}")
    return units


# --------------------------------------------------------------------------- verify


def verify(bundle_dir: Path) -> int:
    manifest_path = bundle_dir / MANIFEST_NAME
    if not manifest_path.exists():
        logger.error("no %s in %s", MANIFEST_NAME, bundle_dir)
        return 2
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    logger.info("bundle %s built %s from %s%s",
                manifest["sweep"]["name"], manifest["created_utc"],
                manifest["git"]["short"], " (DIRTY)" if manifest["git"]["dirty"] else "")

    failures: list[str] = []
    for entry in manifest["entries"]:
        path = bundle_dir / entry["path"]
        if not path.exists():
            failures.append(f"{entry['path']}: MISSING")
            continue
        size = path.stat().st_size
        if size != entry["bytes"]:
            failures.append(f"{entry['path']}: size {size} != {entry['bytes']}")
            continue
        digest = sha256_file(path)
        if digest != entry["sha256"]:
            failures.append(
                f"{entry['path']}: sha256 {digest[:16]}... != {entry['sha256'][:16]}..."
            )
            continue
        logger.info("  OK  %-38s %8.3f GB  %7d members",
                    entry["path"], size / 1e9, entry["members"])

    if failures:
        for f in failures:
            logger.error("  FAIL %s", f)
        logger.error("VERIFY FAILED: %d/%d entries bad", len(failures), len(manifest["entries"]))
        return 1
    logger.info("VERIFY OK: %d entries, %.2f GB, %d members",
                len(manifest["entries"]), manifest["totals"]["bytes"] / 1e9,
                manifest["totals"]["members"])
    return 0


# --------------------------------------------------------------------------- unpack


def unpack(bundle_dir: Path, dest: Path) -> int:
    manifest = json.loads((bundle_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    repo = dest / "repo"
    repo.mkdir(parents=True, exist_ok=True)

    for entry in manifest["entries"]:
        target = {"repo": repo, "bundle": dest}.get(entry["unpack_to"])
        if target is None:
            continue
        src = bundle_dir / entry["path"]
        logger.info("unpacking %-38s -> %s", entry["path"], target)
        with tarfile.open(src, "r:*") as tar:
            _safe_extract(tar, target)

    logger.info("unpacked to %s (repo tree: %s)", dest, repo)
    return 0


def _safe_extract(tar: tarfile.TarFile, target: Path) -> None:
    """Reject absolute paths and ../ escapes before writing anything."""
    resolved = target.resolve()
    for member in tar.getmembers():
        out = (resolved / member.name).resolve()
        if not str(out).startswith(str(resolved)):
            raise RuntimeError(f"unsafe tar member escapes destination: {member.name}")
    if hasattr(tarfile, "data_filter"):  # py3.12+; py3.11 has no filter kwarg
        tar.extractall(resolved, filter="data")
    else:
        tar.extractall(resolved)


# --------------------------------------------------------------------------- cli


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="stage + hash + archive the bundle (dev box)")
    b.add_argument("--config", default=DEFAULT_CONFIG)
    b.add_argument("--out-dir", type=Path, default=None)
    b.add_argument("--no-archive", action="store_true",
                   help="stop at the staged directory (transfer it with rsync instead)")
    b.add_argument("--allow-dirty", action="store_true",
                   help="build from HEAD even though the worktree has uncommitted changes")
    b.add_argument("--refresh-lock", action="store_true", help="recompile requirements-lock.txt")
    b.add_argument("--refresh-wheelhouse", action="store_true",
                   help="re-download data/wheelhouse from the lock")
    b.add_argument("--pip-python", default=sys.executable,
                   help="interpreter that owns pip for --refresh-wheelhouse")

    v = sub.add_parser("verify", help="re-check every manifest sha256 (cluster side)")
    v.add_argument("--bundle-dir", type=Path, default=Path("."))

    u = sub.add_parser("unpack", help="expand staged tars into <dest>/repo (cluster side)")
    u.add_argument("--bundle-dir", type=Path, default=Path("."))
    u.add_argument("--dest", type=Path, default=Path("."))

    c = sub.add_parser("check-wheelhouse",
                       help="resolve the lock offline against the wheelhouse (dev box)")
    c.add_argument("--pip-python", default=sys.executable)

    args = ap.parse_args()

    if args.cmd == "check-wheelhouse":
        return check_wheelhouse(ROOT, args.pip_python)

    if args.cmd == "build":
        if args.refresh_lock:
            refresh_lock(ROOT)
        if args.refresh_wheelhouse:
            refresh_wheelhouse(ROOT, args.pip_python)
        build(config=args.config, out_dir=args.out_dir, archive=not args.no_archive,
              allow_dirty=args.allow_dirty)
        return 0
    if args.cmd == "verify":
        return verify(args.bundle_dir)
    return unpack(args.bundle_dir, args.dest)


if __name__ == "__main__":
    sys.exit(main())
