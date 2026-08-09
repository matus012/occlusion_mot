"""D47: the HPC bundle must ship every input the sweep config declares.

The guard that matters is `test_bundle_plan_covers_every_required_input`: it derives
the requirement set from configs/sweep/perun_full.yaml and asserts the bundle layout
covers all of it. A new arm, source, detector model or data mix added to the config
therefore fails HERE, on the dev box, instead of on a compute node with no internet
40 hours into an allocation.

The build/verify/unpack round-trip runs against a synthetic repo fixture, so it
exercises the real code paths without staging ~11 GB.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import make_hpc_bundle as mhb  # noqa: E402
from sweep_common import (  # noqa: E402
    assert_budget,
    budget_table,
    enumerate_units,
    load_config,
    unit_class,
)

CONFIG = ROOT / "configs" / "sweep" / "perun_full.yaml"


@pytest.fixture(scope="module")
def cfg() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


# ----------------------------------------------------------------- completeness


def test_bundle_plan_covers_every_required_input(cfg: dict) -> None:
    gaps = mhb.coverage_gaps(cfg, ROOT)
    assert not gaps, (
        f"sweep config declares input(s) the bundle does not ship: {gaps}. "
        f"Add them to make_hpc_bundle.plan_entries()."
    )


def test_every_requirement_names_a_real_path(cfg: dict) -> None:
    """A requirement pointing at a path that does not exist on the dev box would
    surface as a build-time FileNotFoundError; catch it as a test failure first."""
    missing = [r.path for r in mhb.required_inputs(cfg, ROOT) if not (ROOT / r.path).exists()]
    # assets/ is staged on demand by build(); everything else must already be here.
    missing = [m for m in missing if not m.startswith("assets/")]
    assert not missing, f"required bundle input(s) absent from the dev tree: {missing}"


def test_reid_sources_match_arm_definitions(cfg: dict) -> None:
    """Arm A trains on nothing but evaluates the null over every source, so the
    bundle must carry all four re-ID trees even though no arm lists all four."""
    covered = {r.key for r in mhb.required_inputs(cfg, ROOT) if r.kind == "reid_source"}
    assert covered == {"reid:mot17_dev", "reid:sim", "reid:mot20", "reid:market1501"}


def test_detection_cache_required_for_every_mot17_sequence(cfg: dict) -> None:
    """FIXED DETECTIONS: one yolo11x cache per FRCNN train sequence, no exceptions."""
    seqs = mhb.mot17_sequences(ROOT)
    assert len(seqs) == 7, f"expected 7 FRCNN train sequences, found {seqs}"
    keys = {r.key for r in mhb.required_inputs(cfg, ROOT) if r.kind == "detection_cache"}
    assert keys == {f"detcache:{s}" for s in seqs}


def test_inlined_unit_enumeration_matches_sweep_common(cfg: dict) -> None:
    """make_hpc_bundle inlines enumerate_units so `verify`/`unpack` stay stdlib-only
    on the cluster (no yaml import). That is a deliberate duplication -- this test is
    what stops the two copies from drifting."""
    assert mhb._enumerate_units(cfg) == enumerate_units(cfg)


def test_plan_entries_have_no_duplicate_coverage(cfg: dict) -> None:
    seen: dict[str, str] = {}
    for entry in mhb.plan_entries(cfg, ROOT):
        for key in entry.covers:
            assert key not in seen, f"{key} covered by both {seen[key]} and {entry.dest}"
            seen[key] = entry.dest


def test_gitignored_weights_are_shipped_explicitly(cfg: dict) -> None:
    """*.pt is gitignored, so repo.tar cannot carry the detector base weights; they
    must appear as their own payload entry or the detector arm dies offline."""
    covered = {k for e in mhb.plan_entries(cfg, ROOT) for k in e.covers}
    for model in cfg["detector"]["models"]:
        assert f"weights:{model}.pt" in covered
    assert "weights:yolo26n.pt" in covered, "ultralytics check_amp() loads yolo26n.pt"


def test_every_source_module_is_tracked_by_git() -> None:
    """repo.tar is `git archive HEAD`, so an ignored source file is simply ABSENT from
    the bundle — and from the published repo.

    This is not hypothetical: `.gitignore`'s unanchored `data/` matched src/omot/data/,
    so MOTSequence/load_split/half_split_frames were never committed. Everything kept
    working locally because the files existed in the worktree; the bundled tree failed
    at `ModuleNotFoundError: No module named 'omot.data'`. Anchored patterns fixed it —
    this test stops the next unanchored pattern from doing the same thing quietly.
    """
    on_disk = {
        p.relative_to(ROOT).as_posix()
        for p in (ROOT / "src").rglob("*.py")
        if "__pycache__" not in p.parts
    }
    tracked = set(
        subprocess.run(["git", "ls-files", "src"], cwd=ROOT, capture_output=True,
                       text=True, check=True).stdout.split()
    )
    assert not (on_disk - tracked), (
        f"source file(s) present on disk but NOT tracked by git: "
        f"{sorted(on_disk - tracked)} -- they will be missing from repo.tar"
    )


def test_no_gitignore_pattern_shadows_a_source_directory() -> None:
    """An unanchored directory pattern matches at ANY depth.

    That is correct and wanted for build noise (`__pycache__/`, `.venv/`, `build/`), so
    the rule is not "anchor everything" -- it is "no unanchored pattern may name a
    directory that actually exists under src/". `data/` broke exactly that rule.
    """
    src = ROOT / "src"
    shadowed = []
    for line in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if not entry or entry.startswith(("#", "/", "!", "*")):
            continue
        if not (entry.endswith("/") and entry.count("/") == 1):
            continue  # anchored, or a multi-segment path pattern
        name = entry.rstrip("/")
        if name == "__pycache__":
            continue  # legitimately ignored at every depth
        if any(d.is_dir() for d in src.rglob(name)):
            shadowed.append(entry)
    assert not shadowed, (
        f"unanchored .gitignore pattern(s) {shadowed} match a real directory under "
        f"src/ -- those sources would be missing from repo.tar. Prefix with '/'."
    )


# ----------------------------------------------------------------- budget / SLURM


def test_unit_classes_partition_the_grid(cfg: dict) -> None:
    units = enumerate_units(cfg)
    table = budget_table(cfg)
    assert sum(c["n_units"] for c in table["classes"].values()) == len(units)
    assert {unit_class(u) for u in units} <= set(mhb_unit_classes())


def mhb_unit_classes() -> tuple[str, ...]:
    return ("embedder", "embedder_null", "detector")


def test_grid_fits_the_ceiling_at_the_low_estimate(cfg: dict) -> None:
    table = assert_budget(cfg)  # raises if the low estimate busts the ceiling
    assert table["total_low_h"] <= table["ceiling_h"]


def test_wall_limits_exceed_the_estimates_they_cap(cfg: dict) -> None:
    """A wall limit below its own estimate would kill healthy units."""
    for name, cls in budget_table(cfg)["classes"].items():
        h, m, _ = (int(x) for x in cls["time_limit"].split(":"))
        assert h + m / 60 >= cls["est_high_h"], f"{name}: limit under its high estimate"


# ----------------------------------------------------------------- round trip


def _fixture_repo(tmp_path: Path) -> tuple[Path, dict]:
    """A miniature repo with the same shape as the real one, tracked by real git."""
    root = tmp_path / "repo"
    files = {
        "configs/sweep/tiny.yaml": "",  # filled below
        "results/occlusion_segments.json": '{"partial": false}',
        "requirements-lock.txt": "numpy==1.26.4\n",
        "data/reid/sim/index.json": '{"identities": {}}',
        "data/reid/sim/crops/a.jpg": "jpegbytes",
        "data/reid/mot17_dev/index.json": '{"identities": {}}',
        "data/MOT17/train/MOT17-02-FRCNN/seqinfo.ini": "[Sequence]\n",
        "data/cache/detections/MOT17-02-FRCNN__yolo11x.npz": "npz",
        "data/models/reid_conv.pt": "ckpt-conv",
        "data/models/reid_proto.pt": "ckpt-proto",
        "data/wheelhouse/numpy-1.26.4.whl": "wheel",
        "yolo11s.pt": "weights-s",
        "yolo26n.pt": "weights-n",
        "assets/torch/hub/checkpoints/resnet18-f37072fd.pth": "imagenet",
        "assets/ultralytics/Arial.ttf": "font",
    }
    cfg = {
        "name": "tiny", "half": "dev", "gate_probe": [0.45], "epochs": 1,
        "batches_per_epoch": 1, "eval_every": 1, "seeds": [0], "pools": [None],
        "arms": {"A": {"sources": None}, "D": {"sources": ["mot17_dev", "sim"]}},
        "mcnemar_pairs": [["D", "A"]],
        "detector": {"enabled": True, "models": ["yolo11s"], "mixes": ["mot17dev"],
                     "base_weights": None, "epochs": 1, "imgsz": 64, "batch": 1},
        "slurm": {"partition": "p", "account": "a", "budget_ceiling_h": 40},
        "env": {"device": "cuda"},
    }
    files["configs/sweep/tiny.yaml"] = yaml.safe_dump(cfg, sort_keys=False)
    for rel, content in files.items():
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")
    # data/reid mirrors the real config's four sources for arm A's eval sweep
    for src in ("mot20", "market1501"):
        (root / "data" / "reid" / src).mkdir(parents=True, exist_ok=True)
        (root / "data" / "reid" / src / "index.json").write_text('{"identities": {}}',
                                                                 encoding="utf-8")

    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=root, check=True)
    # only the tracked, non-data files land in repo.tar
    for rel in ("configs/sweep/tiny.yaml", "results/occlusion_segments.json",
                "requirements-lock.txt"):
        subprocess.run(["git", "add", "-f", rel], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=root, check=True)
    return root, cfg


def test_build_verify_unpack_round_trip(tmp_path: Path) -> None:
    root, cfg = _fixture_repo(tmp_path)

    bundle = mhb.build(config="configs/sweep/tiny.yaml", root=root,
                       out_dir=tmp_path / "dist", archive=False)
    manifest = json.loads((bundle / mhb.MANIFEST_NAME).read_text(encoding="utf-8"))

    assert manifest["sweep"]["name"] == "tiny"
    assert manifest["git"]["dirty"] is False
    assert {e["path"] for e in manifest["entries"]} >= {
        "repo.tar", "wheelhouse.tar", "payload/data_reid.tar", "payload/models.tar",
        "payload/assets.tar", "payload/base_weights.tar",
    }
    assert mhb.verify(bundle) == 0

    dest = tmp_path / "out"
    assert mhb.unpack(bundle, dest) == 0
    for rel in ("repo/results/occlusion_segments.json", "repo/data/reid/sim/crops/a.jpg",
                "repo/data/models/reid_conv.pt", "repo/yolo11s.pt",
                "repo/assets/ultralytics/Arial.ttf", "wheelhouse/numpy-1.26.4.whl"):
        assert (dest / rel).exists(), f"unpack lost {rel}"


def test_verify_detects_corruption(tmp_path: Path) -> None:
    root, _ = _fixture_repo(tmp_path)
    bundle = mhb.build(config="configs/sweep/tiny.yaml", root=root,
                       out_dir=tmp_path / "dist", archive=False)
    target = bundle / "payload" / "models.tar"
    data = bytearray(target.read_bytes())
    data[-1024] ^= 0xFF  # single flipped byte inside a member, size unchanged
    target.write_bytes(bytes(data))
    assert mhb.verify(bundle) == 1


def test_build_refuses_a_dirty_worktree(tmp_path: Path) -> None:
    root, _ = _fixture_repo(tmp_path)
    (root / "requirements-lock.txt").write_text("numpy==9.9.9\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="dirty worktree"):
        mhb.build(config="configs/sweep/tiny.yaml", root=root,
                  out_dir=tmp_path / "dist", archive=False)


def test_repo_tar_excludes_gitignored_data(tmp_path: Path) -> None:
    """repo.tar is `git archive HEAD`: untracked crops/weights must not sneak in
    (they would double the bundle and, for crops, violate D37 license hygiene)."""
    root, _ = _fixture_repo(tmp_path)
    bundle = mhb.build(config="configs/sweep/tiny.yaml", root=root,
                       out_dir=tmp_path / "dist", archive=False)
    with tarfile.open(bundle / "repo.tar", "r:") as tar:
        names = tar.getnames()
    assert not [n for n in names if n.startswith("data/")], names
    assert not [n for n in names if n.endswith(".pt")], names
    assert "results/occlusion_segments.json" in names


# ----------------------------------------------------------------- emitted scripts


def test_emitted_sbatch_is_lf_and_resume_safe(tmp_path: Path) -> None:
    """CRLF in an sbatch script reaches the cluster as `#!/bin/bash\\r` and dies with
    "bad interpreter" -- the exact failure this repo shipped before D47."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import sweep_launcher as sl

    cfg = load_config(CONFIG)
    out_dir = tmp_path / "sweep"
    sl.run_slurm(cfg, CONFIG, out_dir)

    for name in ("submit.sbatch", "submit_smoke.sbatch"):
        raw = (out_dir / cfg["name"] / name).read_bytes()
        assert b"\r\n" not in raw, f"{name} has CRLF line endings"
        assert raw.startswith(b"#!/bin/bash\n")
        assert b"\\" not in raw.split(b"UNITS=(")[0] or name == "submit_smoke.sbatch"

    text = (out_dir / cfg["name"] / "submit.sbatch").read_text(encoding="utf-8")
    assert 'if [ -f "$RESULT" ]; then' in text, "array is not resume-safe"
    assert "timeout --signal=TERM" in text, "per-unit wall limit not enforced"
    assert f"results/sweep/{cfg['name']}/logs/" in text
    assert (out_dir / cfg["name"] / "logs").is_dir(), "SLURM will not create --output dirs"

    smoke = (out_dir / cfg["name"] / "submit_smoke.sbatch").read_text(encoding="utf-8")
    assert "torch.cuda.is_available()" in smoke, "smoke job must probe the GPU"
    for mix in cfg["detector"]["mixes"]:
        assert f"--prep-only --mix {mix}" in smoke, f"{mix} dataset prep not serialised"


def test_sbatch_array_covers_every_unit_once() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    import sweep_launcher as sl

    cfg = load_config(CONFIG)
    units = enumerate_units(cfg)
    text = sl.render_sbatch(cfg, CONFIG)
    assert f"#SBATCH --array=0-{len(units) - 1}%" in text
    for unit in units:
        assert f'"{unit}"' in text, f"{unit} missing from the emitted array"


def test_sbatch_carries_no_windows_paths() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    import sweep_launcher as sl

    cfg = load_config(CONFIG)
    for text in (sl.render_sbatch(cfg, CONFIG), sl.render_smoke_sbatch(cfg, CONFIG)):
        assert "\\" not in text.replace("\\\n", ""), "backslash path leaked into sbatch"
        assert ".venv/Scripts" not in text
        assert "C:/" not in text and "C:\\" not in text
