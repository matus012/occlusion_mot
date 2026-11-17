"""Guards for the Stage-1 detector launcher (perun_detector_v1.md, D56).

The D49 regression is re-asserted here for the new emitter: coreutils `timeout` takes
NUMBER[smhd] and rejects SLURM's HH:MM:SS. That bug killed all 23 tasks of array 77126 in
under a second and the suite did not catch it, because it only ever checked the
`#SBATCH --time` line.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import detector_launcher as dl  # noqa: E402
from detector_unit import enumerate_units, load_config, parse_unit  # noqa: E402

CONFIG = ROOT / "configs" / "sweep" / "perun_detector.yaml"


@pytest.fixture
def cfg():
    return load_config(CONFIG)


def test_grid_is_exactly_the_frozen_twelve(cfg):
    """D53 decision 3: no widening without a dated amendment."""
    units = enumerate_units(cfg)
    assert len(units) == 12, units
    assert {parse_unit(u)[0] for u in units} == {"yolo11s", "yolo11m"}
    assert {parse_unit(u)[1] for u in units} == {"mot20", "mot20_carla"}
    assert {parse_unit(u)[2] for u in units} == {0, 1, 2}


def test_every_mix_is_mot17_disjoint(cfg):
    """The honesty contract: Stage 1 must never train on the evaluation dataset."""
    assert not [m for m in cfg["mixes"] if "mot17" in m.lower()]


def test_timeout_limits_use_coreutils_duration_grammar(cfg, tmp_path):
    text = dl.render_sbatch(cfg, CONFIG, tmp_path)
    block = re.search(r"^LIMITS=\(\n(.*?)^\)", text, re.S | re.M)
    assert block, "no LIMITS array emitted"
    entries = re.findall(r'"([^"]+)"', block.group(1))
    assert len(entries) == 12
    grammar = re.compile(r"^\d+(\.\d+)?[smhd]?$")
    for e in entries:
        assert grammar.match(e), f"{e!r} is not a coreutils timeout DURATION"
        assert ":" not in e, f"{e!r} looks like SLURM HH:MM:SS, which timeout rejects"


def test_slurm_time_and_timeout_encode_the_same_wall(cfg):
    for row in dl.budget_table(cfg)["rows"]:
        h, m, s = (int(v) for v in row["wall_hms"].split(":"))
        assert int(row["wall_timeout"].rstrip("smhd")) == h * 3600 + m * 60 + s


def test_worst_case_fits_the_remaining_ceiling(cfg):
    b = dl.assert_budget(cfg)
    assert b["fits"]
    assert b["worst_case_h"] <= b["ceiling_h"]
    assert b["nominal_h"] < b["worst_case_h"], "caps must sit above the estimates"


def test_budget_assertion_actually_fires_when_the_grid_is_too_big(cfg):
    """A budget guard that cannot fail is not a guard."""
    fat = dict(cfg)
    fat["slurm"] = dict(cfg["slurm"], budget_ceiling_h=1.0)
    with pytest.raises(AssertionError, match="exceeds the remaining ceiling"):
        dl.assert_budget(fat)


def test_concurrency_cap_is_never_raised(cfg, tmp_path):
    assert cfg["slurm"]["max_concurrent"] == 8
    text = dl.render_sbatch(cfg, CONFIG, tmp_path)
    assert re.search(r"#SBATCH --array=0-11%8\b", text), "array cap must stay at %8"


def test_sbatch_is_posix_and_repo_relative(cfg, tmp_path):
    text = dl.render_sbatch(cfg, CONFIG, tmp_path)
    # A trailing backslash is a bash line continuation and is legitimate. What must never
    # appear is a Windows separator or drive letter -- the dev box is Windows, the cluster
    # is not.
    without_continuations = text.replace("\\\n", "\n")
    assert "\\" not in without_continuations, "a Windows path separator leaked"
    assert not re.search(r"[A-Za-z]:[\\/]", text), "a drive-letter path leaked"
    assert str(ROOT) not in text
    assert "configs/sweep/perun_detector.yaml" in text


def test_sbatch_carries_resume_skip(cfg, tmp_path):
    text = dl.render_sbatch(cfg, CONFIG, tmp_path)
    assert 'if [ -f "$RESULT" ]' in text, "re-submitting must skip finished units"


def test_ols_recovers_a_known_slope():
    xs = [0.0, 1.0, 2.0, 3.0, 4.0]
    fit = dl.ols_with_ci(xs, [1.0 + 2.0 * x for x in xs])
    assert fit["slope"] == pytest.approx(2.0)
    assert fit["r2"] == pytest.approx(1.0)
    assert fit["ci_lo"] <= 2.0 <= fit["ci_hi"]


def test_ols_flat_curve_ci_contains_zero():
    """The kill criterion reads exactly this: a flat curve must NOT look significant."""
    xs = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    ys = [0.5, 0.51, 0.49, 0.5, 0.52, 0.48]
    fit = dl.ols_with_ci(xs, ys)
    assert fit["ci_lo"] < 0 < fit["ci_hi"], "noise around a constant must not exclude zero"
