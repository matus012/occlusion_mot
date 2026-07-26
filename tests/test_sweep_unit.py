"""Synthetic tests for scripts/sweep_unit.py's pure command-builder / parsing logic
(no subprocess execution, no model/GPU/data needed)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, rel_path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / rel_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


su = _load_module("sweep_unit_test", "scripts/sweep_unit.py")


def _base_cfg() -> dict[str, Any]:
    return {
        "name": "dryrun_test",
        "half": "dev",
        "gate_probe": [0.45],
        "epochs": 3,
        "batches_per_epoch": 20,
        "seeds": [0],
        "pools": [300],
        "arms": {
            "A": {"sources": None},
            "B": {"sources": ["sim"]},
            "C": {"sources": ["mot17_dev", "mot20", "market1501"]},
            "D": {"sources": ["mot17_dev", "sim", "mot20", "market1501"]},
        },
        "mcnemar_pairs": [["D", "A"], ["D", "C"], ["C", "A"]],
        "detector": {
            "enabled": True, "models": ["yolo11s"], "mixes": ["mot17dev"],
            "base_weights": None, "epochs": 1, "imgsz": 640, "batch": 4,
        },
    }


# ---------------------------------------------------------------------------
# command builders
# ---------------------------------------------------------------------------


def test_train_reid_cmd_shape() -> None:
    cfg = _base_cfg()
    cmd = su.train_reid_cmd(cfg, "B", None, 0, device="cuda")
    assert cmd[0].endswith("train_reid.py")
    assert "--sources" in cmd and "sim" in cmd
    assert "--epochs" in cmd and str(cfg["epochs"]) in cmd
    assert "--identity-frac" in cmd
    assert "--tag" in cmd
    assert "sweep_dryrun_test_B_pfull_s0" in cmd
    assert "--device" in cmd and "cuda" in cmd


def test_train_reid_cmd_omits_device_when_none() -> None:
    cfg = _base_cfg()
    cmd = su.train_reid_cmd(cfg, "B", None, 0, device=None)
    assert "--device" not in cmd


def test_train_reid_cmd_rejects_arm_a() -> None:
    cfg = _base_cfg()
    with pytest.raises(AssertionError):
        su.train_reid_cmd(cfg, "A", 300, 0, device=None)


def test_cache_embeddings_cmd_with_and_without_weights(tmp_path: Path) -> None:
    ckpt = tmp_path / "reid_x.pt"
    cmd_a = su.cache_embeddings_cmd("imagenet", None, None)
    assert "--weights" not in cmd_a
    assert "--tag" in cmd_a and "imagenet" in cmd_a

    cmd_b = su.cache_embeddings_cmd("sweep_x", ckpt, "cpu")
    assert "--weights" in cmd_b and str(ckpt) in cmd_b
    assert "--device" in cmd_b and "cpu" in cmd_b


def test_run_hidden_cmd_includes_gate_and_embedder_tag() -> None:
    cfg = _base_cfg()
    cmd = su.run_hidden_cmd(cfg, "D", 300, 0, 0.45, "sweep_dryrun_test_D_p300_s0")
    assert "--half" in cmd and cfg["half"] in cmd
    assert "--app-gate-lost" in cmd and "0.45" in cmd
    assert "--app-gate-recover" in cmd and "0.45" in cmd
    assert "--embedder-tag" in cmd and "sweep_dryrun_test_D_p300_s0" in cmd
    assert "--skip-trackeval" in cmd
    for flag in su.GATE_CONFIG:
        assert f"--{flag}" in cmd


def test_finetune_detector_cmd_uses_model_pt_default() -> None:
    cfg = _base_cfg()
    cmd = su.finetune_detector_cmd(cfg, "yolo11s", "sweep_tag", "cuda", 0)
    assert "--base-weights" in cmd
    idx = cmd.index("--base-weights")
    assert cmd[idx + 1] == "yolo11s.pt"
    assert "--epochs" in cmd and "1" in cmd
    assert "--imgsz" in cmd and "640" in cmd
    assert "--batch" in cmd and "4" in cmd


def test_finetune_detector_cmd_respects_explicit_base_weights() -> None:
    cfg = _base_cfg()
    cfg["detector"]["base_weights"] = "custom.pt"
    cmd = su.finetune_detector_cmd(cfg, "yolo11s", "sweep_tag", None, 0)
    idx = cmd.index("--base-weights")
    assert cmd[idx + 1] == "custom.pt"


# ---------------------------------------------------------------------------
# ultralytics results.csv parsing
# ---------------------------------------------------------------------------


def test_parse_yolo_results_csv_reads_last_row(tmp_path: Path) -> None:
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "epoch,metrics/mAP50(B),metrics/recall(B)\n"
        "1,0.500,0.600\n"
        "2,0.700,0.800\n",
        encoding="utf-8",
    )
    metrics = su.parse_yolo_results_csv(csv_path)
    assert metrics["epoch"] == pytest.approx(2.0)
    assert metrics["metrics/mAP50(B)"] == pytest.approx(0.700)
    assert metrics["metrics/recall(B)"] == pytest.approx(0.800)


def test_parse_yolo_results_csv_missing_file_fails_fast(tmp_path: Path) -> None:
    with pytest.raises(AssertionError):
        su.parse_yolo_results_csv(tmp_path / "nope.csv")


def test_parse_yolo_results_csv_empty_rows_fails_fast(tmp_path: Path) -> None:
    csv_path = tmp_path / "results.csv"
    csv_path.write_text("epoch,metrics/mAP50(B)\n", encoding="utf-8")
    with pytest.raises(AssertionError):
        su.parse_yolo_results_csv(csv_path)


def test_parse_yolo_results_csv_skips_non_numeric_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "epoch,note,metrics/mAP50(B)\n1,ok,0.5\n",
        encoding="utf-8",
    )
    metrics = su.parse_yolo_results_csv(csv_path)
    assert "note" not in metrics
    assert metrics["metrics/mAP50(B)"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# unit parsing / device dispatch (no execution)
# ---------------------------------------------------------------------------


def test_parse_unit_dispatches_embedder_vs_detector() -> None:
    assert su.parse_unit("D:300:0")[0] == "embedder"
    assert su.parse_unit("detector:yolo11s:mot17dev")[0] == "detector"
