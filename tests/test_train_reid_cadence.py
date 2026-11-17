"""D43-delta(a): synthetic tests for scripts/train_reid.py's --eval-every cadence
logic (eval_epochs) -- no torch/model/GPU/data needed."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load_module(name: str, rel_path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / rel_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tr = _load_module("train_reid_cadence_test", "scripts/train_reid.py")


def test_eval_every_one_evaluates_every_epoch() -> None:
    assert tr.eval_epochs(epochs=5, eval_every=1) == {1, 2, 3, 4, 5}


def test_eval_every_five_over_sixty_epochs() -> None:
    assert tr.eval_epochs(epochs=60, eval_every=5) == set(range(5, 61, 5))


def test_final_epoch_always_included_even_when_not_a_multiple() -> None:
    # 7 epochs, eval_every=3 -> multiples {3, 6} plus the final epoch 7 (not a multiple)
    assert tr.eval_epochs(epochs=7, eval_every=3) == {3, 6, 7}


def test_eval_every_larger_than_epochs_still_evaluates_final_epoch() -> None:
    assert tr.eval_epochs(epochs=3, eval_every=10) == {3}


def test_eval_epochs_rejects_non_positive_inputs() -> None:
    with pytest.raises(AssertionError):
        tr.eval_epochs(epochs=0, eval_every=1)
    with pytest.raises(AssertionError):
        tr.eval_epochs(epochs=5, eval_every=0)
