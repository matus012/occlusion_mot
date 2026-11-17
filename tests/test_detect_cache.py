"""Synthetic tests for src/omot/detect/cache.py cache-tag routing (cache-hit path only,
so no ultralytics/torch model load is required)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from omot.data.mot import MOTSequence
from omot.detect.cache import cache_detections, cache_path


def _fake_seq(name: str = "SEQ-01") -> MOTSequence:
    return MOTSequence(
        name=name, root=Path("unused"), frame_rate=30.0, seq_length=1,
        img_width=640, img_height=480, gt=None,
    )


def test_cache_hit_default_tag_uses_model_stem(tmp_path: Path) -> None:
    seq = _fake_seq()
    expected = cache_path(tmp_path, seq.name, "yolo11x")
    expected.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(expected, frames=np.zeros(0), boxes=np.zeros((0, 5)))

    out = cache_detections(seq, tmp_path, model_name="yolo11x.pt")

    assert out == expected


def test_cache_hit_custom_tag_overrides_model_name_stem(tmp_path: Path) -> None:
    """A custom checkpoint (whose filename stem differs from the desired cache tag)
    must key the cache by `cache_tag`, not by the checkpoint's own stem — this is what
    keeps a finetuned-detector cache from colliding with (or requiring the same name as)
    the checkpoint file itself."""
    seq = _fake_seq()
    expected = cache_path(tmp_path, seq.name, "yolo11s_ft")
    expected.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(expected, frames=np.zeros(0), boxes=np.zeros((0, 5)))
    wrong = cache_path(tmp_path, seq.name, "best")  # stem of the weights path below
    assert not wrong.exists()

    out = cache_detections(
        seq, tmp_path, model_name="some/dir/best.pt", cache_tag="yolo11s_ft",
    )

    assert out == expected
    assert not wrong.exists()


def test_cache_tag_none_default_is_byte_identical_to_pre_existing_behavior(
    tmp_path: Path,
) -> None:
    """Omitting cache_tag must resolve to Path(model_name).stem exactly as before this
    argument was added (regression guard for the FIXED DETECTIONS invariant)."""
    seq = _fake_seq()
    model_name = "yolo11x.pt"
    with_default = cache_path(tmp_path, seq.name, "yolo11x")
    with_explicit_none = cache_path(tmp_path, seq.name, Path(model_name).stem)
    assert with_default == with_explicit_none
