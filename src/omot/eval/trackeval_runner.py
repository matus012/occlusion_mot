"""TrackEval wrapper -> {hota, idf1, mota, idsw} JSON (the schema gates.yaml consumes).

Layout expected:
  gt_folder/<seq>/gt/gt.txt  +  gt_folder/<seq>/seqinfo.ini (optional; SEQ_INFO overrides)
  trackers_folder/<tracker_name>/<seq>.txt
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def run_trackeval(
    gt_folder: Path,
    trackers_folder: Path,
    tracker_name: str,
    seq_info: dict[str, int],
    output_json: Path | None = None,
) -> dict[str, float | int]:
    """Evaluate one tracker on the given sequences; returns combined pedestrian metrics."""
    # TrackEval (pinned, unmaintained) still uses np.float/np.int/np.bool, removed in
    # numpy>=1.24 — restore the aliases before use (context.md D5). hasattr on these
    # emits a FutureWarning on numpy 1.26, hence the suppression.
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        for alias, builtin in (("float", float), ("int", int), ("bool", bool)):
            if not hasattr(np, alias):
                setattr(np, alias, builtin)

    import trackeval  # local: heavy import, optional dependency at module level

    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config.update(
        {
            "PRINT_RESULTS": False,
            "PRINT_ONLY_COMBINED": True,
            "PRINT_CONFIG": False,
            "TIME_PROGRESS": False,
            "DISPLAY_LESS_PROGRESS": True,
            "OUTPUT_SUMMARY": False,
            "OUTPUT_EMPTY_CLASSES": False,
            "OUTPUT_DETAILED": False,
            "PLOT_CURVES": False,
            "USE_PARALLEL": False,
            "LOG_ON_ERROR": None,
        }
    )
    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config.update(
        {
            "GT_FOLDER": str(gt_folder),
            "TRACKERS_FOLDER": str(trackers_folder),
            "TRACKERS_TO_EVAL": [tracker_name],
            "TRACKER_SUB_FOLDER": "",
            "OUTPUT_FOLDER": None,
            "SEQ_INFO": {name: int(length) for name, length in seq_info.items()},
            "SKIP_SPLIT_FOL": True,
            "DO_PREPROC": True,
            "PRINT_CONFIG": False,
        }
    )
    dataset = trackeval.datasets.MotChallenge2DBox(dataset_config)
    metrics = [
        trackeval.metrics.HOTA(),
        trackeval.metrics.CLEAR({"PRINT_CONFIG": False}),
        trackeval.metrics.Identity({"PRINT_CONFIG": False}),
    ]
    evaluator = trackeval.Evaluator(eval_config)
    res, _ = evaluator.evaluate([dataset], metrics)
    combined = res["MotChallenge2DBox"][tracker_name]["COMBINED_SEQ"]["pedestrian"]

    out: dict[str, float | int] = {
        "hota": float(np.mean(combined["HOTA"]["HOTA"])) * 100.0,
        "idf1": float(combined["Identity"]["IDF1"]) * 100.0,
        "mota": float(combined["CLEAR"]["MOTA"]) * 100.0,
        "idsw": int(combined["CLEAR"]["IDSW"]),
        "n_seqs": len(seq_info),
    }
    logger.info("trackeval[%s]: %s", tracker_name, out)
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out
