# Third-party notices

This project (`omot`) is licensed **AGPL-3.0-only** (see `LICENSE`, `pyproject.toml`,
and `context.md` D9). That choice is not incidental: the direct dependency
**ultralytics** (used for YOLO detector inference and fine-tuning) is itself
**AGPL-3.0**, and AGPL is a strong-copyleft license whose obligations propagate to
projects that depend on it. Licensing the whole repository AGPL-3.0-only, rather
than trying to isolate the detector-caching path, was the deliberate decision
recorded in `context.md` D9.

Licences below were read from installed package metadata in this project's `.venv`
(`importlib.metadata`, `License` field and `Classifier` trove entries) at
2026-09-26. Where metadata carried no machine-readable licence, the package is
marked "see upstream" rather than guessed.

## Python dependencies

| Package | Licence | Notes |
|---|---|---|
| ultralytics | **AGPL-3.0** | Drives this repo's own licence (see above). |
| ultralytics-thop | AGPL-3.0 | Ultralytics sub-dependency. |
| trackeval | MIT | Pinned git commit (`requirements.txt`); eval metrics only. |
| torch | see upstream | PyTorch — BSD-3-Clause upstream. |
| torchvision | BSD | |
| numpy | see upstream | BSD-3-Clause upstream. |
| scipy | see upstream | BSD-3-Clause upstream. |
| pandas | see upstream | BSD-3-Clause upstream. |
| polars | see upstream | MIT upstream. |
| matplotlib | see upstream | PSF-based / BSD-style upstream (matplotlib licence). |
| pillow | see upstream | MIT-CMU (HPND) upstream. |
| opencv-python | Apache 2.0 | |
| supervision | see upstream | MIT upstream. |
| lapx | MIT | |
| motmetrics | MIT | |
| pycocotools | FreeBSD | |
| pydeprecate | Apache-2.0 | |
| huggingface-hub | Apache-2.0 | |
| hf-transfer | see upstream | Apache-2.0 upstream. |
| hf-xet | see upstream | Apache Software License. |
| requests | Apache-2.0 | |
| urllib3 | see upstream | MIT upstream. |
| httpx | BSD-3-Clause | |
| httpcore | see upstream | BSD License. |
| h11 | MIT | |
| anyio | see upstream | MIT upstream. |
| idna | see upstream | BSD-3-Clause upstream. |
| certifi | MPL-2.0 | |
| charset-normalizer | MIT | |
| click | see upstream | BSD-3-Clause upstream. |
| colorama | see upstream | BSD License. |
| contourpy | BSD-3-Clause | |
| coverage | Apache-2.0 | |
| cycler | BSD-3-Clause | |
| defusedxml | PSFL | |
| filelock | see upstream | MIT License. |
| fonttools | MIT | |
| fsspec | see upstream | BSD-3-Clause upstream. |
| imageio-ffmpeg | BSD-2-Clause | |
| iniconfig | see upstream | MIT upstream. |
| jinja2 | see upstream | BSD License. |
| kiwisolver | BSD-3-Clause | |
| markupsafe | see upstream | BSD-3-Clause upstream. |
| mpmath | BSD | |
| networkx | see upstream | BSD-3-Clause upstream. |
| nvidia-ml-py | BSD | |
| packaging | see upstream | Apache-2.0 OR BSD-2-Clause upstream (dual). |
| pluggy | MIT | |
| psutil | BSD-3-Clause | |
| pygments | see upstream | BSD-2-Clause upstream. |
| pyparsing | see upstream | MIT upstream. |
| pytest | see upstream | MIT upstream. |
| pytest-cov | see upstream | MIT License. |
| pytest-timeout | MIT | |
| python-dateutil | Dual (BSD / Apache-2.0) | |
| pyyaml | MIT | |
| ruff | see upstream | MIT upstream. |
| setuptools | see upstream | MIT License. |
| six | MIT | |
| sympy | BSD | |
| tqdm | MPL-2.0 AND MIT | |
| typing-extensions | see upstream | PSF-2.0 upstream. |
| tzdata | Apache-2.0 | |
| xmltodict | see upstream | MIT upstream. |

Full pinned versions: `requirements.txt` (dev box, cu126) and
`requirements-lock.txt` (PERUN/HPC, linux cu126, generated from
`requirements-hpc.in` by `scripts/make_hpc_bundle.py --refresh-lock`).

## Datasets

| Dataset | Licence | How used here |
|---|---|---|
| MOT17 | Non-commercial research licence (motchallenge.net) | Evaluation (train half-split dev/val protocol) and detector fine-tuning experiments (`scripts/finetune_detector.py`, MOT17 dev-half). **No weights trained on MOT17 are released** — see `context.md` D37/D41 and `tests/test_license_guard.py`; trained checkpoints stay local under gitignored `data/models/` and are never committed or attached to a release. |
| MOT20 | Non-commercial research licence (motchallenge.net) | Detector fine-tuning experiments (`scripts/detector_unit.py`) and re-ID crop extraction (`scripts/extract_reid_mot20.py`). Same no-redistribution rule as MOT17. |
| CrowdHuman | Non-commercial research licence | Referenced as an auxiliary detector-training source alongside MOT17/MOT20 in the disjoint-training detector sweep (`perun_detector_v1.md`). No weights trained on it are released. |

No dataset-derived pixels (crops, frames, clips) or dataset-trained model weights are
tracked in this repository or attached to any GitHub release; this is enforced by
`tests/test_license_guard.py` (D37/D41) and independently verified for this document
by inspecting `git ls-files`, `.gitignore`, and `gh release list` — see `context.md`
for the dated entry.

## Simulator

| Component | Licence | Notes |
|---|---|---|
| CARLA simulator (0.9.15) | Code: MIT. Assets/content: CC-BY | Used as the P2 occlusion-scenario feeder (`sim_driver.py` and the CARLA render pipeline). CARLA itself, and its rendered output, are **not redistributed**: `data/sim/` and `showcase/renders/` are gitignored; only our own loaders, manifests, and metric JSONs derived from CARLA runs are tracked. |

## Compute

| Resource | Notes |
|---|---|
| TUKE PERUN (208x NVIDIA H200 cluster) | University HPC cluster used for the detector-scale and re-ID sweeps under a 40 H200-h budget (18.46 h spent, ceiling never moved). Acknowledged as the compute provider; no PERUN code or data is redistributed here beyond our own SLURM scripts (`slurm/`) and the reproducible bundle builder (`scripts/make_hpc_bundle.py`). |

## Runtime-downloaded model checkpoints

Stock Ultralytics YOLO checkpoints (`yolo11x.pt`, `yolo11m.pt`, `yolo11s.pt`,
`yolo26n.pt`) are downloaded at runtime by the `ultralytics` package (or fetched
once and kept local) and are **not committed to this repository or attached to any
release** — they are excluded by `.gitignore` (`*.pt`) and confirmed absent from
`git ls-files` and from `gh release list -R matus012/occlusion_mot`.
