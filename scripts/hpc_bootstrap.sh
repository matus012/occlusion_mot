#!/bin/bash
# Create the sweep venv from the bundled wheelhouse -- no network required (D47).
#
#   bash scripts/hpc_bootstrap.sh [--wheelhouse DIR] [--python PYTHON3.11]
#
# Default wheelhouse is ../wheelhouse relative to the repo root, which is exactly
# where `make_hpc_bundle.py unpack` puts it (beside repo/, never inside it).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

WHEELHOUSE="${REPO_ROOT}/../wheelhouse"
BASE_PYTHON=""
while [ $# -gt 0 ]; do
  case "$1" in
    --wheelhouse) WHEELHOUSE="$2"; shift 2 ;;
    --python)     BASE_PYTHON="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

echo "=== 1/6 glibc floor ==="
# torch 2.13 publishes no wheel below manylinux_2_28. A node under glibc 2.28 cannot
# run this stack at all -- say so now, not after a clean install fails at import.
GLIBC="$(ldd --version | head -1 | grep -oE '[0-9]+\.[0-9]+$')"
if [ "$(printf '%s\n2.28\n' "${GLIBC}" | sort -V | head -1)" != "2.28" ]; then
  echo "ERROR: glibc ${GLIBC} < 2.28; the bundled manylinux_2_28 wheels cannot run here." >&2
  echo "       Load a newer toolchain module, or rebuild the wheelhouse for this node." >&2
  exit 1
fi
echo "glibc ${GLIBC} OK"

echo "=== 2/6 interpreter ==="
if [ -z "${BASE_PYTHON}" ]; then
  for cand in python3.11 python3; do
    if command -v "${cand}" >/dev/null 2>&1; then BASE_PYTHON="${cand}"; break; fi
  done
fi
[ -n "${BASE_PYTHON}" ] || { echo "ERROR: no python3 on PATH (module load Python/3.11?)" >&2; exit 1; }
PYVER="$("${BASE_PYTHON}" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
if [ "${PYVER}" != "3.11" ]; then
  echo "ERROR: ${BASE_PYTHON} is Python ${PYVER}; the wheelhouse is cp311-only." >&2
  echo "       Point --python at a 3.11 interpreter (e.g. module load Python/3.11.x)." >&2
  exit 1
fi
echo "${BASE_PYTHON} -> Python ${PYVER} OK"

echo "=== 3/6 wheelhouse ==="
[ -d "${WHEELHOUSE}" ] || { echo "ERROR: no wheelhouse at ${WHEELHOUSE}" >&2; exit 1; }
N_WHEELS="$(find "${WHEELHOUSE}" -name '*.whl' | wc -l)"
[ "${N_WHEELS}" -gt 0 ] || { echo "ERROR: ${WHEELHOUSE} holds no .whl files" >&2; exit 1; }
echo "${N_WHEELS} wheels at ${WHEELHOUSE}"

echo "=== 4/6 venv ==="
if [ ! -x "${REPO_ROOT}/.venv/bin/python" ]; then
  "${BASE_PYTHON}" -m venv "${REPO_ROOT}/.venv"
fi
VENV_PY="${REPO_ROOT}/.venv/bin/python"
"${VENV_PY}" -m pip install --quiet --no-index --find-links "${WHEELHOUSE}" \
  --upgrade pip setuptools wheel 2>/dev/null || \
  echo "(pip/setuptools/wheel not in the wheelhouse -- using the venv's bundled pip)"

echo "=== 5/6 install (offline: --no-index) ==="
"${VENV_PY}" -m pip install --no-index --find-links "${WHEELHOUSE}" -r requirements-lock.txt

# opencv: exactly one cv2, and it must be the headless build.
#
# ultralytics and supervision both depend on `opencv-python`, so pinning
# `opencv-python-headless` does NOT replace it -- the lock legitimately contains both,
# they install into the SAME cv2/ directory, and whichever lands last wins. The
# non-headless build dlopen()s libGL.so.1 at `import cv2`, which compute nodes
# routinely lack. Resolve it deterministically instead of leaving it to install order.
if "${VENV_PY}" -m pip show opencv-python >/dev/null 2>&1; then
  echo "removing opencv-python in favour of the headless build (no libGL on compute nodes)"
  "${VENV_PY}" -m pip uninstall -y opencv-python
  "${VENV_PY}" -m pip install --no-index --find-links "${WHEELHOUSE}" --no-deps \
    --force-reinstall opencv-python-headless
fi

echo "=== 6/6 self-test ==="
# shellcheck source=scripts/hpc_env.sh
source "${REPO_ROOT}/scripts/hpc_env.sh"
"${PYTHON}" - <<'PY'
import os
import sys
from pathlib import Path

from importlib.metadata import distributions

import cv2
import numpy
import torch
import torchvision
import ultralytics
import yaml  # noqa: F401

installed = {d.metadata["Name"].lower() for d in distributions()}
assert "opencv-python" not in installed, (
    "non-headless opencv-python is installed; it dlopen()s libGL.so.1 at import "
    "and will fail on a compute node"
)

print(f"python      {sys.version.split()[0]}")
print(f"torch       {torch.__version__}  (cuda build {torch.version.cuda})")
print(f"torchvision {torchvision.__version__}")
print(f"numpy       {numpy.__version__}")
print(f"opencv      {cv2.__version__}")
print(f"ultralytics {ultralytics.__version__}")

root = Path(os.environ["REPO_ROOT"])
missing = [str(p) for p in [
    root / "assets/torch/hub/checkpoints/resnet18-f37072fd.pth",
    root / "assets/ultralytics/Arial.ttf",
    root / "data/reid/mot17_dev/index.json",
    root / "results/occlusion_segments.json",
] if not p.exists()]
assert not missing, f"staged input(s) missing after unpack: {missing}"
print("staged inputs OK")

# Login nodes have no GPU; this is informational here. The smoke JOB asserts it.
print(f"cuda available (this host): {torch.cuda.is_available()}")
PY

echo
echo "BOOTSTRAP OK -> ${REPO_ROOT}/.venv"
echo "Next: fill slurm.partition + slurm.account in configs/sweep/perun_full.yaml"
