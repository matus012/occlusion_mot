#!/bin/bash
# Offline runtime environment for the PERUN sweep (D47).
#
# SOURCED, never executed: by scripts/hpc_bootstrap.sh and by every emitted
# submit*.sbatch. One place decides where the interpreter and the pre-staged runtime
# assets live, so no sbatch template has to repeat it.
#
# Compute nodes are assumed to have NO internet. Everything here points a library at
# a bundled asset it would otherwise fetch on first use:
#   TORCH_HOME       torchvision resnet18 ImageNet weights (every train_reid.py trunk
#                    init and the whole of arm A) -> assets/torch/hub/checkpoints/
#   YOLO_CONFIG_DIR  ultralytics Arial.ttf + its writable settings.json
#   MPLCONFIGDIR     matplotlib font cache; unset, matplotlib writes into $HOME and
#                    warns (or stalls) when $HOME is a slow parallel filesystem
#   HF_HUB_OFFLINE   huggingface_hub is an ultralytics dependency; make the failure
#                    mode "raises immediately" rather than "hangs on a DNS timeout"

_HPC_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="$(cd "${_HPC_ENV_DIR}/.." && pwd)"

# The bootstrap venv. Override by exporting PYTHON before submitting.
export PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"

export TORCH_HOME="${REPO_ROOT}/assets/torch"
export YOLO_CONFIG_DIR="${REPO_ROOT}/assets/ultralytics"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${REPO_ROOT}/.cache/matplotlib}"
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

# Compile caches default under $HOME. Keep them in-tree so a quota'd or read-only
# home cannot fail a unit for a reason that has nothing to do with the sweep.
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${REPO_ROOT}/.cache/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${REPO_ROOT}/.cache/inductor}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${REPO_ROOT}/.cache/torch_extensions}"
export ULTRALYTICS_OFFLINE=1

# One GPU per task: keep BLAS/OMP from oversubscribing the cores SLURM granted us.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"

mkdir -p "${MPLCONFIGDIR}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"

# Fail loudly here rather than 20 minutes into a unit that silently re-downloads.
if [ ! -f "${TORCH_HOME}/hub/checkpoints/resnet18-f37072fd.pth" ]; then
  echo "WARNING: ImageNet resnet18 not staged at ${TORCH_HOME}/hub/checkpoints/ --" >&2
  echo "         every embedder unit will try to reach download.pytorch.org." >&2
fi
