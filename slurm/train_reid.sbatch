#!/bin/bash
#SBATCH --job-name=omot-reid
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm_%x_%j.out

# Full re-ID embedding training on TUKE PERUN (D8/D26 queue item 1).
# Prereqs on the cluster (one-time): clone the repo, rsync data/reid/ from the
# workstation, `uv venv --python 3.11 .venv && uv pip install -r requirements.txt -e .`
# (CPU/GPU wheels resolve per cluster CUDA; torch cu126 index works for H200).
#
# Three ablation arms (submit one per --sources choice; D24):
#   sbatch slurm/train_reid.sbatch mot17_dev
#   sbatch slurm/train_reid.sbatch sim
#   sbatch slurm/train_reid.sbatch mot17_dev sim

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
SOURCES=("$@")
if [ ${#SOURCES[@]} -eq 0 ]; then SOURCES=(mot17_dev sim); fi
TAG=$(IFS=- ; echo "${SOURCES[*]}")

.venv/bin/python scripts/train_reid.py \
  --sources "${SOURCES[@]}" \
  --epochs 40 \
  --batches-per-epoch 400 \
  --batch-p 24 --batch-k 4 \
  --tag "perun_${TAG}" \
  --device cuda \
  --seed 0

echo "checkpoint: data/models/reid_perun_${TAG}.pt — rsync back and run:"
echo "  cache_embeddings.py --weights data/models/reid_perun_${TAG}.pt --tag perun_${TAG}"
echo "  tune_hidden.py (EMBEDDER_TAG=perun_${TAG}) on dev"
