#!/bin/bash
#SBATCH --job-name=gold_logcnn2d
#SBATCH --output=logs/gold_logcnn2d_%j.out
#SBATCH --error=logs/gold_logcnn2d_%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/beegfs/home/asahai2024/max-planck-project/project}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_DIR/data/Max Planck Data/Gold Particle Labelling/analyzed synapses}"

EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-2e-4}"
PATCH_SIZE="${PATCH_SIZE:-33}"
SIGMAS="${SIGMAS:-1.2,1.6,2.0,2.4,2.8}"
LOG_THRESHOLD="${LOG_THRESHOLD:-0.015}"
MIN_DISTANCE="${MIN_DISTANCE:-5}"
MAX_CANDIDATES_PER_IMAGE="${MAX_CANDIDATES_PER_IMAGE:-800}"
MATCH_DIST="${MATCH_DIST:-5.0}"
SAVE_PATH="${SAVE_PATH:-checkpoints/logcnn_${SLURM_JOB_ID}.pt}"

cd "$PROJECT_DIR"
mkdir -p logs checkpoints

if command -v module >/dev/null 2>&1; then
  module purge || true
  module load python3/3.9.16 || true
  module load cuda/11.8 || true
fi

BASE_TMP="${SLURM_TMPDIR:-/tmp}"
VENV_DIR="${BASE_TMP}/venv_logcnn2d_${SLURM_JOB_ID}"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade "pip<24"
python -m pip install imagecodecs numpy tifffile matplotlib scipy
python -m pip install torch --index-url https://download.pytorch.org/whl/cu118

python -u train_log_cnn.py \
  --data_root "$DATA_ROOT" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --patch_size "$PATCH_SIZE" \
  --sigmas "$SIGMAS" \
  --log_threshold "$LOG_THRESHOLD" \
  --min_distance "$MIN_DISTANCE" \
  --max_candidates_per_image "$MAX_CANDIDATES_PER_IMAGE" \
  --match_dist "$MATCH_DIST" \
  --save_path "$SAVE_PATH"
