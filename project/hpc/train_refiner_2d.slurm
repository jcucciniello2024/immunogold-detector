#!/bin/bash
#SBATCH --job-name=gold_refiner2d
#SBATCH --output=logs/gold_refiner2d_%j.out
#SBATCH --error=logs/gold_refiner2d_%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/beegfs/home/asahai2024/max-planck-project/project}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_DIR/data/Max Planck Data/Gold Particle Labelling/analyzed synapses}"

EPOCHS="${EPOCHS:-25}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-2e-4}"
PATCH_SIZE="${PATCH_SIZE:-33}"
TRAIN_SAMPLES_PER_EPOCH="${TRAIN_SAMPLES_PER_EPOCH:-20000}"
VAL_SAMPLES_PER_EPOCH="${VAL_SAMPLES_PER_EPOCH:-4000}"
POS_FRACTION="${POS_FRACTION:-0.5}"
NEG_MIN_DIST="${NEG_MIN_DIST:-8.0}"
SAVE_PATH="${SAVE_PATH:-checkpoints/refiner_${SLURM_JOB_ID}.pt}"

cd "$PROJECT_DIR"
mkdir -p logs checkpoints

if command -v module >/dev/null 2>&1; then
  module purge || true
  module load python3/3.9.16 || true
  module load cuda/11.8 || true
fi

BASE_TMP="${SLURM_TMPDIR:-/tmp}"
VENV_DIR="${BASE_TMP}/venv_refiner2d_${SLURM_JOB_ID}"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade "pip<24"
python -m pip install imagecodecs numpy tifffile matplotlib scipy
python -m pip install torch --index-url https://download.pytorch.org/whl/cu118

python -u train_refiner.py \
  --data_root "$DATA_ROOT" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --patch_size "$PATCH_SIZE" \
  --train_samples_per_epoch "$TRAIN_SAMPLES_PER_EPOCH" \
  --val_samples_per_epoch "$VAL_SAMPLES_PER_EPOCH" \
  --pos_fraction "$POS_FRACTION" \
  --neg_min_dist "$NEG_MIN_DIST" \
  --save_path "$SAVE_PATH"

