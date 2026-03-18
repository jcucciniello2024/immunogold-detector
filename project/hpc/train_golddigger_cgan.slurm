#!/bin/bash
#SBATCH --job-name=gold_cgan
#SBATCH --output=logs/gold_cgan_%j.out
#SBATCH --error=logs/gold_cgan_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/beegfs/home/asahai2024/max-planck-project/project}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_DIR/data/Max Planck Data/Gold Particle Labelling/analyzed synapses}"

EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-2e-4}"
PATCH_SIZE="${PATCH_SIZE:-256}"
TRAIN_SAMPLES_PER_EPOCH="${TRAIN_SAMPLES_PER_EPOCH:-2048}"
VAL_SAMPLES_PER_EPOCH="${VAL_SAMPLES_PER_EPOCH:-512}"
POS_FRACTION="${POS_FRACTION:-0.7}"
RADIUS_6NM="${RADIUS_6NM:-3}"
RADIUS_12NM="${RADIUS_12NM:-5}"
LAMBDA_L1="${LAMBDA_L1:-50.0}"
SAVE_DIR="${SAVE_DIR:-checkpoints/golddigger_cgan_${SLURM_JOB_ID}}"

cd "$PROJECT_DIR"
mkdir -p logs checkpoints "$SAVE_DIR"

if command -v module >/dev/null 2>&1; then
  module purge || true
  module load python3/3.9.16 || true
  module load cuda/11.8 || true
fi

BASE_TMP="${SLURM_TMPDIR:-/tmp}"
VENV_DIR="${BASE_TMP}/venv_golddigger_cgan_${SLURM_JOB_ID}"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade "pip<24"
python -m pip install imagecodecs numpy tifffile matplotlib scipy
python -m pip install torch --index-url https://download.pytorch.org/whl/cu118

python -u train_golddigger_cgan.py \
  --data_root "$DATA_ROOT" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --patch_size "$PATCH_SIZE" \
  --train_samples_per_epoch "$TRAIN_SAMPLES_PER_EPOCH" \
  --val_samples_per_epoch "$VAL_SAMPLES_PER_EPOCH" \
  --pos_fraction "$POS_FRACTION" \
  --radius_6nm "$RADIUS_6NM" \
  --radius_12nm "$RADIUS_12NM" \
  --lambda_l1 "$LAMBDA_L1" \
  --save_dir "$SAVE_DIR"
