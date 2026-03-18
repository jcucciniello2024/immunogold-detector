#!/bin/bash
#SBATCH --job-name=gold_detector2d_semi
#SBATCH --output=logs/gold_detector2d_semi_%j.out
#SBATCH --error=logs/gold_detector2d_semi_%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/beegfs/home/asahai2024/max-planck-project/project}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_DIR/data/Max Planck Data}"
UNLABELED_DIR="${UNLABELED_DIR:-$PROJECT_DIR/data/Max Planck Data/Gold Particle Labelling/labeled replica - Test Data from Synapses}"
TEACHER_CKPT="${TEACHER_CKPT:-$PROJECT_DIR/checkpoints/4567335/detector_best.pt}"

EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-2e-4}"
PATCH_H="${PATCH_H:-384}"
PATCH_W="${PATCH_W:-384}"
TRAIN_SAMPLES_PER_EPOCH="${TRAIN_SAMPLES_PER_EPOCH:-4096}"
VAL_SAMPLES_PER_EPOCH="${VAL_SAMPLES_PER_EPOCH:-768}"
SIGMA="${SIGMA:-2.0}"
POS_FRACTION="${POS_FRACTION:-0.8}"
BASE_CHANNELS="${BASE_CHANNELS:-24}"
PSEUDO_THRESHOLD="${PSEUDO_THRESHOLD:-0.24}"
PSEUDO_MIN_DISTANCE="${PSEUDO_MIN_DISTANCE:-5}"
PSEUDO_MIN_SUPPORT="${PSEUDO_MIN_SUPPORT:-2}"
PSEUDO_MERGE_DIST="${PSEUDO_MERGE_DIST:-4.0}"
MAX_PSEUDO_PER_CLASS="${MAX_PSEUDO_PER_CLASS:-80}"
SAVE_DIR="${SAVE_DIR:-checkpoints/semi_${SLURM_JOB_ID}}"

cd "$PROJECT_DIR"
mkdir -p logs checkpoints "$SAVE_DIR"

if command -v module >/dev/null 2>&1; then
  module purge || true
  module load python3/3.9.16 || true
  module load cuda/11.8 || true
fi

BASE_TMP="${SLURM_TMPDIR:-/tmp}"
VENV_DIR="${BASE_TMP}/venv_detector2d_semi_${SLURM_JOB_ID}"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade "pip<24"
python -m pip install imagecodecs numpy tifffile matplotlib scipy
python -m pip install torch --index-url https://download.pytorch.org/whl/cu118

python -u train_detector_semi.py \
  --data_root "$DATA_ROOT" \
  --unlabeled_dir "$UNLABELED_DIR" \
  --teacher_ckpt "$TEACHER_CKPT" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --patch_h "$PATCH_H" \
  --patch_w "$PATCH_W" \
  --train_samples_per_epoch "$TRAIN_SAMPLES_PER_EPOCH" \
  --val_samples_per_epoch "$VAL_SAMPLES_PER_EPOCH" \
  --sigma "$SIGMA" \
  --pos_fraction "$POS_FRACTION" \
  --base_channels "$BASE_CHANNELS" \
  --pseudo_threshold "$PSEUDO_THRESHOLD" \
  --pseudo_min_distance "$PSEUDO_MIN_DISTANCE" \
  --pseudo_min_support "$PSEUDO_MIN_SUPPORT" \
  --pseudo_merge_dist "$PSEUDO_MERGE_DIST" \
  --max_pseudo_per_class "$MAX_PSEUDO_PER_CLASS" \
  --save_dir "$SAVE_DIR"

