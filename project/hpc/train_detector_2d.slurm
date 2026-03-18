#!/bin/bash
#SBATCH --job-name=gold_detector2d
#SBATCH --output=logs/gold_detector2d_%j.out
#SBATCH --error=logs/gold_detector2d_%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/beegfs/home/asahai2024/max-planck-project/project}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_DIR/data/Max Planck Data/Gold Particle Labelling/analyzed synapses}"

EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-5e-4}"
PATCH_H="${PATCH_H:-512}"
PATCH_W="${PATCH_W:-512}"
TRAIN_SAMPLES_PER_EPOCH="${TRAIN_SAMPLES_PER_EPOCH:-1024}"
VAL_SAMPLES_PER_EPOCH="${VAL_SAMPLES_PER_EPOCH:-128}"
POS_FRACTION="${POS_FRACTION:-0.6}"
SIGMA="${SIGMA:-2.0}"
TARGET_TYPE="${TARGET_TYPE:-gaussian}"
TARGET_RADIUS="${TARGET_RADIUS:-3}"
BASE_CHANNELS="${BASE_CHANNELS:-32}"
LOSS_TYPE="${LOSS_TYPE:-focal_bce}"
LOSS_POS_WEIGHT="${LOSS_POS_WEIGHT:-30.0}"
LOSS_NEG_WEIGHT="${LOSS_NEG_WEIGHT:-1.0}"
LOSS_POS_POWER="${LOSS_POS_POWER:-1.0}"
FOCAL_GAMMA="${FOCAL_GAMMA:-2.0}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
# New augmentation and training flags
MODEL_TYPE="${MODEL_TYPE:-unet_deep}"
USE_CLAHE="${USE_CLAHE:-true}"
SIGMA_JITTER="${SIGMA_JITTER:-true}"
CONSISTENCY_WEIGHT="${CONSISTENCY_WEIGHT:-0.1}"
SCHED="${SCHED:-cosine}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
MIXED_PRECISION="${MIXED_PRECISION:-true}"
RESUME_PATH="${RESUME_PATH:-}"
SAVE_DIR="${SAVE_DIR:-checkpoints/${SLURM_JOB_ID}}"

cd "$PROJECT_DIR"
mkdir -p logs checkpoints "$SAVE_DIR"

if command -v module >/dev/null 2>&1; then
  module purge || true
  module load python3/3.9.16 || true
  module load cuda/11.8 || true
fi

BASE_TMP="${SLURM_TMPDIR:-/tmp}"
VENV_DIR="${BASE_TMP}/venv_detector2d_${SLURM_JOB_ID}"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade "pip<24"
python -m pip install imagecodecs numpy tifffile matplotlib scipy
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

CMD=(
  python -u train_detector.py
  --data_root "$DATA_ROOT"
  --model_type "$MODEL_TYPE"
  --epochs "$EPOCHS"
  --batch_size "$BATCH_SIZE"
  --lr "$LR"
  --patch_h "$PATCH_H"
  --patch_w "$PATCH_W"
  --train_samples_per_epoch "$TRAIN_SAMPLES_PER_EPOCH"
  --val_samples_per_epoch "$VAL_SAMPLES_PER_EPOCH"
  --pos_fraction "$POS_FRACTION"
  --sigma "$SIGMA"
  --target_type "$TARGET_TYPE"
  --target_radius "$TARGET_RADIUS"
  --base_channels "$BASE_CHANNELS"
  --loss_type "$LOSS_TYPE"
  --loss_pos_weight "$LOSS_POS_WEIGHT"
  --loss_neg_weight "$LOSS_NEG_WEIGHT"
  --loss_pos_power "$LOSS_POS_POWER"
  --focal_gamma "$FOCAL_GAMMA"
  --weight_decay "$WEIGHT_DECAY"
  --grad_clip "$GRAD_CLIP"
  --consistency_weight "$CONSISTENCY_WEIGHT"
  --sched "$SCHED"
  --warmup_epochs "$WARMUP_EPOCHS"
  --save_dir "$SAVE_DIR"
)

if [ "$USE_CLAHE" = "true" ]; then
  CMD+=(--use_clahe)
fi

if [ "$SIGMA_JITTER" = "true" ]; then
  CMD+=(--sigma_jitter)
fi

if [ "$MIXED_PRECISION" = "true" ]; then
  CMD+=(--mixed_precision)
fi

if [ -n "$RESUME_PATH" ]; then
  CMD+=(--resume "$RESUME_PATH")
fi

"${CMD[@]}"
