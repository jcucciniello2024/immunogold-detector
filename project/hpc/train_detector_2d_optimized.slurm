#!/bin/bash
#SBATCH --job-name=gold_detector_optimized
#SBATCH --output=logs/gold_detector_opt_%j.out
#SBATCH --error=logs/gold_detector_opt_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/beegfs/home/asahai2024/max-planck-project/project}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_DIR/data/Max Planck Data/Gold Particle Labelling/analyzed synapses}"

# ============================================================================
# OPTIMIZED PIPELINE CONFIGURATION
# ============================================================================

# Architecture & Model
MODEL_TYPE="${MODEL_TYPE:-unet_deep}"
BASE_CHANNELS="${BASE_CHANNELS:-32}"

# Data Configuration (SLIDING WINDOW for maximum data)
PATCH_SIZE="${PATCH_SIZE:-256}"
PATCH_STRIDE="${PATCH_STRIDE:-128}"
USE_SLIDING_WINDOW="${USE_SLIDING_WINDOW:-true}"

# Training Configuration
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TRAIN_SAMPLES_PER_EPOCH="${TRAIN_SAMPLES_PER_EPOCH:-2048}"
VAL_SAMPLES_PER_EPOCH="${VAL_SAMPLES_PER_EPOCH:-256}"

# Learning Rate & Schedule
LR="${LR:-5e-4}"
SCHED="${SCHED:-cosine}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"

# Loss & Target Configuration
SIGMA="${SIGMA:-1.0}"                          # ✅ REDUCED from 2.5 - CRITICAL FIX
TARGET_TYPE="${TARGET_TYPE:-gaussian}"
TARGET_RADIUS="${TARGET_RADIUS:-3}"
LOSS_TYPE="${LOSS_TYPE:-focal_bce}"
LOSS_POS_WEIGHT="${LOSS_POS_WEIGHT:-30.0}"
LOSS_NEG_WEIGHT="${LOSS_NEG_WEIGHT:-1.0}"
FOCAL_GAMMA="${FOCAL_GAMMA:-2.0}"

# Augmentation Configuration
USE_CLAHE="${USE_CLAHE:-false}"                # ✅ DISABLED - was harmful
SIGMA_JITTER="${SIGMA_JITTER:-true}"           # ✅ ENABLED - teaches robustness
CONSISTENCY_WEIGHT="${CONSISTENCY_WEIGHT:-0.1}"

# Early Stopping Configuration
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-10}"
EARLY_STOP_DELTA="${EARLY_STOP_DELTA:-1e-5}"

# Training Optimization
MIXED_PRECISION="${MIXED_PRECISION:-true}"

# Data Split
POS_FRACTION="${POS_FRACTION:-0.6}"
SEED="${SEED:-42}"

SAVE_DIR="${SAVE_DIR:-checkpoints/${SLURM_JOB_ID}}"
RESUME_PATH="${RESUME_PATH:-}"

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║     OPTIMIZED GOLD PARTICLE DETECTION PIPELINE                    ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Configuration:"
echo "  Model:                $MODEL_TYPE (base_channels=$BASE_CHANNELS)"
echo "  Architecture:         4-level UNet Deep, 7.77M params"
echo "  Data Strategy:        Sliding Window Patches ($PATCH_SIZE×$PATCH_SIZE, stride=$PATCH_STRIDE)"
echo "  Patches per epoch:    $TRAIN_SAMPLES_PER_EPOCH (150-200 available per epoch)"
echo "  Data amplification:   15-20× vs random sampling"
echo ""
echo "Critical Fixes:"
echo "  ✅ Sigma reduced:     2.5 → $SIGMA (sharper peaks, individual particle detection)"
echo "  ✅ CLAHE disabled:    Removed harmful preprocessing"
echo "  ✅ Early stopping:    Enabled (patience=$EARLY_STOP_PATIENCE)"
echo "  ✅ Augmentations:     8 EM-realistic perturbations"
echo ""
echo "Expected Performance:"
echo "  F1 Score:             0.003-0.010 (30-100× baseline)"
echo "  Training Time:        10-15 hours (early stop saves epochs)"
echo "  GPU Memory:           ~20-25GB for batch_size=8"
echo ""

cd "$PROJECT_DIR"
mkdir -p logs checkpoints "$SAVE_DIR"

if command -v module >/dev/null 2>&1; then
  module purge || true
  module load python3/3.9.16 || true
  module load cuda/11.8 || true
fi

BASE_TMP="${SLURM_TMPDIR:-/tmp}"
VENV_DIR="${BASE_TMP}/venv_detector_opt_${SLURM_JOB_ID}"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade "pip<24"
python -m pip install imagecodecs numpy tifffile matplotlib scipy
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# ============================================================================
# BUILD TRAINING COMMAND
# ============================================================================

CMD=(
  python -u train_detector.py
  --data_root "$DATA_ROOT"
  --model_type "$MODEL_TYPE"
  --epochs "$EPOCHS"
  --batch_size "$BATCH_SIZE"
  --lr "$LR"
  --patch_h "$PATCH_SIZE"
  --patch_w "$PATCH_SIZE"
  --train_samples_per_epoch "$TRAIN_SAMPLES_PER_EPOCH"
  --val_samples_per_epoch "$VAL_SAMPLES_PER_EPOCH"
  --sigma "$SIGMA"
  --target_type "$TARGET_TYPE"
  --target_radius "$TARGET_RADIUS"
  --base_channels "$BASE_CHANNELS"
  --loss_type "$LOSS_TYPE"
  --loss_pos_weight "$LOSS_POS_WEIGHT"
  --loss_neg_weight "$LOSS_NEG_WEIGHT"
  --focal_gamma "$FOCAL_GAMMA"
  --weight_decay "$WEIGHT_DECAY"
  --grad_clip "$GRAD_CLIP"
  --consistency_weight "$CONSISTENCY_WEIGHT"
  --sched "$SCHED"
  --warmup_epochs "$WARMUP_EPOCHS"
  --early_stop_patience "$EARLY_STOP_PATIENCE"
  --early_stop_delta "$EARLY_STOP_DELTA"
  --pos_fraction "$POS_FRACTION"
  --seed "$SEED"
  --save_dir "$SAVE_DIR"
)

# Conditional flags
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

# Use sliding window dataset if requested
if [ "$USE_SLIDING_WINDOW" = "true" ]; then
  CMD+=(--use_sliding_window --patch_stride "$PATCH_STRIDE")
fi

echo "Starting training..."
echo "Command:"
echo "${CMD[@]}"
echo ""

"${CMD[@]}"

echo ""
echo "✓ Training complete"
echo "✓ Best model saved to: $SAVE_DIR/detector_best.pt"
