#!/bin/bash
#SBATCH --job-name=gold_v2_fixed
#SBATCH --output=logs/gold_v2_%j.out
#SBATCH --error=logs/gold_v2_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/beegfs/home/asahai2024/max-planck-project/project}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_DIR/data/Max Planck Data/Gold Particle Labelling/analyzed synapses}"

# ============================================================================
# V2 FIXED PIPELINE — All critical bugs resolved
# ============================================================================
#
# FIXES APPLIED:
#   1. Model: unet_deep (was using shallow unet at inference)
#   2. Sigma: 1.5 (was 2.5 — too wide for ~1px particles)
#   3. pos_weight: 300 (was 30 — far too low for 99.9% background)
#   4. Epochs: 150 (was 20 — too few to converge)
#   5. LR: 3e-4 (was 1e-3 — too high for small dataset)
#   6. Blur: reduced sigma to 0.3-0.8, prob to 15% (was erasing particles)
#   7. Flips/rot90: raised to 50% (was 10% — EM has no orientation)
#   8. Cutout: no longer zeros heatmap (was teaching wrong signal)
#   9. Mantis local contrast: new preprocessing
#  10. Inference: now loads correct model architecture
# ============================================================================

MODEL_TYPE="unet_deep"
BASE_CHANNELS=32
PATCH_SIZE=256
PATCH_STRIDE=128
EPOCHS=150
BATCH_SIZE=8
LR=3e-4
SIGMA=1.5
LOSS_POS_WEIGHT=300.0
SCHED="cosine"
WARMUP_EPOCHS=10
EARLY_STOP_PATIENCE=15
TRAIN_SAMPLES=2048
VAL_SAMPLES=256
SEED=42

SAVE_DIR="checkpoints/v2_fixed_${SLURM_JOB_ID}"

echo "================================================================"
echo "  IMMUNOGOLD DETECTOR V2 — FIXED PIPELINE"
echo "================================================================"
echo ""
echo "  Critical fixes:"
echo "    sigma:       2.5 -> $SIGMA (match particle scale)"
echo "    pos_weight:  30  -> $LOSS_POS_WEIGHT (fix class imbalance)"
echo "    epochs:      20  -> $EPOCHS (enough to converge)"
echo "    lr:          1e-3 -> $LR (stable for small data)"
echo "    blur:        sigma 0.5-2.0 -> 0.3-0.8 (stop erasing particles)"
echo "    flips/rot90: 10% -> 50% (physically justified)"
echo "    model:       unet -> $MODEL_TYPE (deeper + BatchNorm)"
echo "    mantis:      NEW local contrast preprocessing"
echo ""

cd "$PROJECT_DIR"
mkdir -p logs checkpoints "$SAVE_DIR"

# Environment setup
if command -v module >/dev/null 2>&1; then
  module purge || true
  module load python3/3.9.16 || true
  module load cuda/11.8 || true
fi

BASE_TMP="${SLURM_TMPDIR:-/tmp}"
VENV_DIR="${BASE_TMP}/venv_v2_${SLURM_JOB_ID}"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade "pip<24"
python -m pip install imagecodecs numpy tifffile matplotlib scipy
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# ============================================================================
# TRAIN
# ============================================================================
python -u train_detector.py \
  --data_root "$DATA_ROOT" \
  --model_type "$MODEL_TYPE" \
  --base_channels "$BASE_CHANNELS" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --patch_h "$PATCH_SIZE" \
  --patch_w "$PATCH_SIZE" \
  --train_samples_per_epoch "$TRAIN_SAMPLES" \
  --val_samples_per_epoch "$VAL_SAMPLES" \
  --sigma "$SIGMA" \
  --target_type gaussian \
  --loss_type focal_bce \
  --loss_pos_weight "$LOSS_POS_WEIGHT" \
  --focal_gamma 2.0 \
  --weight_decay 1e-4 \
  --grad_clip 1.0 \
  --sched "$SCHED" \
  --warmup_epochs "$WARMUP_EPOCHS" \
  --early_stop_patience "$EARLY_STOP_PATIENCE" \
  --seed "$SEED" \
  --save_dir "$SAVE_DIR" \
  --use_sliding_window \
  --patch_stride "$PATCH_STRIDE" \
  --use_mantis \
  --sigma_jitter

echo ""
echo "Training complete. Best model: $SAVE_DIR/detector_best.pt"
