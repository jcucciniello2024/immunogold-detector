#!/bin/bash
#SBATCH --job-name=gold_unet
#SBATCH --output=logs/gold_unet_%j.out
#SBATCH --error=logs/gold_unet_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

set -euo pipefail

# Project root on HPC after syncing.
PROJECT_DIR="${PROJECT_DIR:-$HOME/max-planck-project/project}"
IMAGE_TIF="${IMAGE_TIF:-$PROJECT_DIR/data/Max Planck Data/Gold Particle Labelling/analyzed synapses/S1/S1 MBTt FFRIL01 R1Bg1d Wt 8wk AMPA6nm_NR1_12nm_vGlut2_18nm S1.tif}"
MASK_TIF="${MASK_TIF:-$PROJECT_DIR/data/Max Planck Data/Gold Particle Labelling/analyzed synapses/S1/S1 MBTt FFRIL01 R1Bg1d Wt 8wk AMPA6nm_NR1_12nm_vGlut2_18nm S1 mask.tif}"

EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LR="${LR:-1e-3}"
SIGMA="${SIGMA:-2.0}"
ALIGN_NUM_SLICES="${ALIGN_NUM_SLICES:-20}"
MAX_SLICES="${MAX_SLICES:-}"

cd "$PROJECT_DIR"
mkdir -p logs checkpoints

if command -v module >/dev/null 2>&1; then
  module purge || true
fi

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch numpy tifffile matplotlib scipy

python -u train.py \
  --image_tif "$IMAGE_TIF" \
  --mask_tif "$MASK_TIF" \
  --auto_align_crop \
  --align_num_slices "$ALIGN_NUM_SLICES" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --sigma "$SIGMA" \
  ${MAX_SLICES:+--max_slices "$MAX_SLICES"} \
  --save_model_path "checkpoints/gold_unet_${SLURM_JOB_ID}.pt"
