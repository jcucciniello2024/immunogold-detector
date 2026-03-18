#!/bin/bash
#SBATCH --job-name=gold_unet3d
#SBATCH --output=logs/gold_unet3d_%j.out
#SBATCH --error=logs/gold_unet3d_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/max-planck-project/project}"
IMAGE_TIF="${IMAGE_TIF:-$PROJECT_DIR/data/Max Planck Data/Gold Particle Labelling/analyzed synapses/S1/S1 MBTt FFRIL01 R1Bg1d Wt 8wk AMPA6nm_NR1_12nm_vGlut2_18nm S1.tif}"
MASK_TIF="${MASK_TIF:-$PROJECT_DIR/data/Max Planck Data/Gold Particle Labelling/analyzed synapses/S1/S1 MBTt FFRIL01 R1Bg1d Wt 8wk AMPA6nm_NR1_12nm_vGlut2_18nm S1 mask.tif}"

EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LR="${LR:-1e-4}"
MAX_SLICES="${MAX_SLICES:-}"
ALIGN_NUM_SLICES="${ALIGN_NUM_SLICES:-20}"

PATCH_D="${PATCH_D:-16}"
PATCH_H="${PATCH_H:-128}"
PATCH_W="${PATCH_W:-128}"
SAMPLES_PER_EPOCH="${SAMPLES_PER_EPOCH:-256}"
POS_FRACTION="${POS_FRACTION:-0.5}"
SIGMA_XY="${SIGMA_XY:-2.0}"
SIGMA_Z="${SIGMA_Z:-1.5}"

cd "$PROJECT_DIR"
mkdir -p logs checkpoints

if command -v module >/dev/null 2>&1; then
  module purge || true
  module load python3/3.9.16 || true
  module load cuda/11.8 || true
fi

BASE_TMP="${SLURM_TMPDIR:-/tmp}"
VENV_DIR="${BASE_TMP}/venv_gold5d_${SLURM_JOB_ID}"
python3 -m venv "$VENV_DIR"

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade "pip<24"

if python - <<'PY'
import importlib.util
import sys
mods = ["torch", "numpy", "tifffile", "matplotlib", "scipy"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
sys.exit(1 if missing else 0)
PY
then
  echo "Python dependencies already present in $VENV_DIR"
else
  echo "Installing Python dependencies in $VENV_DIR"
  python -m pip install numpy tifffile matplotlib scipy
  python -m pip install torch --index-url https://download.pytorch.org/whl/cu118
fi

python -u train_3d.py \
  --image_tif "$IMAGE_TIF" \
  --mask_tif "$MASK_TIF" \
  --auto_align_crop \
  --align_num_slices "$ALIGN_NUM_SLICES" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --patch_d "$PATCH_D" \
  --patch_h "$PATCH_H" \
  --patch_w "$PATCH_W" \
  --samples_per_epoch "$SAMPLES_PER_EPOCH" \
  --pos_fraction "$POS_FRACTION" \
  --sigma_xy "$SIGMA_XY" \
  --sigma_z "$SIGMA_Z" \
  ${MAX_SLICES:+--max_slices "$MAX_SLICES"} \
  --save_model_path "checkpoints/gold_unet3d_${SLURM_JOB_ID}.pt"
