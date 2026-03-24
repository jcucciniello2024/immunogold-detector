#!/bin/bash
#SBATCH --job-name=golddigger_full
#SBATCH --output=logs/golddigger_full_%j.out
#SBATCH --error=logs/golddigger_full_%j.err
#SBATCH --partition=longq7-eng
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1

# Train Gold Digger–style pix2pix cGAN, then run inference + overlays + multi-class eval.
# Results: SAVE_DIR/generator_best.pt, OUT_DIR/pred_golddigger.csv, OUT_DIR/vis/*.png, OUT_DIR/eval.txt

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/beegfs/home/asahai2024/max-planck-project/project}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_DIR/data/Max Planck Data/Gold Particle Labelling/analyzed synapses}"
SAVE_DIR="${SAVE_DIR:-checkpoints/golddigger_cgan_${SLURM_JOB_ID}}"
OUT_DIR="${OUT_DIR:-hpc_eval_golddigger_${SLURM_JOB_ID}}"

EPOCHS="${EPOCHS:-35}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-2e-4}"
PATCH_SIZE="${PATCH_SIZE:-256}"

cd "$PROJECT_DIR"
mkdir -p logs "$SAVE_DIR" "$OUT_DIR"

if command -v module >/dev/null 2>&1; then
  module purge || true
  module load python/3.12.4 || true
  module load cuda/11.8 || true
fi

BASE_TMP="${SLURM_TMPDIR:-/tmp}"
VENV_DIR="${BASE_TMP}/venv_golddigger_full_${SLURM_JOB_ID}"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install -q --upgrade "pip<24" || true
python -m pip install -q imagecodecs numpy tifffile matplotlib scipy pandas 2>/dev/null || true
python -m pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu118 2>/dev/null || true

echo "=== Training GoldDigger cGAN -> $SAVE_DIR ==="
python -u train_golddigger_cgan.py \
  --data_root "$DATA_ROOT" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --patch_size "$PATCH_SIZE" \
  --train_samples_per_epoch 2048 \
  --val_samples_per_epoch 512 \
  --pos_fraction 0.7 \
  --radius_6nm 3 \
  --radius_12nm 5 \
  --lambda_l1 50.0 \
  --save_dir "$SAVE_DIR"

CKPT="$SAVE_DIR/generator_best.pt"
if [[ ! -f "$CKPT" ]]; then
  echo "ERROR: missing $CKPT"
  exit 1
fi

echo "=== Inference + overlays -> $OUT_DIR ==="
python -u infer_golddigger_cgan.py \
  --data_root "$DATA_ROOT" \
  --generator_ckpt "$CKPT" \
  --out_csv "$OUT_DIR/pred_golddigger.csv" \
  --out_vis_dir "$OUT_DIR/vis" \
  --tile_h 512 --tile_w 512 \
  --stride_h 384 --stride_w 384 \
  --threshold_6nm 0.35 \
  --threshold_12nm 0.35 \
  --save_vis \
  --save_heatmap

echo "=== Multi-class evaluation (per-class F1) ==="
python evaluate_detector.py \
  --data_root "$DATA_ROOT" \
  --pred_csv "$OUT_DIR/pred_golddigger.csv" \
  --match_dist 15.0 \
  --threshold_sweep "0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.5" \
  > "$OUT_DIR/eval_multiclass.txt"

echo "=== Done ==="
echo "Checkpoint: $CKPT"
echo "CSV: $OUT_DIR/pred_golddigger.csv"
echo "Vis: $OUT_DIR/vis/"
tail -40 "$OUT_DIR/eval_multiclass.txt" || true
