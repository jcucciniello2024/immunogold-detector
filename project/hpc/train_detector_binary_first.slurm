#!/bin/bash
#SBATCH --job-name=ig_binary_first_v2
#SBATCH --output=logs/ig_binary_first_v2_%j.out
#SBATCH --error=logs/ig_binary_first_v2_%j.err
#SBATCH --partition=longq7-eng
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/beegfs/home/asahai2024/max-planck-project/project}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_DIR/data/Max Planck Data/Gold Particle Labelling/analyzed synapses}"
SAVE_DIR="${SAVE_DIR:-checkpoints/binary_first_${SLURM_JOB_ID}}"
OUT_DIR="${OUT_DIR:-hpc_eval_binary_first_${SLURM_JOB_ID}}"

cd "$PROJECT_DIR"
mkdir -p logs "$SAVE_DIR" "$OUT_DIR"

if command -v module >/dev/null 2>&1; then
  module purge || true
  module load python/3.12.4 || true
  module load cuda/11.8 || true
fi

BASE_TMP="${SLURM_TMPDIR:-/tmp}"
VENV_DIR="${BASE_TMP}/venv_binary_first_${SLURM_JOB_ID}"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install -q --upgrade "pip<24" || true
python -m pip install -q imagecodecs numpy tifffile scipy scikit-image pandas 2>/dev/null || true
python -m pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu118 2>/dev/null || true

python train_detector.py \
  --data_root "$DATA_ROOT" \
  --epochs 40 \
  --batch_size 4 \
  --lr 3e-4 \
  --patch_h 384 \
  --patch_w 384 \
  --train_samples_per_epoch 4096 \
  --val_samples_per_epoch 768 \
  --sigma 2.0 \
  --target_type gaussian \
  --pos_fraction 0.8 \
  --loss_type focal_bce \
  --loss_pos_weight 40 \
  --focal_gamma 2.0 \
  --weight_decay 1e-4 \
  --grad_clip 1.0 \
  --model_type unet \
  --base_channels 24 \
  --sched cosine \
  --warmup_epochs 5 \
  --mixed_precision \
  --early_stop_patience 8 \
  --binary_mode \
  --mask_bottom_px 120 \
  --save_dir "$SAVE_DIR"

python infer_detector.py \
  --data_root "$DATA_ROOT" \
  --checkpoint "$SAVE_DIR/detector_best.pt" \
  --out_csv "$OUT_DIR/pred_binary.csv" \
  --out_vis_dir "$OUT_DIR/vis" \
  --tile_h 512 --tile_w 512 \
  --stride_h 384 --stride_w 384 \
  --threshold 0.02 \
  --min_distance 6 \
  --max_detections_per_class 250 \
  --binary_mode \
  --ignore_bottom_px 120 \
  --hard_filter \
  --hard_window 21 \
  --hard_center_radius 2 \
  --hard_ring_inner 4 \
  --hard_ring_outer 8 \
  --hard_max_center_intensity 0.45 \
  --hard_min_ring_contrast 0.02 \
  --hard_dark_quantile 0.20 \
  --hard_min_equiv_diameter 2.0 \
  --hard_max_equiv_diameter 16.0 \
  --hard_max_anisotropy 3.5 \
  --save_vis \
  --save_heatmap

python evaluate_detector_binary.py \
  --data_root "$DATA_ROOT" \
  --pred_csv "$OUT_DIR/pred_binary.csv" \
  --match_dist 15.0 \
  --threshold_sweep "0.0,0.01,0.02,0.03,0.05,0.08,0.10,0.12,0.15,0.20,0.30" \
  > "$OUT_DIR/eval_binary.txt"

echo "=== Binary-first evaluation ==="
grep -E "^(threshold=|binary\\.f1=|binary\\.precision=|binary\\.recall=|best_threshold=)" "$OUT_DIR/eval_binary.txt" | tail -60
