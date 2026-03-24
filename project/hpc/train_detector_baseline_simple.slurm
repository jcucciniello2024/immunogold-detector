#!/bin/bash
#SBATCH --job-name=ig_baseline_simple
#SBATCH --output=logs/ig_baseline_simple_%j.out
#SBATCH --error=logs/ig_baseline_simple_%j.err
#SBATCH --partition=longq7-eng
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1

# Baseline: binary Gaussian heatmap U-Net — no hard particle filter, no bottom masking.
# Goal: maximize usable F1 first; overlays + heatmaps land in OUT_DIR/vis/

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/beegfs/home/asahai2024/max-planck-project/project}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_DIR/data/Max Planck Data/Gold Particle Labelling/analyzed synapses}"
SAVE_DIR="${SAVE_DIR:-checkpoints/baseline_simple_${SLURM_JOB_ID}}"
OUT_DIR="${OUT_DIR:-hpc_eval_baseline_simple_${SLURM_JOB_ID}}"

cd "$PROJECT_DIR"
mkdir -p logs "$SAVE_DIR" "$OUT_DIR"

if command -v module >/dev/null 2>&1; then
  module purge || true
  module load python/3.12.4 || true
  module load cuda/11.8 || true
fi

BASE_TMP="${SLURM_TMPDIR:-/tmp}"
VENV_DIR="${BASE_TMP}/venv_baseline_simple_${SLURM_JOB_ID}"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install -q --upgrade "pip<24" || true
python -m pip install -q imagecodecs numpy tifffile scipy scikit-image pandas matplotlib 2>/dev/null || true
python -m pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu118 2>/dev/null || true

python train_detector.py \
  --data_root "$DATA_ROOT" \
  --epochs 35 \
  --batch_size 4 \
  --lr 1e-3 \
  --patch_h 512 \
  --patch_w 512 \
  --train_samples_per_epoch 2048 \
  --val_samples_per_epoch 512 \
  --sigma 2.5 \
  --target_type gaussian \
  --pos_fraction 0.55 \
  --loss_type focal_bce \
  --loss_pos_weight 35 \
  --focal_gamma 2.0 \
  --weight_decay 1e-4 \
  --grad_clip 1.0 \
  --model_type unet \
  --base_channels 24 \
  --sched cosine \
  --warmup_epochs 3 \
  --early_stop_patience 10 \
  --binary_mode \
  --save_dir "$SAVE_DIR"

# Peak decode: local maxima + light blur (fewer duplicate X's per blob) + optional line thinning.
# Confidences = heatmap values; evaluate_detector_binary.py sweeps confidence thresholds.
python infer_detector.py \
  --data_root "$DATA_ROOT" \
  --checkpoint "$SAVE_DIR/detector_best.pt" \
  --out_csv "$OUT_DIR/pred_binary.csv" \
  --out_vis_dir "$OUT_DIR/vis" \
  --tile_h 512 --tile_w 512 \
  --stride_h 384 --stride_w 384 \
  --threshold 0.05 \
  --min_distance 8 \
  --max_detections_per_class 2000 \
  --peak_blur_sigma 0.8 \
  --binary_mode \
  --save_vis \
  --save_heatmap

python evaluate_detector_binary.py \
  --data_root "$DATA_ROOT" \
  --pred_csv "$OUT_DIR/pred_binary.csv" \
  --match_dist 15.0 \
  --threshold_sweep "0.0,0.05,0.08,0.10,0.12,0.15,0.18,0.20,0.25,0.30,0.35,0.40,0.50" \
  > "$OUT_DIR/eval_binary.txt"

echo "=== Baseline binary evaluation (no hard filter) ==="
grep -E "^(threshold=|binary\\.f1=|binary\\.precision=|binary\\.recall=|best_threshold=)" "$OUT_DIR/eval_binary.txt" | tail -80
echo ""
echo "Overlays: $OUT_DIR/vis/*_detections.png"
echo "Heatmaps: $OUT_DIR/vis/*_heatmap.png"
