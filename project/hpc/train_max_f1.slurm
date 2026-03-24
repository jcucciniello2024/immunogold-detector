#!/bin/bash
#SBATCH --job-name=gold_max_f1
#SBATCH --output=logs/gold_max_f1_%j.out
#SBATCH --error=logs/gold_max_f1_%j.err
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/beegfs/home/asahai2024/max-planck-project/project}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_DIR/data/Max Planck Data/Gold Particle Labelling/analyzed synapses}"
SAVE_DIR="checkpoints/max_f1_${SLURM_JOB_ID}"

cd "$PROJECT_DIR"
mkdir -p logs checkpoints "$SAVE_DIR" "$SAVE_DIR/vis"

if command -v module >/dev/null 2>&1; then
  module purge || true
  module load python3/3.9.16 || true
  module load cuda/11.8 || true
fi

BASE_TMP="${SLURM_TMPDIR:-/tmp}"
VENV_DIR="${BASE_TMP}/venv_maxf1_${SLURM_JOB_ID}"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade "pip<24"
python -m pip install imagecodecs numpy tifffile matplotlib scipy
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

echo "============================================================"
echo "  MAX F1 PIPELINE"
echo "============================================================"

# ============================================================
# PHASE 0: SANITY CHECK — abort early if data/model is broken
# ============================================================
echo ""
echo "=== PHASE 0: SANITY CHECK ==="
python -u sanity_check.py --data_root "$DATA_ROOT" --out_dir "$SAVE_DIR/sanity"

# Check if overfit test passed (pred_max > 0.05)
if grep -q "FAIL" "$SAVE_DIR/sanity/overfit_test.png" 2>/dev/null; then
  echo "Sanity check output saved. Check $SAVE_DIR/sanity/ for visualizations."
fi
echo "Sanity check complete. Review $SAVE_DIR/sanity/ for GT overlays."
echo ""

# ============================================================
# PHASE 1: TRAIN
# ============================================================
echo ""
echo "=== PHASE 1: TRAINING ==="
python -u train_detector.py \
  --data_root "$DATA_ROOT" \
  --model_type unet_deep \
  --base_channels 32 \
  --epochs 200 \
  --batch_size 8 \
  --lr 3e-4 \
  --patch_h 256 \
  --patch_w 256 \
  --train_samples_per_epoch 2048 \
  --val_samples_per_epoch 256 \
  --sigma 1.5 \
  --target_type gaussian \
  --loss_type focal_bce \
  --loss_pos_weight 300.0 \
  --loss_neg_weight 1.0 \
  --focal_gamma 2.0 \
  --weight_decay 1e-4 \
  --grad_clip 1.0 \
  --sched cosine \
  --warmup_epochs 10 \
  --early_stop_patience 20 \
  --early_stop_delta 1e-6 \
  --seed 42 \
  --save_dir "$SAVE_DIR" \
  --use_sliding_window \
  --patch_stride 128 \
  --use_mantis \
  --sigma_jitter

echo ""
echo "Training complete."

# ============================================================
# PHASE 2: INFERENCE AT MULTIPLE THRESHOLDS
# ============================================================
echo ""
echo "=== PHASE 2: INFERENCE ==="
for T in 0.01 0.02 0.05 0.08 0.1 0.15 0.2 0.25 0.3 0.4; do
  python -u infer_detector.py \
    --data_root "$DATA_ROOT" \
    --checkpoint "$SAVE_DIR/detector_best.pt" \
    --model_type unet_deep \
    --base_channels 32 \
    --out_csv "$SAVE_DIR/preds_t${T}.csv" \
    --tile_h 256 \
    --tile_w 256 \
    --stride_h 128 \
    --stride_w 128 \
    --threshold "$T" \
    --min_distance 3 \
    --use_mantis
done

# Also save visualizations for the 0.1 threshold
python -u infer_detector.py \
  --data_root "$DATA_ROOT" \
  --checkpoint "$SAVE_DIR/detector_best.pt" \
  --model_type unet_deep \
  --base_channels 32 \
  --out_csv "$SAVE_DIR/preds_vis.csv" \
  --out_vis_dir "$SAVE_DIR/vis" \
  --tile_h 256 \
  --tile_w 256 \
  --stride_h 128 \
  --stride_w 128 \
  --threshold 0.1 \
  --min_distance 3 \
  --use_mantis \
  --save_vis

# ============================================================
# PHASE 3: EVALUATE ALL THRESHOLDS, FIND BEST
# ============================================================
echo ""
echo "=== PHASE 3: EVALUATION ==="
echo ""

BEST_F1=0
BEST_T=""

for T in 0.01 0.02 0.05 0.08 0.1 0.15 0.2 0.25 0.3 0.4; do
  echo "--- threshold=$T ---"
  python -u evaluate_detector.py \
    --data_root "$DATA_ROOT" \
    --pred_csv "$SAVE_DIR/preds_t${T}.csv" \
    --match_dist 5 \
    --threshold 0.0 \
    | tee "$SAVE_DIR/eval_t${T}.txt"
  echo ""
done

# Fine-grained sweep on the 0.01 threshold predictions (captures all detections)
echo "=== FINE-GRAINED THRESHOLD SWEEP ==="
python -u evaluate_detector.py \
  --data_root "$DATA_ROOT" \
  --pred_csv "$SAVE_DIR/preds_t0.01.csv" \
  --match_dist 5 \
  --sweep_start 0.0 \
  --sweep_end 0.5 \
  --sweep_steps 100 \
  | tee "$SAVE_DIR/eval_sweep.txt"

echo ""
echo "============================================================"
echo "  DONE — Results in: $SAVE_DIR"
echo "  Visualizations in: $SAVE_DIR/vis"
echo "  Best threshold: check $SAVE_DIR/eval_sweep.txt"
echo "============================================================"
