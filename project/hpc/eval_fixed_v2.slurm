#!/bin/bash
#SBATCH --job-name=gold_v2_eval
#SBATCH --output=logs/gold_v2_eval_%j.out
#SBATCH --error=logs/gold_v2_eval_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gres=gpu:1

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/beegfs/home/asahai2024/max-planck-project/project}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_DIR/data/Max Planck Data/Gold Particle Labelling/analyzed synapses}"

# ============================================================================
# INFERENCE + EVALUATION FOR V2 FIXED MODEL
# ============================================================================
# IMPORTANT: Must match training config exactly:
#   - model_type = unet_deep
#   - base_channels = 32
#   - use_mantis = true (same preprocessing as training)
# ============================================================================

CHECKPOINT="${CHECKPOINT:-checkpoints/v2_fixed/detector_best.pt}"
MODEL_TYPE="unet_deep"
BASE_CHANNELS=32
OUT_DIR="eval_results/v2_fixed_${SLURM_JOB_ID}"

echo "================================================================"
echo "  IMMUNOGOLD V2 — INFERENCE + EVALUATION"
echo "================================================================"
echo "  Checkpoint: $CHECKPOINT"
echo "  Model:      $MODEL_TYPE (base_channels=$BASE_CHANNELS)"
echo "  Mantis:     enabled"
echo ""

cd "$PROJECT_DIR"
mkdir -p logs "$OUT_DIR"

# Environment setup
if command -v module >/dev/null 2>&1; then
  module purge || true
  module load python3/3.9.16 || true
  module load cuda/11.8 || true
fi

BASE_TMP="${SLURM_TMPDIR:-/tmp}"
VENV_DIR="${BASE_TMP}/venv_eval_${SLURM_JOB_ID}"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade "pip<24"
python -m pip install imagecodecs numpy tifffile matplotlib scipy
python -m pip install torch --index-url https://download.pytorch.org/whl/cu118

# ============================================================================
# INFERENCE — sweep multiple thresholds
# ============================================================================
for THRESH in 0.05 0.1 0.15 0.2 0.3; do
  echo ""
  echo "--- Inference with threshold=$THRESH ---"
  python -u infer_detector.py \
    --data_root "$DATA_ROOT" \
    --checkpoint "$CHECKPOINT" \
    --model_type "$MODEL_TYPE" \
    --base_channels "$BASE_CHANNELS" \
    --out_csv "$OUT_DIR/preds_t${THRESH}.csv" \
    --out_vis_dir "$OUT_DIR/vis_t${THRESH}" \
    --tile_h 256 \
    --tile_w 256 \
    --stride_h 192 \
    --stride_w 192 \
    --threshold "$THRESH" \
    --min_distance 3 \
    --use_mantis \
    --save_vis
done

# ============================================================================
# EVALUATION — for each threshold file
# ============================================================================
echo ""
echo "================================================================"
echo "  EVALUATION RESULTS"
echo "================================================================"

BEST_F1=0
BEST_THRESH=""

for THRESH in 0.05 0.1 0.15 0.2 0.3; do
  echo ""
  echo "=== Threshold: $THRESH ==="
  python -u evaluate_detector.py \
    --data_root "$DATA_ROOT" \
    --pred_csv "$OUT_DIR/preds_t${THRESH}.csv" \
    --match_dist 5 \
    --threshold 0.0 \
    | tee "$OUT_DIR/eval_t${THRESH}.txt"
done

# ============================================================================
# FINE-GRAINED THRESHOLD SWEEP on best predictions
# ============================================================================
echo ""
echo "=== Fine-grained threshold sweep (t=0.1 predictions) ==="
python -u evaluate_detector.py \
  --data_root "$DATA_ROOT" \
  --pred_csv "$OUT_DIR/preds_t0.1.csv" \
  --match_dist 5 \
  --sweep_start 0.0 \
  --sweep_end 0.5 \
  --sweep_steps 50 \
  | tee "$OUT_DIR/eval_sweep.txt"

echo ""
echo "All results saved to: $OUT_DIR"
