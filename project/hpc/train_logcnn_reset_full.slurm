#!/bin/bash
#SBATCH --job-name=ig_reset_logcnn
#SBATCH --output=logs/ig_reset_logcnn_%j.out
#SBATCH --error=logs/ig_reset_logcnn_%j.err
#SBATCH --partition=longq7-eng
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/beegfs/home/asahai2024/max-planck-project/project}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_DIR/data/Max Planck Data/Gold Particle Labelling/analyzed synapses}"
SAVE_PATH="${SAVE_PATH:-checkpoints/logcnn_reset_${SLURM_JOB_ID}.pt}"
OUT_DIR="${OUT_DIR:-hpc_eval_logcnn_reset_${SLURM_JOB_ID}}"

cd "$PROJECT_DIR"
mkdir -p logs checkpoints "$OUT_DIR"

if command -v module >/dev/null 2>&1; then
  module purge || true
  module load python/3.12.4 || true
  module load cuda/11.8 || true
fi

BASE_TMP="${SLURM_TMPDIR:-/tmp}"
VENV_DIR="${BASE_TMP}/venv_logcnn_reset_${SLURM_JOB_ID}"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install -q --upgrade "pip<24" || true
python -m pip install -q imagecodecs numpy tifffile scipy scikit-image pandas matplotlib 2>/dev/null || true
python -m pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu118 2>/dev/null || true

echo "=== RESET PIPELINE: Train LoG+CNN ==="
python -u train_log_cnn.py \
  --data_root "$DATA_ROOT" \
  --epochs 30 \
  --batch_size 128 \
  --lr 2e-4 \
  --patch_size 33 \
  --sigmas "0.8,1.0,1.2,1.4,1.6,1.8" \
  --log_threshold 0.010 \
  --min_distance 5 \
  --max_candidates_per_image 300 \
  --match_dist 8.0 \
  --save_path "$SAVE_PATH"

echo "=== Inference with strict caps ==="
python -u infer_log_cnn.py \
  --data_root "$DATA_ROOT" \
  --classifier_ckpt "$SAVE_PATH" \
  --out_csv "$OUT_DIR/pred_logcnn_reset.csv" \
  --sigmas "0.8,1.0,1.2,1.4,1.6,1.8" \
  --log_threshold 0.010 \
  --candidate_min_distance 5 \
  --max_candidates_per_image 300 \
  --patch_size 33 \
  --class_threshold 0.35 \
  --final_min_distance 14

echo "=== Evaluate multiclass and joint ==="
python evaluate_detector.py \
  --data_root "$DATA_ROOT" \
  --pred_csv "$OUT_DIR/pred_logcnn_reset.csv" \
  --match_dist 15.0 \
  --threshold_sweep "0.0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.50" \
  --report_joint \
  > "$OUT_DIR/eval_multiclass.txt"

echo "=== Key metrics ==="
grep -E "^(threshold=|all\\.f1=|macro\\.f1=|joint\\.|best_threshold=)" "$OUT_DIR/eval_multiclass.txt" | tail -120
echo ""
echo "Model: $SAVE_PATH"
echo "Predictions: $OUT_DIR/pred_logcnn_reset.csv"
echo "Eval: $OUT_DIR/eval_multiclass.txt"
