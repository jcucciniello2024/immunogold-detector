#!/bin/bash
#SBATCH --job-name=gold_sweep
#SBATCH --output=logs/gold_sweep_%A_%a.out
#SBATCH --error=logs/gold_sweep_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --array=0-5

# ============================================================================
# HYPERPARAMETER SWEEP — 6 configurations to find best settings
# ============================================================================
# Submit with:  sbatch hpc/train_hyperparam_sweep.slurm
# ============================================================================

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/beegfs/home/asahai2024/max-planck-project/project}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_DIR/data/Max Planck Data/Gold Particle Labelling/analyzed synapses}"

# Define sweep configurations
# Format: SIGMA|POS_WEIGHT|LR|LABEL
CONFIGS=(
  "1.0|300|3e-4|sigma1.0_pw300"
  "1.5|300|3e-4|sigma1.5_pw300"
  "2.0|300|3e-4|sigma2.0_pw300"
  "1.5|500|3e-4|sigma1.5_pw500"
  "1.5|300|1e-4|sigma1.5_lr1e4"
  "1.5|200|5e-4|sigma1.5_pw200_lr5e4"
)

CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
IFS='|' read -r SIGMA POS_WEIGHT LR LABEL <<< "$CONFIG"

SAVE_DIR="checkpoints/sweep_${SLURM_ARRAY_JOB_ID}_${LABEL}"

echo "================================================================"
echo "  SWEEP CONFIG $SLURM_ARRAY_TASK_ID: $LABEL"
echo "  sigma=$SIGMA  pos_weight=$POS_WEIGHT  lr=$LR"
echo "================================================================"

cd "$PROJECT_DIR"
mkdir -p logs checkpoints "$SAVE_DIR"

if command -v module >/dev/null 2>&1; then
  module purge || true
  module load python3/3.9.16 || true
  module load cuda/11.8 || true
fi

BASE_TMP="${SLURM_TMPDIR:-/tmp}"
VENV_DIR="${BASE_TMP}/venv_sweep_${SLURM_ARRAY_TASK_ID}_${SLURM_JOB_ID}"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade "pip<24"
python -m pip install imagecodecs numpy tifffile matplotlib scipy
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Train
python -u train_detector.py \
  --data_root "$DATA_ROOT" \
  --model_type unet_deep \
  --base_channels 32 \
  --epochs 100 \
  --batch_size 8 \
  --lr "$LR" \
  --patch_h 256 \
  --patch_w 256 \
  --train_samples_per_epoch 2048 \
  --val_samples_per_epoch 256 \
  --sigma "$SIGMA" \
  --target_type gaussian \
  --loss_type focal_bce \
  --loss_pos_weight "$POS_WEIGHT" \
  --focal_gamma 2.0 \
  --weight_decay 1e-4 \
  --grad_clip 1.0 \
  --sched cosine \
  --warmup_epochs 10 \
  --early_stop_patience 15 \
  --save_dir "$SAVE_DIR" \
  --use_sliding_window \
  --patch_stride 128 \
  --use_mantis \
  --sigma_jitter

# Quick eval
python -u infer_detector.py \
  --data_root "$DATA_ROOT" \
  --checkpoint "$SAVE_DIR/detector_best.pt" \
  --model_type unet_deep \
  --base_channels 32 \
  --out_csv "$SAVE_DIR/predictions.csv" \
  --threshold 0.1 \
  --min_distance 3 \
  --use_mantis

python -u evaluate_detector.py \
  --data_root "$DATA_ROOT" \
  --pred_csv "$SAVE_DIR/predictions.csv" \
  --match_dist 5 \
  --sweep_start 0.0 \
  --sweep_end 0.5 \
  --sweep_steps 25 \
  | tee "$SAVE_DIR/eval_results.txt"

echo ""
echo "Config $LABEL complete. Results in $SAVE_DIR/eval_results.txt"
