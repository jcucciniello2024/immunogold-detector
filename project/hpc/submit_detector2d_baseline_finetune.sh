#!/bin/bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-asahai2024@athene-login.hpc.fau.edu}"
REMOTE_BASE="${REMOTE_BASE:-/mnt/beegfs/home/asahai2024/max-planck-project}"
LOCAL_PROJECT_DIR="${LOCAL_PROJECT_DIR:-/Users/aniksahai/Desktop/Max Planck Project/project}"
REMOTE_PROJECT_DIR="$REMOTE_BASE/project"
SSH_OPTS="-o ServerAliveInterval=30 -o ServerAliveCountMax=10"

echo "Syncing code and dataset to Athene..."
ssh $SSH_OPTS "$REMOTE_HOST" "mkdir -p \"$REMOTE_PROJECT_DIR\""
rsync -avz --progress --partial --inplace -e "ssh $SSH_OPTS" \
  "$LOCAL_PROJECT_DIR/" "$REMOTE_HOST:$REMOTE_PROJECT_DIR/"

echo "Submitting baseline detector job..."
BASE_JOB_ID=$(ssh $SSH_OPTS "$REMOTE_HOST" \
  "cd \"$REMOTE_PROJECT_DIR\" && sbatch --partition=shortq7-gpu --time=06:00:00 --job-name=det2d_base \
   hpc/train_detector_2d.slurm | awk '{print \$4}'")
echo "Baseline job: $BASE_JOB_ID"

echo "Submitting fine-tune detector job (after baseline)..."
FT_JOB_ID=$(ssh $SSH_OPTS "$REMOTE_HOST" \
  "cd \"$REMOTE_PROJECT_DIR\" && \
   EPOCHS=6 LR=2e-4 RESUME_PATH=\"$REMOTE_PROJECT_DIR/checkpoints/detector2d_${BASE_JOB_ID}.pt\" \
   sbatch --dependency=afterok:${BASE_JOB_ID} --partition=shortq7-gpu --time=06:00:00 --job-name=det2d_ft \
   hpc/train_detector_2d.slurm | awk '{print \$4}'")
echo "Fine-tune job: $FT_JOB_ID"

echo "Monitor with:"
echo "ssh $REMOTE_HOST 'squeue -j $BASE_JOB_ID,$FT_JOB_ID -o \"%i %j %T %M %R\"'"
