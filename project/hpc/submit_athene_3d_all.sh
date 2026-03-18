#!/bin/bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-asahai2024@athene-login.hpc.fau.edu}"
REMOTE_BASE="${REMOTE_BASE:-/mnt/beegfs/home/asahai2024/max-planck-project}"
LOCAL_PROJECT_DIR="${LOCAL_PROJECT_DIR:-/Users/aniksahai/Desktop/Max Planck Project/project}"
REMOTE_PROJECT_DIR="$REMOTE_BASE/project"
IMAGE_FILE="${IMAGE_FILE:-Max Planck Data/Gold Particle Labelling/analyzed synapses/S1/S1 MBTt FFRIL01 R1Bg1d Wt 8wk AMPA6nm_NR1_12nm_vGlut2_18nm S1.tif}"
MASK_FILE="${MASK_FILE:-Max Planck Data/Gold Particle Labelling/analyzed synapses/S1/S1 MBTt FFRIL01 R1Bg1d Wt 8wk AMPA6nm_NR1_12nm_vGlut2_18nm S1 mask.tif}"
MAX_RETRIES="${MAX_RETRIES:-4}"

retry() {
  local n=1
  local cmd="$1"
  until eval "$cmd"; do
    if [ "$n" -ge "$MAX_RETRIES" ]; then
      echo "Command failed after $n attempts"
      return 1
    fi
    n=$((n + 1))
    echo "Retrying ($n/$MAX_RETRIES)..."
    sleep 5
  done
}

SSH_OPTS="-o ServerAliveInterval=30 -o ServerAliveCountMax=10"
RSYNC_SSH="ssh $SSH_OPTS"

echo "Preparing remote directories..."
retry "ssh $SSH_OPTS \"$REMOTE_HOST\" \"mkdir -p $REMOTE_PROJECT_DIR/data $REMOTE_PROJECT_DIR/logs $REMOTE_PROJECT_DIR/checkpoints\""

echo "Syncing project code..."
retry "rsync -avz --progress --partial --inplace -e \"$RSYNC_SSH\" \
  --exclude \"data/\" --exclude \"alignment_checks/\" --exclude \"__pycache__/\" \
  \"$LOCAL_PROJECT_DIR/\" \"$REMOTE_HOST:$REMOTE_PROJECT_DIR/\""

echo "Syncing required TIFF files..."
retry "ssh $SSH_OPTS \"$REMOTE_HOST\" \"mkdir -p \\\"$(dirname "$REMOTE_PROJECT_DIR/data/$IMAGE_FILE")\\\" \\\"$(dirname "$REMOTE_PROJECT_DIR/data/$MASK_FILE")\\\"\""
retry "rsync -avz --progress --partial --inplace -e \"$RSYNC_SSH\" \
  \"$LOCAL_PROJECT_DIR/data/$IMAGE_FILE\" \"$REMOTE_HOST:$REMOTE_PROJECT_DIR/data/\""
retry "rsync -avz --progress --partial --inplace -e \"$RSYNC_SSH\" \
  \"$LOCAL_PROJECT_DIR/data/$MASK_FILE\" \"$REMOTE_HOST:$REMOTE_PROJECT_DIR/data/\""

echo "Submitting 3D jobs..."
retry "ssh $SSH_OPTS \"$REMOTE_HOST\" \"cd $REMOTE_PROJECT_DIR && \
EPOCHS=5 BATCH_SIZE=1 LR=1e-4 MAX_SLICES=50 PATCH_D=8 PATCH_H=64 PATCH_W=64 SAMPLES_PER_EPOCH=128 POS_FRACTION=0.6 sbatch hpc/train_gold_particles_3d.slurm\""

retry "ssh $SSH_OPTS \"$REMOTE_HOST\" \"cd $REMOTE_PROJECT_DIR && \
EPOCHS=10 BATCH_SIZE=1 LR=1e-4 MAX_SLICES=100 PATCH_D=16 PATCH_H=96 PATCH_W=96 SAMPLES_PER_EPOCH=192 POS_FRACTION=0.5 sbatch hpc/train_gold_particles_3d.slurm\""

retry "ssh $SSH_OPTS \"$REMOTE_HOST\" \"cd $REMOTE_PROJECT_DIR && \
EPOCHS=15 BATCH_SIZE=1 LR=5e-5 PATCH_D=16 PATCH_H=128 PATCH_W=128 SAMPLES_PER_EPOCH=256 POS_FRACTION=0.5 sbatch hpc/train_gold_particles_3d.slurm\""

echo "All 3D jobs submitted."
echo "Monitor with:"
echo "ssh $REMOTE_HOST 'squeue -u \$USER'"
