#!/bin/bash
set -euo pipefail

REMOTE_HOST="asahai2024@athene-login.hpc.fau.edu"
REMOTE_BASE="${REMOTE_BASE:-/mnt/beegfs/home/asahai2024/max-planck-project}"
LOCAL_PROJECT_DIR="${LOCAL_PROJECT_DIR:-/Users/aniksahai/Desktop/Max Planck Project/project}"
REMOTE_PROJECT_DIR="$REMOTE_BASE/project"

# Default sanity-run settings (override via environment if needed).
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-1e-4}"
MAX_SLICES="${MAX_SLICES:-50}"
IMAGE_FILE="${IMAGE_FILE:-Max Planck Data/Gold Particle Labelling/analyzed synapses/S1/S1 MBTt FFRIL01 R1Bg1d Wt 8wk AMPA6nm_NR1_12nm_vGlut2_18nm S1.tif}"
MASK_FILE="${MASK_FILE:-Max Planck Data/Gold Particle Labelling/analyzed synapses/S1/S1 MBTt FFRIL01 R1Bg1d Wt 8wk AMPA6nm_NR1_12nm_vGlut2_18nm S1 mask.tif}"

echo "Syncing local project to Athene..."
ssh "$REMOTE_HOST" "mkdir -p $REMOTE_PROJECT_DIR/data $REMOTE_PROJECT_DIR/logs $REMOTE_PROJECT_DIR/checkpoints"
rsync -avz --progress \
  --exclude "data/" \
  --exclude "alignment_checks/" \
  --exclude "__pycache__/" \
  "$LOCAL_PROJECT_DIR/" \
  "$REMOTE_HOST:$REMOTE_PROJECT_DIR/"

echo "Syncing required TIFF files..."
ssh "$REMOTE_HOST" "mkdir -p \"$(dirname "$REMOTE_PROJECT_DIR/data/$IMAGE_FILE")\" \"$(dirname "$REMOTE_PROJECT_DIR/data/$MASK_FILE")\""
rsync -avz --progress \
  "$LOCAL_PROJECT_DIR/data/$IMAGE_FILE" \
  "$REMOTE_HOST:$REMOTE_PROJECT_DIR/data/"
rsync -avz --progress \
  "$LOCAL_PROJECT_DIR/data/$MASK_FILE" \
  "$REMOTE_HOST:$REMOTE_PROJECT_DIR/data/"

echo "Submitting Slurm job..."
ssh "$REMOTE_HOST" "\
cd $REMOTE_PROJECT_DIR && \
EPOCHS=$EPOCHS BATCH_SIZE=$BATCH_SIZE LR=$LR MAX_SLICES=$MAX_SLICES sbatch hpc/train_gold_particles.slurm"

echo "Done. Monitor with:"
echo "ssh $REMOTE_HOST 'squeue -u \$USER'"
