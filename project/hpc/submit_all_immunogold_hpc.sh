#!/bin/bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-asahai2024@athene-login.hpc.fau.edu}"
REMOTE_BASE="${REMOTE_BASE:-/mnt/beegfs/home/asahai2024/max-planck-project}"
LOCAL_PROJECT_DIR="${LOCAL_PROJECT_DIR:-/Users/aniksahai/Desktop/Max Planck Project/project}"
REMOTE_PROJECT_DIR="$REMOTE_BASE/project"
SSH_OPTS="-o ServerAliveInterval=30 -o ServerAliveCountMax=10"

echo "Syncing project to Athene..."
ssh $SSH_OPTS "$REMOTE_HOST" "mkdir -p \"$REMOTE_PROJECT_DIR\""
rsync -avz --progress --partial --inplace -e "ssh $SSH_OPTS" \
  "$LOCAL_PROJECT_DIR/" "$REMOTE_HOST:$REMOTE_PROJECT_DIR/"

echo "Submitting U-Net detector training..."
UNET_JOB_ID=$(ssh $SSH_OPTS "$REMOTE_HOST" \
  "cd \"$REMOTE_PROJECT_DIR\" && \
   DATA_ROOT=\"$REMOTE_PROJECT_DIR/data/Max Planck Data/Gold Particle Labelling/analyzed synapses\" \
   SAVE_DIR=\"checkpoints/hpc_unet_tuned\" \
   EPOCHS=20 BATCH_SIZE=4 LR=6e-4 PATCH_H=384 PATCH_W=384 \
   TRAIN_SAMPLES_PER_EPOCH=4096 VAL_SAMPLES_PER_EPOCH=768 \
   POS_FRACTION=0.8 SIGMA=2.0 LOSS_TYPE=focal_bce LOSS_POS_WEIGHT=35 FOCAL_GAMMA=2.0 \
   sbatch --partition=shortq7-gpu --job-name=ig_unet hpc/train_detector_2d.slurm | awk '{print \$4}'")
echo "UNet job: $UNET_JOB_ID"

echo "Submitting refiner training..."
REFINER_JOB_ID=$(ssh $SSH_OPTS "$REMOTE_HOST" \
  "cd \"$REMOTE_PROJECT_DIR\" && \
   DATA_ROOT=\"$REMOTE_PROJECT_DIR/data/Max Planck Data/Gold Particle Labelling/analyzed synapses\" \
   SAVE_PATH=\"checkpoints/hpc_refiner_best.pt\" \
   EPOCHS=25 BATCH_SIZE=64 TRAIN_SAMPLES_PER_EPOCH=20000 VAL_SAMPLES_PER_EPOCH=4000 \
   sbatch --partition=shortq7-gpu --job-name=ig_refiner hpc/train_refiner_2d.slurm | awk '{print \$4}'")
echo "Refiner job: $REFINER_JOB_ID"

echo "Submitting LoG+CNN training..."
LOGCNN_JOB_ID=$(ssh $SSH_OPTS "$REMOTE_HOST" \
  "cd \"$REMOTE_PROJECT_DIR\" && \
   DATA_ROOT=\"$REMOTE_PROJECT_DIR/data/Max Planck Data/Gold Particle Labelling/analyzed synapses\" \
   SAVE_PATH=\"checkpoints/hpc_logcnn_best.pt\" \
   EPOCHS=25 BATCH_SIZE=128 LOG_THRESHOLD=0.015 \
   sbatch --partition=shortq7-gpu --job-name=ig_logcnn hpc/train_log_cnn_2d.slurm | awk '{print \$4}'")
echo "LoG+CNN job: $LOGCNN_JOB_ID"

echo "Submitting semi-supervised detector training (depends on U-Net)..."
SEMI_JOB_ID=$(ssh $SSH_OPTS "$REMOTE_HOST" \
  "cd \"$REMOTE_PROJECT_DIR\" && \
   DATA_ROOT=\"$REMOTE_PROJECT_DIR/data/Max Planck Data/Gold Particle Labelling/analyzed synapses\" \
   UNLABELED_DIR=\"$REMOTE_PROJECT_DIR/data/Max Planck Data/Gold Particle Labelling/labeled replica - Test Data from Synapses\" \
   TEACHER_CKPT=\"$REMOTE_PROJECT_DIR/checkpoints/hpc_unet_tuned/detector_best.pt\" \
   SAVE_DIR=\"checkpoints/hpc_semi_tuned\" \
   EPOCHS=20 BATCH_SIZE=4 PSEUDO_THRESHOLD=0.22 PSEUDO_MIN_DISTANCE=5 \
   PSEUDO_MIN_SUPPORT=2 PSEUDO_MERGE_DIST=4.0 MAX_PSEUDO_PER_CLASS=80 \
   sbatch --dependency=afterok:${UNET_JOB_ID} --partition=shortq7-gpu --job-name=ig_semi hpc/train_detector_semi.slurm | awk '{print \$4}'")
echo "Semi job: $SEMI_JOB_ID"

echo "Submitting Gold Digger cGAN training..."
CGAN_JOB_ID=$(ssh $SSH_OPTS "$REMOTE_HOST" \
  "cd \"$REMOTE_PROJECT_DIR\" && \
   DATA_ROOT=\"$REMOTE_PROJECT_DIR/data/Max Planck Data/Gold Particle Labelling/analyzed synapses\" \
   SAVE_DIR=\"checkpoints/hpc_golddigger_cgan\" \
   EPOCHS=30 BATCH_SIZE=4 PATCH_SIZE=256 \
   sbatch --partition=shortq7-gpu --job-name=ig_cgan hpc/train_golddigger_cgan.slurm | awk '{print \$4}'")
echo "cGAN job: $CGAN_JOB_ID"

echo "Submitted jobs:"
echo "  UNet:     $UNET_JOB_ID"
echo "  Refiner:  $REFINER_JOB_ID"
echo "  LoG+CNN:  $LOGCNN_JOB_ID"
echo "  Semi:     $SEMI_JOB_ID"
echo "  cGAN:     $CGAN_JOB_ID"
echo ""
echo "Monitor:"
echo "ssh $REMOTE_HOST 'squeue -j $UNET_JOB_ID,$REFINER_JOB_ID,$LOGCNN_JOB_ID,$SEMI_JOB_ID,$CGAN_JOB_ID -o \"%i %j %T %M %R\"'"
