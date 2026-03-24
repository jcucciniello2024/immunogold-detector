#!/bin/bash
# ============================================================================
# SUBMIT V2 FIXED PIPELINE
# ============================================================================
# Usage:
#   bash hpc/submit_v2_pipeline.sh
#
# This submits:
#   1. Training job (train_fixed_v2.slurm)
#   2. Evaluation job (eval_fixed_v2.slurm) — starts AFTER training completes
#
# To override the data root:
#   DATA_ROOT=/path/to/data bash hpc/submit_v2_pipeline.sh
# ============================================================================

set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs checkpoints

echo "Submitting V2 Fixed Pipeline..."
echo ""

# Submit training
TRAIN_JOB=$(sbatch --parsable hpc/train_fixed_v2.slurm)
echo "Submitted training job: $TRAIN_JOB"

# Submit evaluation — depends on training completing successfully
EVAL_JOB=$(sbatch --parsable --dependency=afterok:${TRAIN_JOB} \
  --export=ALL,CHECKPOINT="checkpoints/v2_fixed_${TRAIN_JOB}/detector_best.pt" \
  hpc/eval_fixed_v2.slurm)
echo "Submitted evaluation job: $EVAL_JOB (depends on $TRAIN_JOB)"

echo ""
echo "Pipeline submitted:"
echo "  Training:   $TRAIN_JOB"
echo "  Evaluation: $EVAL_JOB (starts after training)"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f logs/gold_v2_${TRAIN_JOB}.out"
