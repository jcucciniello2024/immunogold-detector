#!/usr/bin/env bash
# Pull baseline run outputs (CSVs + overlay PNGs) from Athene to this machine.
# Usage:
#   ./hpc/pull_baseline_overlays.sh JOB_ID
# Example:
#   ./hpc/pull_baseline_overlays.sh 4597500

set -euo pipefail
JOB_ID="${1:?Usage: $0 SLURM_JOB_ID}"
REMOTE_USER_HOST="${REMOTE_USER_HOST:-asahai2024@athene-login.hpc.fau.edu}"
REMOTE_PROJECT="${REMOTE_PROJECT:-/mnt/beegfs/home/asahai2024/max-planck-project/project}"
LOCAL_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_OUT="${REMOTE_PROJECT}/hpc_eval_baseline_simple_${JOB_ID}"

rsync -avz -e "ssh -o BatchMode=yes" \
  "${REMOTE_USER_HOST}:${REMOTE_OUT}/" \
  "${LOCAL_ROOT}/hpc_eval_baseline_simple_${JOB_ID}/"

echo "Downloaded to: ${LOCAL_ROOT}/hpc_eval_baseline_simple_${JOB_ID}/"
echo "Open vis/: *_detections.png and *_heatmap.png"
