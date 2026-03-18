#!/bin/bash
# Pull training logs and evaluation metrics from HPC

set -euo pipefail

HPC_HOST="${HPC_HOST:-login.cluster.de}"
HPC_USER="${HPC_USER:-asahai2024}"
HPC_PROJECT_DIR="/mnt/beegfs/home/asahai2024/max-planck-project/project"
LOCAL_DIR="./hpc_results"

echo "📥 Pulling metrics from HPC cluster..."
mkdir -p "$LOCAL_DIR"

# Pull training logs
echo "  Fetching training logs..."
scp -r "$HPC_USER@$HPC_HOST:$HPC_PROJECT_DIR/logs/*.out" "$LOCAL_DIR/" 2>/dev/null || true
scp -r "$HPC_USER@$HPC_HOST:$HPC_PROJECT_DIR/logs/*.err" "$LOCAL_DIR/" 2>/dev/null || true

# Pull evaluation results
echo "  Fetching evaluation results..."
scp "$HPC_USER@$HPC_HOST:$HPC_PROJECT_DIR/eval_results_*.txt" "$LOCAL_DIR/" 2>/dev/null || true
scp "$HPC_USER@$HPC_HOST:$HPC_PROJECT_DIR/predictions_*.csv" "$LOCAL_DIR/" 2>/dev/null || true

echo "✅ Saved to: $LOCAL_DIR/"
echo ""
echo "Latest training logs:"
ls -lht "$LOCAL_DIR"/*.out 2>/dev/null | head -3 || echo "  (no logs yet)"

echo ""
echo "Latest evaluation results:"
ls -lht "$LOCAL_DIR"/eval_results_*.txt 2>/dev/null | head -3 || echo "  (no results yet)"

echo ""
echo "To view latest training output:"
echo "  tail -100 $LOCAL_DIR/gold_detector2d_<jobid>.out"

echo ""
echo "To view evaluation metrics:"
echo "  cat $LOCAL_DIR/eval_results_<jobid>.txt"
