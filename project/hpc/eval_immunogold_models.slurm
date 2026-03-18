#!/bin/bash
#SBATCH --job-name=ig_eval
#SBATCH --output=logs/ig_eval_%j.out
#SBATCH --error=logs/ig_eval_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gres=gpu:1

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/beegfs/home/asahai2024/max-planck-project/project}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_DIR/data/Max Planck Data/Gold Particle Labelling/analyzed synapses}"
OUT_DIR="${OUT_DIR:-hpc_eval_results}"

cd "$PROJECT_DIR"
mkdir -p "$OUT_DIR" logs

if command -v module >/dev/null 2>&1; then
  module purge || true
  module load python3/3.9.16 || true
  module load cuda/11.8 || true
fi

BASE_TMP="${SLURM_TMPDIR:-/tmp}"
VENV_DIR="${BASE_TMP}/venv_ig_eval_${SLURM_JOB_ID}"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade "pip<24"
python -m pip install imagecodecs numpy tifffile matplotlib scipy
python -m pip install torch --index-url https://download.pytorch.org/whl/cu118

python infer_detector.py \
  --data_root "$DATA_ROOT" \
  --checkpoint checkpoints/hpc_unet_tuned/detector_best.pt \
  --out_csv "$OUT_DIR/hpc_unet.csv" \
  --threshold 0.02 --min_distance 5
python evaluate_detector.py \
  --data_root "$DATA_ROOT" \
  --pred_csv "$OUT_DIR/hpc_unet.csv" \
  --match_dist 5 \
  --threshold_sweep 0.0,0.01,0.02,0.03,0.05 \
  > "$OUT_DIR/hpc_unet_eval.txt"

python infer_detector.py \
  --data_root "$DATA_ROOT" \
  --checkpoint checkpoints/hpc_semi_tuned/detector_semi_best.pt \
  --out_csv "$OUT_DIR/hpc_semi.csv" \
  --threshold 0.02 --min_distance 5
python evaluate_detector.py \
  --data_root "$DATA_ROOT" \
  --pred_csv "$OUT_DIR/hpc_semi.csv" \
  --match_dist 5 \
  --threshold_sweep 0.0,0.01,0.02,0.03,0.05 \
  > "$OUT_DIR/hpc_semi_eval.txt"

python infer_log_cnn.py \
  --data_root "$DATA_ROOT" \
  --classifier_ckpt checkpoints/hpc_logcnn_best.pt \
  --out_csv "$OUT_DIR/hpc_logcnn.csv" \
  --log_threshold 0.015 \
  --class_threshold 0.35
python evaluate_detector.py \
  --data_root "$DATA_ROOT" \
  --pred_csv "$OUT_DIR/hpc_logcnn.csv" \
  --match_dist 5 \
  --threshold_sweep 0.0,0.01,0.02,0.03,0.05 \
  > "$OUT_DIR/hpc_logcnn_eval.txt"

python infer_two_stage.py \
  --data_root "$DATA_ROOT" \
  --heatmap_ckpt checkpoints/hpc_unet_tuned/detector_best.pt \
  --refiner_ckpt checkpoints/hpc_refiner_best.pt \
  --out_csv "$OUT_DIR/hpc_two_stage.csv" \
  --proposal_threshold 0.02 \
  --proposal_min_distance 5 \
  --refiner_keep_threshold 0.5
python evaluate_detector.py \
  --data_root "$DATA_ROOT" \
  --pred_csv "$OUT_DIR/hpc_two_stage.csv" \
  --match_dist 5 \
  --threshold_sweep 0.0,0.01,0.02,0.03,0.05 \
  > "$OUT_DIR/hpc_two_stage_eval.txt"

python infer_golddigger_cgan.py \
  --data_root "$DATA_ROOT" \
  --generator_ckpt checkpoints/hpc_golddigger_cgan/generator_best.pt \
  --out_csv "$OUT_DIR/hpc_cgan.csv" \
  --threshold_6nm 0.35 \
  --threshold_12nm 0.35
python evaluate_detector.py \
  --data_root "$DATA_ROOT" \
  --pred_csv "$OUT_DIR/hpc_cgan.csv" \
  --match_dist 5 \
  --threshold_sweep 0.0,0.01,0.02,0.03,0.05 \
  > "$OUT_DIR/hpc_cgan_eval.txt"

python - <<'PY'
import os, re, json
out_dir = "hpc_eval_results"
files = [
    ("unet", f"{out_dir}/hpc_unet_eval.txt"),
    ("semi", f"{out_dir}/hpc_semi_eval.txt"),
    ("logcnn", f"{out_dir}/hpc_logcnn_eval.txt"),
    ("two_stage", f"{out_dir}/hpc_two_stage_eval.txt"),
    ("cgan", f"{out_dir}/hpc_cgan_eval.txt"),
]
rows = []
for name, path in files:
    txt = open(path).read().splitlines()
    cur = {}
    best = None
    for line in txt:
        m = re.match(r"threshold=([0-9.]+)$", line)
        if m:
            cur = {"threshold": float(m.group(1))}
            continue
        if line.startswith("all.f1="):
            cur["all_f1"] = float(line.split("=")[1])
        elif line.startswith("class_6nm.f1="):
            cur["f1_6nm"] = float(line.split("=")[1])
        elif line.startswith("class_12nm.f1="):
            cur["f1_12nm"] = float(line.split("=")[1])
        elif line.startswith("macro.f1="):
            cur["macro_f1"] = float(line.split("=")[1])
            if best is None or cur["macro_f1"] > best["macro_f1"]:
                best = dict(cur)
    if best is None:
        best = {"threshold": None, "all_f1": 0.0, "f1_6nm": 0.0, "f1_12nm": 0.0, "macro_f1": 0.0}
    best["model"] = name
    rows.append(best)

rows.sort(key=lambda r: r["macro_f1"], reverse=True)
with open(f"{out_dir}/summary.json", "w") as f:
    json.dump({"best": rows[0], "all": rows}, f, indent=2)
print("Best model:", rows[0])
PY
