# Project Organization

This repository was cleaned up with a non-breaking structure:

- Core code remains at root (`train_*`, `infer_*`, `evaluate_*`, `model_*`, `dataset_*`).
- Cluster scripts remain in `hpc/`.
- Generated artifacts were moved under `outputs/`.

## outputs/

- `outputs/predictions/` - prediction CSVs (`predictions*.csv`)
- `outputs/visualizations/` - generated PNG visualizations (`pred_vis*`, augmentation/model debug images)
- `outputs/evaluations/` - text eval artifacts (`eval_protocol_baseline.txt`)
- `outputs/debug/` - debug CSV/TXT outputs (`debug_unet_after_nms*`)
- `outputs/hpc_runs/` - downloaded HPC result folders (`hpc_eval_*`)
- `outputs/benchmark/` - benchmark output folders (`benchmark_runs`, `benchmark_smoke`)

## Notes

- No training/inference code paths were moved.
- Existing job scripts in `hpc/` are unchanged.
- Checkpoints remain in `checkpoints/`.
