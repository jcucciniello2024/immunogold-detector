from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence

from evaluate_detector import evaluate_subset, filter_predictions_by_threshold, load_predictions
from prepare_labels import discover_image_records


@dataclass
class EvalRow:
    pipeline: str
    run_name: str
    pred_csv: str
    threshold: float
    all_f1: float
    macro_f1: float
    f1_6nm: float
    f1_12nm: float
    precision_all: float
    recall_all: float


def run_cmd(cmd: Sequence[str]) -> None:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            f"Command failed:\n{' '.join(cmd)}\n\nstdout:\n{p.stdout}\n\nstderr:\n{p.stderr}"
        )


def parse_grid(spec: str) -> List[float]:
    vals = [float(x.strip()) for x in spec.split(",") if x.strip()]
    if not vals:
        raise ValueError(f"Empty grid: {spec}")
    return vals


def evaluate_prediction_csv(
    gt_map: Dict[str, Dict[int, object]],
    image_ids: List[str],
    pred_csv: str,
    match_dist: float,
    thresholds: Sequence[float],
    pipeline: str,
    run_name: str,
) -> List[EvalRow]:
    pred_map_raw = load_predictions(pred_csv)
    rows: List[EvalRow] = []
    for thr in thresholds:
        pred_map = filter_predictions_by_threshold(pred_map_raw, threshold=thr)
        m = evaluate_subset(gt_map=gt_map, pred_map=pred_map, match_dist=match_dist, image_ids=image_ids)
        rows.append(
            EvalRow(
                pipeline=pipeline,
                run_name=run_name,
                pred_csv=pred_csv,
                threshold=float(thr),
                all_f1=float(m["all"].f1),
                macro_f1=float(m["macro"].f1),
                f1_6nm=float(m["6nm"].f1),
                f1_12nm=float(m["12nm"].f1),
                precision_all=float(m["all"].precision),
                recall_all=float(m["all"].recall),
            )
        )
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Tune/compare detector pipelines under a unified evaluation protocol.")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="benchmark_runs")
    p.add_argument("--match_dist", type=float, default=5.0)
    p.add_argument("--eval_thresholds", type=str, default="0.0,0.01,0.02,0.05,0.1,0.2,0.3")

    # U-Net heatmap pipeline.
    p.add_argument("--unet_checkpoint", type=str, default="")
    p.add_argument("--unet_min_dist_grid", type=str, default="4,5,6")
    p.add_argument("--unet_infer_threshold", type=float, default=0.01)

    # Two-stage pipeline.
    p.add_argument("--two_stage_heatmap_ckpt", type=str, default="")
    p.add_argument("--two_stage_refiner_ckpt", type=str, default="")
    p.add_argument("--two_stage_keep_grid", type=str, default="0.45,0.55,0.65,0.75")
    p.add_argument("--two_stage_proposal_threshold", type=float, default=0.01)

    # LoG+CNN pipeline.
    p.add_argument("--logcnn_checkpoint", type=str, default="")
    p.add_argument("--logcnn_class_threshold_grid", type=str, default="0.35,0.45,0.55,0.65")
    p.add_argument("--logcnn_log_threshold", type=float, default=0.015)

    # Optional existing CSV files for quick benchmarking.
    p.add_argument("--extra_pred_csvs", type=str, default="", help="Comma-separated CSV paths to include directly.")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    thresholds = parse_grid(args.eval_thresholds)

    records = discover_image_records(args.data_root)
    gt_map = {r.image_id: r.points for r in records}
    image_ids = sorted(gt_map.keys())
    all_rows: List[EvalRow] = []

    if args.unet_checkpoint:
        min_dist_grid = parse_grid(args.unet_min_dist_grid)
        for min_dist in min_dist_grid:
            run_name = f"unet_md{int(min_dist)}"
            out_csv = os.path.join(args.out_dir, f"{run_name}.csv")
            run_cmd(
                [
                    "python3",
                    "infer_detector.py",
                    "--data_root",
                    args.data_root,
                    "--checkpoint",
                    args.unet_checkpoint,
                    "--out_csv",
                    out_csv,
                    "--threshold",
                    str(args.unet_infer_threshold),
                    "--min_distance",
                    str(int(min_dist)),
                ]
            )
            all_rows.extend(
                evaluate_prediction_csv(
                    gt_map=gt_map,
                    image_ids=image_ids,
                    pred_csv=out_csv,
                    match_dist=args.match_dist,
                    thresholds=thresholds,
                    pipeline="unet",
                    run_name=run_name,
                )
            )

    if args.two_stage_heatmap_ckpt and args.two_stage_refiner_ckpt:
        keep_grid = parse_grid(args.two_stage_keep_grid)
        for keep in keep_grid:
            run_name = f"two_stage_keep{str(keep).replace('.', 'p')}"
            out_csv = os.path.join(args.out_dir, f"{run_name}.csv")
            run_cmd(
                [
                    "python3",
                    "infer_two_stage.py",
                    "--data_root",
                    args.data_root,
                    "--heatmap_ckpt",
                    args.two_stage_heatmap_ckpt,
                    "--refiner_ckpt",
                    args.two_stage_refiner_ckpt,
                    "--out_csv",
                    out_csv,
                    "--proposal_threshold",
                    str(args.two_stage_proposal_threshold),
                    "--refiner_keep_threshold",
                    str(keep),
                ]
            )
            all_rows.extend(
                evaluate_prediction_csv(
                    gt_map=gt_map,
                    image_ids=image_ids,
                    pred_csv=out_csv,
                    match_dist=args.match_dist,
                    thresholds=thresholds,
                    pipeline="two_stage",
                    run_name=run_name,
                )
            )

    if args.logcnn_checkpoint:
        class_grid = parse_grid(args.logcnn_class_threshold_grid)
        for cthr in class_grid:
            run_name = f"logcnn_cls{str(cthr).replace('.', 'p')}"
            out_csv = os.path.join(args.out_dir, f"{run_name}.csv")
            run_cmd(
                [
                    "python3",
                    "infer_log_cnn.py",
                    "--data_root",
                    args.data_root,
                    "--classifier_ckpt",
                    args.logcnn_checkpoint,
                    "--out_csv",
                    out_csv,
                    "--log_threshold",
                    str(args.logcnn_log_threshold),
                    "--class_threshold",
                    str(cthr),
                ]
            )
            all_rows.extend(
                evaluate_prediction_csv(
                    gt_map=gt_map,
                    image_ids=image_ids,
                    pred_csv=out_csv,
                    match_dist=args.match_dist,
                    thresholds=thresholds,
                    pipeline="logcnn",
                    run_name=run_name,
                )
            )

    if args.extra_pred_csvs.strip():
        for i, path in enumerate([x.strip() for x in args.extra_pred_csvs.split(",") if x.strip()]):
            run_name = f"extra_{i + 1}"
            all_rows.extend(
                evaluate_prediction_csv(
                    gt_map=gt_map,
                    image_ids=image_ids,
                    pred_csv=path,
                    match_dist=args.match_dist,
                    thresholds=thresholds,
                    pipeline="extra",
                    run_name=run_name,
                )
            )

    if not all_rows:
        raise ValueError("No pipelines were evaluated. Provide checkpoints and/or --extra_pred_csvs.")

    rows_sorted = sorted(all_rows, key=lambda r: r.macro_f1, reverse=True)

    csv_path = os.path.join(args.out_dir, "benchmark_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows_sorted[0]).keys()))
        w.writeheader()
        for row in rows_sorted:
            w.writerow(asdict(row))

    topk = rows_sorted[:10]
    summary = {
        "best": asdict(rows_sorted[0]),
        "top10": [asdict(r) for r in topk],
        "num_rows": len(rows_sorted),
    }
    summary_path = os.path.join(args.out_dir, "benchmark_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved benchmark table: {csv_path}")
    print(f"Saved benchmark summary: {summary_path}")
    print(f"Best -> pipeline={rows_sorted[0].pipeline} run={rows_sorted[0].run_name} "
          f"threshold={rows_sorted[0].threshold:.4f} macro_f1={rows_sorted[0].macro_f1:.4f}")


if __name__ == "__main__":
    main()
