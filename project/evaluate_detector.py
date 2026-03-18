import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from prepare_labels import discover_image_records


@dataclass
class Metrics:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    mean_localization_error: float


def load_predictions(path: str) -> Dict[str, Dict[int, List[Tuple[float, float, float]]]]:
    out: Dict[str, Dict[int, List[Tuple[float, float, float]]]] = defaultdict(lambda: {0: [], 1: []})
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row["image_id"]
            x = float(row["x"])
            y = float(row["y"])
            cls = int(row["class_id"])
            conf = float(row["confidence"])
            out[image_id][cls].append((x, y, conf))
    return out


def greedy_match(
    gt: np.ndarray, pred: List[Tuple[float, float, float]], max_dist: float
) -> Tuple[int, int, int, List[float]]:
    if len(gt) == 0:
        return 0, len(pred), 0, []
    pred_xy = np.array([[p[0], p[1]] for p in pred], dtype=np.float32) if pred else np.zeros((0, 2), np.float32)
    used = np.zeros(len(pred_xy), dtype=bool)
    tp = 0
    dists: List[float] = []
    for g in gt:
        if len(pred_xy) == 0:
            continue
        dist = np.sqrt(((pred_xy - g[None, :]) ** 2).sum(axis=1))
        dist[used] = 1e9
        j = int(np.argmin(dist))
        if dist[j] < max_dist:
            used[j] = True
            tp += 1
            dists.append(float(dist[j]))
    fp = int((~used).sum())
    fn = int(len(gt) - tp)
    return tp, fp, fn, dists


def calc_metrics(tp: int, fp: int, fn: int, loc_errors: Sequence[float]) -> Metrics:
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    mean_loc = float(np.mean(loc_errors)) if len(loc_errors) > 0 else float("nan")
    return Metrics(
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        tp=int(tp),
        fp=int(fp),
        fn=int(fn),
        mean_localization_error=mean_loc,
    )


def filter_predictions_by_threshold(
    pred_map: Dict[str, Dict[int, List[Tuple[float, float, float]]]], threshold: float
) -> Dict[str, Dict[int, List[Tuple[float, float, float]]]]:
    out: Dict[str, Dict[int, List[Tuple[float, float, float]]]] = defaultdict(lambda: {0: [], 1: []})
    for image_id, cls_map in pred_map.items():
        for cls in [0, 1]:
            out[image_id][cls] = [p for p in cls_map.get(cls, []) if p[2] >= threshold]
    return out


def evaluate_subset(
    gt_map: Dict[str, Dict[int, np.ndarray]],
    pred_map: Dict[str, Dict[int, List[Tuple[float, float, float]]]],
    match_dist: float,
    image_ids: Sequence[str],
) -> Dict[str, Metrics]:
    per_cls = {
        0: {"tp": 0, "fp": 0, "fn": 0, "loc_errors": []},
        1: {"tp": 0, "fp": 0, "fn": 0, "loc_errors": []},
    }
    total_tp = total_fp = total_fn = 0
    loc_errors_all: List[float] = []

    for image_id in image_ids:
        points = gt_map[image_id]
        preds = pred_map.get(image_id, {0: [], 1: []})
        for cls in [0, 1]:
            tp, fp, fn, d = greedy_match(points[cls], preds.get(cls, []), max_dist=match_dist)
            per_cls[cls]["tp"] += tp
            per_cls[cls]["fp"] += fp
            per_cls[cls]["fn"] += fn
            per_cls[cls]["loc_errors"].extend(d)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            loc_errors_all.extend(d)

    m_all = calc_metrics(total_tp, total_fp, total_fn, loc_errors_all)
    m_6 = calc_metrics(
        int(per_cls[0]["tp"]),
        int(per_cls[0]["fp"]),
        int(per_cls[0]["fn"]),
        per_cls[0]["loc_errors"],
    )
    m_12 = calc_metrics(
        int(per_cls[1]["tp"]),
        int(per_cls[1]["fp"]),
        int(per_cls[1]["fn"]),
        per_cls[1]["loc_errors"],
    )
    macro_f1 = 0.5 * (m_6.f1 + m_12.f1)
    return {
        "all": m_all,
        "6nm": m_6,
        "12nm": m_12,
        "macro": Metrics(
            precision=float("nan"),
            recall=float("nan"),
            f1=float(macro_f1),
            tp=0,
            fp=0,
            fn=0,
            mean_localization_error=float("nan"),
        ),
    }


def build_grouped_folds(image_ids: Sequence[str], k_folds: int, seed: int) -> List[List[str]]:
    if k_folds < 2:
        return [list(image_ids)]
    ids = np.array(sorted(image_ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    folds: List[List[str]] = [[] for _ in range(k_folds)]
    for i, image_id in enumerate(ids.tolist()):
        folds[i % k_folds].append(image_id)
    return folds


def parse_thresholds(
    threshold: float,
    threshold_sweep: str,
    sweep_start: float,
    sweep_end: float,
    sweep_steps: int,
) -> List[float]:
    if threshold_sweep.strip():
        vals = [float(s.strip()) for s in threshold_sweep.split(",") if s.strip()]
        return sorted(set(vals))
    if sweep_steps > 1:
        vals = np.linspace(sweep_start, sweep_end, int(sweep_steps)).tolist()
        return sorted(set(float(v) for v in vals))
    return [float(threshold)]


def print_metrics_block(name: str, m: Metrics) -> None:
    print(f"{name}.precision={m.precision:.4f}")
    print(f"{name}.recall={m.recall:.4f}")
    print(f"{name}.f1={m.f1:.4f}")
    print(f"{name}.mean_localization_error={m.mean_localization_error:.4f}")
    print(f"{name}.tp={m.tp} {name}.fp={m.fp} {name}.fn={m.fn}")


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate keypoint detections against GT coordinates.")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--pred_csv", type=str, required=True)
    p.add_argument("--match_dist", type=float, default=5.0)
    p.add_argument("--threshold", type=float, default=0.0, help="Minimum prediction confidence.")
    p.add_argument(
        "--threshold_sweep",
        type=str,
        default="",
        help="Comma-separated confidence thresholds (e.g. 0.01,0.02,0.03).",
    )
    p.add_argument("--sweep_start", type=float, default=0.0)
    p.add_argument("--sweep_end", type=float, default=0.6)
    p.add_argument("--sweep_steps", type=int, default=0)
    p.add_argument("--k_folds", type=int, default=1, help="Grouped K-fold by image_id.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    records = discover_image_records(args.data_root)
    gt_map = {r.image_id: r.points for r in records}
    pred_map_raw = load_predictions(args.pred_csv)
    image_ids = sorted(gt_map.keys())
    folds = build_grouped_folds(image_ids, args.k_folds, args.seed)
    thresholds = parse_thresholds(
        threshold=args.threshold,
        threshold_sweep=args.threshold_sweep,
        sweep_start=args.sweep_start,
        sweep_end=args.sweep_end,
        sweep_steps=args.sweep_steps,
    )

    best_row = None
    best_macro = -1.0
    for thr in thresholds:
        pred_map = filter_predictions_by_threshold(pred_map_raw, threshold=thr)
        if len(folds) == 1:
            metrics = evaluate_subset(gt_map, pred_map, args.match_dist, image_ids)
            print(f"threshold={thr:.6f}")
            print_metrics_block("all", metrics["all"])
            print_metrics_block("class_6nm", metrics["6nm"])
            print_metrics_block("class_12nm", metrics["12nm"])
            print(f"macro.f1={metrics['macro'].f1:.4f}")
            macro_f1 = metrics["macro"].f1
        else:
            fold_macros: List[float] = []
            fold_f1_all: List[float] = []
            for i, fold_ids in enumerate(folds):
                metrics = evaluate_subset(gt_map, pred_map, args.match_dist, fold_ids)
                fold_macros.append(metrics["macro"].f1)
                fold_f1_all.append(metrics["all"].f1)
                print(
                    f"threshold={thr:.6f} fold={i + 1}/{len(folds)} "
                    f"n_images={len(fold_ids)} all_f1={metrics['all'].f1:.4f} "
                    f"macro_f1={metrics['macro'].f1:.4f}"
                )
            macro_f1 = float(np.mean(fold_macros))
            print(
                f"threshold={thr:.6f} grouped_cv_mean_all_f1={float(np.mean(fold_f1_all)):.4f} "
                f"grouped_cv_mean_macro_f1={macro_f1:.4f}"
            )

        if macro_f1 > best_macro:
            best_macro = macro_f1
            best_row = (thr, macro_f1)

    if best_row is not None and len(thresholds) > 1:
        print(f"best_threshold={best_row[0]:.6f} best_macro_f1={best_row[1]:.4f}")


if __name__ == "__main__":
    main()
