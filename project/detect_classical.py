"""
Classical immunogold particle detector using Laplacian of Gaussians (LoG).

Zero training data needed. Detects dark circular blobs and classifies
by diameter into 6nm/12nm size classes.

Usage:
  python detect_classical.py --data_root "path/to/analyzed synapses"
  python detect_classical.py --data_root "path" --visualize --out_dir results/
"""

import argparse
import csv
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import tifffile
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_labels import discover_image_records


def mantis_local_contrast(image: np.ndarray, kernel_sigma: float = 15.0) -> np.ndarray:
    """Enhance local contrast — makes gold particles pop."""
    local_mean = ndimage.gaussian_filter(image, sigma=kernel_sigma)
    local_sq_mean = ndimage.gaussian_filter(image ** 2, sigma=kernel_sigma)
    local_std = np.sqrt(np.maximum(local_sq_mean - local_mean ** 2, 1e-8))
    enhanced = (image - local_mean) / (local_std + 1e-8)
    emin, emax = enhanced.min(), enhanced.max()
    if emax > emin:
        enhanced = (enhanced - emin) / (emax - emin)
    return enhanced


def log_blob_detect(
    image: np.ndarray,
    sigmas: List[float],
    threshold: float = 0.01,
    min_distance: int = 3,
) -> List[Tuple[float, float, float, float]]:
    """
    Laplacian of Gaussians blob detection for dark particles.

    Returns list of (x, y, sigma, response) for each detection.
    """
    h, w = image.shape

    # Compute LoG response at each scale
    log_stack = np.zeros((len(sigmas), h, w), dtype=np.float32)
    for i, sigma in enumerate(sigmas):
        # LoG = sigma^2 * (laplacian of gaussian-smoothed image)
        smoothed = ndimage.gaussian_filter(image, sigma=sigma)
        laplacian = ndimage.laplace(smoothed)
        # Normalize by sigma^2 for scale invariance
        log_stack[i] = -(sigma ** 2) * laplacian  # negative because particles are dark

    # Find peaks across all scales
    detections = []
    for i, sigma in enumerate(sigmas):
        response = log_stack[i]
        candidates = np.where(response > threshold)
        ys, xs = candidates

        if len(xs) == 0:
            continue

        scores = response[ys, xs]
        # Sort by score descending
        order = np.argsort(scores)[::-1]

        # Check that this scale is the best scale for each candidate
        for idx in order:
            y, x = int(ys[idx]), int(xs[idx])
            score = float(scores[idx])

            # Is this the best scale at this location?
            best_scale = True
            for j in range(len(sigmas)):
                if j != i and log_stack[j, y, x] > score:
                    best_scale = False
                    break
            if not best_scale:
                continue

            detections.append((float(x), float(y), sigma, score))

    # Greedy NMS
    if not detections:
        return []

    detections.sort(key=lambda d: d[3], reverse=True)
    suppressed = np.zeros((h, w), dtype=bool)
    r = max(1, min_distance)
    kept = []

    for x, y, sigma, score in detections:
        ix, iy = int(round(x)), int(round(y))
        if iy < 0 or iy >= h or ix < 0 or ix >= w:
            continue
        if suppressed[iy, ix]:
            continue
        kept.append((x, y, sigma, score))
        y0, y1 = max(0, iy - r), min(h, iy + r + 1)
        x0, x1 = max(0, ix - r), min(w, ix + r + 1)
        suppressed[y0:y1, x0:x1] = True

    return kept


def classify_by_diameter(
    detections: List[Tuple[float, float, float, float]],
    sigma_boundary: float = 1.8,
) -> Tuple[List[Tuple[float, float, float]], List[Tuple[float, float, float]]]:
    """
    Split detections into 6nm and 12nm classes based on detected sigma.

    sigma_boundary: sigmas below this → 6nm, above → 12nm
    """
    class_6nm = []
    class_12nm = []
    for x, y, sigma, score in detections:
        if sigma <= sigma_boundary:
            class_6nm.append((x, y, score))
        else:
            class_12nm.append((x, y, score))
    return class_6nm, class_12nm


def greedy_match(
    gt: np.ndarray, pred: List[Tuple[float, float, float]], max_dist: float
) -> Tuple[int, int, int]:
    """Match predictions to ground truth, return tp, fp, fn."""
    if len(gt) == 0:
        return 0, len(pred), 0
    if len(pred) == 0:
        return 0, 0, len(gt)

    pred_xy = np.array([[p[0], p[1]] for p in pred], dtype=np.float32)
    used = np.zeros(len(pred_xy), dtype=bool)
    tp = 0
    for g in gt:
        dist = np.sqrt(((pred_xy - g[None, :]) ** 2).sum(axis=1))
        dist[used] = 1e9
        j = int(np.argmin(dist))
        if dist[j] < max_dist:
            used[j] = True
            tp += 1
    fp = int((~used).sum())
    fn = int(len(gt) - tp)
    return tp, fp, fn


def main():
    p = argparse.ArgumentParser(description="Classical LoG immunogold particle detector")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--out_csv", type=str, default="predictions_classical.csv")
    p.add_argument("--out_dir", type=str, default="classical_results")
    p.add_argument("--threshold", type=float, default=0.005,
                   help="LoG response threshold (lower = more detections)")
    p.add_argument("--min_distance", type=int, default=3)
    p.add_argument("--sigma_boundary", type=float, default=1.8,
                   help="Sigma boundary between 6nm and 12nm classes")
    p.add_argument("--match_dist", type=float, default=5.0)
    p.add_argument("--visualize", action="store_true")
    # Sigma range to scan — covers expected particle sizes
    p.add_argument("--sigma_min", type=float, default=0.8)
    p.add_argument("--sigma_max", type=float, default=4.0)
    p.add_argument("--sigma_steps", type=int, default=12)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    records = discover_image_records(args.data_root)
    print(f"Found {len(records)} images")

    sigmas = np.linspace(args.sigma_min, args.sigma_max, args.sigma_steps).tolist()
    print(f"LoG sigmas: {[f'{s:.2f}' for s in sigmas]}")

    # CSV output
    csv_rows = [["image_id", "x", "y", "class_id", "confidence", "sigma"]]

    # Metrics accumulators
    total_tp = {0: 0, 1: 0}
    total_fp = {0: 0, 1: 0}
    total_fn = {0: 0, 1: 0}

    if args.visualize:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

    # ================================================================
    # THRESHOLD SWEEP — find best threshold automatically
    # ================================================================
    print("\n=== Threshold Sweep ===")
    thresholds_to_try = [0.001, 0.002, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.05]
    best_f1 = 0
    best_thresh = args.threshold

    for thresh in thresholds_to_try:
        sweep_tp, sweep_fp, sweep_fn = 0, 0, 0
        for rec in records:
            img = tifffile.imread(rec.image_path)
            if img.ndim == 3:
                img = img.mean(axis=2)
            img = img.astype(np.float32)
            mn, mx = img.min(), img.max()
            if mx > mn:
                img = (img - mn) / (mx - mn)

            enhanced = mantis_local_contrast(img)
            # Invert — LoG looks for bright blobs, particles are dark
            inverted = 1.0 - enhanced

            dets = log_blob_detect(inverted, sigmas, threshold=thresh, min_distance=args.min_distance)
            det_6nm, det_12nm = classify_by_diameter(dets, args.sigma_boundary)

            tp6, fp6, fn6 = greedy_match(rec.points[0], det_6nm, args.match_dist)
            tp12, fp12, fn12 = greedy_match(rec.points[1], det_12nm, args.match_dist)
            sweep_tp += tp6 + tp12
            sweep_fp += fp6 + fp12
            sweep_fn += fn6 + fn12

        prec = sweep_tp / max(1, sweep_tp + sweep_fp)
        rec_val = sweep_tp / max(1, sweep_tp + sweep_fn)
        f1 = 2 * prec * rec_val / max(1e-8, prec + rec_val)
        print(f"  thresh={thresh:.4f}  tp={sweep_tp}  fp={sweep_fp}  fn={sweep_fn}  "
              f"P={prec:.4f}  R={rec_val:.4f}  F1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    print(f"\nBest threshold: {best_thresh:.4f} (F1={best_f1:.4f})")

    # ================================================================
    # RUN WITH BEST THRESHOLD
    # ================================================================
    print(f"\n=== Running with threshold={best_thresh:.4f} ===\n")

    for rec in records:
        img = tifffile.imread(rec.image_path)
        if img.ndim == 3:
            img = img.mean(axis=2)
        img = img.astype(np.float32)
        mn, mx = img.min(), img.max()
        if mx > mn:
            img = (img - mn) / (mx - mn)

        enhanced = mantis_local_contrast(img)
        inverted = 1.0 - enhanced

        dets = log_blob_detect(inverted, sigmas, threshold=best_thresh, min_distance=args.min_distance)
        det_6nm, det_12nm = classify_by_diameter(dets, args.sigma_boundary)

        # Evaluate
        tp6, fp6, fn6 = greedy_match(rec.points[0], det_6nm, args.match_dist)
        tp12, fp12, fn12 = greedy_match(rec.points[1], det_12nm, args.match_dist)
        total_tp[0] += tp6; total_fp[0] += fp6; total_fn[0] += fn6
        total_tp[1] += tp12; total_fp[1] += fp12; total_fn[1] += fn12

        n_det = len(det_6nm) + len(det_12nm)
        n_gt = len(rec.points[0]) + len(rec.points[1])
        print(f"  {rec.image_id}: detected={n_det} gt={n_gt} "
              f"6nm(tp={tp6} fp={fp6} fn={fn6}) "
              f"12nm(tp={tp12} fp={fp12} fn={fn12})")

        # Save to CSV
        for x, y, score in det_6nm:
            sigma_val = next((d[2] for d in dets if d[0] == x and d[1] == y), 0)
            csv_rows.append([rec.image_id, f"{x:.2f}", f"{y:.2f}", "0", f"{score:.4f}", f"{sigma_val:.2f}"])
        for x, y, score in det_12nm:
            sigma_val = next((d[2] for d in dets if d[0] == x and d[1] == y), 0)
            csv_rows.append([rec.image_id, f"{x:.2f}", f"{y:.2f}", "1", f"{score:.4f}", f"{sigma_val:.2f}"])

        # Visualize
        if args.visualize:
            fig, axes = plt.subplots(1, 3, figsize=(21, 7))

            axes[0].imshow(img, cmap="gray")
            axes[0].set_title(f"{rec.image_id} — Raw")

            axes[1].imshow(enhanced, cmap="gray")
            axes[1].set_title("Mantis Enhanced")

            axes[2].imshow(enhanced, cmap="gray")
            # GT points
            if len(rec.points[0]) > 0:
                axes[2].scatter(rec.points[0][:, 0], rec.points[0][:, 1],
                               s=40, facecolors="none", edgecolors="lime", linewidths=0.8, label="GT 6nm")
            if len(rec.points[1]) > 0:
                axes[2].scatter(rec.points[1][:, 0], rec.points[1][:, 1],
                               s=60, facecolors="none", edgecolors="green", linewidths=0.8, label="GT 12nm")
            # Detected points
            if det_6nm:
                dx = [d[0] for d in det_6nm]
                dy = [d[1] for d in det_6nm]
                axes[2].scatter(dx, dy, s=20, c="cyan", marker="+", linewidths=0.7, label=f"Det 6nm ({len(det_6nm)})")
            if det_12nm:
                dx = [d[0] for d in det_12nm]
                dy = [d[1] for d in det_12nm]
                axes[2].scatter(dx, dy, s=30, c="magenta", marker="+", linewidths=0.7, label=f"Det 12nm ({len(det_12nm)})")

            axes[2].legend(fontsize=8)
            axes[2].set_title(f"Detections (tp6={tp6} tp12={tp12})")

            for ax in axes:
                ax.axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(args.out_dir, f"{rec.image_id}_classical.png"), dpi=150)
            plt.close()

    # ================================================================
    # FINAL METRICS
    # ================================================================
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    for cls_name, cls_id in [("6nm", 0), ("12nm", 1)]:
        tp = total_tp[cls_id]
        fp = total_fp[cls_id]
        fn = total_fn[cls_id]
        prec = tp / max(1, tp + fp)
        rec_val = tp / max(1, tp + fn)
        f1 = 2 * prec * rec_val / max(1e-8, prec + rec_val)
        print(f"  {cls_name}: tp={tp} fp={fp} fn={fn} P={prec:.4f} R={rec_val:.4f} F1={f1:.4f}")

    all_tp = total_tp[0] + total_tp[1]
    all_fp = total_fp[0] + total_fp[1]
    all_fn = total_fn[0] + total_fn[1]
    all_prec = all_tp / max(1, all_tp + all_fp)
    all_rec = all_tp / max(1, all_tp + all_fn)
    all_f1 = 2 * all_prec * all_rec / max(1e-8, all_prec + all_rec)
    macro_f1 = 0.5 * (
        2 * total_tp[0] / max(1, 2 * total_tp[0] + total_fp[0] + total_fn[0]) +
        2 * total_tp[1] / max(1, 2 * total_tp[1] + total_fp[1] + total_fn[1])
    )

    print(f"\n  ALL:  tp={all_tp} fp={all_fp} fn={all_fn} P={all_prec:.4f} R={all_rec:.4f} F1={all_f1:.4f}")
    print(f"  Macro F1: {macro_f1:.4f}")

    # Save CSV
    csv_path = os.path.join(args.out_dir, args.out_csv)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)
    print(f"\nPredictions saved to: {csv_path}")
    print(f"Visualizations in: {args.out_dir}/")


if __name__ == "__main__":
    main()
