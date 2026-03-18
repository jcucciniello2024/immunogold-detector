"""Complete evaluation pipeline: inference + metrics on test set."""
import os
import sys
import csv
import numpy as np
import torch
import tifffile
from scipy import ndimage

sys.path.insert(0, os.path.dirname(__file__))

from dataset_points import _to_chw_01
from model_unet_deep import UNetDeepKeypointDetector
from prepare_labels import discover_image_records

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Load latest best model
    model = UNetDeepKeypointDetector(in_channels=3, out_channels=2, base_channels=32).to(device)

    # Find the latest checkpoint directory
    checkpoint_dirs = sorted([d for d in os.listdir("checkpoints") if os.path.isdir(f"checkpoints/{d}")])
    if not checkpoint_dirs:
        print("ERROR: No checkpoints found!")
        return

    latest_dir = checkpoint_dirs[-1]
    model_path = f"checkpoints/{latest_dir}/detector_best.pt"

    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded model from: {model_path}\n")

    # Discover records and split into train/val/test
    records = discover_image_records("data/Max Planck Data/Gold Particle Labelling/analyzed synapses")
    print(f"Discovered {len(records)} images")

    rng = np.random.default_rng(42)
    idx = np.arange(len(records))
    rng.shuffle(idx)
    n_train = max(1, int(round(len(records) * 0.7)))
    n_val = max(1, int(round(len(records) * 0.15)))

    test_idx = idx[n_train + n_val:]
    test_records = [records[i] for i in test_idx]
    print(f"Test set: {len(test_records)} images\n")

    # Ground truth
    gt_map = {}
    for rec in records:
        gt_map[rec.image_id] = {
            0: rec.points[0].astype(np.float32),  # 6nm
            1: rec.points[1].astype(np.float32),  # 12nm
        }

    # Generate predictions
    print("Running inference on test set...")
    predictions = []
    pred_map = {}

    with torch.no_grad():
        for i, rec in enumerate(test_records):
            try:
                img = _to_chw_01(tifffile.imread(rec.image_path))
                img_t = torch.from_numpy(img).float().unsqueeze(0).to(device)
                logits = model(img_t)
                probs = torch.sigmoid(logits)

                pred_map[rec.image_id] = {0: [], 1: []}

                # Peak detection per class
                for cls_id in [0, 1]:
                    hm = probs[0, cls_id].cpu().numpy()

                    # Local maxima with threshold
                    local_max = ndimage.maximum_filter(hm, size=5) == hm
                    peaks = np.argwhere(local_max & (hm > 0.2))

                    for y, x in peaks:
                        conf = float(hm[y, x])
                        predictions.append({
                            'image_id': rec.image_id,
                            'class_id': cls_id,
                            'x': float(x),
                            'y': float(y),
                            'confidence': conf
                        })
                        pred_map[rec.image_id][cls_id].append((float(x), float(y), conf))

                if (i + 1) % max(1, len(test_records) // 5) == 0:
                    print(f"  {i+1}/{len(test_records)} images processed")

            except Exception as e:
                print(f"  ERROR processing {rec.image_id}: {e}")

    print(f"\nGenerated {len(predictions)} predictions\n")

    # Save predictions CSV
    pred_csv = f"predictions_{latest_dir}.csv"
    with open(pred_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['image_id', 'class_id', 'x', 'y', 'confidence'])
        writer.writeheader()
        writer.writerows(predictions)
    print(f"Saved predictions to: {pred_csv}")

    # Calculate metrics
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60 + "\n")

    match_dist = 15.0  # pixels

    for cls_id, cls_name in [(0, "6nm particles"), (1, "12nm particles")]:
        tp = fp = fn = 0
        loc_errors = []

        for image_id in [r.image_id for r in test_records]:
            gt_pts = gt_map.get(image_id, {}).get(cls_id, np.zeros((0, 2)))
            pred_pts = pred_map.get(image_id, {}).get(cls_id, [])

            if len(gt_pts) == 0:
                fp += len(pred_pts)
                continue

            # Greedy matching
            used = np.zeros(len(pred_pts), dtype=bool)
            for g in gt_pts:
                if len(pred_pts) == 0:
                    continue
                dists = np.sqrt(((np.array([p[:2] for p in pred_pts]) - g) ** 2).sum(axis=1))
                dists[used] = 1e9
                j = int(np.argmin(dists))
                if dists[j] < match_dist:
                    used[j] = True
                    tp += 1
                    loc_errors.append(float(dists[j]))

            fp += int((~used).sum())
            fn += int(len(gt_pts) - tp)

        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-8, precision + recall)
        mean_loc_err = float(np.mean(loc_errors)) if loc_errors else np.nan

        print(f"{cls_name}:")
        print(f"  TP: {tp}, FP: {fp}, FN: {fn}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  Mean Localization Error: {mean_loc_err:.2f} px")
        print()

    # Overall metrics
    total_tp = total_fp = total_fn = 0
    all_loc_errors = []

    for cls_id in [0, 1]:
        tp = fp = fn = 0
        for image_id in [r.image_id for r in test_records]:
            gt_pts = gt_map.get(image_id, {}).get(cls_id, np.zeros((0, 2)))
            pred_pts = pred_map.get(image_id, {}).get(cls_id, [])

            if len(gt_pts) == 0:
                fp += len(pred_pts)
                continue

            used = np.zeros(len(pred_pts), dtype=bool)
            for g in gt_pts:
                if len(pred_pts) == 0:
                    continue
                dists = np.sqrt(((np.array([p[:2] for p in pred_pts]) - g) ** 2).sum(axis=1))
                dists[used] = 1e9
                j = int(np.argmin(dists))
                if dists[j] < match_dist:
                    used[j] = True
                    tp += 1
                    all_loc_errors.append(float(dists[j]))

            fp += int((~used).sum())
            fn += int(len(gt_pts) - tp)

        total_tp += tp
        total_fp += fp
        total_fn += fn

    overall_precision = total_tp / max(1, total_tp + total_fp)
    overall_recall = total_tp / max(1, total_tp + total_fn)
    overall_f1 = 2 * overall_precision * overall_recall / max(1e-8, overall_precision + overall_recall)
    mean_loc_all = float(np.mean(all_loc_errors)) if all_loc_errors else np.nan

    print("-" * 60)
    print(f"OVERALL (both particle types):")
    print(f"  TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")
    print(f"  Precision: {overall_precision:.4f}")
    print(f"  Recall:    {overall_recall:.4f}")
    print(f"  F1 Score:  {overall_f1:.4f}")
    print(f"  Mean Localization Error: {mean_loc_all:.2f} px")
    print("=" * 60)

    # Save results
    results_file = f"eval_results_{latest_dir}.txt"
    with open(results_file, 'w') as f:
        f.write("EVALUATION RESULTS\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Test set: {len(test_records)} images\n")
        f.write(f"Predictions: {pred_csv}\n\n")
        f.write(f"Overall F1: {overall_f1:.4f}\n")
        f.write(f"Overall Precision: {overall_precision:.4f}\n")
        f.write(f"Overall Recall: {overall_recall:.4f}\n")
        f.write(f"Mean Localization Error: {mean_loc_all:.2f} px\n")

    print(f"\nResults saved to: {results_file}")

if __name__ == "__main__":
    main()
