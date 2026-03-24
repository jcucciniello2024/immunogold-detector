"""
Sanity check script — run this BEFORE the full training job.
Validates that data loads correctly, heatmaps are in the right place,
and the model can overfit on a single image.

Usage:
  python sanity_check.py --data_root "path/to/analyzed synapses"

Expected output:
  1. Prints image sizes and particle counts
  2. Saves ground truth heatmap overlay images to sanity_check_output/
  3. Trains for 50 epochs on 1 image and prints whether pred_max > 0.1
  4. Saves predicted heatmap overlay

If pred_max stays near 0 after 50 epochs, something is fundamentally broken.
If the GT overlay shows dots in the wrong place, labels are bad.
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

# Ensure project dir is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prepare_labels import discover_image_records, gaussian_heatmap
from model_unet_deep import UNetDeepKeypointDetector
from augmentations import MantisLocalContrast
import tifffile


def to_chw_01(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    image = image.astype(np.float32)
    mn, mx = float(image.min()), float(image.max())
    if mx > mn:
        image = (image - mn) / (mx - mn)
    return np.transpose(image, (2, 0, 1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="sanity_check_output")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ================================================================
    # CHECK 1: Data loading
    # ================================================================
    print("=" * 60)
    print("CHECK 1: Data Loading")
    print("=" * 60)

    records = discover_image_records(args.data_root)
    print(f"Found {len(records)} images")

    if len(records) == 0:
        print("FATAL: No images found! Check data_root path.")
        return

    total_6nm = 0
    total_12nm = 0
    for r in records:
        n6 = len(r.points[0])
        n12 = len(r.points[1])
        total_6nm += n6
        total_12nm += n12
        print(f"  {r.image_id}: {r.width}x{r.height}, 6nm={n6}, 12nm={n12}")

    print(f"\nTotal: {total_6nm} 6nm + {total_12nm} 12nm = {total_6nm + total_12nm} particles")

    if total_6nm + total_12nm == 0:
        print("FATAL: No particles found! Check CSV label files.")
        return

    # ================================================================
    # CHECK 2: Ground truth heatmap visualization
    # ================================================================
    print("\n" + "=" * 60)
    print("CHECK 2: Ground Truth Heatmap Visualization")
    print("=" * 60)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mantis = MantisLocalContrast(kernel_sigma=15.0, strength=0.5)

    for r in records[:3]:  # First 3 images
        img = tifffile.imread(r.image_path)
        chw = to_chw_01(img)

        # Apply Mantis
        dummy = np.zeros((2, chw.shape[1], chw.shape[2]), dtype=np.float32)
        chw_mantis, _ = mantis(chw, dummy)

        # Generate heatmaps
        hm6 = gaussian_heatmap((r.height, r.width), r.points[0], sigma=1.5)
        hm12 = gaussian_heatmap((r.height, r.width), r.points[1], sigma=1.5)

        # Visualize
        fig, axes = plt.subplots(1, 3, figsize=(21, 7))

        # Original image with particle locations
        vis = np.transpose(chw, (1, 2, 0))
        axes[0].imshow(vis)
        if len(r.points[0]) > 0:
            axes[0].scatter(r.points[0][:, 0], r.points[0][:, 1], s=20, c="cyan", marker="+", linewidths=0.5)
        if len(r.points[1]) > 0:
            axes[0].scatter(r.points[1][:, 0], r.points[1][:, 1], s=30, c="magenta", marker="+", linewidths=0.5)
        axes[0].set_title(f"{r.image_id} - Raw + Labels")

        # Mantis-enhanced
        vis_m = np.transpose(chw_mantis, (1, 2, 0))
        axes[1].imshow(vis_m)
        if len(r.points[0]) > 0:
            axes[1].scatter(r.points[0][:, 0], r.points[0][:, 1], s=20, c="cyan", marker="+", linewidths=0.5)
        if len(r.points[1]) > 0:
            axes[1].scatter(r.points[1][:, 0], r.points[1][:, 1], s=30, c="magenta", marker="+", linewidths=0.5)
        axes[1].set_title(f"{r.image_id} - Mantis Enhanced")

        # Heatmap
        combined_hm = np.maximum(hm6, hm12)
        axes[2].imshow(vis, alpha=0.7)
        axes[2].imshow(combined_hm, cmap="hot", alpha=0.5)
        axes[2].set_title(f"{r.image_id} - GT Heatmap (sigma=1.5)")

        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        path = os.path.join(args.out_dir, f"gt_{r.image_id}.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved: {path}")

        # Check if particles are within image bounds
        for cls_name, cls_id in [("6nm", 0), ("12nm", 1)]:
            pts = r.points[cls_id]
            if len(pts) == 0:
                continue
            oob = ((pts[:, 0] < 0) | (pts[:, 0] >= r.width) |
                   (pts[:, 1] < 0) | (pts[:, 1] >= r.height))
            if oob.any():
                print(f"  WARNING: {oob.sum()} {cls_name} points out of bounds in {r.image_id}!")
            else:
                print(f"  OK: All {cls_name} points within bounds")

    # ================================================================
    # CHECK 3: Overfit on 1 image
    # ================================================================
    print("\n" + "=" * 60)
    print("CHECK 3: Overfit Test (50 epochs on 1 image)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = UNetDeepKeypointDetector(in_channels=3, out_channels=2, base_channels=32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    # Use the record with most particles
    best_rec = max(records, key=lambda r: len(r.points[0]) + len(r.points[1]))
    print(f"Using: {best_rec.image_id} ({len(best_rec.points[0])} 6nm + {len(best_rec.points[1])} 12nm)")

    img = tifffile.imread(best_rec.image_path)
    chw = to_chw_01(img)
    dummy = np.zeros((2, chw.shape[1], chw.shape[2]), dtype=np.float32)
    chw, _ = mantis(chw, dummy)

    # Generate target heatmap
    hm = np.zeros((2, best_rec.height, best_rec.width), dtype=np.float32)
    hm[0] = gaussian_heatmap((best_rec.height, best_rec.width), best_rec.points[0], sigma=1.5)
    hm[1] = gaussian_heatmap((best_rec.height, best_rec.width), best_rec.points[1], sigma=1.5)

    # Extract a patch centered on a particle
    if len(best_rec.points[0]) > 0:
        cx, cy = int(best_rec.points[0][0, 0]), int(best_rec.points[0][0, 1])
    else:
        cx, cy = int(best_rec.points[1][0, 0]), int(best_rec.points[1][0, 1])

    # Clamp to valid region
    ps = 256
    x0 = max(0, min(cx - ps // 2, best_rec.width - ps))
    y0 = max(0, min(cy - ps // 2, best_rec.height - ps))

    patch_img = torch.from_numpy(chw[:, y0:y0+ps, x0:x0+ps].copy()).unsqueeze(0).float().to(device)
    patch_hm = torch.from_numpy(hm[:, y0:y0+ps, x0:x0+ps].copy()).unsqueeze(0).float().to(device)

    hm_max_gt = float(patch_hm.max())
    print(f"Patch region: ({x0},{y0}) to ({x0+ps},{y0+ps})")
    print(f"GT heatmap max in patch: {hm_max_gt:.4f}")

    if hm_max_gt < 0.01:
        print("WARNING: No particles in this patch! Trying another location...")
        # Try to find a patch with particles
        all_pts = np.concatenate([best_rec.points[0], best_rec.points[1]], axis=0)
        for pt in all_pts:
            cx, cy = int(pt[0]), int(pt[1])
            x0 = max(0, min(cx - ps // 2, best_rec.width - ps))
            y0 = max(0, min(cy - ps // 2, best_rec.height - ps))
            patch_hm_test = hm[:, y0:y0+ps, x0:x0+ps]
            if patch_hm_test.max() > 0.5:
                patch_img = torch.from_numpy(chw[:, y0:y0+ps, x0:x0+ps].copy()).unsqueeze(0).float().to(device)
                patch_hm = torch.from_numpy(patch_hm_test.copy()).unsqueeze(0).float().to(device)
                print(f"Found good patch at ({x0},{y0}), hm_max={patch_hm_test.max():.4f}")
                break

    # Focal BCE loss with high pos_weight
    from train_detector import FocalBCELoss
    criterion = FocalBCELoss(pos_weight=300.0, neg_weight=1.0, gamma=2.0)

    model.train()
    for epoch in range(1, 51):
        optimizer.zero_grad()
        logits = model(patch_img)
        loss = criterion(logits, patch_hm)
        loss.backward()
        optimizer.step()

        pred = torch.sigmoid(logits)
        pred_max = float(pred.max())
        pred_mean = float(pred.mean())

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}: loss={loss.item():.6f} pred_max={pred_max:.4f} pred_mean={pred_mean:.6f}")

    # Final check
    model.eval()
    with torch.no_grad():
        logits = model(patch_img)
        pred = torch.sigmoid(logits)
        pred_max = float(pred.max())

    print(f"\nFinal pred_max: {pred_max:.4f}")

    if pred_max > 0.3:
        print("PASS: Model can learn to predict non-zero. Training pipeline is working.")
    elif pred_max > 0.05:
        print("PARTIAL: Model is learning but slowly. May need more epochs or higher pos_weight.")
    else:
        print("FAIL: Model still predicting near-zero. Something is fundamentally wrong.")
        print("  Check: Are labels correct? Is the heatmap in the right location?")

    # Save predicted heatmap visualization
    pred_np = pred[0].cpu().numpy()
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    patch_vis = patch_img[0].cpu().numpy().transpose(1, 2, 0)
    gt_vis = patch_hm[0].cpu().numpy()

    axes[0].imshow(patch_vis)
    axes[0].set_title("Input Patch")
    axes[1].imshow(np.maximum(gt_vis[0], gt_vis[1]), cmap="hot")
    axes[1].set_title(f"GT Heatmap (max={hm_max_gt:.3f})")
    axes[2].imshow(np.maximum(pred_np[0], pred_np[1]), cmap="hot")
    axes[2].set_title(f"Predicted Heatmap (max={pred_max:.3f})")

    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    path = os.path.join(args.out_dir, "overfit_test.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

    print("\n" + "=" * 60)
    print("SANITY CHECK COMPLETE")
    print(f"All outputs in: {args.out_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
