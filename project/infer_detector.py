from __future__ import annotations

import argparse
import csv
import os
from typing import List, Tuple

import numpy as np
import tifffile
import torch

from model_unet import UNetKeypointDetector
from model_unet_deep import UNetDeepKeypointDetector
from prepare_labels import ID_TO_CLASS, discover_image_records


def image_to_chw_01(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    img = image.astype(np.float32)
    mn, mx = float(img.min()), float(img.max())
    if mx > mn:
        img = (img - mn) / (mx - mn)
    else:
        img = np.zeros_like(img, dtype=np.float32)
    return np.transpose(img, (2, 0, 1))


def tiled_inference(
    model: torch.nn.Module, image_chw: np.ndarray, tile_hw: Tuple[int, int], stride_hw: Tuple[int, int], device: torch.device
) -> np.ndarray:
    c, h, w = image_chw.shape
    th, tw = tile_hw
    sh, sw = stride_hw
    out = np.zeros((2, h, w), dtype=np.float32)
    cnt = np.zeros((1, h, w), dtype=np.float32)

    ys = list(range(0, max(1, h - th + 1), sh))
    xs = list(range(0, max(1, w - tw + 1), sw))
    if not ys or ys[-1] != h - th:
        ys.append(max(0, h - th))
    if not xs or xs[-1] != w - tw:
        xs.append(max(0, w - tw))

    model.eval()
    with torch.no_grad():
        for y0 in ys:
            for x0 in xs:
                patch = image_chw[:, y0 : y0 + th, x0 : x0 + tw]
                t = torch.from_numpy(patch[None]).float().to(device)
                pred = torch.sigmoid(model(t))[0].cpu().numpy()
                out[:, y0 : y0 + th, x0 : x0 + tw] += pred
                cnt[:, y0 : y0 + th, x0 : x0 + tw] += 1.0
    out /= np.maximum(cnt, 1e-6)
    return out


def peak_detect(
    heatmap: np.ndarray,
    threshold: float = 0.5,
    min_distance: int = 5,
    max_peaks: int = 2000,
) -> List[Tuple[float, float, float]]:
    """
    Greedy NMS on candidate pixels above threshold.
    This avoids plateau explosions that can produce extreme FP counts.
    """
    h, w = heatmap.shape
    candidates = np.where(heatmap >= float(threshold))
    ys = candidates[0]
    xs = candidates[1]
    if len(xs) == 0:
        return []

    scores = heatmap[ys, xs]
    order = np.argsort(scores)[::-1]
    suppressed = np.zeros((h, w), dtype=bool)
    r = int(max(1, min_distance))
    dets: List[Tuple[float, float, float]] = []

    for idx in order:
        y = int(ys[idx])
        x = int(xs[idx])
        if suppressed[y, x]:
            continue
        conf = float(heatmap[y, x])
        dets.append((float(x), float(y), conf))
        if 0 < max_peaks <= len(dets):
            break
        y0, y1 = max(0, y - r), min(h, y + r + 1)
        x0, x1 = max(0, x - r), min(w, x + r + 1)
        suppressed[y0:y1, x0:x1] = True
    return dets


def main() -> None:
    p = argparse.ArgumentParser(description="Infer keypoint detections from trained heatmap detector.")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--out_csv", type=str, default="predictions.csv")
    p.add_argument("--out_vis_dir", type=str, default="pred_vis")
    p.add_argument("--tile_h", type=int, default=512)
    p.add_argument("--tile_w", type=int, default=512)
    p.add_argument("--stride_h", type=int, default=384)
    p.add_argument("--stride_w", type=int, default=384)
    p.add_argument("--base_channels", type=int, default=32)
    p.add_argument("--model_type", type=str, default="unet_deep", choices=["unet", "unet_deep"])
    p.add_argument("--threshold", type=float, default=0.1)
    p.add_argument("--min_distance", type=int, default=5)
    p.add_argument("--max_detections_per_class", type=int, default=2000)
    p.add_argument("--use_mantis", action="store_true", help="Apply Mantis local contrast preprocessing")
    p.add_argument("--save_vis", action="store_true", help="Save per-image detection overlays.")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model_type == "unet_deep":
        model = UNetDeepKeypointDetector(in_channels=3, out_channels=2, base_channels=args.base_channels).to(device)
    else:
        model = UNetKeypointDetector(in_channels=3, out_channels=2, base_channels=args.base_channels).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    print(f"Loaded {args.model_type} model from {args.checkpoint}")

    # Setup Mantis preprocessing if requested
    mantis_filter = None
    if args.use_mantis:
        from augmentations import MantisLocalContrast
        mantis_filter = MantisLocalContrast(kernel_sigma=15.0, strength=0.5)
        print("Mantis local contrast preprocessing enabled")

    if args.save_vis:
        import matplotlib.pyplot as plt

        os.makedirs(args.out_vis_dir, exist_ok=True)
    records = discover_image_records(args.data_root)

    rows: List[List[str]] = [["image_id", "x", "y", "class_id", "confidence"]]
    for r in records:
        img = tifffile.imread(r.image_path)
        chw = image_to_chw_01(img)
        if mantis_filter is not None:
            dummy_hm = np.zeros((2, chw.shape[1], chw.shape[2]), dtype=np.float32)
            chw, _ = mantis_filter(chw, dummy_hm)
        pred = tiled_inference(model, chw, (args.tile_h, args.tile_w), (args.stride_h, args.stride_w), device)

        dets_all = []
        for cls in [0, 1]:
            dets = peak_detect(
                pred[cls],
                threshold=args.threshold,
                min_distance=args.min_distance,
                max_peaks=args.max_detections_per_class,
            )
            for x, y, conf in dets:
                rows.append([r.image_id, f"{x:.2f}", f"{y:.2f}", str(cls), f"{conf:.4f}"])
                dets_all.append((x, y, cls))

        if args.save_vis:
            vis = np.transpose(chw, (1, 2, 0))
            plt.figure(figsize=(7, 7))
            plt.imshow(vis)
            for x, y, cls in dets_all:
                color = "cyan" if cls == 0 else "magenta"
                plt.scatter([x], [y], s=10, c=color)
            plt.title(f"{r.image_id} detections")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(args.out_vis_dir, f"{r.image_id}_detections.png"), dpi=150)
            plt.close()

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"Saved detections to {args.out_csv}")


if __name__ == "__main__":
    main()
