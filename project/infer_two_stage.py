from __future__ import annotations

import argparse
import csv
import os
from typing import List, Tuple

import numpy as np
import tifffile
import torch
import torch.nn.functional as F

from infer_detector import image_to_chw_01, peak_detect, tiled_inference
from model_refiner import PatchRefinerCNN
from model_unet import UNetKeypointDetector
from prepare_labels import discover_image_records


def _extract_patch(chw: np.ndarray, x: float, y: float, patch_size: int) -> np.ndarray:
    _, h, w = chw.shape
    r = patch_size // 2
    xc = int(round(float(x)))
    yc = int(round(float(y)))
    x0, x1 = xc - r, xc + r
    y0, y1 = yc - r, yc + r
    out = np.zeros((chw.shape[0], patch_size, patch_size), dtype=np.float32)
    sx0, sx1 = max(0, x0), min(w, x1)
    sy0, sy1 = max(0, y0), min(h, y1)
    if sx1 <= sx0 or sy1 <= sy0:
        return out
    dx0, dy0 = sx0 - x0, sy0 - y0
    dx1, dy1 = dx0 + (sx1 - sx0), dy0 + (sy1 - sy0)
    out[:, dy0:dy1, dx0:dx1] = chw[:, sy0:sy1, sx0:sx1]
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Two-stage inference: heatmap proposals + refiner classifier.")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--heatmap_ckpt", type=str, required=True)
    p.add_argument("--refiner_ckpt", type=str, required=True)
    p.add_argument("--out_csv", type=str, default="predictions_two_stage.csv")
    p.add_argument("--base_channels", type=int, default=24)
    p.add_argument("--tile_h", type=int, default=384)
    p.add_argument("--tile_w", type=int, default=384)
    p.add_argument("--stride_h", type=int, default=288)
    p.add_argument("--stride_w", type=int, default=288)
    p.add_argument("--proposal_threshold", type=float, default=0.10)
    p.add_argument("--proposal_min_distance", type=int, default=5)
    p.add_argument("--refiner_patch_size", type=int, default=33)
    p.add_argument("--refiner_keep_threshold", type=float, default=0.75)
    p.add_argument("--refiner_batch_size", type=int, default=512)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    heatmap_model = UNetKeypointDetector(in_channels=3, out_channels=2, base_channels=args.base_channels).to(device)
    heatmap_model.load_state_dict(torch.load(args.heatmap_ckpt, map_location=device))
    heatmap_model.eval()

    refiner = PatchRefinerCNN(in_channels=3, num_classes=3, base_channels=32).to(device)
    refiner.load_state_dict(torch.load(args.refiner_ckpt, map_location=device))
    refiner.eval()

    records = discover_image_records(args.data_root)
    rows: List[List[str]] = [["image_id", "x", "y", "class_id", "confidence"]]

    with torch.no_grad():
        for r in records:
            img = tifffile.imread(r.image_path)
            chw = image_to_chw_01(img)
            pred = tiled_inference(
                heatmap_model,
                chw,
                (args.tile_h, args.tile_w),
                (args.stride_h, args.stride_w),
                device,
            )

            proposals: List[Tuple[float, float, int, float]] = []
            for cls in [0, 1]:
                dets = peak_detect(
                    pred[cls],
                    threshold=args.proposal_threshold,
                    min_distance=args.proposal_min_distance,
                )
                for x, y, conf in dets:
                    proposals.append((x, y, cls, conf))

            if not proposals:
                continue

            patch_buf: List[np.ndarray] = []
            proposal_buf: List[Tuple[float, float, int, float]] = []
            for x, y, cls, conf in proposals:
                patch = _extract_patch(chw, x, y, args.refiner_patch_size)
                patch_buf.append(patch)
                proposal_buf.append((x, y, cls, conf))

            for i in range(0, len(patch_buf), args.refiner_batch_size):
                pb = patch_buf[i : i + args.refiner_batch_size]
                meta = proposal_buf[i : i + args.refiner_batch_size]
                x_batch = torch.from_numpy(np.stack(pb, axis=0)).float().to(device)
                logits = refiner(x_batch)
                probs = F.softmax(logits, dim=1).cpu().numpy()
                for (x, y, coarse_cls, coarse_conf), pr in zip(meta, probs):
                    refined_label = int(np.argmax(pr))  # 0=bg, 1=6nm, 2=12nm
                    refined_conf = float(pr[refined_label])
                    if refined_label == 0:
                        continue
                    if refined_conf < args.refiner_keep_threshold:
                        continue
                    class_id = refined_label - 1
                    final_conf = float(coarse_conf * refined_conf)
                    rows.append([r.image_id, f"{x:.2f}", f"{y:.2f}", str(class_id), f"{final_conf:.6f}"])

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"Saved detections to {args.out_csv}")


if __name__ == "__main__":
    main()

