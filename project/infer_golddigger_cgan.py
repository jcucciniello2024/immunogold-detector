from __future__ import annotations

import argparse
import csv
from typing import List, Tuple

import numpy as np
import tifffile
import torch
from scipy.ndimage import center_of_mass, label

from infer_detector import image_to_chw_01, tiled_inference
from model_golddigger_cgan import GoldDiggerGenerator
from prepare_labels import discover_image_records


def components_to_points(
    prob_map: np.ndarray,
    threshold: float,
    min_area: int,
    max_area: int,
) -> List[Tuple[float, float, float]]:
    bw = prob_map >= float(threshold)
    cc, n = label(bw)
    if n <= 0:
        return []
    out: List[Tuple[float, float, float]] = []
    for i in range(1, n + 1):
        m = cc == i
        area = int(m.sum())
        if area < int(min_area):
            continue
        if max_area > 0 and area > int(max_area):
            continue
        cy, cx = center_of_mass(m.astype(np.float32))
        conf = float(prob_map[m].mean())
        out.append((float(cx), float(cy), conf))
    out.sort(key=lambda t: t[2], reverse=True)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inference for Gold Digger-style cGAN detector.")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--generator_ckpt", type=str, required=True)
    p.add_argument("--out_csv", type=str, default="predictions_golddigger_cgan.csv")
    p.add_argument("--tile_h", type=int, default=512)
    p.add_argument("--tile_w", type=int, default=512)
    p.add_argument("--stride_h", type=int, default=384)
    p.add_argument("--stride_w", type=int, default=384)
    p.add_argument("--threshold_6nm", type=float, default=0.35)
    p.add_argument("--threshold_12nm", type=float, default=0.35)
    p.add_argument("--min_area_6nm", type=int, default=4)
    p.add_argument("--max_area_6nm", type=int, default=150)
    p.add_argument("--min_area_12nm", type=int, default=8)
    p.add_argument("--max_area_12nm", type=int, default=250)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GoldDiggerGenerator(in_channels=3, out_channels=2, base_channels=64).to(device)
    model.load_state_dict(torch.load(args.generator_ckpt, map_location=device))
    model.eval()

    records = discover_image_records(args.data_root)
    rows: List[List[str]] = [["image_id", "x", "y", "class_id", "confidence"]]
    for r in records:
        img = tifffile.imread(r.image_path)
        chw = image_to_chw_01(img)
        pred = tiled_inference(
            model=model,
            image_chw=chw,
            tile_hw=(args.tile_h, args.tile_w),
            stride_hw=(args.stride_h, args.stride_w),
            device=device,
        )  # (2,H,W), already sigmoid in tiled_inference

        det6 = components_to_points(
            prob_map=pred[0],
            threshold=args.threshold_6nm,
            min_area=args.min_area_6nm,
            max_area=args.max_area_6nm,
        )
        det12 = components_to_points(
            prob_map=pred[1],
            threshold=args.threshold_12nm,
            min_area=args.min_area_12nm,
            max_area=args.max_area_12nm,
        )
        for x, y, conf in det6:
            rows.append([r.image_id, f"{x:.2f}", f"{y:.2f}", "0", f"{conf:.6f}"])
        for x, y, conf in det12:
            rows.append([r.image_id, f"{x:.2f}", f"{y:.2f}", "1", f"{conf:.6f}"])

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"Saved detections to {args.out_csv}")


if __name__ == "__main__":
    main()
