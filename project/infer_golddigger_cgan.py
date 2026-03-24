from __future__ import annotations

import argparse
import csv
import os
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
    p.add_argument("--out_vis_dir", type=str, default="golddigger_vis")
    p.add_argument("--save_vis", action="store_true", help="Save EM + detection overlays per image.")
    p.add_argument(
        "--save_heatmap",
        action="store_true",
        help="Save predicted mask channels (6nm / 12nm) and max-projection heatmap.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GoldDiggerGenerator(in_channels=3, out_channels=2, base_channels=64).to(device)
    model.load_state_dict(torch.load(args.generator_ckpt, map_location=device))
    model.eval()

    if args.save_vis or args.save_heatmap:
        import matplotlib.pyplot as plt

        os.makedirs(args.out_vis_dir, exist_ok=True)

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

        if args.save_vis:
            import matplotlib.pyplot as plt

            vis = np.transpose(chw, (1, 2, 0))
            plt.figure(figsize=(7, 7))
            plt.imshow(vis)
            if det6:
                plt.scatter(
                    [d[0] for d in det6],
                    [d[1] for d in det6],
                    s=18,
                    c="cyan",
                    marker="x",
                    linewidths=0.9,
                    label="6 nm",
                )
            if det12:
                plt.scatter(
                    [d[0] for d in det12],
                    [d[1] for d in det12],
                    s=18,
                    c="magenta",
                    marker="+",
                    linewidths=0.9,
                    label="12 nm",
                )
            if det6 or det12:
                plt.legend(loc="upper right", fontsize=8)
            plt.title(f"{r.image_id} GoldDigger (cyan=6nm, magenta=12nm)")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(args.out_vis_dir, f"{r.image_id}_golddigger_detections.png"), dpi=150)
            plt.close()

        if args.save_heatmap:
            import matplotlib.pyplot as plt

            vis = np.transpose(chw, (1, 2, 0))
            p6, p12 = pred[0], pred[1]
            vmax = max(0.2, float(max(p6.max(), p12.max())))
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(vis, cmap="gray")
            axes[0].set_title("EM")
            axes[0].axis("off")
            im1 = axes[1].imshow(p6, cmap="magma", vmin=0.0, vmax=vmax)
            axes[1].set_title("6 nm channel (sigmoid)")
            axes[1].axis("off")
            fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
            im2 = axes[2].imshow(p12, cmap="magma", vmin=0.0, vmax=vmax)
            axes[2].set_title("12 nm channel (sigmoid)")
            axes[2].axis("off")
            fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
            fig.suptitle(f"{r.image_id}: Gold Digger cGAN masks")
            fig.tight_layout()
            fig.savefig(os.path.join(args.out_vis_dir, f"{r.image_id}_golddigger_heatmap.png"), dpi=150)
            plt.close(fig)

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"Saved detections to {args.out_csv}")
    if args.save_vis or args.save_heatmap:
        print(f"Saved visualizations under {args.out_vis_dir}/")


if __name__ == "__main__":
    main()
