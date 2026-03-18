from __future__ import annotations

import argparse
import csv
from typing import Dict, List, Sequence, Tuple

import numpy as np
import tifffile
import torch
import torch.nn.functional as F

from dataset_guard import enforce_allowed_data_root
from log_detector import multiscale_log_candidates
from model_refiner import PatchRefinerCNN
from prepare_labels import discover_image_records


def _to_chw_01(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    image = image.astype(np.float32)
    mn, mx = float(image.min()), float(image.max())
    if mx > mn:
        image = (image - mn) / (mx - mn)
    else:
        image = np.zeros_like(image, dtype=np.float32)
    return np.transpose(image, (2, 0, 1))


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


def _nms_xy_conf(points: Sequence[Tuple[float, float, float]], min_distance: int) -> List[Tuple[float, float, float]]:
    out: List[Tuple[float, float, float]] = []
    md2 = float(min_distance * min_distance)
    for x, y, c in points:
        keep = True
        for ox, oy, _ in out:
            if (x - ox) * (x - ox) + (y - oy) * (y - oy) < md2:
                keep = False
                break
        if keep:
            out.append((x, y, c))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inference: LoG candidates + CNN classifier.")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--classifier_ckpt", type=str, required=True)
    p.add_argument("--out_csv", type=str, default="predictions_log_cnn.csv")
    p.add_argument("--sigmas", type=str, default="1.2,1.6,2.0,2.4,2.8")
    p.add_argument("--log_threshold", type=float, default=0.02)
    p.add_argument("--candidate_min_distance", type=int, default=5)
    p.add_argument("--max_candidates_per_image", type=int, default=600)
    p.add_argument("--patch_size", type=int, default=33)
    p.add_argument("--class_threshold", type=float, default=0.55)
    p.add_argument("--final_min_distance", type=int, default=5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.data_root = enforce_allowed_data_root(args.data_root)
    sigmas = [float(s.strip()) for s in args.sigmas.split(",") if s.strip()]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PatchRefinerCNN(in_channels=3, num_classes=3, base_channels=32).to(device)
    model.load_state_dict(torch.load(args.classifier_ckpt, map_location=device))
    model.eval()

    records = discover_image_records(args.data_root)
    rows: List[List[str]] = [["image_id", "x", "y", "class_id", "confidence"]]

    with torch.no_grad():
        for r in records:
            img = tifffile.imread(r.image_path)
            gray = img.mean(axis=2) if img.ndim == 3 else img.astype(np.float32)
            gray = gray.astype(np.float32)
            mn, mx = float(gray.min()), float(gray.max())
            if mx > mn:
                gray = (gray - mn) / (mx - mn)
            else:
                gray = np.zeros_like(gray, dtype=np.float32)

            chw = _to_chw_01(img)
            cands = multiscale_log_candidates(
                gray,
                sigmas=sigmas,
                threshold=args.log_threshold,
                min_distance=args.candidate_min_distance,
                max_candidates=args.max_candidates_per_image,
            )
            if not cands:
                continue

            patches = np.stack([_extract_patch(chw, x, y, args.patch_size) for x, y, _, _ in cands], axis=0)
            probs = F.softmax(model(torch.from_numpy(patches).float().to(device)), dim=1).cpu().numpy()

            by_class: Dict[int, List[Tuple[float, float, float]]] = {0: [], 1: []}
            for (x, y, score, _), p in zip(cands, probs):
                cls = int(np.argmax(p))
                if cls == 0:
                    continue
                conf = float(p[cls] * score)
                if conf < float(args.class_threshold):
                    continue
                by_class[cls - 1].append((x, y, conf))

            for class_id in [0, 1]:
                det = by_class[class_id]
                det.sort(key=lambda t: t[2], reverse=True)
                det = _nms_xy_conf(det, min_distance=args.final_min_distance)
                for x, y, conf in det:
                    rows.append([r.image_id, f"{x:.2f}", f"{y:.2f}", str(class_id), f"{conf:.6f}"])

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"Saved detections to {args.out_csv}")


if __name__ == "__main__":
    main()

