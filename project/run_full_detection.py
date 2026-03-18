from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from typing import Dict, List, Sequence, Tuple

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from scipy.ndimage import center_of_mass, label

from infer_detector import image_to_chw_01, peak_detect, tiled_inference
from log_detector import multiscale_log_candidates
from model_golddigger_cgan import GoldDiggerGenerator
from model_refiner import PatchRefinerCNN
from model_unet import UNetKeypointDetector


def list_tifs(scan_root: str) -> List[str]:
    out: List[str] = []
    for dp, _, files in os.walk(scan_root):
        for f in files:
            if f.lower().endswith(".tif"):
                out.append(os.path.join(dp, f))
    return sorted(out)


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


def components_to_points(prob_map: np.ndarray, threshold: float, min_area: int, max_area: int) -> List[Tuple[float, float, float]]:
    bw = prob_map >= float(threshold)
    cc, n = label(bw)
    out: List[Tuple[float, float, float]] = []
    for i in range(1, int(n) + 1):
        m = cc == i
        area = int(m.sum())
        if area < min_area:
            continue
        if max_area > 0 and area > max_area:
            continue
        cy, cx = center_of_mass(m.astype(np.float32))
        conf = float(prob_map[m].mean())
        out.append((float(cx), float(cy), conf))
    out.sort(key=lambda t: t[2], reverse=True)
    return out


def detect_unet(
    model: UNetKeypointDetector,
    chw: np.ndarray,
    device: torch.device,
    threshold: float,
    min_distance: int,
    tile_hw: Tuple[int, int],
    stride_hw: Tuple[int, int],
) -> Dict[int, List[Tuple[float, float, float]]]:
    pred = tiled_inference(model, chw, tile_hw, stride_hw, device)
    out = {0: [], 1: []}
    for cls in [0, 1]:
        out[cls] = peak_detect(pred[cls], threshold=threshold, min_distance=min_distance)
    return out


def detect_two_stage(
    heatmap_model: UNetKeypointDetector,
    refiner: PatchRefinerCNN,
    chw: np.ndarray,
    device: torch.device,
    proposal_threshold: float,
    proposal_min_distance: int,
    refiner_keep_threshold: float,
    refiner_patch_size: int,
    tile_hw: Tuple[int, int],
    stride_hw: Tuple[int, int],
) -> Dict[int, List[Tuple[float, float, float]]]:
    pred = tiled_inference(heatmap_model, chw, tile_hw, stride_hw, device)
    proposals: List[Tuple[float, float, int, float]] = []
    for cls in [0, 1]:
        for x, y, conf in peak_detect(pred[cls], threshold=proposal_threshold, min_distance=proposal_min_distance):
            proposals.append((x, y, cls, conf))

    out = {0: [], 1: []}
    if not proposals:
        return out
    patches = np.stack([_extract_patch(chw, x, y, refiner_patch_size) for x, y, _, _ in proposals], axis=0)
    with torch.no_grad():
        probs = F.softmax(refiner(torch.from_numpy(patches).float().to(device)), dim=1).cpu().numpy()
    for (x, y, _, coarse_conf), p in zip(proposals, probs):
        cls_raw = int(np.argmax(p))  # 0=bg, 1=6nm, 2=12nm
        if cls_raw == 0:
            continue
        refined_conf = float(p[cls_raw])
        if refined_conf < refiner_keep_threshold:
            continue
        class_id = cls_raw - 1
        out[class_id].append((x, y, float(coarse_conf * refined_conf)))
    return out


def detect_logcnn(
    classifier: PatchRefinerCNN,
    image_raw: np.ndarray,
    chw: np.ndarray,
    device: torch.device,
    sigmas: Sequence[float],
    log_threshold: float,
    candidate_min_distance: int,
    max_candidates_per_image: int,
    patch_size: int,
    class_threshold: float,
) -> Dict[int, List[Tuple[float, float, float]]]:
    gray = image_raw.mean(axis=2) if image_raw.ndim == 3 else image_raw.astype(np.float32)
    gray = gray.astype(np.float32)
    mn, mx = float(gray.min()), float(gray.max())
    if mx > mn:
        gray = (gray - mn) / (mx - mn)
    else:
        gray = np.zeros_like(gray, dtype=np.float32)

    cands = multiscale_log_candidates(
        gray,
        sigmas=sigmas,
        threshold=log_threshold,
        min_distance=candidate_min_distance,
        max_candidates=max_candidates_per_image,
    )
    out = {0: [], 1: []}
    if not cands:
        return out
    patches = np.stack([_extract_patch(chw, x, y, patch_size) for x, y, _, _ in cands], axis=0)
    with torch.no_grad():
        probs = F.softmax(classifier(torch.from_numpy(patches).float().to(device)), dim=1).cpu().numpy()
    for (x, y, score, _), p in zip(cands, probs):
        cls_raw = int(np.argmax(p))  # 0=bg, 1=6nm, 2=12nm
        if cls_raw == 0:
            continue
        conf = float(score * p[cls_raw])
        if conf < class_threshold:
            continue
        out[cls_raw - 1].append((x, y, conf))
    return out


def detect_golddigger_cgan(
    model: GoldDiggerGenerator,
    chw: np.ndarray,
    device: torch.device,
    tile_hw: Tuple[int, int],
    stride_hw: Tuple[int, int],
    threshold_6nm: float,
    threshold_12nm: float,
    min_area_6nm: int,
    max_area_6nm: int,
    min_area_12nm: int,
    max_area_12nm: int,
) -> Dict[int, List[Tuple[float, float, float]]]:
    pred = tiled_inference(model, chw, tile_hw, stride_hw, device)
    return {
        0: components_to_points(pred[0], threshold_6nm, min_area_6nm, max_area_6nm),
        1: components_to_points(pred[1], threshold_12nm, min_area_12nm, max_area_12nm),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="One-command full-folder immunogold detection runner.")
    p.add_argument("--scan_root", type=str, required=True, help="Folder tree containing TIFF images.")
    p.add_argument("--method", type=str, default="unet", choices=["unet", "two_stage", "logcnn", "golddigger_cgan"])
    p.add_argument("--out_dir", type=str, default="full_detection_runs")
    p.add_argument("--run_name", type=str, default="")

    p.add_argument("--tile_h", type=int, default=512)
    p.add_argument("--tile_w", type=int, default=512)
    p.add_argument("--stride_h", type=int, default=384)
    p.add_argument("--stride_w", type=int, default=384)

    # U-Net / two-stage shared heatmap model.
    p.add_argument("--heatmap_ckpt", type=str, default="")
    p.add_argument("--base_channels", type=int, default=24)
    p.add_argument("--threshold", type=float, default=0.02)
    p.add_argument("--min_distance", type=int, default=5)

    # Two-stage.
    p.add_argument("--refiner_ckpt", type=str, default="")
    p.add_argument("--proposal_threshold", type=float, default=0.02)
    p.add_argument("--proposal_min_distance", type=int, default=5)
    p.add_argument("--refiner_keep_threshold", type=float, default=0.6)
    p.add_argument("--refiner_patch_size", type=int, default=33)

    # LoG+CNN.
    p.add_argument("--logcnn_ckpt", type=str, default="")
    p.add_argument("--sigmas", type=str, default="1.2,1.6,2.0,2.4,2.8")
    p.add_argument("--log_threshold", type=float, default=0.02)
    p.add_argument("--candidate_min_distance", type=int, default=5)
    p.add_argument("--max_candidates_per_image", type=int, default=600)
    p.add_argument("--class_threshold", type=float, default=0.55)

    # Gold Digger cGAN.
    p.add_argument("--generator_ckpt", type=str, default="")
    p.add_argument("--threshold_6nm", type=float, default=0.35)
    p.add_argument("--threshold_12nm", type=float, default=0.35)
    p.add_argument("--min_area_6nm", type=int, default=4)
    p.add_argument("--max_area_6nm", type=int, default=150)
    p.add_argument("--min_area_12nm", type=int, default=8)
    p.add_argument("--max_area_12nm", type=int, default=250)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tifs = list_tifs(args.scan_root)
    if not tifs:
        raise ValueError(f"No .tif files found under: {args.scan_root}")

    os.makedirs(args.out_dir, exist_ok=True)
    if args.run_name.strip():
        run_name = args.run_name.strip()
    else:
        run_name = f"{args.method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Found {len(tifs)} TIFF files.")

    heatmap_model = None
    refiner = None
    logcnn = None
    cgan = None

    if args.method in {"unet", "two_stage"}:
        if not args.heatmap_ckpt:
            raise ValueError("--heatmap_ckpt is required for method unet/two_stage")
        heatmap_model = UNetKeypointDetector(in_channels=3, out_channels=2, base_channels=args.base_channels).to(device)
        heatmap_model.load_state_dict(torch.load(args.heatmap_ckpt, map_location=device))
        heatmap_model.eval()
    if args.method == "two_stage":
        if not args.refiner_ckpt:
            raise ValueError("--refiner_ckpt is required for method two_stage")
        refiner = PatchRefinerCNN(in_channels=3, num_classes=3, base_channels=32).to(device)
        refiner.load_state_dict(torch.load(args.refiner_ckpt, map_location=device))
        refiner.eval()
    if args.method == "logcnn":
        if not args.logcnn_ckpt:
            raise ValueError("--logcnn_ckpt is required for method logcnn")
        logcnn = PatchRefinerCNN(in_channels=3, num_classes=3, base_channels=32).to(device)
        logcnn.load_state_dict(torch.load(args.logcnn_ckpt, map_location=device))
        logcnn.eval()
    if args.method == "golddigger_cgan":
        if not args.generator_ckpt:
            raise ValueError("--generator_ckpt is required for method golddigger_cgan")
        cgan = GoldDiggerGenerator(in_channels=3, out_channels=2, base_channels=64).to(device)
        cgan.load_state_dict(torch.load(args.generator_ckpt, map_location=device))
        cgan.eval()

    rows: List[List[str]] = [["image_id", "x", "y", "class_id", "confidence", "image_path"]]
    tile_hw = (args.tile_h, args.tile_w)
    stride_hw = (args.stride_h, args.stride_w)
    sigmas = [float(s.strip()) for s in args.sigmas.split(",") if s.strip()]

    with torch.no_grad():
        for idx, tif_path in enumerate(tifs, start=1):
            img = tifffile.imread(tif_path)
            chw = image_to_chw_01(img)
            image_id = os.path.splitext(os.path.basename(tif_path))[0]

            if args.method == "unet":
                det = detect_unet(
                    model=heatmap_model,
                    chw=chw,
                    device=device,
                    threshold=args.threshold,
                    min_distance=args.min_distance,
                    tile_hw=tile_hw,
                    stride_hw=stride_hw,
                )
            elif args.method == "two_stage":
                det = detect_two_stage(
                    heatmap_model=heatmap_model,
                    refiner=refiner,
                    chw=chw,
                    device=device,
                    proposal_threshold=args.proposal_threshold,
                    proposal_min_distance=args.proposal_min_distance,
                    refiner_keep_threshold=args.refiner_keep_threshold,
                    refiner_patch_size=args.refiner_patch_size,
                    tile_hw=tile_hw,
                    stride_hw=stride_hw,
                )
            elif args.method == "logcnn":
                det = detect_logcnn(
                    classifier=logcnn,
                    image_raw=img,
                    chw=chw,
                    device=device,
                    sigmas=sigmas,
                    log_threshold=args.log_threshold,
                    candidate_min_distance=args.candidate_min_distance,
                    max_candidates_per_image=args.max_candidates_per_image,
                    patch_size=args.refiner_patch_size,
                    class_threshold=args.class_threshold,
                )
            else:
                det = detect_golddigger_cgan(
                    model=cgan,
                    chw=chw,
                    device=device,
                    tile_hw=tile_hw,
                    stride_hw=stride_hw,
                    threshold_6nm=args.threshold_6nm,
                    threshold_12nm=args.threshold_12nm,
                    min_area_6nm=args.min_area_6nm,
                    max_area_6nm=args.max_area_6nm,
                    min_area_12nm=args.min_area_12nm,
                    max_area_12nm=args.max_area_12nm,
                )

            for class_id in [0, 1]:
                for x, y, conf in det[class_id]:
                    rows.append(
                        [
                            image_id,
                            f"{x:.2f}",
                            f"{y:.2f}",
                            str(class_id),
                            f"{float(conf):.6f}",
                            tif_path,
                        ]
                    )
            if idx % 10 == 0 or idx == len(tifs):
                print(f"Processed {idx}/{len(tifs)} images")

    pred_csv = os.path.join(run_dir, "detections.csv")
    with open(pred_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    cfg = vars(args).copy()
    cfg["run_name"] = run_name
    cfg["num_images"] = len(tifs)
    cfg["created_at"] = datetime.now().isoformat()
    cfg_path = os.path.join(run_dir, "run_config.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"Saved detections: {pred_csv}")
    print(f"Saved run config: {cfg_path}")


if __name__ == "__main__":
    main()
