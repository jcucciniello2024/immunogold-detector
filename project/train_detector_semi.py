from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset_guard import enforce_allowed_data_root
from dataset_points import PointPatchDataset
from infer_detector import image_to_chw_01, peak_detect, tiled_inference
from model_unet import UNetKeypointDetector
from prepare_labels import ImageRecord, discover_image_records


class FocalBCELoss(nn.Module):
    def __init__(self, pos_weight: float = 25.0, neg_weight: float = 1.0, gamma: float = 2.0) -> None:
        super().__init__()
        self.pos_weight = float(pos_weight)
        self.neg_weight = float(neg_weight)
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        t = torch.clamp(targets, 0.0, 1.0)
        probs = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
        pt = probs * t + (1.0 - probs) * (1.0 - t)
        focal = torch.pow(1.0 - pt, self.gamma)
        alpha = self.neg_weight + self.pos_weight * t
        return (alpha * focal * bce).mean()


def split_by_image(
    records: List[ImageRecord], train_ratio: float = 0.7, val_ratio: float = 0.15, seed: int = 42
) -> Tuple[List[ImageRecord], List[ImageRecord], List[ImageRecord]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(records))
    rng.shuffle(idx)
    n = len(records)
    n_train = max(1, int(round(n * train_ratio)))
    n_val = max(1, int(round(n * val_ratio)))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
    train = [records[i] for i in idx[:n_train]]
    val = [records[i] for i in idx[n_train : n_train + n_val]]
    test = [records[i] for i in idx[n_train + n_val :]]
    return train, val, test


def _discover_unlabeled_tifs(unlabeled_dir: str) -> List[str]:
    out: List[str] = []
    for f in sorted(os.listdir(unlabeled_dir)):
        low = f.lower()
        if not low.endswith(".tif"):
            continue
        # Keep synapse-like images from replica folder.
        if f.strip().upper().startswith("S"):
            out.append(os.path.join(unlabeled_dir, f))
    return out


def _predict_heatmaps_with_tta(
    model: UNetKeypointDetector,
    chw: np.ndarray,
    device: torch.device,
) -> List[np.ndarray]:
    """
    Returns list of (2,H,W) heatmaps in original orientation:
    [original, hflip-aug restored, vflip-aug restored].
    """
    preds: List[np.ndarray] = []
    p0 = tiled_inference(model, chw, (384, 384), (288, 288), device)
    preds.append(p0)

    chw_h = chw[:, :, ::-1].copy()
    ph = tiled_inference(model, chw_h, (384, 384), (288, 288), device)
    ph = ph[:, :, ::-1].copy()
    preds.append(ph)

    chw_v = chw[:, ::-1, :].copy()
    pv = tiled_inference(model, chw_v, (384, 384), (288, 288), device)
    pv = pv[:, ::-1, :].copy()
    preds.append(pv)
    return preds


def _merge_consistent_points(
    det_lists: List[List[Tuple[float, float, float]]],
    min_support: int,
    merge_dist: float,
    max_points: int,
) -> np.ndarray:
    """
    Merge detections from multiple TTA views and keep clusters seen repeatedly.
    """
    clusters: List[Dict[str, object]] = []
    md2 = float(merge_dist * merge_dist)

    for view_idx, dets in enumerate(det_lists):
        for x, y, conf in dets:
            best_i = -1
            best_d2 = float("inf")
            for i, c in enumerate(clusters):
                cx = float(c["x"])
                cy = float(c["y"])
                d2 = (x - cx) * (x - cx) + (y - cy) * (y - cy)
                if d2 < md2 and d2 < best_d2:
                    best_d2 = d2
                    best_i = i
            if best_i < 0:
                clusters.append(
                    {
                        "x": float(x),
                        "y": float(y),
                        "sum_conf": float(conf),
                        "count": 1,
                        "views": {view_idx},
                    }
                )
            else:
                c = clusters[best_i]
                n = int(c["count"]) + 1
                c["x"] = (float(c["x"]) * float(c["count"]) + float(x)) / float(n)
                c["y"] = (float(c["y"]) * float(c["count"]) + float(y)) / float(n)
                c["sum_conf"] = float(c["sum_conf"]) + float(conf)
                c["count"] = n
                views = c["views"]
                views.add(view_idx)

    kept = []
    for c in clusters:
        support = len(c["views"])
        if support < int(min_support):
            continue
        avg_conf = float(c["sum_conf"]) / max(1, int(c["count"]))
        kept.append((float(c["x"]), float(c["y"]), avg_conf, support))
    kept.sort(key=lambda t: (t[2], t[3]), reverse=True)
    if max_points > 0:
        kept = kept[:max_points]
    out = np.array([[k[0], k[1]] for k in kept], dtype=np.float32).reshape(-1, 2)
    return out


def build_pseudo_records(
    unlabeled_dir: str,
    teacher_ckpt: str,
    base_channels: int,
    pseudo_threshold: float,
    min_distance: int,
    max_pseudo_per_class: int,
    pseudo_min_support: int,
    pseudo_merge_dist: float,
) -> List[ImageRecord]:
    tif_paths = _discover_unlabeled_tifs(unlabeled_dir)
    if not tif_paths:
        return []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = UNetKeypointDetector(in_channels=3, out_channels=2, base_channels=base_channels).to(device)
    teacher.load_state_dict(torch.load(teacher_ckpt, map_location=device))
    teacher.eval()

    pseudo: List[ImageRecord] = []
    for i, p in enumerate(tif_paths):
        img = tifffile.imread(p)
        h, w = img.shape[:2]
        chw = image_to_chw_01(img)
        pred_list = _predict_heatmaps_with_tta(teacher, chw, device)

        pts = {}
        for cls in [0, 1]:
            det_lists = [
                peak_detect(pred[cls], threshold=pseudo_threshold, min_distance=min_distance)
                for pred in pred_list
            ]
            arr = _merge_consistent_points(
                det_lists=det_lists,
                min_support=pseudo_min_support,
                merge_dist=pseudo_merge_dist,
                max_points=max_pseudo_per_class,
            )
            pts[cls] = arr

        pseudo.append(
            ImageRecord(
                image_id=f"PSEUDO_{i+1:03d}",
                image_path=p,
                width=w,
                height=h,
                points=pts,
            )
        )
    return pseudo


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> Tuple[float, float]:
    train_mode = optimizer is not None
    model.train(train_mode)
    total = 0.0
    pred_mean = 0.0
    n = 0
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        with torch.set_grad_enabled(train_mode):
            logits = model(images)
            loss = criterion(logits, targets)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            probs = torch.sigmoid(logits)
        bs = images.size(0)
        total += float(loss.item()) * bs
        pred_mean += float(probs.mean().item()) * bs
        n += bs
    denom = max(1, n)
    return total / denom, pred_mean / denom


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Semi-supervised detector training with pseudo-labels from unlabeled Max Planck TIFFs.")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--unlabeled_dir", type=str, required=True)
    p.add_argument("--teacher_ckpt", type=str, required=True)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--patch_h", type=int, default=384)
    p.add_argument("--patch_w", type=int, default=384)
    p.add_argument("--train_samples_per_epoch", type=int, default=4096)
    p.add_argument("--val_samples_per_epoch", type=int, default=768)
    p.add_argument("--sigma", type=float, default=2.0)
    p.add_argument("--pos_fraction", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--base_channels", type=int, default=24)
    p.add_argument("--pseudo_threshold", type=float, default=0.24)
    p.add_argument("--pseudo_min_distance", type=int, default=5)
    p.add_argument("--max_pseudo_per_class", type=int, default=80)
    p.add_argument(
        "--pseudo_min_support",
        type=int,
        default=2,
        help="Minimum number of TTA views agreeing on a pseudo point (1-3).",
    )
    p.add_argument(
        "--pseudo_merge_dist",
        type=float,
        default=4.0,
        help="Distance (px) for merging TTA pseudo detections.",
    )
    p.add_argument("--save_dir", type=str, default="checkpoints")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.data_root = enforce_allowed_data_root(args.data_root)
    args.unlabeled_dir = enforce_allowed_data_root(args.unlabeled_dir)

    labeled = discover_image_records(args.data_root)
    train_r, val_r, test_r = split_by_image(labeled, seed=args.seed)
    print(f"Labeled split -> train={len(train_r)} val={len(val_r)} test={len(test_r)}")

    pseudo = build_pseudo_records(
        unlabeled_dir=args.unlabeled_dir,
        teacher_ckpt=args.teacher_ckpt,
        base_channels=args.base_channels,
        pseudo_threshold=args.pseudo_threshold,
        min_distance=args.pseudo_min_distance,
        max_pseudo_per_class=args.max_pseudo_per_class,
        pseudo_min_support=args.pseudo_min_support,
        pseudo_merge_dist=args.pseudo_merge_dist,
    )
    n_p0 = int(sum(len(r.points[0]) for r in pseudo))
    n_p1 = int(sum(len(r.points[1]) for r in pseudo))
    print(f"Pseudo records={len(pseudo)} pseudo_points_6nm={n_p0} pseudo_points_12nm={n_p1}")

    train_all = train_r + pseudo
    train_ds = PointPatchDataset(
        train_all,
        patch_size=(args.patch_h, args.patch_w),
        samples_per_epoch=args.train_samples_per_epoch,
        pos_fraction=args.pos_fraction,
        sigma=args.sigma,
        target_type="gaussian",
        target_radius=3,
        augment=True,
        seed=args.seed,
    )
    val_ds = PointPatchDataset(
        val_r,
        patch_size=(args.patch_h, args.patch_w),
        samples_per_epoch=args.val_samples_per_epoch,
        pos_fraction=args.pos_fraction,
        sigma=args.sigma,
        target_type="gaussian",
        target_radius=3,
        augment=False,
        seed=args.seed + 1,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = UNetKeypointDetector(in_channels=3, out_channels=2, base_channels=args.base_channels).to(device)
    model.load_state_dict(torch.load(args.teacher_ckpt, map_location=device))
    print(f"Initialized from teacher checkpoint: {args.teacher_ckpt}")

    criterion = FocalBCELoss(pos_weight=20.0, neg_weight=1.0, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    os.makedirs(args.save_dir, exist_ok=True)
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_mean = run_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_mean = run_epoch(model, val_loader, criterion, None, device)
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} "
            f"train={tr_loss:.6f} val={va_loss:.6f} "
            f"train_pred_mean={tr_mean:.6f} val_pred_mean={va_mean:.6f}"
        )
        torch.save(model.state_dict(), os.path.join(args.save_dir, "detector_semi_last.pt"))
        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, "detector_semi_best.pt"))
            print(f"New best semi checkpoint: val={best_val:.6f}")


if __name__ == "__main__":
    main()

