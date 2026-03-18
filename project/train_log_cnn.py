from __future__ import annotations

import argparse
import os
from typing import List, Sequence, Tuple

import numpy as np
import tifffile
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from dataset_guard import enforce_allowed_data_root
from log_detector import multiscale_log_candidates
from model_refiner import PatchRefinerCNN
from prepare_labels import ImageRecord, discover_image_records


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


def _label_candidate(x: float, y: float, points0: np.ndarray, points1: np.ndarray, match_dist: float) -> int:
    best_cls = 0
    best_d = float("inf")
    for cls_idx, pts in [(1, points0), (2, points1)]:
        if len(pts) == 0:
            continue
        d2 = (pts[:, 0] - x) ** 2 + (pts[:, 1] - y) ** 2
        d = float(np.sqrt(float(d2.min())))
        if d < best_d:
            best_d = d
            best_cls = cls_idx
    return best_cls if best_d <= float(match_dist) else 0


class CandidatePatchDataset(Dataset):
    def __init__(self, patches: np.ndarray, labels: np.ndarray, augment: bool = False, seed: int = 42) -> None:
        self.patches = patches.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.augment = bool(augment)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return int(self.patches.shape[0])

    def _augment(self, x: np.ndarray) -> np.ndarray:
        if self.rng.random() < 0.5:
            x = x[:, :, ::-1].copy()
        if self.rng.random() < 0.5:
            x = x[:, ::-1, :].copy()
        if self.rng.random() < 0.5:
            c = float(self.rng.uniform(0.9, 1.1))
            b = float(self.rng.uniform(-0.05, 0.05))
            x = np.clip(x * c + b, 0.0, 1.0)
        return x

    def __getitem__(self, idx: int):
        x = self.patches[idx]
        y = self.labels[idx]
        if self.augment:
            x = self._augment(x)
        return torch.from_numpy(x).float(), torch.tensor(y, dtype=torch.long)


def build_candidate_dataset(
    records: Sequence[ImageRecord],
    patch_size: int,
    sigmas: Sequence[float],
    log_threshold: float,
    min_distance: int,
    max_candidates_per_image: int,
    match_dist: float,
    include_gt_points: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    patches: List[np.ndarray] = []
    labels: List[int] = []

    for r in records:
        img = tifffile.imread(r.image_path)
        img_gray = img.mean(axis=2) if img.ndim == 3 else img.astype(np.float32)
        img_gray = img_gray.astype(np.float32)
        mn, mx = float(img_gray.min()), float(img_gray.max())
        if mx > mn:
            img_gray = (img_gray - mn) / (mx - mn)
        else:
            img_gray = np.zeros_like(img_gray, dtype=np.float32)

        chw = _to_chw_01(img)
        cands = multiscale_log_candidates(
            img_gray,
            sigmas=sigmas,
            threshold=log_threshold,
            min_distance=min_distance,
            max_candidates=max_candidates_per_image,
        )

        if include_gt_points:
            for x, y in r.points[0]:
                cands.append((float(x), float(y), 1.0, 0.0))
            for x, y in r.points[1]:
                cands.append((float(x), float(y), 1.0, 0.0))

        for x, y, _, _ in cands:
            lab = _label_candidate(x, y, r.points[0], r.points[1], match_dist=match_dist)
            patches.append(_extract_patch(chw, x, y, patch_size))
            labels.append(lab)

    if not patches:
        raise ValueError("No candidate patches generated.")
    return np.stack(patches, axis=0), np.asarray(labels, dtype=np.int64)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> Tuple[float, float]:
    train_mode = optimizer is not None
    model.train(train_mode)
    loss_total = 0.0
    correct = 0
    n = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        with torch.set_grad_enabled(train_mode):
            logits = model(x)
            loss = criterion(logits, y)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        pred = torch.argmax(logits, dim=1)
        bs = x.size(0)
        loss_total += float(loss.item()) * bs
        correct += int((pred == y).sum().item())
        n += bs
    denom = max(1, n)
    return loss_total / denom, correct / denom


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LoG+CNN classifier for immunogold candidates.")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--patch_size", type=int, default=33)
    p.add_argument("--sigmas", type=str, default="1.2,1.6,2.0,2.4,2.8")
    p.add_argument("--log_threshold", type=float, default=0.015)
    p.add_argument("--min_distance", type=int, default=5)
    p.add_argument("--max_candidates_per_image", type=int, default=800)
    p.add_argument("--match_dist", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_path", type=str, default="checkpoints/log_cnn_best.pt")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.data_root = enforce_allowed_data_root(args.data_root)
    sigmas = [float(s.strip()) for s in args.sigmas.split(",") if s.strip()]

    records = discover_image_records(args.data_root)
    train_r, val_r, test_r = split_by_image(records, seed=args.seed)
    print(f"Image split -> train={len(train_r)} val={len(val_r)} test={len(test_r)}")

    x_train, y_train = build_candidate_dataset(
        train_r,
        patch_size=args.patch_size,
        sigmas=sigmas,
        log_threshold=args.log_threshold,
        min_distance=args.min_distance,
        max_candidates_per_image=args.max_candidates_per_image,
        match_dist=args.match_dist,
        include_gt_points=True,
    )
    x_val, y_val = build_candidate_dataset(
        val_r,
        patch_size=args.patch_size,
        sigmas=sigmas,
        log_threshold=args.log_threshold,
        min_distance=args.min_distance,
        max_candidates_per_image=args.max_candidates_per_image,
        match_dist=args.match_dist,
        include_gt_points=True,
    )
    print(f"Train candidates={len(y_train)} class_counts={np.bincount(y_train, minlength=3).tolist()}")
    print(f"Val candidates={len(y_val)} class_counts={np.bincount(y_val, minlength=3).tolist()}")

    train_ds = CandidatePatchDataset(x_train, y_train, augment=True, seed=args.seed)
    val_ds = CandidatePatchDataset(x_val, y_val, augment=False, seed=args.seed + 1)

    class_counts = np.bincount(y_train, minlength=3).astype(np.float32)
    class_weights = 1.0 / np.maximum(class_counts, 1.0)
    sample_weights = class_weights[y_train]
    sampler = WeightedRandomSampler(weights=torch.from_numpy(sample_weights).double(), num_samples=len(y_train), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = PatchRefinerCNN(in_channels=3, num_classes=3, base_channels=32).to(device)

    ce_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=ce_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, None, device)
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} "
            f"train_loss={tr_loss:.5f} train_acc={tr_acc:.4f} "
            f"val_loss={va_loss:.5f} val_acc={va_acc:.4f}"
        )
        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), args.save_path)
            print(f"Saved best: {args.save_path} (val_loss={best_val:.5f})")


if __name__ == "__main__":
    main()

