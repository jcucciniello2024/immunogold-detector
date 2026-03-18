from __future__ import annotations

import argparse
import os
from typing import List, Sequence, Tuple

import numpy as np
import tifffile
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from model_refiner import PatchRefinerCNN
from prepare_labels import ImageRecord, discover_image_records


def _split_by_image(
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

    sx0 = max(0, x0)
    sx1 = min(w, x1)
    sy0 = max(0, y0)
    sy1 = min(h, y1)
    if sx1 <= sx0 or sy1 <= sy0:
        return out

    dx0 = sx0 - x0
    dy0 = sy0 - y0
    dx1 = dx0 + (sx1 - sx0)
    dy1 = dy0 + (sy1 - sy0)
    out[:, dy0:dy1, dx0:dx1] = chw[:, sy0:sy1, sx0:sx1]
    return out


def _min_dist_to_points(x: float, y: float, points: np.ndarray) -> float:
    if len(points) == 0:
        return 1e9
    d2 = (points[:, 0] - x) ** 2 + (points[:, 1] - y) ** 2
    return float(np.sqrt(float(d2.min())))


class RefinerPatchDataset(Dataset):
    def __init__(
        self,
        records: Sequence[ImageRecord],
        patch_size: int = 33,
        samples_per_epoch: int = 4096,
        pos_fraction: float = 0.5,
        neg_min_dist: float = 8.0,
        augment: bool = True,
        seed: int = 42,
    ) -> None:
        self.patch_size = int(patch_size)
        self.samples_per_epoch = int(samples_per_epoch)
        self.pos_fraction = float(pos_fraction)
        self.neg_min_dist = float(neg_min_dist)
        self.augment = bool(augment)
        self.rng = np.random.default_rng(seed)

        self.images: List[np.ndarray] = []
        self.points0: List[np.ndarray] = []
        self.points1: List[np.ndarray] = []
        self.points_all: List[np.ndarray] = []
        for r in records:
            img = _to_chw_01(tifffile.imread(r.image_path))
            p0 = r.points[0].astype(np.float32)
            p1 = r.points[1].astype(np.float32)
            all_pts = np.concatenate([p0, p1], axis=0) if (len(p0) + len(p1)) > 0 else np.zeros((0, 2), np.float32)
            self.images.append(img)
            self.points0.append(p0)
            self.points1.append(p1)
            self.points_all.append(all_pts)

        if not self.images:
            raise ValueError("No image records for refiner dataset.")

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _augment(self, patch: np.ndarray) -> np.ndarray:
        if self.rng.random() < 0.5:
            patch = patch[:, :, ::-1].copy()
        if self.rng.random() < 0.5:
            patch = patch[:, ::-1, :].copy()
        if self.rng.random() < 0.6:
            contrast = float(self.rng.uniform(0.9, 1.1))
            brightness = float(self.rng.uniform(-0.05, 0.05))
            patch = np.clip(patch * contrast + brightness, 0.0, 1.0)
        return patch

    def __getitem__(self, index: int):
        del index
        i = int(self.rng.integers(0, len(self.images)))
        img = self.images[i]
        p0 = self.points0[i]
        p1 = self.points1[i]
        pall = self.points_all[i]
        _, h, w = img.shape

        is_pos = len(pall) > 0 and self.rng.random() < self.pos_fraction
        if is_pos:
            cls = int(self.rng.integers(0, 2))
            pool = p0 if cls == 0 else p1
            if len(pool) == 0:
                pool = p1 if cls == 0 else p0
                cls = 1 - cls
            if len(pool) == 0:
                is_pos = False
            else:
                x, y = pool[int(self.rng.integers(0, len(pool)))]
                x += float(self.rng.normal(0.0, 1.5))
                y += float(self.rng.normal(0.0, 1.5))
                label = cls + 1
        if not is_pos:
            label = 0
            tries = 0
            while True:
                x = float(self.rng.uniform(0, max(1, w - 1)))
                y = float(self.rng.uniform(0, max(1, h - 1)))
                if _min_dist_to_points(x, y, pall) >= self.neg_min_dist or tries >= 20:
                    break
                tries += 1

        patch = _extract_patch(img, x, y, self.patch_size)
        if self.augment:
            patch = self._augment(patch)

        return torch.from_numpy(patch).float(), torch.tensor(label, dtype=torch.long)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> Tuple[float, float]:
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss = 0.0
    total_correct = 0
    total_n = 0
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
        total_correct += int((pred == y).sum().item())
        bs = x.size(0)
        total_loss += float(loss.item()) * bs
        total_n += bs
    denom = max(1, total_n)
    return total_loss / denom, total_correct / denom


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train patch refiner (bg/6nm/12nm) for two-stage detection.")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--patch_size", type=int, default=33)
    p.add_argument("--train_samples_per_epoch", type=int, default=20000)
    p.add_argument("--val_samples_per_epoch", type=int, default=4000)
    p.add_argument("--pos_fraction", type=float, default=0.5)
    p.add_argument("--neg_min_dist", type=float, default=8.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_path", type=str, default="checkpoints/refiner_best.pt")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    records = discover_image_records(args.data_root)
    train_r, val_r, test_r = _split_by_image(records, seed=args.seed)
    print(f"Images split -> train={len(train_r)} val={len(val_r)} test={len(test_r)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds = RefinerPatchDataset(
        train_r,
        patch_size=args.patch_size,
        samples_per_epoch=args.train_samples_per_epoch,
        pos_fraction=args.pos_fraction,
        neg_min_dist=args.neg_min_dist,
        augment=True,
        seed=args.seed,
    )
    val_ds = RefinerPatchDataset(
        val_r,
        patch_size=args.patch_size,
        samples_per_epoch=args.val_samples_per_epoch,
        pos_fraction=args.pos_fraction,
        neg_min_dist=args.neg_min_dist,
        augment=False,
        seed=args.seed + 1,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = PatchRefinerCNN(in_channels=3, num_classes=3, base_channels=32).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.25, 1.0, 1.0], dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = _run_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = _run_epoch(model, val_loader, criterion, None, device)
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} "
            f"train_loss={tr_loss:.5f} train_acc={tr_acc:.4f} "
            f"val_loss={va_loss:.5f} val_acc={va_acc:.4f}"
        )
        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), args.save_path)
            print(f"Saved best refiner: {args.save_path} (val_loss={best_val:.5f})")


if __name__ == "__main__":
    main()

