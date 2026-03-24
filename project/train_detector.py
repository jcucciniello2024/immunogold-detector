
import argparse
import math
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset_points import PointPatchDataset
from model_unet import UNetKeypointDetector
from model_unet_deep import UNetDeepKeypointDetector
from prepare_labels import ImageRecord, discover_image_records


class WeightedHeatmapLoss(nn.Module):
    def __init__(self, pos_weight: float = 30.0, neg_weight: float = 1.0, pos_power: float = 1.0) -> None:
        super().__init__()
        self.pos_weight = float(pos_weight)
        self.neg_weight = float(neg_weight)
        self.pos_power = float(pos_power)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        t = torch.clamp(targets, 0.0, 1.0)
        weights = self.neg_weight + self.pos_weight * torch.pow(t, self.pos_power)
        return ((preds - t) ** 2 * weights).mean()


class FocalBCELoss(nn.Module):
    def __init__(self, pos_weight: float = 30.0, neg_weight: float = 1.0, gamma: float = 2.0) -> None:
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train 2D heatmap detector for immunogold keypoints.")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--patch_h", type=int, default=256)
    p.add_argument("--patch_w", type=int, default=256)
    p.add_argument("--train_samples_per_epoch", type=int, default=2048)
    p.add_argument("--val_samples_per_epoch", type=int, default=256)
    p.add_argument("--sigma", type=float, default=1.5)
    p.add_argument("--target_type", type=str, default="gaussian", choices=["gaussian", "disk"])
    p.add_argument("--target_radius", type=int, default=3)
    p.add_argument("--pos_fraction", type=float, default=0.6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_dir", type=str, default="checkpoints")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--base_channels", type=int, default=32)
    p.add_argument("--loss_type", type=str, default="focal_bce", choices=["focal_bce", "weighted_mse"])
    p.add_argument("--loss_pos_weight", type=float, default=300.0)
    p.add_argument("--loss_neg_weight", type=float, default=1.0)
    p.add_argument("--loss_pos_power", type=float, default=1.0)
    p.add_argument("--focal_gamma", type=float, default=2.0)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    # New augmentation and training flags
    p.add_argument("--model_type", type=str, default="unet_deep", choices=["unet", "unet_deep"])
    p.add_argument("--use_clahe", action="store_true", help="Enable CLAHE preprocessing (not recommended)")
    p.add_argument("--use_mantis", action="store_true", help="Enable Mantis local contrast preprocessing")
    p.add_argument("--sigma_jitter", action="store_true", help="Enable sigma jittering at heatmap generation")
    p.add_argument("--consistency_weight", type=float, default=0.0, help="Weight for consistency loss (0 to disable)")
    p.add_argument("--sched", type=str, default="step", choices=["none", "step", "cosine"], help="LR schedule type")
    p.add_argument("--warmup_epochs", type=int, default=5, help="Warmup epochs for cosine schedule")
    p.add_argument("--mixed_precision", action="store_true", help="Enable mixed precision training")
    p.add_argument("--early_stop_patience", type=int, default=15, help="Early stopping patience (0 to disable)")
    p.add_argument("--early_stop_delta", type=float, default=1e-5, help="Minimum val loss improvement to reset patience")
    p.add_argument("--use_sliding_window", action="store_true", help="Use SlidingWindowPatchDataset for maximum data coverage")
    p.add_argument("--patch_stride", type=int, default=128, help="Stride for sliding window patches")
    return p.parse_args()


class WarmupCosineScheduler:
    """Cosine annealing with linear warmup."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_lr: float,
        warmup_epochs: int,
        total_epochs: int,
    ) -> None:
        self.optimizer = optimizer
        self.base_lr = float(base_lr)
        self.warmup_epochs = int(warmup_epochs)
        self.total_epochs = int(total_epochs)
        self.current_epoch = 0

    def step(self, epoch: int) -> None:
        """Update learning rate for the given epoch."""
        self.current_epoch = epoch
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * epoch / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.base_lr * 0.5 * (1 + math.cos(math.pi * progress))
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    grad_clip: float = 0.0,
    amp_scaler: Optional[torch.cuda.amp.GradScaler] = None,
    consistency_weight: float = 0.0,
) -> Tuple[float, float, float]:
    train_mode = optimizer is not None
    model.train(train_mode)
    total = 0.0
    pred_total = 0.0
    pred_max = 0.0
    n = 0

    for batch in loader:
        # Handle both regular batch and consistency pair batch
        if len(batch) == 4:
            images, targets, images2, targets2 = batch
            images = images.to(device)
            targets = targets.to(device)
            images2 = images2.to(device)
            targets2 = targets2.to(device)
            has_consistency = True
        else:
            images, targets = batch
            images = images.to(device)
            targets = targets.to(device)
            has_consistency = False

        with torch.set_grad_enabled(train_mode):
            if train_mode and amp_scaler is not None:
                # Mixed precision training
                with torch.cuda.amp.autocast():
                    logits = model(images)
                    loss = criterion(logits, targets)

                    if has_consistency and consistency_weight > 0:
                        logits2 = model(images2)
                        probs = torch.sigmoid(logits)
                        probs2 = torch.sigmoid(logits2)
                        consistency_loss = F.mse_loss(probs, probs2.detach())
                        loss = loss + consistency_weight * consistency_loss

                optimizer.zero_grad()
                amp_scaler.scale(loss).backward()
                if grad_clip > 0.0:
                    amp_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                amp_scaler.step(optimizer)
                amp_scaler.update()
            else:
                # Standard training (no mixed precision)
                logits = model(images)
                loss = criterion(logits, targets)

                if has_consistency and consistency_weight > 0:
                    logits2 = model(images2)
                    probs = torch.sigmoid(logits)
                    probs2 = torch.sigmoid(logits2)
                    consistency_loss = F.mse_loss(probs, probs2.detach())
                    loss = loss + consistency_weight * consistency_loss

                if train_mode:
                    optimizer.zero_grad()
                    loss.backward()
                    if grad_clip > 0.0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

            probs = torch.sigmoid(logits)

        bs = images.size(0)
        total += loss.item() * bs
        pred_total += float(probs.mean().item()) * bs
        pred_max = max(pred_max, float(probs.max().item()))
        n += bs

    denom = max(1, n)
    return total / denom, pred_total / denom, pred_max


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    records = discover_image_records(args.data_root)
    print(f"Discovered images: {len(records)}")
    train_r, val_r, test_r = split_by_image(records, seed=args.seed)
    print(f"Split by image -> train {len(train_r)}, val {len(val_r)}, test {len(test_r)}")

    # Use sliding window dataset if requested, otherwise fall back to random sampling
    if args.use_sliding_window:
        from dataset_points_sliding_window import SlidingWindowPatchDataset
        train_ds = SlidingWindowPatchDataset(
            train_r,
            patch_size=(args.patch_h, args.patch_w),
            patch_stride=args.patch_stride,
            samples_per_epoch=args.train_samples_per_epoch,
            pos_fraction=args.pos_fraction,
            sigma=args.sigma,
            target_type=args.target_type,
            target_radius=args.target_radius,
            augment=True,
            seed=args.seed,
            preprocess=args.use_clahe,
            mantis_preprocess=args.use_mantis,
            sigma_jitter=args.sigma_jitter,
            consistency_pairs=(args.consistency_weight > 0),
        )
    else:
        train_ds = PointPatchDataset(
            train_r,
            patch_size=(args.patch_h, args.patch_w),
            samples_per_epoch=args.train_samples_per_epoch,
            pos_fraction=args.pos_fraction,
            sigma=args.sigma,
            target_type=args.target_type,
            target_radius=args.target_radius,
            augment=True,
            seed=args.seed,
            preprocess=args.use_clahe,
            sigma_jitter=args.sigma_jitter,
            consistency_pairs=(args.consistency_weight > 0),
        )
    val_ds = PointPatchDataset(
        val_r,
        patch_size=(args.patch_h, args.patch_w),
        samples_per_epoch=args.val_samples_per_epoch,
        pos_fraction=args.pos_fraction,
        sigma=args.sigma,
        target_type=args.target_type,
        target_radius=args.target_radius,
        augment=False,
        seed=args.seed + 1,
        preprocess=args.use_clahe,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Create model based on model_type flag
    if args.model_type == "unet_deep":
        model = UNetDeepKeypointDetector(in_channels=3, out_channels=2, base_channels=args.base_channels).to(device)
    else:
        model = UNetKeypointDetector(in_channels=3, out_channels=2, base_channels=args.base_channels).to(device)

    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"Resumed from: {args.resume}")

    if args.loss_type == "focal_bce":
        criterion = FocalBCELoss(
            pos_weight=args.loss_pos_weight,
            neg_weight=args.loss_neg_weight,
            gamma=args.focal_gamma,
        )
    else:
        criterion = WeightedHeatmapLoss(
            pos_weight=args.loss_pos_weight,
            neg_weight=args.loss_neg_weight,
            pos_power=args.loss_pos_power,
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Create LR scheduler if needed
    scheduler = None
    if args.sched == "cosine":
        scheduler = WarmupCosineScheduler(
            optimizer, base_lr=args.lr, warmup_epochs=args.warmup_epochs, total_epochs=args.epochs
        )

    # Create mixed precision scaler if needed
    amp_scaler = None
    if args.mixed_precision and torch.cuda.is_available():
        amp_scaler = torch.cuda.amp.GradScaler()

    print(
        f"Model={args.model_type} base_channels={args.base_channels} "
        f"Loss={args.loss_type} pos_w={args.loss_pos_weight} neg_w={args.loss_neg_weight} "
        f"focal_gamma={args.focal_gamma} weight_decay={args.weight_decay} grad_clip={args.grad_clip} "
        f"target_type={args.target_type} target_radius={args.target_radius} sigma={args.sigma} "
        f"use_clahe={args.use_clahe} sigma_jitter={args.sigma_jitter} "
        f"consistency_weight={args.consistency_weight} sched={args.sched} "
        f"mixed_precision={args.mixed_precision}"
    )

    os.makedirs(args.save_dir, exist_ok=True)
    best_val = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        # Update LR scheduler
        if scheduler is not None:
            scheduler.step(epoch - 1)

        tr, tr_pred_mean, tr_pred_max = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            grad_clip=args.grad_clip,
            amp_scaler=amp_scaler,
            consistency_weight=args.consistency_weight,
        )
        va, va_pred_mean, va_pred_max = run_epoch(model, val_loader, criterion, None, device)

        # Check for overfitting
        train_val_ratio = tr / max(1e-8, va)
        overfit_warning = ""
        if train_val_ratio < 0.5:
            overfit_warning = " ⚠️  OVERFITTING DETECTED (train loss << val loss)"

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} "
            f"train={tr:.6f} val={va:.6f} "
            f"train_pred_mean={tr_pred_mean:.6f} train_pred_max={tr_pred_max:.6f} "
            f"val_pred_mean={va_pred_mean:.6f} val_pred_max={va_pred_max:.6f}{overfit_warning}"
        )
        ckpt = os.path.join(args.save_dir, f"detector_epoch{epoch:02d}.pt")
        torch.save(model.state_dict(), ckpt)
        torch.save(model.state_dict(), os.path.join(args.save_dir, "detector_last.pt"))
        if va < best_val - args.early_stop_delta:
            best_val = va
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(args.save_dir, "detector_best.pt"))
            print(f"New best checkpoint: val={best_val:.6f}")
        else:
            patience_counter += 1
            if args.early_stop_patience > 0 and patience_counter >= args.early_stop_patience:
                print(f"\n🛑 Early stopping triggered (patience {args.early_stop_patience} reached)")
                break


if __name__ == "__main__":
    main()
