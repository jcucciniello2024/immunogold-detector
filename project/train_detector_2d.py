from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_points import PointHeatmapPatchDataset
from model_detector_2d import SmallUNetDetector2D
from particle_data import discover_synapse_samples, split_samples


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train 2D particle center detector from synapse TIFF+CSV labels.")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patch_h", type=int, default=512)
    p.add_argument("--patch_w", type=int, default=512)
    p.add_argument("--train_samples_per_epoch", type=int, default=512)
    p.add_argument("--val_samples_per_epoch", type=int, default=128)
    p.add_argument("--pos_fraction", type=float, default=0.6)
    p.add_argument("--sigma", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_dir", type=str, default="checkpoints")
    p.add_argument("--save_name", type=str, default="detector2d.pt")
    p.add_argument("--resume_path", type=str, default=None)
    return p.parse_args()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total = 0.0
    n = 0
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        with torch.set_grad_enabled(is_train):
            preds = model(images)
            loss = criterion(preds, targets)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        bs = images.size(0)
        total += loss.item() * bs
        n += bs
    return total / max(1, n)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    samples = discover_synapse_samples(args.data_root)
    print(f"Discovered {len(samples)} synapse samples.")
    train_s, val_s, test_s = split_samples(samples, seed=args.seed)
    print(f"Split sizes -> train: {len(train_s)}, val: {len(val_s)}, test: {len(test_s)}")

    train_ds = PointHeatmapPatchDataset(
        train_s,
        patch_size=(args.patch_h, args.patch_w),
        samples_per_epoch=args.train_samples_per_epoch,
        pos_fraction=args.pos_fraction,
        sigma=args.sigma,
        seed=args.seed,
    )
    val_ds = PointHeatmapPatchDataset(
        val_s,
        patch_size=(args.patch_h, args.patch_w),
        samples_per_epoch=args.val_samples_per_epoch,
        pos_fraction=args.pos_fraction,
        sigma=args.sigma,
        seed=args.seed + 1,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = SmallUNetDetector2D(in_channels=3, out_channels=2, base_channels=16).to(device)
    if args.resume_path:
        state = torch.load(args.resume_path, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded resume checkpoint: {args.resume_path}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.save_dir, exist_ok=True)
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = run_epoch(model, val_loader, criterion, None, device)
        print(f"Epoch {epoch:03d}/{args.epochs:03d} train={train_loss:.6f} val={val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            save_path = os.path.join(args.save_dir, args.save_name)
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")


if __name__ == "__main__":
    main()

