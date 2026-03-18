from __future__ import annotations

import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_utils import (
    crop_stack_to_shape,
    estimate_crop_offset_from_mask,
    extract_particle_centers,
    load_tiff_stack,
    verify_matching_shapes,
    visualize_random_slice_overlay,
)
from dataset import GoldParticleDataset
from model import SmallUNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a minimal U-Net on EM gold particle heatmaps.")
    parser.add_argument("--image_tif", type=str, required=True, help="Path to grayscale multipage TIFF stack.")
    parser.add_argument("--mask_tif", type=str, required=True, help="Path to segmentation multipage TIFF stack.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--sigma", type=float, default=2.0, help="Gaussian sigma for target heatmaps.")
    parser.add_argument(
        "--save_model_path",
        type=str,
        default=None,
        help="Optional path to save final model weights (.pt).",
    )
    parser.add_argument(
        "--max_slices",
        type=int,
        default=None,
        help="Optional cap on number of slices (for fast smoke tests).",
    )
    parser.add_argument(
        "--auto_align_crop",
        action="store_true",
        help=(
            "If shapes mismatch but slice counts match, estimate (y0, x0) crop offset from mask "
            "and crop image stack to mask shape."
        ),
    )
    parser.add_argument(
        "--align_num_slices",
        type=int,
        default=20,
        help="Number of mask-rich slices used for automatic crop-offset estimation.",
    )
    parser.add_argument(
        "--no_visualize",
        action="store_true",
        help="Disable random slice overlay visualization.",
    )
    return parser.parse_args()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        preds = model(images)
        loss = criterion(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading TIFF stacks...")
    image_stack = load_tiff_stack(args.image_tif)
    mask_stack = load_tiff_stack(args.mask_tif)

    if image_stack.shape != mask_stack.shape:
        if args.auto_align_crop:
            if image_stack.shape[0] != mask_stack.shape[0]:
                raise ValueError(
                    "Cannot auto-align crop because slice counts differ: "
                    f"image={image_stack.shape[0]}, mask={mask_stack.shape[0]}"
                )
            print("Shape mismatch detected. Estimating crop offset from segmentation mask...")
            y0, x0, score = estimate_crop_offset_from_mask(
                image_stack=image_stack,
                mask_stack=mask_stack,
                num_slices=args.align_num_slices,
            )
            print(f"Estimated crop offset: y0={y0}, x0={x0}, score={score:.6f}")
            image_stack = crop_stack_to_shape(
                image_stack,
                target_hw=mask_stack.shape[1:],
                y0=y0,
                x0=x0,
            )
            print(f"Cropped image stack shape: {image_stack.shape}")
        else:
            verify_matching_shapes(image_stack, mask_stack)

    verify_matching_shapes(image_stack, mask_stack)

    if args.max_slices is not None:
        if args.max_slices <= 0:
            raise ValueError("--max_slices must be a positive integer.")
        keep = min(args.max_slices, image_stack.shape[0])
        image_stack = image_stack[:keep]
        mask_stack = mask_stack[:keep]
        print(f"Using first {keep} slices for this run.")

    print(f"Loaded stacks with shape: {image_stack.shape}")

    if not args.no_visualize:
        print("Visualizing random slice overlay...")
        visualize_random_slice_overlay(image_stack, mask_stack)

    print("Extracting particle centers from segmentation...")
    centers_3d = extract_particle_centers(mask_stack)
    print(f"Detected {len(centers_3d)} particles across all slices.")

    dataset = GoldParticleDataset(image_stack=image_stack, centers_3d=centers_3d, sigma=args.sigma)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = SmallUNet(in_channels=1, out_channels=1, base_channels=16).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        epoch_loss = train_one_epoch(model, loader, criterion, optimizer, device)
        print(f"Epoch {epoch:03d}/{args.epochs:03d} - Loss: {epoch_loss:.6f}")

    if args.save_model_path:
        torch.save(model.state_dict(), args.save_model_path)
        print(f"Saved model weights to: {args.save_model_path}")


if __name__ == "__main__":
    main()
