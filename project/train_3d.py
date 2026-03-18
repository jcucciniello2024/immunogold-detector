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
)
from dataset_3d import GoldParticle3DPatchDataset
from model_3d import SmallUNet3D


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small 3D U-Net (5D tensors: N,C,D,H,W).")
    parser.add_argument("--image_tif", type=str, required=True, help="Path to grayscale multipage TIFF stack.")
    parser.add_argument("--mask_tif", type=str, required=True, help="Path to segmentation multipage TIFF stack.")

    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for 3D patches.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")

    parser.add_argument("--patch_d", type=int, default=16, help="Patch depth.")
    parser.add_argument("--patch_h", type=int, default=128, help="Patch height.")
    parser.add_argument("--patch_w", type=int, default=128, help="Patch width.")
    parser.add_argument("--samples_per_epoch", type=int, default=256, help="Number of patches sampled per epoch.")
    parser.add_argument("--pos_fraction", type=float, default=0.5, help="Fraction of positive-centered patches.")

    parser.add_argument("--sigma_xy", type=float, default=2.0, help="Gaussian sigma in X/Y.")
    parser.add_argument("--sigma_z", type=float, default=1.5, help="Gaussian sigma along Z.")

    parser.add_argument("--max_slices", type=int, default=None, help="Optional cap on number of slices.")
    parser.add_argument(
        "--auto_align_crop",
        action="store_true",
        help="Estimate crop offset from mask and crop image stack to mask shape if needed.",
    )
    parser.add_argument("--align_num_slices", type=int, default=20, help="Slices used in crop offset estimation.")
    parser.add_argument("--save_model_path", type=str, default=None, help="Optional path to save final model.")
    return parser.parse_args()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total = 0.0
    count = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        preds = model(images)
        loss = criterion(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = images.size(0)
        total += loss.item() * bs
        count += bs

    return total / max(count, 1)


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
            y0, x0, score = estimate_crop_offset_from_mask(image_stack, mask_stack, args.align_num_slices)
            print(f"Estimated crop offset: y0={y0}, x0={x0}, score={score:.6f}")
            image_stack = crop_stack_to_shape(image_stack, mask_stack.shape[1:], y0, x0)
            print(f"Cropped image stack shape: {image_stack.shape}")
        else:
            verify_matching_shapes(image_stack, mask_stack)

    verify_matching_shapes(image_stack, mask_stack)

    if args.max_slices is not None:
        if args.max_slices <= 0:
            raise ValueError("--max_slices must be positive.")
        keep = min(args.max_slices, image_stack.shape[0])
        image_stack = image_stack[:keep]
        mask_stack = mask_stack[:keep]
        print(f"Using first {keep} slices.")

    print(f"Aligned stack shape: {image_stack.shape}")
    centers_3d = extract_particle_centers(mask_stack)
    print(f"Detected {len(centers_3d)} particle centers.")

    dataset = GoldParticle3DPatchDataset(
        image_stack=image_stack,
        centers_3d=centers_3d,
        patch_size=(args.patch_d, args.patch_h, args.patch_w),
        samples_per_epoch=args.samples_per_epoch,
        pos_fraction=args.pos_fraction,
        sigma_xy=args.sigma_xy,
        sigma_z=args.sigma_z,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = SmallUNet3D(in_channels=1, out_channels=1, base_channels=8).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print("Starting 3D training...")
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, loader, criterion, optimizer, device)
        print(f"Epoch {epoch:03d}/{args.epochs:03d} - Loss: {loss:.6f}")

    if args.save_model_path:
        torch.save(model.state_dict(), args.save_model_path)
        print(f"Saved model weights to: {args.save_model_path}")


if __name__ == "__main__":
    main()
