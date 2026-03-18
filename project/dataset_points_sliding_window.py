"""
Sliding window patch dataset - Extract 256×256 patches with stride to maximize data usage.

This variant of PointPatchDataset pre-computes all possible patch locations
and samples from them, effectively using 15-20× more data per epoch.
"""

from typing import List, Sequence, Tuple

import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset

from prepare_labels import ImageRecord, gaussian_heatmap
from augmentations import apply_augmentation, CLAHEPreprocess, MultiScaleSigmaJitter


def binary_disk_map(image_hw: Tuple[int, int], points: np.ndarray, radius: int = 3) -> np.ndarray:
    h, w = image_hw
    out = np.zeros((h, w), dtype=np.float32)
    if len(points) == 0:
        return out
    rr = int(max(1, radius))
    yy = np.arange(-rr, rr + 1, dtype=np.int32)[:, None]
    xx = np.arange(-rr, rr + 1, dtype=np.int32)[None, :]
    mask = (xx * xx + yy * yy) <= (rr * rr)
    dy, dx = np.where(mask)
    dy = dy - rr
    dx = dx - rr
    for x, y in points:
        cx = int(round(float(x)))
        cy = int(round(float(y)))
        xs = cx + dx
        ys = cy + dy
        keep = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        out[ys[keep], xs[keep]] = 1.0
    return out


def _to_chw_01(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    if image.ndim != 3:
        raise ValueError(f"Expected image with shape (H,W) or (H,W,C), got {image.shape}")
    image = image.astype(np.float32)
    mn, mx = float(image.min()), float(image.max())
    if mx > mn:
        image = (image - mn) / (mx - mn)
    else:
        image = np.zeros_like(image, dtype=np.float32)
    return np.transpose(image, (2, 0, 1))


class SlidingWindowPatchDataset(Dataset):
    """
    Extract 256×256 patches with sliding window to maximize training data.

    For a 2048×2048 image with stride=128, generates ~15-20 patches.
    10 images × 15-20 patches = 150-200 patches per epoch.

    This is 15-20× more data than random sampling (10 patches).
    """

    def __init__(
        self,
        records: Sequence[ImageRecord],
        patch_size: Tuple[int, int] = (256, 256),
        patch_stride: int = 128,
        samples_per_epoch: int = 256,
        pos_fraction: float = 0.6,
        sigma: float = 2.5,
        target_type: str = "gaussian",
        target_radius: int = 3,
        augment: bool = False,
        seed: int = 42,
        preprocess: bool = False,
        sigma_jitter: bool = False,
        consistency_pairs: bool = False,
    ) -> None:
        self.patch_h, self.patch_w = patch_size
        self.patch_stride = int(patch_stride)
        self.samples_per_epoch = int(samples_per_epoch)
        self.pos_fraction = float(pos_fraction)
        self.sigma = float(sigma)
        self.target_type = str(target_type)
        self.target_radius = int(target_radius)
        self.augment = augment
        self.preprocess = bool(preprocess)
        self.sigma_jitter = bool(sigma_jitter)
        self.consistency_pairs = bool(consistency_pairs)
        self.rng = np.random.default_rng(seed)

        # Setup preprocessing
        if self.preprocess:
            self.clahe = CLAHEPreprocess(tile_size=64, clip_limit=2.0)
        else:
            self.clahe = None

        # Setup sigma jitter
        if self.sigma_jitter:
            self.sigma_jitter_obj = MultiScaleSigmaJitter(sigma_range=(1.5, 3.5))
        else:
            self.sigma_jitter_obj = None

        # Pre-compute all patch locations
        self.patch_locations = []  # List of (image_idx, y0, x0)

        for img_idx, rec in enumerate(records):
            try:
                img = _to_chw_01(tifffile.imread(rec.image_path))
                if self.preprocess:
                    dummy_hm = np.zeros((2, img.shape[1], img.shape[2]), dtype=np.float32)
                    img, _ = self.clahe(img, dummy_hm)
                _, h, w = img.shape

                # Enumerate all sliding window positions
                for y0 in range(0, h - self.patch_h + 1, self.patch_stride):
                    for x0 in range(0, w - self.patch_w + 1, self.patch_stride):
                        self.patch_locations.append((img_idx, y0, x0, rec))

            except Exception as e:
                print(f"Warning: Could not load image {rec.image_id}: {e}")
                continue

        if not self.patch_locations:
            raise ValueError("No valid patch locations found!")

        print(f"SlidingWindowPatchDataset initialized:")
        print(f"  Records: {len(records)}")
        print(f"  Total patch locations: {len(self.patch_locations)}")
        print(f"  Patches per epoch (samples_per_epoch): {self.samples_per_epoch}")
        print(f"  Patch size: {self.patch_h}×{self.patch_w}, stride: {self.patch_stride}")

        # Cache images
        self.images = []
        self.points6_list = []
        self.points12_list = []
        self.points_all_list = []

        for rec in records:
            try:
                img = _to_chw_01(tifffile.imread(rec.image_path))
                if self.preprocess:
                    dummy_hm = np.zeros((2, img.shape[1], img.shape[2]), dtype=np.float32)
                    img, _ = self.clahe(img, dummy_hm)
                p6 = rec.points[0].astype(np.float32)
                p12 = rec.points[1].astype(np.float32)
                pall = np.concatenate([p6, p12], axis=0) if (len(p6) + len(p12)) > 0 else np.zeros((0, 2), np.float32)
                self.images.append(img)
                self.points6_list.append(p6)
                self.points12_list.append(p12)
                self.points_all_list.append(pall)
            except Exception as e:
                print(f"Warning: Could not cache image {rec.image_id}: {e}")
                continue

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _crop_points(self, points: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
        if len(points) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        m = (
            (points[:, 0] >= x0 - 3 * self.sigma)
            & (points[:, 0] < x1 + 3 * self.sigma)
            & (points[:, 1] >= y0 - 3 * self.sigma)
            & (points[:, 1] < y1 + 3 * self.sigma)
        )
        out = points[m].copy()
        out[:, 0] -= x0
        out[:, 1] -= y0
        return out

    def _augment(self, image: np.ndarray, heatmap: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply rich augmentation pipeline."""
        return apply_augmentation(image, heatmap, self.rng)

    def __getitem__(self, index: int):
        del index

        # Randomly select a patch location
        loc_idx = int(self.rng.integers(0, len(self.patch_locations)))
        img_idx, y0, x0, _ = self.patch_locations[loc_idx]

        image = self.images[img_idx]
        points6 = self.points6_list[img_idx]
        points12 = self.points12_list[img_idx]

        x1, y1 = x0 + self.patch_w, y0 + self.patch_h

        img_patch = image[:, y0:y1, x0:x1]
        p6 = self._crop_points(points6, x0, y0, x1, y1)
        p12 = self._crop_points(points12, x0, y0, x1, y1)

        # Generate heatmap with optional sigma jitter
        sigma_to_use = self.sigma
        if self.sigma_jitter and self.augment:
            sigma_to_use = self.sigma_jitter_obj.sample_sigma(self.rng)

        hm = np.zeros((2, self.patch_h, self.patch_w), dtype=np.float32)
        if self.target_type == "gaussian":
            hm[0] = gaussian_heatmap((self.patch_h, self.patch_w), p6, sigma=sigma_to_use)
            hm[1] = gaussian_heatmap((self.patch_h, self.patch_w), p12, sigma=sigma_to_use)
        else:
            hm[0] = binary_disk_map((self.patch_h, self.patch_w), p6, radius=self.target_radius)
            hm[1] = binary_disk_map((self.patch_h, self.patch_w), p12, radius=self.target_radius)

        # Apply augmentation
        if self.augment:
            img_patch, hm = self._augment(img_patch, hm)

        # For consistency pairs
        if self.consistency_pairs and self.augment:
            img_patch2, hm2 = self._augment(img_patch.copy(), hm.copy())
            return (
                torch.from_numpy(img_patch).float(),
                torch.from_numpy(hm).float(),
                torch.from_numpy(img_patch2).float(),
                torch.from_numpy(hm2).float(),
            )

        return torch.from_numpy(img_patch).float(), torch.from_numpy(hm).float()
