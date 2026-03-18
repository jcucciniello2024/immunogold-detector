
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


class PointPatchDataset(Dataset):
    """
    Returns:
      image: (3, 512, 512)
      heatmap: (2, 512, 512)
    """

    def __init__(
        self,
        records: Sequence[ImageRecord],
        patch_size: Tuple[int, int] = (512, 512),
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
        if self.target_type not in {"gaussian", "disk"}:
            raise ValueError(f"Unsupported target_type={self.target_type}; use 'gaussian' or 'disk'.")

        self.clahe = CLAHEPreprocess() if self.preprocess else None
        self.sigma_jitter_obj = MultiScaleSigmaJitter() if self.sigma_jitter else None

        self.images: List[np.ndarray] = []
        self.points6: List[np.ndarray] = []
        self.points12: List[np.ndarray] = []
        self.points_all: List[np.ndarray] = []
        for r in records:
            img = _to_chw_01(tifffile.imread(r.image_path))
            _, h, w = img.shape
            if h < self.patch_h or w < self.patch_w:
                continue
            # Apply CLAHE preprocessing if enabled
            if self.preprocess:
                dummy_hm = np.zeros((1, h, w), dtype=np.float32)
                img, _ = self.clahe(img, dummy_hm)
            p6 = r.points[0].astype(np.float32)
            p12 = r.points[1].astype(np.float32)
            pall = np.concatenate([p6, p12], axis=0) if (len(p6) + len(p12)) > 0 else np.zeros((0, 2), np.float32)
            self.images.append(img)
            self.points6.append(p6)
            self.points12.append(p12)
            self.points_all.append(pall)
        if not self.images:
            raise ValueError("No valid records for patch dataset.")

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
        """Apply rich augmentation pipeline using augmentations module."""
        return apply_augmentation(image, heatmap, self.rng)

    def __getitem__(self, index: int):
        del index
        i = int(self.rng.integers(0, len(self.images)))
        image = self.images[i]
        points_all = self.points_all[i]
        points6 = self.points6[i]
        points12 = self.points12[i]
        _, h, w = image.shape

        use_pos = len(points_all) > 0 and self.rng.random() < self.pos_fraction
        if use_pos:
            positive_pools = []
            if len(points6) > 0:
                positive_pools.append(points6)
            if len(points12) > 0:
                positive_pools.append(points12)
            chosen_pool = positive_pools[int(self.rng.integers(0, len(positive_pools)))]
            x, y = chosen_pool[int(self.rng.integers(0, len(chosen_pool)))]
            x0 = int(round(x - self.patch_w / 2 + self.rng.integers(-24, 25)))
            y0 = int(round(y - self.patch_h / 2 + self.rng.integers(-24, 25)))
            x0 = max(0, min(x0, w - self.patch_w))
            y0 = max(0, min(y0, h - self.patch_h))
        else:
            x0 = int(self.rng.integers(0, w - self.patch_w + 1))
            y0 = int(self.rng.integers(0, h - self.patch_h + 1))
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

        # For consistency pairs, return two independently augmented views of same patch
        if self.consistency_pairs and self.augment:
            img_patch2, hm2 = self._augment(img_patch.copy(), hm.copy())
            return (
                torch.from_numpy(img_patch).float(),
                torch.from_numpy(hm).float(),
                torch.from_numpy(img_patch2).float(),
                torch.from_numpy(hm2).float(),
            )

        return torch.from_numpy(img_patch).float(), torch.from_numpy(hm).float()

