from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from data_utils import Center3D, generate_gaussian_heatmap_3d


class GoldParticle3DPatchDataset(Dataset):
    """
    3D patch dataset for volumetric training.

    Returns tensors with shape:
    - image: (1, D, H, W)
    - target heatmap: (1, D, H, W)
    """

    def __init__(
        self,
        image_stack: np.ndarray,
        centers_3d: Sequence[Center3D],
        patch_size: Tuple[int, int, int] = (16, 128, 128),
        samples_per_epoch: int = 256,
        pos_fraction: float = 0.5,
        sigma_xy: float = 2.0,
        sigma_z: float = 1.5,
        seed: int = 42,
    ) -> None:
        if image_stack.ndim != 3:
            raise ValueError(f"Expected image_stack shape (Z, H, W), got {image_stack.shape}")

        self.image_stack = image_stack.astype(np.float32)
        self.d, self.h, self.w = self.image_stack.shape
        self.patch_d, self.patch_h, self.patch_w = patch_size
        self.samples_per_epoch = int(samples_per_epoch)
        self.pos_fraction = float(pos_fraction)
        self.sigma_xy = float(sigma_xy)
        self.sigma_z = float(sigma_z)
        self.rng = np.random.default_rng(seed)

        if not (0.0 <= self.pos_fraction <= 1.0):
            raise ValueError("pos_fraction must be in [0, 1].")
        if self.patch_d > self.d or self.patch_h > self.h or self.patch_w > self.w:
            raise ValueError(
                f"Patch size {patch_size} cannot exceed volume size {(self.d, self.h, self.w)}."
            )

        # Normalize to [0, 1] for stable training.
        min_val = float(self.image_stack.min())
        max_val = float(self.image_stack.max())
        if max_val > min_val:
            self.image_stack = (self.image_stack - min_val) / (max_val - min_val)
        else:
            self.image_stack = np.zeros_like(self.image_stack, dtype=np.float32)

        if len(centers_3d) > 0:
            self.centers = np.array(centers_3d, dtype=np.float32)  # columns: z, x, y
        else:
            self.centers = np.zeros((0, 3), dtype=np.float32)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _random_start(self) -> Tuple[int, int, int]:
        z0 = int(self.rng.integers(0, self.d - self.patch_d + 1))
        y0 = int(self.rng.integers(0, self.h - self.patch_h + 1))
        x0 = int(self.rng.integers(0, self.w - self.patch_w + 1))
        return z0, y0, x0

    def _positive_start(self) -> Tuple[int, int, int]:
        # Center patch around a random positive point with small jitter.
        idx = int(self.rng.integers(0, len(self.centers)))
        zc, xc, yc = self.centers[idx]
        zc, xc, yc = int(round(zc)), int(round(xc)), int(round(yc))

        z0 = zc - self.patch_d // 2 + int(self.rng.integers(-2, 3))
        y0 = yc - self.patch_h // 2 + int(self.rng.integers(-8, 9))
        x0 = xc - self.patch_w // 2 + int(self.rng.integers(-8, 9))

        z0 = max(0, min(z0, self.d - self.patch_d))
        y0 = max(0, min(y0, self.h - self.patch_h))
        x0 = max(0, min(x0, self.w - self.patch_w))
        return z0, y0, x0

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        del idx  # sampled stochastically

        use_positive = (len(self.centers) > 0) and (self.rng.random() < self.pos_fraction)
        if use_positive:
            z0, y0, x0 = self._positive_start()
        else:
            z0, y0, x0 = self._random_start()

        z1, y1, x1 = z0 + self.patch_d, y0 + self.patch_h, x0 + self.patch_w
        image_patch = self.image_stack[z0:z1, y0:y1, x0:x1]

        # Keep only centers near this patch. Small margin avoids truncated blobs.
        margin_xy = 3.0 * self.sigma_xy
        margin_z = 3.0 * self.sigma_z
        patch_centers: list[Center3D] = []
        if len(self.centers) > 0:
            for zc, xc, yc in self.centers:
                if (
                    z0 - margin_z <= zc < z1 + margin_z
                    and y0 - margin_xy <= yc < y1 + margin_xy
                    and x0 - margin_xy <= xc < x1 + margin_xy
                ):
                    patch_centers.append((float(zc - z0), float(xc - x0), float(yc - y0)))

        target_patch = generate_gaussian_heatmap_3d(
            volume_shape=(self.patch_d, self.patch_h, self.patch_w),
            centers=patch_centers,
            sigma_xy=self.sigma_xy,
            sigma_z=self.sigma_z,
        )

        image_tensor = torch.from_numpy(image_patch).unsqueeze(0).to(torch.float32)
        target_tensor = torch.from_numpy(target_patch).unsqueeze(0).to(torch.float32)
        return image_tensor, target_tensor
