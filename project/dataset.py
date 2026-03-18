from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from data_utils import Center3D, centers_to_slice_dict, generate_gaussian_heatmap


class GoldParticleDataset(Dataset):
    """
    Minimal dataset for EM slice -> Gaussian heatmap regression.
    Returns:
        image_tensor: (1, H, W), float32 in [0, 1]
        heatmap_tensor: (1, H, W), float32 in [0, 1]
    """

    def __init__(
        self,
        image_stack: np.ndarray,
        centers_3d: Sequence[Center3D],
        sigma: float = 2.0,
    ) -> None:
        if image_stack.ndim != 3:
            raise ValueError(f"Expected image_stack shape (Z, H, W), got {image_stack.shape}")

        self.image_stack = image_stack.astype(np.float32)
        self.num_slices, self.height, self.width = self.image_stack.shape
        self.sigma = float(sigma)

        # Normalize entire stack to [0, 1].
        min_val = float(self.image_stack.min())
        max_val = float(self.image_stack.max())
        if max_val > min_val:
            self.image_stack = (self.image_stack - min_val) / (max_val - min_val)
        else:
            self.image_stack = np.zeros_like(self.image_stack, dtype=np.float32)

        self.centers_per_slice = centers_to_slice_dict(centers_3d, self.num_slices)

    def __len__(self) -> int:
        return self.num_slices

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = self.image_stack[idx]
        centers = self.centers_per_slice[idx]
        heatmap = generate_gaussian_heatmap((self.height, self.width), centers, sigma=self.sigma)

        image_tensor = torch.from_numpy(image).unsqueeze(0).to(torch.float32)
        heatmap_tensor = torch.from_numpy(heatmap).unsqueeze(0).to(torch.float32)
        return image_tensor, heatmap_tensor
