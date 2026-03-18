from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy import ndimage as ndi
from scipy.signal import fftconvolve


Center3D = Tuple[int, float, float]  # (slice_index, x, y)
Center2D = Tuple[float, float]  # (x, y)


def load_tiff_stack(path: str) -> np.ndarray:
    """Load a multipage TIFF stack as a 3D array: (num_slices, H, W)."""
    stack = tifffile.imread(path)
    if stack.ndim != 3:
        raise ValueError(f"Expected 3D stack (Z, H, W), got shape {stack.shape}")
    return stack


def verify_matching_shapes(image_stack: np.ndarray, mask_stack: np.ndarray) -> None:
    """Raise an error if image and mask stack shapes do not match."""
    if image_stack.shape != mask_stack.shape:
        raise ValueError(
            "Image and segmentation stacks must have identical shapes. "
            f"Got image={image_stack.shape}, mask={mask_stack.shape}"
        )


def visualize_random_slice_overlay(
    image_stack: np.ndarray,
    mask_stack: np.ndarray,
    alpha: float = 0.35,
    random_seed: int | None = None,
) -> int:
    """Display a random slice overlay of grayscale image and segmentation mask."""
    verify_matching_shapes(image_stack, mask_stack)
    rng = np.random.default_rng(random_seed)
    slice_idx = int(rng.integers(0, image_stack.shape[0]))

    image_slice = image_stack[slice_idx]
    mask_slice = mask_stack[slice_idx] > 0

    plt.figure(figsize=(7, 7))
    plt.imshow(image_slice, cmap="gray")
    plt.imshow(np.ma.masked_where(~mask_slice, mask_slice), cmap="autumn", alpha=alpha)
    plt.title(f"Slice {slice_idx}: image + segmentation overlay")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    return slice_idx


def extract_particle_centers(mask_stack: np.ndarray) -> List[Center3D]:
    """
    Convert a binary/label mask stack into particle centers.

    Returns:
        List of (slice_index, x, y), one row per connected component.
    """
    if mask_stack.ndim != 3:
        raise ValueError(f"Expected 3D mask stack (Z, H, W), got shape {mask_stack.shape}")

    centers: List[Center3D] = []
    structure = np.ones((3, 3), dtype=np.uint8)  # 8-connectivity in 2D

    for z in range(mask_stack.shape[0]):
        binary = mask_stack[z] > 0
        labeled, num_objects = ndi.label(binary, structure=structure)
        if num_objects == 0:
            continue

        # center_of_mass returns (y, x) for 2D slices
        centroids_yx = ndi.center_of_mass(binary, labeled, range(1, num_objects + 1))
        for y, x in centroids_yx:
            centers.append((z, float(x), float(y)))

    return centers


def centers_to_slice_dict(centers: Sequence[Center3D], num_slices: int) -> Dict[int, List[Center2D]]:
    """Map global (z, x, y) center list to per-slice [(x, y), ...]."""
    per_slice: Dict[int, List[Center2D]] = {z: [] for z in range(num_slices)}
    for z, x, y in centers:
        per_slice[z].append((x, y))
    return per_slice


def generate_gaussian_heatmap(
    image_shape: Tuple[int, int],
    centers: Sequence[Center2D],
    sigma: float = 2.0,
) -> np.ndarray:
    """
    Generate a 2D heatmap with Gaussian blobs centered at particle coordinates.

    Args:
        image_shape: (H, W)
        centers: list of (x, y) centers for one slice
        sigma: Gaussian standard deviation in pixels
    """
    h, w = image_shape
    yy = np.arange(h, dtype=np.float32)[:, None]
    xx = np.arange(w, dtype=np.float32)[None, :]
    heatmap = np.zeros((h, w), dtype=np.float32)

    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    denom = 2.0 * sigma * sigma
    for x, y in centers:
        gaussian = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / denom)
        # Merge overlapping particles without unbounded accumulation.
        heatmap = np.maximum(heatmap, gaussian.astype(np.float32))

    return heatmap


def generate_gaussian_heatmap_3d(
    volume_shape: Tuple[int, int, int],
    centers: Sequence[Center3D],
    sigma_xy: float = 2.0,
    sigma_z: float = 1.5,
) -> np.ndarray:
    """
    Generate a 3D heatmap with Gaussian blobs centered at (z, x, y).

    Args:
        volume_shape: (D, H, W)
        centers: list of (slice_index, x, y)
        sigma_xy: Gaussian std in X/Y
        sigma_z: Gaussian std along Z
    """
    d, h, w = volume_shape
    zz = np.arange(d, dtype=np.float32)[:, None, None]
    yy = np.arange(h, dtype=np.float32)[None, :, None]
    xx = np.arange(w, dtype=np.float32)[None, None, :]
    heatmap = np.zeros((d, h, w), dtype=np.float32)

    if sigma_xy <= 0 or sigma_z <= 0:
        raise ValueError("sigma_xy and sigma_z must be positive.")

    denom_xy = 2.0 * sigma_xy * sigma_xy
    denom_z = 2.0 * sigma_z * sigma_z

    for z0, x0, y0 in centers:
        gaussian = np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / denom_xy - ((zz - z0) ** 2) / denom_z)
        heatmap = np.maximum(heatmap, gaussian.astype(np.float32))

    return heatmap


def _normalize_slice(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    mean = float(image.mean())
    std = float(image.std())
    if std < 1e-6:
        return np.zeros_like(image, dtype=np.float32)
    return (image - mean) / std


def estimate_crop_offset_from_mask(
    image_stack: np.ndarray,
    mask_stack: np.ndarray,
    num_slices: int = 20,
) -> Tuple[int, int, float]:
    """
    Estimate where the mask frame is located inside a larger image frame.

    Strategy:
    - Pick slices with the most mask pixels.
    - For each selected slice, compute valid cross-correlation:
      score(y, x) = sum( normalized_image_crop(y, x) * mask )
    - Average score maps across slices.
    - Since gold particles are dark in EM, use the minimum average score.

    Returns:
        y0, x0, score_at_best_offset
    """
    if image_stack.ndim != 3 or mask_stack.ndim != 3:
        raise ValueError("Both stacks must be 3D with shape (Z, H, W).")
    if image_stack.shape[0] != mask_stack.shape[0]:
        raise ValueError(
            "Image and mask stacks must have the same slice count. "
            f"Got {image_stack.shape[0]} vs {mask_stack.shape[0]}"
        )

    _, big_h, big_w = image_stack.shape
    _, small_h, small_w = mask_stack.shape
    if small_h > big_h or small_w > big_w:
        raise ValueError(
            "Mask frame must be smaller than or equal to image frame for crop alignment. "
            f"Got image HW=({big_h}, {big_w}), mask HW=({small_h}, {small_w})"
        )

    mask_binary = (mask_stack > 0).astype(np.float32)
    counts = mask_binary.reshape(mask_binary.shape[0], -1).sum(axis=1)
    nonzero_slices = np.where(counts > 0)[0]
    if len(nonzero_slices) == 0:
        raise ValueError("Mask has no positive pixels; cannot estimate crop offset.")

    # Use slices with the highest particle density to improve signal.
    order = np.argsort(counts[nonzero_slices])[::-1]
    selected = nonzero_slices[order[: min(num_slices, len(nonzero_slices))]]

    score_accum: np.ndarray | None = None
    used = 0
    for z in selected:
        mask_slice = mask_binary[z]
        weight = float(mask_slice.sum())
        if weight <= 0:
            continue
        image_slice = _normalize_slice(image_stack[z])
        # Cross-correlation via convolution with flipped template.
        score = fftconvolve(image_slice, mask_slice[::-1, ::-1], mode="valid") / weight
        if score_accum is None:
            score_accum = score
        else:
            score_accum += score
        used += 1

    if score_accum is None or used == 0:
        raise ValueError("No valid slices for crop-offset estimation.")

    score_mean = score_accum / float(used)
    y0, x0 = np.unravel_index(np.argmin(score_mean), score_mean.shape)
    return int(y0), int(x0), float(score_mean[y0, x0])


def crop_stack_to_shape(
    stack: np.ndarray,
    target_hw: Tuple[int, int],
    y0: int,
    x0: int,
) -> np.ndarray:
    """Crop a (Z, H, W) stack at offset (y0, x0) to target (H, W)."""
    if stack.ndim != 3:
        raise ValueError(f"Expected stack shape (Z, H, W), got {stack.shape}")

    target_h, target_w = target_hw
    _, h, w = stack.shape
    if y0 < 0 or x0 < 0 or y0 + target_h > h or x0 + target_w > w:
        raise ValueError(
            "Crop exceeds stack bounds. "
            f"stack HW=({h}, {w}), target HW=({target_h}, {target_w}), offset=({y0}, {x0})"
        )
    return stack[:, y0 : y0 + target_h, x0 : x0 + target_w]
