from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from scipy.ndimage import gaussian_laplace, maximum_filter


def _nms_points(
    points: Sequence[Tuple[float, float, float, float]],
    min_distance: int,
) -> List[Tuple[float, float, float, float]]:
    if not points:
        return []
    md2 = float(min_distance * min_distance)
    kept: List[Tuple[float, float, float, float]] = []
    for x, y, score, sigma in points:
        keep = True
        for kx, ky, _, _ in kept:
            if (x - kx) * (x - kx) + (y - ky) * (y - ky) < md2:
                keep = False
                break
        if keep:
            kept.append((x, y, score, sigma))
    return kept


def multiscale_log_candidates(
    image_hw: np.ndarray,
    sigmas: Sequence[float],
    threshold: float = 0.02,
    min_distance: int = 5,
    max_candidates: int = 500,
) -> List[Tuple[float, float, float, float]]:
    """
    Returns list of (x, y, score, sigma) using scale-normalized LoG maxima.
    """
    if image_hw.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {image_hw.shape}")
    img = image_hw.astype(np.float32)
    h, w = img.shape

    best_resp = np.full((h, w), -np.inf, dtype=np.float32)
    best_sigma = np.zeros((h, w), dtype=np.float32)

    for sigma in sigmas:
        # Dark circular particles become positive peaks after negated LoG.
        resp = -gaussian_laplace(img, sigma=float(sigma))
        # Scale normalization for fair comparison across sigmas.
        resp = (float(sigma) ** 2) * resp
        m = resp > best_resp
        best_resp[m] = resp[m]
        best_sigma[m] = float(sigma)

    local_max = maximum_filter(best_resp, size=2 * min_distance + 1)
    peaks = (best_resp == local_max) & (best_resp >= float(threshold))
    ys, xs = np.where(peaks)

    cand: List[Tuple[float, float, float, float]] = []
    for y, x in zip(ys, xs):
        cand.append((float(x), float(y), float(best_resp[y, x]), float(best_sigma[y, x])))
    cand.sort(key=lambda t: t[2], reverse=True)
    if max_candidates > 0:
        cand = cand[:max_candidates]
    cand = _nms_points(cand, min_distance=min_distance)
    return cand

