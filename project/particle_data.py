from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import tifffile

from dataset_guard import enforce_allowed_data_root


@dataclass
class SynapseSample:
    synapse_id: str
    image_path: str
    points_px_6nm: np.ndarray  # (N, 2) in x,y pixel coordinates
    points_px_12nm: np.ndarray  # (N, 2) in x,y pixel coordinates


def _read_xy_csv(path: str) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=",", skip_header=1, usecols=(1, 2))
    if data.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if data.ndim == 1:
        data = data[None, :]
    return data.astype(np.float32)


def _xy_to_pixels(points_xy: np.ndarray, width: int, height: int) -> np.ndarray:
    if len(points_xy) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    pts = points_xy.copy()
    # In this dataset, CSV XY are usually normalized in [0, 1].
    if np.nanmax(pts) <= 1.5:
        pts[:, 0] = pts[:, 0] * float(width)
        pts[:, 1] = pts[:, 1] * float(height)
    pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
    return pts


def _find_primary_image_tif(synapse_dir: str) -> str | None:
    files = [f for f in os.listdir(synapse_dir) if f.lower().endswith(".tif")]
    candidates = []
    for f in files:
        low = f.lower()
        if "mask" in low or "color" in low or "overlay" in low:
            continue
        candidates.append(f)
    if not candidates:
        return None
    # Prefer shortest base filename for the primary raw image in this folder.
    candidates.sort(key=len)
    return os.path.join(synapse_dir, candidates[0])


def _collect_synapse_dirs(root_dir: str) -> List[str]:
    syn_dirs: List[str] = []
    for dp, _, files in os.walk(root_dir):
        base = os.path.basename(dp)
        if not base.upper().startswith("S"):
            continue
        has_tif = any(f.lower().endswith(".tif") for f in files)
        if has_tif:
            syn_dirs.append(dp)
    return sorted(set(syn_dirs))


def discover_synapse_samples(root_dir: str) -> List[SynapseSample]:
    root_dir = enforce_allowed_data_root(root_dir)
    samples: List[SynapseSample] = []
    synapse_dirs = _collect_synapse_dirs(root_dir)

    for syn_dir in synapse_dirs:
        image_path = _find_primary_image_tif(syn_dir)
        if image_path is None:
            continue

        image = tifffile.imread(image_path)
        h, w = image.shape[:2]

        csv_paths: List[str] = []
        for dp, _, files in os.walk(syn_dir):
            for f in files:
                if f.lower().endswith(".csv"):
                    csv_paths.append(os.path.join(dp, f))

        points_6: List[np.ndarray] = []
        points_12: List[np.ndarray] = []
        for csv_path in csv_paths:
            name = os.path.basename(csv_path).lower()
            pts = _read_xy_csv(csv_path)
            if "6nm" in name:
                points_6.append(pts)
            elif "12nm" in name:
                points_12.append(pts)

        pts6 = np.concatenate(points_6, axis=0) if points_6 else np.zeros((0, 2), dtype=np.float32)
        pts12 = np.concatenate(points_12, axis=0) if points_12 else np.zeros((0, 2), dtype=np.float32)
        pts6 = _xy_to_pixels(pts6, w, h)
        pts12 = _xy_to_pixels(pts12, w, h)

        samples.append(
            SynapseSample(
                synapse_id=os.path.basename(syn_dir),
                image_path=image_path,
                points_px_6nm=pts6,
                points_px_12nm=pts12,
            )
        )
    return samples


def split_samples(
    samples: Sequence[SynapseSample], train_ratio: float = 0.7, val_ratio: float = 0.15, seed: int = 42
) -> Tuple[List[SynapseSample], List[SynapseSample], List[SynapseSample]]:
    if len(samples) < 3:
        raise ValueError("Need at least 3 samples for train/val/test split.")
    rng = np.random.default_rng(seed)
    idx = np.arange(len(samples))
    rng.shuffle(idx)

    n = len(samples)
    n_train = max(1, int(round(n * train_ratio)))
    n_val = max(1, int(round(n * val_ratio)))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
    n_test = n - n_train - n_val
    if n_test <= 0:
        n_test = 1
        n_train = max(1, n_train - 1)

    train = [samples[i] for i in idx[:n_train]]
    val = [samples[i] for i in idx[n_train : n_train + n_val]]
    test = [samples[i] for i in idx[n_train + n_val :]]
    return train, val, test


def gaussian_heatmap_2c(
    image_hw: Tuple[int, int],
    points_6nm: np.ndarray,
    points_12nm: np.ndarray,
    sigma: float = 2.0,
) -> np.ndarray:
    h, w = image_hw
    yy = np.arange(h, dtype=np.float32)[:, None]
    xx = np.arange(w, dtype=np.float32)[None, :]
    out = np.zeros((2, h, w), dtype=np.float32)
    denom = 2.0 * sigma * sigma

    for cls, points in enumerate([points_6nm, points_12nm]):
        for x, y in points:
            g = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / denom)
            out[cls] = np.maximum(out[cls], g.astype(np.float32))
    return out


def image_to_chw_float(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    if image.ndim != 3:
        raise ValueError(f"Expected image shape (H,W) or (H,W,C), got {image.shape}")
    image = image.astype(np.float32)
    mn, mx = float(image.min()), float(image.max())
    if mx > mn:
        image = (image - mn) / (mx - mn)
    else:
        image = np.zeros_like(image, dtype=np.float32)
    return np.transpose(image, (2, 0, 1))

