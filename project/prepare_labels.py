import argparse
import csv
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import tifffile

from dataset_guard import enforce_allowed_data_root


CLASS_TO_ID = {"6nm": 0, "12nm": 1}
ID_TO_CLASS = {0: "6nm", 1: "12nm"}


@dataclass
class ImageRecord:
    image_id: str
    image_path: str
    width: int
    height: int
    points: Dict[int, np.ndarray]  # class_id -> (N, 2), x,y in pixel coords


def _infer_class_from_filename(path: str) -> Optional[int]:
    low = os.path.basename(path).lower()
    if "6nm" in low:
        return 0
    if "12nm" in low:
        return 1
    return None


def _parse_csv_points(path: str, width: int, height: int) -> Dict[int, List[Tuple[float, float]]]:
    """
    Supports:
      1) x,y,particle_type
      2) id,x,y (class inferred from file name)
    """
    class_points: Dict[int, List[Tuple[float, float]]] = {0: [], 1: []}
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return class_points

    header = [h.strip().lower() for h in rows[0]]
    data_rows = rows[1:]
    has_type = "particle_type" in header
    has_x = "x" in header
    has_y = "y" in header

    if has_x and has_y:
        ix = header.index("x")
        iy = header.index("y")
        it = header.index("particle_type") if has_type else -1
    else:
        # Fallback for format: [index, X, Y]
        ix, iy, it = 1, 2, -1

    inferred_class = _infer_class_from_filename(path)
    for row in data_rows:
        if len(row) <= max(ix, iy):
            continue
        try:
            x = float(row[ix])
            y = float(row[iy])
        except ValueError:
            continue

        # Normalize if needed.
        if x <= 1.5 and y <= 1.5:
            x *= width
            y *= height
        x = float(np.clip(x, 0, width - 1))
        y = float(np.clip(y, 0, height - 1))

        if it >= 0 and it < len(row):
            t = row[it].strip().lower()
            if t in CLASS_TO_ID:
                class_points[CLASS_TO_ID[t]].append((x, y))
                continue
        if inferred_class is not None:
            class_points[inferred_class].append((x, y))

    return class_points


def _find_primary_image(syn_dir: str) -> Optional[str]:
    files = [f for f in os.listdir(syn_dir) if f.lower().endswith(".tif")]
    candidates = []
    for f in files:
        low = f.lower()
        if "mask" in low or "color" in low or "overlay" in low:
            continue
        candidates.append(f)
    if not candidates:
        return None
    candidates.sort(key=len)
    return os.path.join(syn_dir, candidates[0])


def _collect_synapse_dirs(data_root: str) -> List[str]:
    syn_dirs: List[str] = []
    for dp, _, files in os.walk(data_root):
        base = os.path.basename(dp)
        if not base.upper().startswith("S"):
            continue
        has_tif = any(f.lower().endswith(".tif") for f in files)
        if has_tif:
            syn_dirs.append(dp)
    return sorted(set(syn_dirs))


def discover_image_records(data_root: str) -> List[ImageRecord]:
    data_root = enforce_allowed_data_root(data_root)
    records: List[ImageRecord] = []
    syn_dirs = _collect_synapse_dirs(data_root)
    for syn_dir in syn_dirs:
        image_path = _find_primary_image(syn_dir)
        if image_path is None:
            continue
        image = tifffile.imread(image_path)
        if image.ndim == 2:
            h, w = image.shape
        else:
            h, w = image.shape[:2]

        csv_paths: List[str] = []
        for dp, _, files in os.walk(syn_dir):
            for f in files:
                if f.lower().endswith(".csv"):
                    csv_paths.append(os.path.join(dp, f))

        points_raw: Dict[int, List[Tuple[float, float]]] = {0: [], 1: []}
        for csv_path in csv_paths:
            parsed = _parse_csv_points(csv_path, width=w, height=h)
            points_raw[0].extend(parsed[0])
            points_raw[1].extend(parsed[1])

        points = {
            0: np.array(points_raw[0], dtype=np.float32).reshape(-1, 2),
            1: np.array(points_raw[1], dtype=np.float32).reshape(-1, 2),
        }
        records.append(
            ImageRecord(
                image_id=os.path.basename(syn_dir),
                image_path=image_path,
                width=w,
                height=h,
                points=points,
            )
        )
    return records


def gaussian_heatmap(
    image_hw: Tuple[int, int], points: np.ndarray, sigma: float = 2.5
) -> np.ndarray:
    h, w = image_hw
    heat = np.zeros((h, w), dtype=np.float32)
    if len(points) == 0:
        return heat
    yy = np.arange(h, dtype=np.float32)[:, None]
    xx = np.arange(w, dtype=np.float32)[None, :]
    denom = 2.0 * sigma * sigma
    for x, y in points:
        g = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / denom)
        heat = np.maximum(heat, g.astype(np.float32))
    return heat


def build_target_heatmap(record: ImageRecord, sigma: float = 2.5) -> np.ndarray:
    out = np.zeros((2, record.height, record.width), dtype=np.float32)
    out[0] = gaussian_heatmap((record.height, record.width), record.points[0], sigma=sigma)
    out[1] = gaussian_heatmap((record.height, record.width), record.points[1], sigma=sigma)
    return out


def save_manifest_and_targets(records: Sequence[ImageRecord], out_dir: str, sigma: float = 2.5) -> None:
    os.makedirs(out_dir, exist_ok=True)
    target_dir = os.path.join(out_dir, "heatmaps")
    os.makedirs(target_dir, exist_ok=True)

    manifest = []
    for r in records:
        target = build_target_heatmap(r, sigma=sigma)
        heat_path = os.path.join(target_dir, f"{r.image_id}_heatmap.npy")
        np.save(heat_path, target)
        manifest.append(
            {
                "image_id": r.image_id,
                "image_path": r.image_path,
                "width": r.width,
                "height": r.height,
                "heatmap_path": heat_path,
                "num_points_6nm": int(len(r.points[0])),
                "num_points_12nm": int(len(r.points[1])),
            }
        )

    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare 2D keypoint heatmap labels from CSV annotations.")
    p.add_argument("--data_root", type=str, required=True, help="Root with synapse subfolders (S1, S2, ...).")
    p.add_argument("--out_dir", type=str, default="prepared_labels")
    p.add_argument("--sigma", type=float, default=2.5)
    args = p.parse_args()

    records = discover_image_records(args.data_root)
    print(f"Discovered {len(records)} image records.")
    save_manifest_and_targets(records, args.out_dir, sigma=args.sigma)
    print(f"Saved manifest + heatmaps to {args.out_dir}")


if __name__ == "__main__":
    main()
