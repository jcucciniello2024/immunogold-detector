from __future__ import annotations

import argparse
import csv
import os
from typing import List, Sequence, Tuple

import numpy as np
import tifffile
import torch
from scipy.ndimage import gaussian_filter, maximum_filter

from model_unet import UNetKeypointDetector
from prepare_labels import ID_TO_CLASS, discover_image_records


def image_to_chw_01(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    img = image.astype(np.float32)
    mn, mx = float(img.min()), float(img.max())
    if mx > mn:
        img = (img - mn) / (mx - mn)
    else:
        img = np.zeros_like(img, dtype=np.float32)
    return np.transpose(img, (2, 0, 1))


def tiled_inference(
    model: torch.nn.Module,
    image_chw: np.ndarray,
    tile_hw: Tuple[int, int],
    stride_hw: Tuple[int, int],
    device: torch.device,
    out_channels: int = 2,
) -> np.ndarray:
    c, h, w = image_chw.shape
    th, tw = tile_hw
    sh, sw = stride_hw
    out = np.zeros((out_channels, h, w), dtype=np.float32)
    cnt = np.zeros((1, h, w), dtype=np.float32)

    ys = list(range(0, max(1, h - th + 1), sh))
    xs = list(range(0, max(1, w - tw + 1), sw))
    if not ys or ys[-1] != h - th:
        ys.append(max(0, h - th))
    if not xs or xs[-1] != w - tw:
        xs.append(max(0, w - tw))

    model.eval()
    with torch.no_grad():
        for y0 in ys:
            for x0 in xs:
                patch = image_chw[:, y0 : y0 + th, x0 : x0 + tw]
                t = torch.from_numpy(patch[None]).float().to(device)
                pred = torch.sigmoid(model(t))[0].cpu().numpy()
                out[:, y0 : y0 + th, x0 : x0 + tw] += pred
                cnt[:, y0 : y0 + th, x0 : x0 + tw] += 1.0
    out /= np.maximum(cnt, 1e-6)
    return out


def peak_detect(
    heatmap: np.ndarray,
    threshold: float = 0.5,
    min_distance: int = 5,
    max_peaks: int = 2000,
    *,
    blur_sigma: float = 0.0,
    local_max_only: bool = True,
    local_max_size: int = 3,
) -> List[Tuple[float, float, float]]:
    """
    Greedy NMS on candidate pixels above threshold.

    If local_max_only is True, only **strict local maxima** (Gold-Digger / blob style)
    are considered. That removes dense clusters of X's on heatmap plateaus and
    streaks of tied pixels along edges.

    Optional blur_sigma>0 merges sub-pixel plateaus into a single peak before max filtering.
    """
    hm = np.asarray(heatmap, dtype=np.float32)
    if blur_sigma and float(blur_sigma) > 0:
        hm = gaussian_filter(hm, sigma=float(blur_sigma))

    if local_max_only:
        k = int(max(3, local_max_size | 1))  # odd >= 3
        mx = maximum_filter(hm, size=k, mode="nearest")
        eps = 1e-6 * max(float(hm.max()), 1.0)
        is_peak = hm >= (mx - eps)
        hm_masked = np.where(is_peak & (hm >= float(threshold)), hm, 0.0)
    else:
        hm_masked = np.where(hm >= float(threshold), hm, 0.0)

    h, w = hm_masked.shape
    candidates = np.where(hm_masked > 0)
    ys = candidates[0]
    xs = candidates[1]
    if len(xs) == 0:
        return []

    scores = hm_masked[ys, xs]
    order = np.argsort(scores)[::-1]
    suppressed = np.zeros((h, w), dtype=bool)
    r = int(max(1, min_distance))
    dets: List[Tuple[float, float, float]] = []

    for idx in order:
        y = int(ys[idx])
        x = int(xs[idx])
        if suppressed[y, x]:
            continue
        conf = float(hm_masked[y, x])
        dets.append((float(x), float(y), conf))
        if 0 < max_peaks <= len(dets):
            break
        y0, y1 = max(0, y - r), min(h, y + r + 1)
        x0, x1 = max(0, x - r), min(w, x + r + 1)
        suppressed[y0:y1, x0:x1] = True
    return dets


def suppress_linear_streaks(
    dets: Sequence[Tuple[float, float, float]],
    neighbor_radius: float = 12.0,
    min_cluster: int = 5,
    eigen_ratio: float = 22.0,
    keep_top_in_line: int = 2,
    mode: str = "thin",
) -> List[Tuple[float, float, float]]:
    """
    Thin out near-collinear chains (common failure mode: edge / membrane streaks
    decoded as many peaks). Real immunogold is roughly isotropic; long 1D chains
    of detections are usually artefacts.

    Groups points by distance < neighbor_radius, then if PCA eigenvalue ratio
    is high (elongated cluster):
      - mode="thin": keep strongest few peaks in that line-like group
      - mode="drop": remove the whole line-like group
    """
    dets = list(dets)
    if len(dets) < min_cluster:
        return dets

    pts = np.array([[d[0], d[1], d[2]] for d in dets], dtype=np.float64)
    n = len(pts)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    rr = float(neighbor_radius) ** 2
    for i in range(n):
        for j in range(i + 1, n):
            dx = pts[i, 0] - pts[j, 0]
            dy = pts[i, 1] - pts[j, 1]
            if dx * dx + dy * dy <= rr:
                union(i, j)

    groups: dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    keep_idx: set[int] = set()
    for _root, idxs in groups.items():
        if len(idxs) < min_cluster:
            keep_idx.update(idxs)
            continue
        xy = pts[idxs, :2]
        if len(idxs) == 1:
            keep_idx.update(idxs)
            continue
        xy_c = xy - xy.mean(axis=0, keepdims=True)
        cov = (xy_c.T @ xy_c) / max(1, len(idxs) - 1)
        eig = np.linalg.eigvalsh(cov)
        eig = np.sort(np.maximum(eig, 1e-12))
        ratio = float(eig[-1] / eig[0])
        if ratio > float(eigen_ratio):
            if str(mode).lower() == "drop":
                # Reject this elongated, likely edge-aligned artefact group.
                continue
            order = sorted(idxs, key=lambda i: pts[i, 2], reverse=True)
            keep_idx.update(order[: max(0, int(keep_top_in_line))])
        else:
            keep_idx.update(idxs)

    return [dets[i] for i in sorted(keep_idx)]


def decode_heatmap_to_detections(
    heatmap: np.ndarray,
    threshold: float,
    min_distance: int,
    max_peaks: int,
    *,
    blur_sigma: float = 0.0,
    local_max_only: bool = True,
    local_max_size: int = 3,
    suppress_line_streaks: bool = False,
    line_neighbor_radius: float = 12.0,
    line_min_cluster: int = 5,
    line_eigen_ratio: float = 22.0,
    line_keep_top_in_line: int = 2,
    line_mode: str = "thin",
) -> List[Tuple[float, float, float]]:
    dets = peak_detect(
        heatmap,
        threshold=threshold,
        min_distance=min_distance,
        max_peaks=max_peaks,
        blur_sigma=blur_sigma,
        local_max_only=local_max_only,
        local_max_size=local_max_size,
    )
    if suppress_line_streaks and len(dets) > 0:
        dets = suppress_linear_streaks(
            dets,
            neighbor_radius=line_neighbor_radius,
            min_cluster=line_min_cluster,
            eigen_ratio=line_eigen_ratio,
            keep_top_in_line=line_keep_top_in_line,
            mode=line_mode,
        )
    return dets


def filter_by_peak_support_size(
    heatmap: np.ndarray,
    dets: Sequence[Tuple[float, float, float]],
    *,
    support_rel_thresh: float = 0.70,
    support_window: int = 11,
    min_equiv_diameter_px: float = 1.0,
    max_equiv_diameter_px: float = 7.0,
) -> List[Tuple[float, float, float]]:
    """
    Keep only detections whose local *heatmap support* size is in a tiny-particle range.
    This rejects broad peaks from large dark artefacts while preserving sharp tiny spots.
    """
    hm = np.asarray(heatmap, dtype=np.float32)
    h, w = hm.shape
    r = int(max(2, support_window // 2))
    out: List[Tuple[float, float, float]] = []
    for x, y, conf in dets:
        xi = int(round(x))
        yi = int(round(y))
        y0, y1 = max(0, yi - r), min(h, yi + r + 1)
        x0, x1 = max(0, xi - r), min(w, xi + r + 1)
        patch = hm[y0:y1, x0:x1]
        if patch.size == 0:
            continue
        peak = float(np.max(patch))
        if peak <= 0:
            continue
        thr = float(np.clip(support_rel_thresh, 0.10, 0.95)) * peak
        area = int((patch >= thr).sum())
        if area <= 0:
            continue
        equiv_d = 2.0 * float(np.sqrt(area / np.pi))
        if equiv_d < float(min_equiv_diameter_px) or equiv_d > float(max_equiv_diameter_px):
            continue
        out.append((x, y, conf))
    return out


def enforce_global_non_touching_cap(
    dets_xyc_cls: Sequence[Tuple[float, float, float, int]],
    *,
    min_distance_px: float,
    max_total: int,
) -> List[Tuple[float, float, float, int]]:
    """
    Final safety gate:
    - globally non-touching across all classes
    - hard cap on total detections per image
    """
    if not dets_xyc_cls:
        return []
    r2 = float(min_distance_px) * float(min_distance_px)
    ordered = sorted(dets_xyc_cls, key=lambda t: t[2], reverse=True)
    kept: List[Tuple[float, float, float, int]] = []
    for x, y, conf, cls in ordered:
        ok = True
        for kx, ky, _kc, _kcls in kept:
            dx = x - kx
            dy = y - ky
            if dx * dx + dy * dy < r2:
                ok = False
                break
        if ok:
            kept.append((x, y, conf, cls))
        if 0 < int(max_total) <= len(kept):
            break
    return kept


def hard_particle_filter(
    image_chw: np.ndarray,
    dets: List[Tuple[float, float, float]],
    window: int = 21,
    center_radius: int = 2,
    ring_inner: int = 4,
    ring_outer: int = 8,
    max_center_intensity: float = 0.40,
    min_ring_contrast: float = 0.05,
    dark_quantile: float = 0.20,
    min_equiv_diameter: float = 2.0,
    max_equiv_diameter: float = 14.0,
    max_anisotropy: float = 2.8,
) -> List[Tuple[float, float, float]]:
    """
    Image-aware filter for immunogold-like particles.
    Keeps candidates that are dark, compact, roughly circular, and not streak-like.
    """
    gray = image_chw[0] if image_chw.ndim == 3 else image_chw
    h, w = gray.shape
    r = window // 2
    kept: List[Tuple[float, float, float]] = []
    eps = 1e-8

    # Precompute coordinate grid in local patch frame.
    yy0, xx0 = np.meshgrid(np.arange(window), np.arange(window), indexing="ij")
    cy = cx = r
    rr = np.sqrt((yy0 - cy) ** 2 + (xx0 - cx) ** 2)
    center_mask_ref = rr <= float(center_radius)
    ring_mask_ref = (rr >= float(ring_inner)) & (rr <= float(ring_outer))

    for x, y, conf in dets:
        xi = int(round(x))
        yi = int(round(y))
        y0, y1 = yi - r, yi + r + 1
        x0, x1 = xi - r, xi + r + 1
        if y0 < 0 or x0 < 0 or y1 > h or x1 > w:
            continue

        patch = gray[y0:y1, x0:x1]
        if patch.shape != (window, window):
            continue

        center_vals = patch[center_mask_ref]
        ring_vals = patch[ring_mask_ref]
        if center_vals.size == 0 or ring_vals.size == 0:
            continue

        center_mean = float(center_vals.mean())
        ring_mean = float(ring_vals.mean())
        # Dark and contrasted center against local ring.
        if center_mean > float(max_center_intensity):
            continue
        if (ring_mean - center_mean) < float(min_ring_contrast):
            continue

        q = float(np.clip(dark_quantile, 0.01, 0.50))
        thr = float(np.quantile(patch, q))
        dark = patch <= thr
        area = int(dark.sum())
        if area < 5:
            continue

        # Size gate on equivalent disk diameter.
        equiv_d = 2.0 * float(np.sqrt(area / np.pi))
        if equiv_d < float(min_equiv_diameter) or equiv_d > float(max_equiv_diameter):
            continue

        ys, xs = np.where(dark)
        if len(xs) < 3:
            continue
        coords = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
        coords = coords - coords.mean(axis=0, keepdims=True)
        cov = (coords.T @ coords) / max(1, len(coords) - 1)
        eig = np.linalg.eigvalsh(cov)
        anisotropy = float(eig[-1] / max(eps, eig[0]))
        # Long streaks/shadows are highly anisotropic.
        if anisotropy > float(max_anisotropy):
            continue

        kept.append((x, y, conf))

    return kept


def main() -> None:
    p = argparse.ArgumentParser(description="Infer keypoint detections from trained heatmap detector.")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--out_csv", type=str, default="predictions.csv")
    p.add_argument("--out_vis_dir", type=str, default="pred_vis")
    p.add_argument("--tile_h", type=int, default=512)
    p.add_argument("--tile_w", type=int, default=512)
    p.add_argument("--stride_h", type=int, default=384)
    p.add_argument("--stride_w", type=int, default=384)
    p.add_argument("--base_channels", type=int, default=24)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--min_distance", type=int, default=5)
    p.add_argument("--max_detections_per_class", type=int, default=2000)
    p.add_argument("--binary_mode", action="store_true", help="Infer as single-class particle detector.")
    p.add_argument("--ignore_bottom_px", type=int, default=0, help="Suppress detections in bottom N pixels.")
    p.add_argument("--hard_filter", action="store_true", help="Enable image-aware hard filtering of candidates.")
    p.add_argument("--hard_window", type=int, default=21)
    p.add_argument("--hard_center_radius", type=int, default=2)
    p.add_argument("--hard_ring_inner", type=int, default=4)
    p.add_argument("--hard_ring_outer", type=int, default=8)
    p.add_argument("--hard_max_center_intensity", type=float, default=0.40)
    p.add_argument("--hard_min_ring_contrast", type=float, default=0.05)
    p.add_argument("--hard_dark_quantile", type=float, default=0.20)
    p.add_argument("--hard_min_equiv_diameter", type=float, default=2.0)
    p.add_argument("--hard_max_equiv_diameter", type=float, default=14.0)
    p.add_argument("--hard_max_anisotropy", type=float, default=2.8)
    p.add_argument("--save_vis", action="store_true", help="Save per-image detection overlays.")
    p.add_argument(
        "--save_heatmap",
        action="store_true",
        help="Save predicted heatmap/confidence visualizations.",
    )
    p.add_argument(
        "--confidence_colormap",
        type=str,
        default="plasma",
        help="Matplotlib colormap for detection confidence in overlays.",
    )
    # Peak decoding (Gold Digger–style: local maxima + blur reduces clusters & streaks)
    p.add_argument(
        "--peak_blur_sigma",
        type=float,
        default=0.8,
        help="Gaussian blur on heatmap before peaks (0=off). Merges plateaus into one peak.",
    )
    p.add_argument(
        "--no_peak_local_max",
        action="store_true",
        help="Disable strict local-maximum filtering (not recommended; causes clustered X's).",
    )
    p.add_argument("--peak_local_max_size", type=int, default=3, help="Window for local maximum (odd >=3).")
    p.add_argument(
        "--no_suppress_line_streaks",
        action="store_true",
        help="Disable near-collinear peak thinning (default: suppression on).",
    )
    p.add_argument("--line_neighbor_radius", type=float, default=12.0)
    p.add_argument("--line_min_cluster", type=int, default=5)
    p.add_argument("--line_eigen_ratio", type=float, default=22.0)
    p.add_argument(
        "--line_keep_top_in_line",
        type=int,
        default=2,
        help="In line-thinning mode, keep at most this many detections per line-like group.",
    )
    p.add_argument(
        "--line_mode",
        type=str,
        default="thin",
        choices=["thin", "drop"],
        help="How to handle line-like clusters: thin (keep a few) or drop (remove all).",
    )
    p.add_argument(
        "--final_max_per_image",
        type=int,
        default=0,
        help="Hard cap on final detections per image after all decoding/filtering (0=off).",
    )
    p.add_argument(
        "--final_min_distance",
        type=float,
        default=0.0,
        help="Global min spacing (px) across final detections after all filtering (0=off).",
    )
    p.add_argument(
        "--tiny_particle_mode",
        action="store_true",
        help="Preset decode/postprocess for very small (about 2-5 px) immunogold particles.",
    )
    p.add_argument("--tiny_max_equiv_diameter_px", type=float, default=7.0)
    p.add_argument("--tiny_min_equiv_diameter_px", type=float, default=1.2)
    p.add_argument("--tiny_support_rel_thresh", type=float, default=0.70)
    p.add_argument("--tiny_support_window", type=int, default=11)
    args = p.parse_args()
    suppress_line_streaks = not bool(args.no_suppress_line_streaks)
    peak_local_max = not bool(args.no_peak_local_max)

    if args.tiny_particle_mode:
        # Conservative defaults for tiny, subtle particles.
        args.min_distance = max(int(args.min_distance), 5)
        args.peak_blur_sigma = max(float(args.peak_blur_sigma), 0.6)
        args.peak_local_max_size = max(int(args.peak_local_max_size), 3)
        args.max_detections_per_class = min(int(args.max_detections_per_class), 800)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_channels = 1 if args.binary_mode else 2
    model = UNetKeypointDetector(in_channels=3, out_channels=out_channels, base_channels=args.base_channels).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    if args.save_vis or args.save_heatmap:
        import matplotlib.pyplot as plt

        os.makedirs(args.out_vis_dir, exist_ok=True)
    records = discover_image_records(args.data_root)

    rows: List[List[str]] = [["image_id", "x", "y", "class_id", "confidence"]]
    for r in records:
        img = tifffile.imread(r.image_path)
        chw = image_to_chw_01(img)
        pred = tiled_inference(
            model,
            chw,
            (args.tile_h, args.tile_w),
            (args.stride_h, args.stride_w),
            device,
            out_channels=out_channels,
        )

        if args.ignore_bottom_px > 0:
            cut = max(0, pred.shape[1] - int(args.ignore_bottom_px))
            pred[:, cut:, :] = 0.0

        def decode_cls(cls_idx: int) -> List[Tuple[float, float, float]]:
            dets = decode_heatmap_to_detections(
                pred[cls_idx],
                threshold=args.threshold,
                min_distance=args.min_distance,
                max_peaks=args.max_detections_per_class,
                blur_sigma=args.peak_blur_sigma,
                local_max_only=peak_local_max,
                local_max_size=args.peak_local_max_size,
                suppress_line_streaks=suppress_line_streaks,
                line_neighbor_radius=args.line_neighbor_radius,
                line_min_cluster=args.line_min_cluster,
                line_eigen_ratio=args.line_eigen_ratio,
                line_keep_top_in_line=args.line_keep_top_in_line,
                line_mode=args.line_mode,
            )
            if args.tiny_particle_mode:
                dets = filter_by_peak_support_size(
                    pred[cls_idx],
                    dets,
                    support_rel_thresh=float(args.tiny_support_rel_thresh),
                    support_window=int(args.tiny_support_window),
                    min_equiv_diameter_px=float(args.tiny_min_equiv_diameter_px),
                    max_equiv_diameter_px=float(args.tiny_max_equiv_diameter_px),
                )
            if args.hard_filter:
                dets = hard_particle_filter(
                    image_chw=chw,
                    dets=dets,
                    window=int(args.hard_window),
                    center_radius=int(args.hard_center_radius),
                    ring_inner=int(args.hard_ring_inner),
                    ring_outer=int(args.hard_ring_outer),
                    max_center_intensity=float(args.hard_max_center_intensity),
                    min_ring_contrast=float(args.hard_min_ring_contrast),
                    dark_quantile=float(args.hard_dark_quantile),
                    min_equiv_diameter=float(args.hard_min_equiv_diameter),
                    max_equiv_diameter=float(args.hard_max_equiv_diameter),
                    max_anisotropy=float(args.hard_max_anisotropy),
                )
            return dets

        dets_all = []
        cls_ids = [0] if args.binary_mode else [0, 1]
        for cls in cls_ids:
            dets = decode_cls(cls)
            for x, y, conf in dets:
                dets_all.append((x, y, conf, cls))

        if args.final_max_per_image > 0 or args.final_min_distance > 0:
            final_dist = float(args.final_min_distance) if args.final_min_distance > 0 else float(args.min_distance)
            final_cap = int(args.final_max_per_image) if args.final_max_per_image > 0 else int(1e9)
            dets_all = enforce_global_non_touching_cap(
                dets_all,
                min_distance_px=final_dist,
                max_total=final_cap,
            )

        for x, y, conf, cls in dets_all:
            rows.append([r.image_id, f"{x:.2f}", f"{y:.2f}", str(cls), f"{conf:.4f}"])

        if args.save_vis:
            vis = np.transpose(chw, (1, 2, 0))
            plt.figure(figsize=(7, 7))
            plt.imshow(vis)
            cmap = plt.get_cmap(args.confidence_colormap)
            for x, y, conf, _cls in dets_all:
                plt.scatter([x], [y], s=14, c=[cmap(float(conf))], marker="x", linewidths=0.9)
            plt.title(f"{r.image_id} detections")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(args.out_vis_dir, f"{r.image_id}_detections.png"), dpi=150)
            plt.close()

        if args.save_heatmap:
            import matplotlib.pyplot as plt

            # For binary mode, channel 0 is the particle map.
            if args.binary_mode:
                heat = pred[0]
            else:
                heat = np.max(pred, axis=0)

            vis = np.transpose(chw, (1, 2, 0))
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(vis, cmap="gray")
            axes[0].set_title("EM image")
            axes[0].axis("off")

            im1 = axes[1].imshow(heat, cmap="magma", vmin=0.0, vmax=max(0.15, float(heat.max())))
            axes[1].set_title("Predicted heatmap/probability")
            axes[1].axis("off")
            fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

            axes[2].imshow(vis, cmap="gray")
            alpha = np.clip(heat / max(1e-6, float(heat.max())), 0.0, 1.0) if float(heat.max()) > 0 else heat
            axes[2].imshow(heat, cmap="magma", alpha=0.45 * alpha)
            axes[2].set_title("Heatmap overlay on EM")
            axes[2].axis("off")

            fig.suptitle(f"{r.image_id}: heatmap + confidence context")
            fig.tight_layout()
            fig.savefig(os.path.join(args.out_vis_dir, f"{r.image_id}_heatmap.png"), dpi=150)
            plt.close(fig)

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"Saved detections to {args.out_csv}")


if __name__ == "__main__":
    main()
