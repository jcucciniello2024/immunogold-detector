from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from dataset_guard import enforce_allowed_data_root
from model_golddigger_cgan import GoldDiggerGenerator, GoldDiggerPatchDiscriminator
from prepare_labels import ImageRecord, discover_image_records


def draw_disk_map(image_hw: Tuple[int, int], points: np.ndarray, radius: int) -> np.ndarray:
    h, w = image_hw
    out = np.zeros((h, w), dtype=np.float32)
    if len(points) == 0:
        return out
    rr = max(1, int(radius))
    yy = np.arange(-rr, rr + 1, dtype=np.int32)[:, None]
    xx = np.arange(-rr, rr + 1, dtype=np.int32)[None, :]
    mask = (xx * xx + yy * yy) <= rr * rr
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


def image_to_chw01(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    img = image.astype(np.float32)
    mn, mx = float(img.min()), float(img.max())
    if mx > mn:
        img = (img - mn) / (mx - mn)
    else:
        img = np.zeros_like(img, dtype=np.float32)
    return np.transpose(img, (2, 0, 1))


def split_by_image(
    records: Sequence[ImageRecord], train_ratio: float = 0.7, val_ratio: float = 0.15, seed: int = 42
) -> Tuple[List[ImageRecord], List[ImageRecord], List[ImageRecord]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(records))
    rng.shuffle(idx)
    n = len(records)
    n_train = max(1, int(round(n * train_ratio)))
    n_val = max(1, int(round(n * val_ratio)))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
    train = [records[i] for i in idx[:n_train]]
    val = [records[i] for i in idx[n_train : n_train + n_val]]
    test = [records[i] for i in idx[n_train + n_val :]]
    return train, val, test


class GoldPatchDataset(Dataset):
    def __init__(
        self,
        records: Sequence[ImageRecord],
        patch_size: int,
        samples_per_epoch: int,
        pos_fraction: float,
        radius_6nm: int,
        radius_12nm: int,
        augment: bool,
        seed: int,
    ) -> None:
        self.patch_size = int(patch_size)
        self.samples_per_epoch = int(samples_per_epoch)
        self.pos_fraction = float(pos_fraction)
        self.radius_6 = int(radius_6nm)
        self.radius_12 = int(radius_12nm)
        self.augment = bool(augment)
        self.rng = np.random.default_rng(seed)

        self.images: List[np.ndarray] = []
        self.p6: List[np.ndarray] = []
        self.p12: List[np.ndarray] = []
        self.p_all: List[np.ndarray] = []
        for r in records:
            img = image_to_chw01(tifffile.imread(r.image_path))
            _, h, w = img.shape
            if h < self.patch_size or w < self.patch_size:
                continue
            a = r.points[0].astype(np.float32)
            b = r.points[1].astype(np.float32)
            allp = np.concatenate([a, b], axis=0) if len(a) + len(b) > 0 else np.zeros((0, 2), dtype=np.float32)
            self.images.append(img)
            self.p6.append(a)
            self.p12.append(b)
            self.p_all.append(allp)
        if not self.images:
            raise ValueError("No valid records for GoldPatchDataset.")

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _crop_points(self, points: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
        if len(points) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        m = (points[:, 0] >= x0) & (points[:, 0] < x1) & (points[:, 1] >= y0) & (points[:, 1] < y1)
        out = points[m].copy()
        out[:, 0] -= x0
        out[:, 1] -= y0
        return out

    def _augment(self, img: np.ndarray, tgt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.rng.random() < 0.5:
            img = img[:, :, ::-1].copy()
            tgt = tgt[:, :, ::-1].copy()
        if self.rng.random() < 0.5:
            img = img[:, ::-1, :].copy()
            tgt = tgt[:, ::-1, :].copy()
        k = int(self.rng.integers(0, 4))
        if k:
            img = np.rot90(img, k=k, axes=(1, 2)).copy()
            tgt = np.rot90(tgt, k=k, axes=(1, 2)).copy()
        if self.rng.random() < 0.5:
            c = float(self.rng.uniform(0.9, 1.1))
            b = float(self.rng.uniform(-0.05, 0.05))
            img = np.clip(img * c + b, 0.0, 1.0)
        return img, tgt

    def __getitem__(self, index: int):
        del index
        i = int(self.rng.integers(0, len(self.images)))
        img = self.images[i]
        allp = self.p_all[i]
        p6 = self.p6[i]
        p12 = self.p12[i]
        _, h, w = img.shape

        if len(allp) > 0 and self.rng.random() < self.pos_fraction:
            anchor = allp[int(self.rng.integers(0, len(allp)))]
            x0 = int(round(float(anchor[0]) - self.patch_size / 2 + self.rng.integers(-20, 21)))
            y0 = int(round(float(anchor[1]) - self.patch_size / 2 + self.rng.integers(-20, 21)))
            x0 = max(0, min(x0, w - self.patch_size))
            y0 = max(0, min(y0, h - self.patch_size))
        else:
            x0 = int(self.rng.integers(0, w - self.patch_size + 1))
            y0 = int(self.rng.integers(0, h - self.patch_size + 1))
        x1, y1 = x0 + self.patch_size, y0 + self.patch_size

        patch = img[:, y0:y1, x0:x1]
        p6c = self._crop_points(p6, x0, y0, x1, y1)
        p12c = self._crop_points(p12, x0, y0, x1, y1)
        tgt = np.zeros((2, self.patch_size, self.patch_size), dtype=np.float32)
        tgt[0] = draw_disk_map((self.patch_size, self.patch_size), p6c, self.radius_6)
        tgt[1] = draw_disk_map((self.patch_size, self.patch_size), p12c, self.radius_12)

        if self.augment:
            patch, tgt = self._augment(patch, tgt)
        return torch.from_numpy(patch).float(), torch.from_numpy(tgt).float()


@dataclass
class TrainStats:
    d_loss: float
    g_adv: float
    g_rec: float
    g_total: float


def run_epoch(
    gen: GoldDiggerGenerator,
    disc: GoldDiggerPatchDiscriminator,
    loader: DataLoader,
    opt_g: torch.optim.Optimizer | None,
    opt_d: torch.optim.Optimizer | None,
    device: torch.device,
    lambda_l1: float,
) -> TrainStats:
    is_train = opt_g is not None and opt_d is not None
    gen.train(is_train)
    disc.train(is_train)

    bce = torch.nn.BCEWithLogitsLoss()
    total_d = total_adv = total_rec = total_g = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        bs = x.shape[0]
        real_label = torch.ones((bs, 1, 30, 30), device=device)
        fake_label = torch.zeros((bs, 1, 30, 30), device=device)

        with torch.set_grad_enabled(is_train):
            fake_logits = gen(x)
            fake_mask = torch.sigmoid(fake_logits)

            pred_real = disc(x, y)
            pred_fake = disc(x, fake_mask.detach())
            if pred_real.shape != real_label.shape:
                real_label = torch.ones_like(pred_real)
                fake_label = torch.zeros_like(pred_real)

            d_loss = 0.5 * (bce(pred_real, real_label) + bce(pred_fake, fake_label))
            if is_train:
                opt_d.zero_grad()
                d_loss.backward()
                opt_d.step()

            pred_fake_for_g = disc(x, fake_mask)
            valid = torch.ones_like(pred_fake_for_g)
            g_adv = bce(pred_fake_for_g, valid)
            g_rec = F.l1_loss(fake_mask, y)
            g_loss = g_adv + lambda_l1 * g_rec
            if is_train:
                opt_g.zero_grad()
                g_loss.backward()
                opt_g.step()

        total_d += float(d_loss.item()) * bs
        total_adv += float(g_adv.item()) * bs
        total_rec += float(g_rec.item()) * bs
        total_g += float(g_loss.item()) * bs
        n += bs

    d = max(1, n)
    return TrainStats(
        d_loss=total_d / d,
        g_adv=total_adv / d,
        g_rec=total_rec / d,
        g_total=total_g / d,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Gold Digger-style pix2pix cGAN for immunogold mask prediction.")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--patch_size", type=int, default=256)
    p.add_argument("--train_samples_per_epoch", type=int, default=2048)
    p.add_argument("--val_samples_per_epoch", type=int, default=512)
    p.add_argument("--pos_fraction", type=float, default=0.7)
    p.add_argument("--radius_6nm", type=int, default=3)
    p.add_argument("--radius_12nm", type=int, default=5)
    p.add_argument("--lambda_l1", type=float, default=50.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_dir", type=str, default="checkpoints/golddigger_cgan")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.data_root = enforce_allowed_data_root(args.data_root)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    records = discover_image_records(args.data_root)
    train_r, val_r, test_r = split_by_image(records, seed=args.seed)
    print(f"Image split -> train={len(train_r)} val={len(val_r)} test={len(test_r)}")

    train_ds = GoldPatchDataset(
        train_r,
        patch_size=args.patch_size,
        samples_per_epoch=args.train_samples_per_epoch,
        pos_fraction=args.pos_fraction,
        radius_6nm=args.radius_6nm,
        radius_12nm=args.radius_12nm,
        augment=True,
        seed=args.seed,
    )
    val_ds = GoldPatchDataset(
        val_r,
        patch_size=args.patch_size,
        samples_per_epoch=args.val_samples_per_epoch,
        pos_fraction=args.pos_fraction,
        radius_6nm=args.radius_6nm,
        radius_12nm=args.radius_12nm,
        augment=False,
        seed=args.seed + 1,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    gen = GoldDiggerGenerator(in_channels=3, out_channels=2, base_channels=64).to(device)
    disc = GoldDiggerPatchDiscriminator(in_channels_image=3, in_channels_mask=2, base_channels=64).to(device)
    opt_g = torch.optim.Adam(gen.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=args.lr, betas=(0.5, 0.999))

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(gen, disc, train_loader, opt_g, opt_d, device, args.lambda_l1)
        va = run_epoch(gen, disc, val_loader, None, None, device, args.lambda_l1)
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} "
            f"train_d={tr.d_loss:.5f} train_g={tr.g_total:.5f} "
            f"val_d={va.d_loss:.5f} val_g={va.g_total:.5f} val_l1={va.g_rec:.5f}"
        )
        torch.save(gen.state_dict(), os.path.join(args.save_dir, "generator_last.pt"))
        torch.save(disc.state_dict(), os.path.join(args.save_dir, "discriminator_last.pt"))
        if va.g_rec < best_val:
            best_val = va.g_rec
            torch.save(gen.state_dict(), os.path.join(args.save_dir, "generator_best.pt"))
            print(f"Saved best generator (val_l1={best_val:.5f})")


if __name__ == "__main__":
    main()
