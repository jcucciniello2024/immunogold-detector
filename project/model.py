from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SmallUNet(nn.Module):
    """Small U-Net style model: 1-channel in, 1-channel heatmap out."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 16) -> None:
        super().__init__()
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4

        self.enc1 = DoubleConv(in_channels, c1)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(c1, c2)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(c2, c3)

        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(c2 + c2, c2)

        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(c1 + c1, c1)

        self.out_conv = nn.Conv2d(c1, out_channels, kernel_size=1)

    @staticmethod
    def _center_crop_to_match(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Center-crop source spatially to target HxW for skip concatenation."""
        _, _, h_s, w_s = source.shape
        _, _, h_t, w_t = target.shape

        if h_s == h_t and w_s == w_t:
            return source

        if h_s < h_t or w_s < w_t:
            raise ValueError(
                f"Cannot crop source {source.shape} to larger target {target.shape}."
            )

        y0 = (h_s - h_t) // 2
        x0 = (w_s - w_t) // 2
        return source[:, :, y0 : y0 + h_t, x0 : x0 + w_t]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_h, in_w = x.shape[-2], x.shape[-1]
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.bottleneck(self.pool2(x2))

        x = self.up2(x3)
        x2 = self._center_crop_to_match(x2, x)
        x = torch.cat([x, x2], dim=1)
        x = self.dec2(x)

        x = self.up1(x)
        x1 = self._center_crop_to_match(x1, x)
        x = torch.cat([x, x1], dim=1)
        x = self.dec1(x)

        # Keep regression target in [0, 1] range.
        x = torch.sigmoid(self.out_conv(x))
        if x.shape[-2:] != (in_h, in_w):
            x = F.interpolate(x, size=(in_h, in_w), mode="bilinear", align_corners=False)
        return x
