from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SmallUNetDetector2D(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 2, base_channels: int = 16) -> None:
        super().__init__()
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4
        self.enc1 = DoubleConv(in_channels, c1)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(c1, c2)
        self.pool2 = nn.MaxPool2d(2)
        self.bot = DoubleConv(c2, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, 2, 2)
        self.dec2 = DoubleConv(c2 + c2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, 2)
        self.dec1 = DoubleConv(c1 + c1, c1)
        self.out_conv = nn.Conv2d(c1, out_channels, 1)

    @staticmethod
    def _crop(src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        _, _, hs, ws = src.shape
        _, _, ht, wt = tgt.shape
        if (hs, ws) == (ht, wt):
            return src
        y0 = max(0, (hs - ht) // 2)
        x0 = max(0, (ws - wt) // 2)
        return src[:, :, y0 : y0 + ht, x0 : x0 + wt]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw = x.shape[-2:]
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        xb = self.bot(self.pool2(x2))
        x = self.up2(xb)
        x2 = self._crop(x2, x)
        x = self.dec2(torch.cat([x, x2], dim=1))
        x = self.up1(x)
        x1 = self._crop(x1, x)
        x = self.dec1(torch.cat([x, x1], dim=1))
        x = torch.sigmoid(self.out_conv(x))
        if x.shape[-2:] != (ih, iw):
            x = F.interpolate(x, size=(ih, iw), mode="bilinear", align_corners=False)
        return x

