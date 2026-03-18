"""Deeper 4-level UNet with BatchNorm and Dropout for robust particle detection."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two convolutions with BatchNorm and ReLU activation."""

    def __init__(
        self, in_channels: int, out_channels: int, dropout_p: float = 0.0
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(p=dropout_p) if dropout_p > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class UNetDeepKeypointDetector(nn.Module):
    """Deeper 4-level UNet with BatchNorm and Dropout.

    Architecture:
    Encoder:
      enc1: DoubleConv(3,   c)      [patch]
      enc2: DoubleConv(c,   2c)     [patch/2]
      enc3: DoubleConv(2c,  4c)     [patch/4]
      enc4: DoubleConv(4c,  8c)     [patch/8]
      bot:  DoubleConv(8c,  16c)    [patch/16] with dropout

    Decoder: symmetric
      up4:  ConvTranspose2d(16c, 8c)
      dec4: DoubleConv(16c, 8c)
      up3:  ConvTranspose2d(8c, 4c)
      dec3: DoubleConv(8c, 4c)
      up2:  ConvTranspose2d(4c, 2c)
      dec2: DoubleConv(4c, 2c)
      up1:  ConvTranspose2d(2c, c)
      dec1: DoubleConv(2c, c)

    Output: Conv2d(c, 2) → logits
    """

    def __init__(
        self, in_channels: int = 3, out_channels: int = 2, base_channels: int = 32
    ) -> None:
        super().__init__()
        c, c2, c4, c8, c16 = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 16,
        )

        # Encoder
        self.enc1 = DoubleConv(in_channels, c)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(c, c2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(c2, c4)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = DoubleConv(c4, c8)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck with dropout
        self.bot = DoubleConv(c8, c16, dropout_p=0.1)

        # Decoder
        self.up4 = nn.ConvTranspose2d(c16, c8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(c8 + c8, c8)

        self.up3 = nn.ConvTranspose2d(c8, c4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(c4 + c4, c4)

        self.up2 = nn.ConvTranspose2d(c4, c2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(c2 + c2, c2)

        self.up1 = nn.ConvTranspose2d(c2, c, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(c + c, c)

        # Output layer (logits, no activation)
        self.out_conv = nn.Conv2d(c, out_channels, kernel_size=1)

    @staticmethod
    def _center_crop(src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """Center-crop src to match tgt size."""
        _, _, hs, ws = src.shape
        _, _, ht, wt = tgt.shape
        if (hs, ws) == (ht, wt):
            return src
        y0 = max(0, (hs - ht) // 2)
        x0 = max(0, (ws - wt) // 2)
        return src[:, :, y0 : y0 + ht, x0 : x0 + wt]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw = x.shape[-2:]

        # Encoder
        x1 = self.enc1(x)
        x = self.pool1(x1)

        x2 = self.enc2(x)
        x = self.pool2(x2)

        x3 = self.enc3(x)
        x = self.pool3(x3)

        x4 = self.enc4(x)
        x = self.pool4(x4)

        # Bottleneck
        xb = self.bot(x)

        # Decoder
        x = self.up4(xb)
        x4 = self._center_crop(x4, x)
        x = self.dec4(torch.cat([x, x4], dim=1))

        x = self.up3(x)
        x3 = self._center_crop(x3, x)
        x = self.dec3(torch.cat([x, x3], dim=1))

        x = self.up2(x)
        x2 = self._center_crop(x2, x)
        x = self.dec2(torch.cat([x, x2], dim=1))

        x = self.up1(x)
        x1 = self._center_crop(x1, x)
        x = self.dec1(torch.cat([x, x1], dim=1))

        # Output (logits)
        x = self.out_conv(x)

        # Ensure output matches input spatial size
        if x.shape[-2:] != (ih, iw):
            x = F.interpolate(x, size=(ih, iw), mode="bilinear", align_corners=False)

        return x
