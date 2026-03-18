from __future__ import annotations

import torch
import torch.nn as nn


def _down(in_ch: int, out_ch: int, use_bn: bool = True) -> nn.Sequential:
    layers = [nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=not use_bn)]
    if use_bn:
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


def _up(in_ch: int, out_ch: int, dropout: float = 0.0) -> nn.Sequential:
    layers = [
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    ]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class GoldDiggerGenerator(nn.Module):
    """
    U-Net style generator:
      input  (B,3,H,W) -> output logits (B,2,H,W)
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 2, base_channels: int = 64) -> None:
        super().__init__()
        c = base_channels
        self.d1 = _down(in_channels, c, use_bn=False)  # 128
        self.d2 = _down(c, c * 2)  # 64
        self.d3 = _down(c * 2, c * 4)  # 32
        self.d4 = _down(c * 4, c * 8)  # 16
        self.d5 = _down(c * 8, c * 8)  # 8

        self.bottleneck = nn.Sequential(
            nn.Conv2d(c * 8, c * 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )  # 4

        self.u1 = _up(c * 8, c * 8, dropout=0.2)  # 8
        self.u2 = _up(c * 16, c * 8, dropout=0.2)  # 16
        self.u3 = _up(c * 16, c * 4)  # 32
        self.u4 = _up(c * 8, c * 2)  # 64
        self.u5 = _up(c * 4, c)  # 128
        self.u6 = nn.ConvTranspose2d(c * 2, out_channels, kernel_size=4, stride=2, padding=1)  # 256

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)
        b = self.bottleneck(d5)

        u1 = self.u1(b)
        u2 = self.u2(torch.cat([u1, d5], dim=1))
        u3 = self.u3(torch.cat([u2, d4], dim=1))
        u4 = self.u4(torch.cat([u3, d3], dim=1))
        u5 = self.u5(torch.cat([u4, d2], dim=1))
        out = self.u6(torch.cat([u5, d1], dim=1))
        return out


class GoldDiggerPatchDiscriminator(nn.Module):
    """
    Patch discriminator over [image, mask] pair.
    Input:
      image: (B,3,H,W), mask: (B,2,H,W)
    Output:
      patch logits (B,1,h,w)
    """

    def __init__(self, in_channels_image: int = 3, in_channels_mask: int = 2, base_channels: int = 64) -> None:
        super().__init__()
        c = base_channels
        in_ch = in_channels_image + in_channels_mask
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, c, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c, c * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c * 2, c * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c * 4, c * 8, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c * 8, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = torch.cat([image, mask], dim=1)
        return self.net(x)

