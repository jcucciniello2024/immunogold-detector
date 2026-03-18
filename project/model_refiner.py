from __future__ import annotations

import torch
import torch.nn as nn


class PatchRefinerCNN(nn.Module):
    """Tiny CNN for classifying candidate patches into bg/6nm/12nm."""

    def __init__(self, in_channels: int = 3, num_classes: int = 3, base_channels: int = 32) -> None:
        super().__init__()
        c = base_channels
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c, c * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c * 2, c * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c * 2, c * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(c * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)

