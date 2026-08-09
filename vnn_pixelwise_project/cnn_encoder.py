from __future__ import annotations

import torch
from torch import nn


class CNNEncoder(nn.Module):
    """Light spatial encoder for cached 8-channel mammography patches."""

    def __init__(self, in_ch: int = 8, embedding_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 48, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(48), nn.GELU(),
            nn.Conv2d(48, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 96, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Sequential(nn.Flatten(), nn.Linear(96, embedding_dim), nn.LayerNorm(embedding_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(self.features(x))

