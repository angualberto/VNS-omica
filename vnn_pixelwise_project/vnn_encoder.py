from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class VolterraLowRankBlock(nn.Module):
    """Linear 1x1 response plus low-rank quadratic U(x) * V(x)."""

    def __init__(self, in_ch: int, out_ch: int, rank: int, stride: int = 1):
        super().__init__()
        self.spatial = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False)
        self.linear = nn.Conv2d(in_ch, out_ch, 1)
        self.u = nn.Conv2d(in_ch, rank, 1, bias=False)
        self.v = nn.Conv2d(in_ch, rank, 1, bias=False)
        self.quadratic = nn.Conv2d(rank, out_ch, 1)
        self.norm = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spatial(x)
        return F.gelu(self.norm(self.linear(x) + self.quadratic(self.u(x) * self.v(x))))


class VNNEncoder(nn.Module):
    def __init__(self, in_ch: int = 8, embedding_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 24, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(24), nn.GELU(),
            VolterraLowRankBlock(24, 32, rank=16, stride=2),
            VolterraLowRankBlock(32, 64, rank=24, stride=2),
            VolterraLowRankBlock(64, 96, rank=32, stride=2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Sequential(nn.Flatten(), nn.Linear(96, embedding_dim), nn.LayerNorm(embedding_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(self.features(x))

