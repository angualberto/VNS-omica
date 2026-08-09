from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VolterraPixelLayer(nn.Module):
    """Pixel-wise Volterra layer using linear and low-rank quadratic 1x1 convolutions.

    Input:  [B, C, H, W]
    Output: [B, out_ch, H, W]
    """

    def __init__(self, in_ch: int, out_ch: int, rank: int = 32):
        super().__init__()
        self.linear = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.U = nn.Conv2d(in_ch, rank, kernel_size=1, bias=False)
        self.V = nn.Conv2d(in_ch, rank, kernel_size=1, bias=False)
        self.quad = nn.Conv2d(rank, out_ch, kernel_size=1)
        self.norm = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lin = self.linear(x)
        quad = self.quad(self.U(x) * self.V(x))
        return F.gelu(self.norm(lin + quad))


class VNNPixelMammo(nn.Module):
    """VNN pixel-wise model for experimental mammography risk maps."""

    def __init__(self, in_ch: int = 8):
        super().__init__()
        self.enc1 = VolterraPixelLayer(in_ch, 32, rank=16)
        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.enc2 = VolterraPixelLayer(32, 64, rank=32)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.enc3 = VolterraPixelLayer(64, 64, rank=32)
        self.out = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.enc1(x)
        x = F.gelu(self.conv1(x))
        x = self.enc2(x)
        x = F.gelu(self.conv2(x))
        x = self.enc3(x)
        return self.out(x)


def smoke_test(batch_size: int = 4, in_ch: int = 8, patch_size: int = 256) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VNNPixelMammo(in_ch=in_ch).to(device)
    x = torch.randn(batch_size, in_ch, patch_size, patch_size, device=device)
    with torch.no_grad():
        y = model(x)
    print("device:", device)
    if device == "cuda":
        print("gpu:", torch.cuda.get_device_name(0))
        torch.cuda.synchronize()
        print("memory_allocated_gb:", torch.cuda.memory_allocated() / 1024**3)
        print("memory_reserved_gb:", torch.cuda.memory_reserved() / 1024**3)
    print("input:", tuple(x.shape))
    print("output:", tuple(y.shape))


if __name__ == "__main__":
    smoke_test()
