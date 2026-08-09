from __future__ import annotations

import torch
from torch import nn

from cnn_encoder import CNNEncoder
from vnn_encoder import VNNEncoder


class AttentionMIL(nn.Module):
    def __init__(self, embedding_dim: int = 128, attention_dim: int = 64):
        super().__init__()
        self.v = nn.Linear(embedding_dim, attention_dim)
        self.w = nn.Linear(attention_dim, 1)

    def forward(self, z: torch.Tensor, valid: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        attention_logits = self.w(torch.tanh(self.v(z))).squeeze(-1)
        if valid is not None:
            attention_logits = attention_logits.masked_fill(~valid, torch.finfo(attention_logits.dtype).min)
        attention = torch.softmax(attention_logits, dim=1)
        return torch.sum(attention.unsqueeze(-1) * z, dim=1), attention


class HybridCNNVNNMIL(nn.Module):
    """Configurable CNN, VNN or fused CNN+VNN MIL image classifier."""

    MODES = {"cnn", "vnn", "hybrid"}

    def __init__(self, mode: str = "hybrid", in_ch: int = 8, embedding_dim: int = 128):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {sorted(self.MODES)}")
        self.mode = mode
        self.cnn = CNNEncoder(in_ch, embedding_dim) if mode in {"cnn", "hybrid"} else None
        self.vnn = VNNEncoder(in_ch, embedding_dim) if mode in {"vnn", "hybrid"} else None
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim), nn.GELU(), nn.LayerNorm(embedding_dim), nn.Dropout(0.1)
        ) if mode == "hybrid" else nn.Identity()
        self.attention = AttentionMIL(embedding_dim, attention_dim=64)
        self.classifier = nn.Sequential(nn.LayerNorm(embedding_dim), nn.Linear(embedding_dim, 1))
        self.cnn_aux = nn.Linear(embedding_dim, 1) if mode == "hybrid" else None
        self.vnn_aux = nn.Linear(embedding_dim, 1) if mode == "hybrid" else None

    def forward(self, x: torch.Tensor, valid: torch.Tensor | None = None, return_details: bool = False):
        batch, bag, channels, height, width = x.shape
        patches = x.reshape(batch * bag, channels, height, width)
        z_cnn = self.cnn(patches).reshape(batch, bag, -1) if self.cnn is not None else None
        z_vnn = self.vnn(patches).reshape(batch, bag, -1) if self.vnn is not None else None
        if self.mode == "cnn":
            z = z_cnn
        elif self.mode == "vnn":
            z = z_vnn
        else:
            z = self.fusion(torch.cat([z_cnn, z_vnn], dim=-1))
        pooled, attention = self.attention(z, valid)
        logits = self.classifier(pooled).squeeze(1)
        if not return_details:
            return logits, attention
        details: dict[str, torch.Tensor | None] = {"attention": attention, "cnn_logits": None, "vnn_logits": None}
        if self.mode == "hybrid":
            cnn_pool, _ = self.attention(z_cnn, valid)
            vnn_pool, _ = self.attention(z_vnn, valid)
            details["cnn_logits"] = self.cnn_aux(cnn_pool).squeeze(1)
            details["vnn_logits"] = self.vnn_aux(vnn_pool).squeeze(1)
        return logits, details

