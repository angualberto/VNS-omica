import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralAttention(nn.Module):
    """Simple transformer-style attention over spectral vector.
    Input: (batch, F) -> reshape to (batch, L, 1) as sequence of length L
    """
    def __init__(self, input_dim, embed_dim=64, nhead=4, num_layers=1, mlp_hidden=64):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.proj = nn.Linear(1, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=embed_dim*2, activation='relu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(mlp_hidden, 1)
        )

    def forward(self, x):
        # x: (batch, F)
        b, f = x.shape
        seq = x.view(b, f, 1)  # (b, L, 1)
        seq = self.proj(seq)  # (b, L, embed)
        # transformer expects (L, b, embed)
        seq = seq.permute(1, 0, 2)
        out = self.transformer(seq)  # (L, b, embed)
        out = out.permute(1, 2, 0)  # (b, embed, L)
        pooled = self.pool(out).squeeze(-1)  # (b, embed)
        logits = self.mlp(pooled).squeeze(1)
        return logits
