"""Sequence models for ISL sign classification from hand-landmark sequences.

Primary model: bidirectional GRU over frame features + additive attention
pooling -> classifier. The attention weights give the model (and the UI) a
per-frame importance signal, which is the temporal-modeling story of this
capstone: signs are distinguished by *motion*, not by any single pose.

Ablation baseline: frame-MLP that averages frame features (no temporal
modeling). Reported side-by-side in the README results table.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import FRAME_FEAT


class AttentionPool(nn.Module):
    """Additive attention over time steps -> weighted sum + attention weights."""

    def __init__(self, hidden: int):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        # x: (B, T, H); mask: (B, T) with 1 = valid
        logits = self.score(x).squeeze(-1)  # (B, T)
        if mask is not None:
            logits = logits.masked_fill(mask == 0, float("-inf"))
        w = F.softmax(logits, dim=-1)
        w = torch.nan_to_num(w, nan=0.0)
        ctx = (w.unsqueeze(-1) * x).sum(dim=1)
        return ctx, w


class GRUAttentionClassifier(nn.Module):
    def __init__(self, num_classes: int, input_dim: int = FRAME_FEAT, hidden: int = 128,
                 layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.ReLU())
        self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True,
                          bidirectional=True, dropout=dropout if layers > 1 else 0.0)
        self.attn = AttentionPool(hidden * 2)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        h = self.input_proj(x)
        out, _ = self.gru(h)
        ctx, attn_w = self.attn(out, mask)
        return self.head(ctx), attn_w


class FrameMLPClassifier(nn.Module):
    """Ablation baseline: mean-pool frames, no temporal modeling."""

    def __init__(self, num_classes: int, input_dim: int = FRAME_FEAT, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        if mask is not None:
            m = mask.unsqueeze(-1).float()
            pooled = (x * m).sum(1) / m.sum(1).clamp_min(1)
        else:
            pooled = x.mean(1)
        return self.net(pooled), None
