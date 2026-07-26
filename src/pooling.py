"""Pooling strategies for extracting sentence vectors from hidden states."""
from __future__ import annotations

import torch
import torch.nn as nn


class BasePooling(nn.Module):
    """Abstract pooling: (batch, seq_len, hidden_dim) -> (batch, hidden_dim)."""

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class MeanPooling(BasePooling):
    """Mask-weighted mean over non-padding tokens. Zero extra parameters."""

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)  # (B, L, 1)
        masked = hidden_states * mask
        summed = masked.sum(dim=1)                         # (B, D)
        counts = mask.sum(dim=1).clamp(min=1)              # (B, 1)
        return summed / counts


class LastTokenPooling(BasePooling):
    """Take the hidden state of the last non-padding token (EOS). Zero extra parameters."""

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # sequence_lengths: (B,) — position of last valid token for each sample
        sequence_lengths = (attention_mask.sum(dim=1) - 1).long()   # 0-indexed
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[batch_indices, sequence_lengths]


class AttentionPooling(BasePooling):
    """Learnable query vector -> softmax-weighted sum over tokens. Extra D params."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # hidden_states: (B, L, D)
        scores = torch.matmul(hidden_states, self.query.to(hidden_states.dtype))   # (B, L)
        # Mask padding positions with -inf
        scores = scores.masked_fill(attention_mask == 0, torch.finfo(hidden_states.dtype).min)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)  # (B, L, 1)
        return (hidden_states * weights).sum(dim=1)            # (B, D)


def build_pooling(strategy: str, hidden_dim: int) -> BasePooling:
    """Factory: return the pooling module for a given strategy name."""
    if strategy == "mean":
        return MeanPooling()
    elif strategy == "last":
        return LastTokenPooling()
    elif strategy == "attention":
        return AttentionPooling(hidden_dim)
    else:
        raise ValueError(f"Unknown pooling strategy: {strategy}. Use 'mean', 'last', or 'attention'.")
