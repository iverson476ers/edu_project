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


class MultiLayerPooling(BasePooling):
    """Mean-pool last num_layers hidden states, concatenate, LayerNorm.

    Uses hidden states from multiple transformer layers for richer representation.
    Output dim = hidden_dim * num_layers. Extra params: LayerNorm only.
    """

    def __init__(self, hidden_dim: int, num_layers: int = 4):
        super().__init__()
        self.num_layers = num_layers
        self.output_dim = hidden_dim * num_layers
        self.norm = nn.LayerNorm(self.output_dim)

    def forward(self, hidden_states, attention_mask: torch.Tensor) -> torch.Tensor:
        # hidden_states: tuple of (batch, seq_len, hidden_dim) from all layers
        pooled = []
        mask = attention_mask.unsqueeze(-1).to(dtype=hidden_states[-1].dtype,
                                                device=hidden_states[-1].device)
        for layer_hidden in hidden_states[-self.num_layers:]:
            masked = layer_hidden * mask
            summed = masked.sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            pooled.append(summed / counts)
        concat = torch.cat(pooled, dim=-1)  # (batch, hidden_dim * num_layers)
        concat = concat.to(torch.float32)    # cast to float32 for LayerNorm
        return self.norm(concat)


def build_pooling(strategy: str, hidden_dim: int, **kwargs) -> BasePooling:
    """Factory: return the pooling module for a given strategy name."""
    if strategy == "mean":
        return MeanPooling()
    elif strategy == "last":
        return LastTokenPooling()
    elif strategy == "attention":
        return AttentionPooling(hidden_dim)
    elif strategy == "multi_layer":
        num_layers = kwargs.get("num_layers", 4)
        return MultiLayerPooling(hidden_dim, num_layers)
    else:
        raise ValueError(f"Unknown pooling strategy: {strategy}. Use 'mean', 'last', 'attention', or 'multi_layer'.")
