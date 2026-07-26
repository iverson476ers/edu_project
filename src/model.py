from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CoralHead(nn.Module):
    """Ordinal regression head: K classes -> K-1 binary classifiers.

    Configurable depth with LayerNorm + GELU + Dropout + Residual connections.
    Set hidden_sizes=[] for the original single-layer behavior.
    """

    def __init__(self, hidden_dim: int, num_classes: int,
                 hidden_sizes: list[int] | None = None,
                 dropout: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_sizes = hidden_sizes or []
        self.dropout = dropout

        layers = []
        in_dim = hidden_dim
        for h in self.hidden_sizes:
            block = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, h),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            layers.append(block)
            # Residual projection if dimensions don't match
            proj = nn.Linear(in_dim, h) if in_dim != h else nn.Identity()
            layers.append(proj)
            in_dim = h

        self.blocks = nn.ModuleList(layers) if layers else None
        self.linear = nn.Linear(in_dim, num_classes - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, hidden_dim)
        x = x.to(self.linear.weight.dtype)  # ensure float32 for all Linear/LayerNorm layers
        if self.blocks is not None:
            for i in range(0, len(self.blocks), 2):
                block = self.blocks[i]       # Sequential: LayerNorm → Linear → GELU → Dropout
                proj = self.blocks[i + 1]    # Projection for residual
                residual = proj(x)
                x = block(x) + residual
        return self.linear(x)  # (batch, K-1)


class DummyBackbone(nn.Module):
    """Returns random embeddings for local testing without loading Qwen."""

    def __init__(self, hidden_dim: int = 2560):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.config = type("Config", (), {"hidden_size": hidden_dim})()

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False):
        batch_size = len(input_ids) if isinstance(input_ids, list) else input_ids.size(0)
        seq_len = 128
        # Simulate multiple hidden layers (embedding + N transformer layers)
        hidden = torch.randn(batch_size, seq_len, self.hidden_dim)
        return type("ModelOutput", (), {
            "last_hidden_state": hidden,
            "hidden_states": (hidden, hidden),  # tuple of (embedding_output, last_hidden)
        })()


class OrdinalScorer(nn.Module):
    """Qwen backbone + Pooling + CoralHead for ordinal regression scoring.

    Args:
        pooling: BasePooling instance (defaults to MeanPooling)
        head_config: kwargs dict for CoralHead (hidden_sizes, dropout)
    """

    def __init__(self, backbone: nn.Module, hidden_dim: int, num_classes: int,
                 pooling: nn.Module | None = None,
                 head_config: dict | None = None):
        super().__init__()
        from src.pooling import MeanPooling
        self.backbone = backbone
        self.pooling = pooling or MeanPooling()
        head_kwargs = head_config or {}
        self.head = CoralHead(hidden_dim, num_classes, **head_kwargs)
        self.num_classes = num_classes

    def forward(self, input_ids, attention_mask=None,
                handcrafted_features: torch.Tensor | None = None):
        outputs = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True
        )
        hidden = outputs.hidden_states[-1]  # (batch, seq_len, hidden_dim)
        if attention_mask is None:
            attention_mask = torch.ones(hidden.size(0), hidden.size(1), device=hidden.device)
        pooled = self.pooling(hidden, attention_mask)  # (batch, hidden_dim)

        # Reserved: concat handcrafted features here in the future
        if handcrafted_features is not None:
            pooled = torch.cat([pooled, handcrafted_features], dim=-1)

        return self.head(pooled)  # (batch, K-1)


def coral_loss(
    logits: torch.Tensor, label_indices: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """CORAL loss: K classes -> K-1 binary classifers.

    For each threshold k (0 <= k < K-1):
      target_k = 1 if label_idx > k else 0
      loss += BCE(logits[:, k], target_k)
    """
    batch_size = logits.size(0)
    targets = torch.zeros_like(logits)  # (batch, K-1)
    for k in range(num_classes - 1):
        targets[:, k] = (label_indices > k).float()
    return F.binary_cross_entropy_with_logits(logits, targets)


def prediction_to_score(probs: torch.Tensor, score_points: list[float]) -> list[float]:
    """Convert K-1 probabilities to final score.

    probs: (batch, K-1) — sigmoid probabilities P(score > threshold_k)
    Returns the score_point whose cumulative probability pattern best matches.
    """
    # Cumulative probability: P(score > threshold_k for each k)
    # Find k* = argmax over cumulative match
    batch_size = probs.size(0)
    K = len(score_points)
    device = probs.device

    results = []
    for i in range(batch_size):
        best_score = score_points[0]
        best_diff = float("inf")
        for k in range(K):
            # For score_points[k], expected target pattern:
            # target_j = 1 if k > j else 0  (i.e., score > threshold_j)
            expected = torch.tensor(
                [1.0 if k > j else 0.0 for j in range(K - 1)], device=device
            )
            diff = (probs[i] - expected).abs().sum().item()
            if diff < best_diff:
                best_diff = diff
                best_score = score_points[k]
        results.append(best_score)
    return results
