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


class RegressionHead(nn.Module):
    """Regression head: outputs a single [0,1] score via sigmoid.

    Same residual block architecture as CoralHead but projects to 1 dimension.
    Multiply by full_score to recover the original scale.
    """

    def __init__(self, hidden_dim: int,
                 hidden_sizes: list[int] | None = None,
                 dropout: float = 0.1):
        super().__init__()
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
            proj = nn.Linear(in_dim, h) if in_dim != h else nn.Identity()
            layers.append(proj)
            in_dim = h

        self.blocks = nn.ModuleList(layers) if layers else None
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, hidden_dim)
        x = x.to(self.linear.weight.dtype)
        if self.blocks is not None:
            for i in range(0, len(self.blocks), 2):
                block = self.blocks[i]
                proj = self.blocks[i + 1]
                residual = proj(x)
                x = block(x) + residual
        return torch.sigmoid(self.linear(x))  # (batch, 1) in [0, 1]


class CoralMixHead(nn.Module):
    """Ordinal + Regression head: K classes -> K logits.

    First K-1 outputs: raw logits for ordinal binary classifiers (same as CoralHead).
    K-th output: sigmoid → [0, 1] regression value.

    Same residual block architecture as CoralHead.
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
            proj = nn.Linear(in_dim, h) if in_dim != h else nn.Identity()
            layers.append(proj)
            in_dim = h

        self.blocks = nn.ModuleList(layers) if layers else None
        self.ord_linear = nn.Linear(in_dim, num_classes - 1)  # K-1 ordinal logits
        self.reg_linear = nn.Linear(in_dim, 1)                # 1 regression logit

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, hidden_dim)
        x = x.to(self.ord_linear.weight.dtype)
        if self.blocks is not None:
            for i in range(0, len(self.blocks), 2):
                block = self.blocks[i]
                proj = self.blocks[i + 1]
                residual = proj(x)
                x = block(x) + residual
        ordinal_logits = self.ord_linear(x)                   # (batch, K-1)
        reg_value = torch.sigmoid(self.reg_linear(x))          # (batch, 1) in [0, 1]
        return ordinal_logits, reg_value


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
    """Qwen backbone + Pooling + Head for scoring.

    Args:
        head_type: "coral" (ordinal, K-1 logits), "regression" (single [0,1] output),
                   or "coral_mix" (K logits: K-1 ordinal + 1 regression).
        pooling: BasePooling instance (defaults to MeanPooling)
        head_config: kwargs dict for the head (hidden_sizes, dropout)
    """

    def __init__(self, backbone: nn.Module, hidden_dim: int, num_classes: int = 2,
                 pooling: nn.Module | None = None,
                 head_config: dict | None = None,
                 head_type: str = "coral"):
        super().__init__()
        from src.pooling import MeanPooling
        self.backbone = backbone
        self.pooling = pooling or MeanPooling()
        self.head_type = head_type
        head_kwargs = head_config or {}
        head_input_dim = getattr(self.pooling, 'output_dim', hidden_dim)

        if head_type == "regression":
            self.head = RegressionHead(head_input_dim, **head_kwargs)
        elif head_type == "coral_mix":
            self.head = CoralMixHead(head_input_dim, num_classes, **head_kwargs)
        else:
            self.head = CoralHead(head_input_dim, num_classes, **head_kwargs)

        self.num_classes = num_classes

    def forward(self, input_ids, attention_mask=None,
                handcrafted_features: torch.Tensor | None = None):
        outputs = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True
        )
        hidden = outputs.hidden_states[-1]  # (batch, seq_len, hidden_dim)
        if attention_mask is None:
            attention_mask = torch.ones(hidden.size(0), hidden.size(1), device=hidden.device)

        from src.pooling import MultiLayerPooling
        if isinstance(self.pooling, MultiLayerPooling):
            pooled = self.pooling(outputs.hidden_states, attention_mask)
        else:
            pooled = self.pooling(hidden, attention_mask)

        # Reserved: concat handcrafted features here in the future
        if handcrafted_features is not None:
            pooled = torch.cat([pooled, handcrafted_features], dim=-1)

        return self.head(pooled)  # (batch, K-1) for coral, (batch, 1) for regression


def coral_loss(
    logits: torch.Tensor, label_indices: torch.Tensor, num_classes: int,
    pos_weight: torch.Tensor | None = None,
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
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)


def coral_mix_loss(
    ordinal_logits: torch.Tensor,
    reg_value: torch.Tensor,
    label_indices: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    full_score: float,
    lambda_reg: float = 0.4,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """CORAL + Huber mixed loss for CoralMixHead.

    coral_part = coral_loss(ordinal_logits, label_indices, num_classes)
    huber_part = SmoothL1(reg_value, labels / full_score)
    total = coral_part + lambda_reg * huber_part
    """
    coral_part = coral_loss(ordinal_logits, label_indices, num_classes, pos_weight=pos_weight)
    labels_norm = labels.float() / full_score
    # mse_part = ((reg_value.squeeze(-1) - labels_norm) ** 2).mean()
    huber_part = F.smooth_l1_loss(reg_value.squeeze(-1), labels_norm, beta=0.2)
    return (1 - lambda_reg) * coral_part + lambda_reg * huber_part


def regression_loss(
    preds: torch.Tensor, labels: torch.Tensor, full_score: float,
    beta: float = 1.0,
) -> torch.Tensor:
    """MSE + margin penalty for regression head.

    L1_loss = beta * max(|pred - labels_norm| - 0.5/full_score, 0)^2
    total = MSE + mean(L1_loss)

    The margin (0.5/full_score) means errors within 0.5 score points
    only incur MSE; errors beyond that get an extra penalty weighted by beta.
    """
    labels_norm = labels.float() / full_score
    preds = preds.squeeze(-1)
    gap = torch.abs(preds - labels_norm)

    mse = ((preds - labels_norm) ** 2).mean()

    margin = 0.5 / full_score
    excess = torch.clamp(gap - margin, min=0.0)
    l1_penalty = beta * (excess ** 2).mean()

    return mse + l1_penalty


def regression_to_score(preds: torch.Tensor, full_score: float) -> list[float]:
    """Convert normalized predictions back to original score scale.

    Args:
        preds: (batch, 1) — sigmoid output in [0,1]
        full_score: maximum possible score
    Returns:
        list of float scores
    """
    return (preds.squeeze(-1) * full_score).tolist()


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
