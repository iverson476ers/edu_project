import torch
import torch.nn as nn
import torch.nn.functional as F


class CoralHead(nn.Module):
    """Ordinal regression head: K classes -> K-1 binary classifiers.

    Each output neuron k predicts P(score > threshold_k).
    """

    def __init__(self, hidden_dim: int, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.linear = nn.Linear(hidden_dim, num_classes - 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (batch, hidden_dim)
        return self.linear(hidden_states)  # (batch, K-1)


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
    """Qwen backbone + CoralHead for ordinal regression scoring."""

    def __init__(self, backbone: nn.Module, hidden_dim: int, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.head = CoralHead(hidden_dim, num_classes)
        self.num_classes = num_classes

    def forward(self, input_ids, attention_mask=None):
        outputs = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True
        )
        # Use last hidden state of last token (or mean pool)
        hidden = outputs.hidden_states[-1]  # (batch, seq_len, hidden_dim)
        # Mean pool over non-padding tokens, or use last token
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            hidden = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            hidden = hidden.mean(dim=1)
        return self.head(hidden)  # (batch, K-1)


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
