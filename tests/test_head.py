"""Tests for CoralHead and OrdinalScorer — runs without GPU."""
import torch
from src.model import CoralHead, coral_loss, prediction_to_score, DummyBackbone, OrdinalScorer
from src.pooling import MeanPooling, LastTokenPooling


def test_coral_head_old_behavior():
    """Empty hidden_sizes should behave like old single Linear head."""
    head = CoralHead(2560, 9, hidden_sizes=[])
    out = head(torch.randn(4, 2560))
    assert out.shape == (4, 8), f"Expected (4, 8), got {out.shape}"


def test_coral_head_single_hidden():
    head = CoralHead(2560, 9, hidden_sizes=[256])
    out = head(torch.randn(4, 2560))
    assert out.shape == (4, 8)


def test_coral_head_deep():
    head = CoralHead(2560, 9, hidden_sizes=[512, 128], dropout=0.1)
    out = head(torch.randn(4, 2560))
    assert out.shape == (4, 8)


def test_coral_head_q9():
    """Test with Q9-like 71 score points."""
    head = CoralHead(2560, 71, hidden_sizes=[512, 128])
    out = head(torch.randn(2, 2560))
    assert out.shape == (2, 70)


def test_coral_loss_with_new_head():
    head = CoralHead(2560, 9, hidden_sizes=[512, 128])
    out = head(torch.randn(4, 2560))
    loss = coral_loss(out, torch.tensor([0, 3, 5, 8]), 9)
    assert loss.item() > 0
    assert loss.requires_grad


def test_ordinal_scorer_with_mean_pooling():
    backbone = DummyBackbone(2560)
    pooling = MeanPooling()
    scorer = OrdinalScorer(backbone, 2560, 9, pooling=pooling, head_config={"hidden_sizes": []})
    out = scorer(torch.randint(0, 1000, (4, 128)))
    assert out.shape == (4, 8)


def test_ordinal_scorer_with_last_pooling():
    backbone = DummyBackbone(2560)
    pooling = LastTokenPooling()
    scorer = OrdinalScorer(backbone, 2560, 9, pooling=pooling, head_config={"hidden_sizes": [512, 128]})
    out = scorer(torch.randint(0, 1000, (4, 128)))
    assert out.shape == (4, 8)


def test_ordinal_scorer_defaults():
    """Backward compat: OrdinalScorer without pooling/head_config should work."""
    backbone = DummyBackbone(2560)
    scorer = OrdinalScorer(backbone, 2560, 9)
    out = scorer(torch.randint(0, 1000, (4, 128)))
    assert out.shape == (4, 8)


def test_ordinal_scorer_handcrafted_reserved():
    """handcrafted_features=None should not break forward."""
    backbone = DummyBackbone(2560)
    scorer = OrdinalScorer(backbone, 2560, 9, head_config={"hidden_sizes": [512, 128]})
    out = scorer(torch.randint(0, 1000, (4, 128)), handcrafted_features=None)
    assert out.shape == (4, 8)
