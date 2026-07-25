"""Tests for src/pooling.py — runs without GPU."""
import torch
from src.pooling import build_pooling


def test_mean_pooling_shape():
    pooling = build_pooling("mean", 2560)
    hidden = torch.randn(4, 128, 2560)
    mask = torch.ones(4, 128)
    mask[:, 100:] = 0  # last 28 tokens are padding
    out = pooling(hidden, mask)
    assert out.shape == (4, 2560), f"Expected (4, 2560), got {out.shape}"


def test_mean_pooling_values():
    pooling = build_pooling("mean", 2560)
    hidden = torch.ones(2, 5, 8)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]], dtype=torch.float)
    out = pooling(hidden, mask)
    # first sample: mean over 3 ones → all 1s; second: mean over 4 ones → all 1s
    assert out.shape == (2, 8)
    assert torch.allclose(out, torch.ones(2, 8))


def test_last_token_pooling_shape():
    pooling = build_pooling("last", 2560)
    hidden = torch.randn(4, 128, 2560)
    mask = torch.ones(4, 128)
    mask[:, 100:] = 0
    out = pooling(hidden, mask)
    assert out.shape == (4, 2560)


def test_last_token_pooling_correct_position():
    pooling = build_pooling("last", 2560)
    hidden = torch.randn(1, 5, 8)
    mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.float)
    out = pooling(hidden, mask)
    # should return hidden[:, 2, :] (index 2 = 3rd token, the last valid one)
    assert torch.allclose(out, hidden[:, 2, :])


def test_attention_pooling_shape():
    pooling = build_pooling("attention", 2560)
    hidden = torch.randn(4, 128, 2560)
    mask = torch.ones(4, 128)
    mask[:, 100:] = 0
    out = pooling(hidden, mask)
    assert out.shape == (4, 2560)


def test_build_pooling_invalid():
    try:
        build_pooling("invalid", 2560)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_all_strategies_same_interface():
    for strategy in ["mean", "last", "attention"]:
        pooling = build_pooling(strategy, 2560)
        hidden = torch.randn(2, 10, 2560)
        mask = torch.ones(2, 10)
        out = pooling(hidden, mask)
        assert out.shape == (2, 2560), f"{strategy} pooling shape mismatch"
