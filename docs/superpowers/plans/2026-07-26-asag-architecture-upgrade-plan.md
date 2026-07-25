# ASAG Architecture Upgrade — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将简单 `backbone → mean pool → Linear head` 架构升级为可配置 Pooling Layer + Deep CoralHead，预留手工特征入口。

**Architecture:** 拆出独立 Pooling 模块（mean/last/attention 三策略），CoralHead 从单层 Linear 升级为可配置深度 MLP + GELU + LayerNorm + Residual，OrdinalScorer 编排 pooling + head 并预留 handcrafted_features 参数。

**Tech Stack:** PyTorch, PEFT/LoRA, Qwen 4B, Transformers, CORAL ordinal regression

## Global Constraints

- 向后兼容：旧 checkpoint（config 中无 pooling 字段）必须可正常加载推理
- 本地可测试：DummyBackbone 模式不受影响，所有新模块可独立用 `torch.randn` 验证
- `TrainingConfig` 新增字段必须有默认值，旧 YAML 不加新字段可正常运行
- checkpoint 格式不变：`{"model_state_dict": ..., "config": TrainingConfig, "score_points": [...]}`
- `strict=False` 的 `load_state_dict` 保证新旧权重兼容
- pooling 默认值 `"mean"`, `head_hidden_sizes` 默认值 `[]`（退化为旧行为）

---

### Task 1: Config 字段 + ASAGDataset raw text

**Files:**
- Modify: `src/config.py:6-27`
- Modify: `src/data_pipeline.py:82-87`

**Interfaces:**
- Consumes: nothing
- Produces: `TrainingConfig.pooling: str`, `TrainingConfig.head_hidden_sizes: list[int]`, `TrainingConfig.head_dropout: float`; `ASAGDataset.__getitem__` returns `"text"` key

- [ ] **Step 1: Add fields to TrainingConfig**

```python
# src/config.py — after line 27 (tolerance field), before the blank line

@dataclass
class TrainingConfig:
    model_name: str = "qwen/Qwen2.5-4B-Instruct"
    max_length: int = 1024
    batch_size: int = 16
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    epochs: int = 5
    test_size: float = 0.2
    seed: int = 42
    output_dir: str = "./output"
    data_dir: str = "./data/101"
    question_id: str = "8"
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    logging_steps: int = 50
    eval_steps: int = 200
    save_steps: int = 500
    tolerance: float = 0.0
    # NEW fields:
    pooling: str = "mean"                              # "mean" | "last" | "attention"
    head_hidden_sizes: list[int] = field(default_factory=list)  # e.g. [512, 128], empty=old behavior
    head_dropout: float = 0.1
```

- [ ] **Step 2: Add raw text to ASAGDataset return dict**

```python
# src/data_pipeline.py — in __getitem__, add "text" key to return dict
# Line 82-87, change to:

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": self.labels[idx],
            "label_idx": self.label_indices[idx],
            "text": self.texts[idx],
        }
```

- [ ] **Step 3: Verify config loads**

Run: `python3 -c "from src.config import load_config; cfg=load_config('configs/q8_config.yaml'); print(f'pooling={cfg.pooling}, head_hidden_sizes={cfg.head_hidden_sizes}, head_dropout={cfg.head_dropout}')"`

Expected: `pooling=mean, head_hidden_sizes=[], head_dropout=0.1` (旧 YAML 无这些字段，使用默认值)

- [ ] **Step 4: Commit**

```bash
git add src/config.py src/data_pipeline.py
git commit -m "feat: add pooling and head config fields, add raw text to ASAGDataset"
```

---

### Task 2: Pooling Layer

**Files:**
- Create: `src/pooling.py`

**Interfaces:**
- Consumes: nothing (standalone module)
- Produces: `build_pooling(strategy: str, hidden_dim: int) -> BasePooling`, `MeanPooling`, `LastTokenPooling`, `AttentionPooling`

- [ ] **Step 1: Write the test script**

Create `tests/test_pooling.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_pooling.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'src.pooling'`

- [ ] **Step 3: Create `src/pooling.py`**

```python
"""Pooling strategies for extracting sentence vectors from hidden states."""
from __future__ import annotations

import torch
import torch.nn as nn


class BasePooling(nn.Module):
    """Abstract pooling: (batch, seq_len, hidden_dim) → (batch, hidden_dim)."""

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class MeanPooling(BasePooling):
    """Mask-weighted mean over non-padding tokens. Zero extra parameters."""

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()        # (B, L, 1)
        masked = hidden_states * mask
        summed = masked.sum(dim=1)                         # (B, D)
        counts = mask.sum(dim=1).clamp(min=1)              # (B, 1)
        return summed / counts


class LastTokenPooling(BasePooling):
    """Take the hidden state of the last non-padding token (EOS). Zero extra parameters."""

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # sequence_lengths: (B,) — position of last valid token for each sample
        sequence_lengths = attention_mask.sum(dim=1) - 1   # 0-indexed
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[batch_indices, sequence_lengths]


class AttentionPooling(BasePooling):
    """Learnable query vector → softmax-weighted sum over tokens. Extra D params."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # hidden_states: (B, L, D)
        scores = torch.matmul(hidden_states, self.query)   # (B, L)
        # Mask padding positions with -inf
        scores = scores.masked_fill(attention_mask == 0, float("-inf"))
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_pooling.py -v`

Expected: 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/pooling.py tests/test_pooling.py
git commit -m "feat: add pooling layer with mean/last/attention strategies"
```

---

### Task 3: CoralHead 重写 + OrdinalScorer 重构

**Files:**
- Modify: `src/model.py:8-64`

**Interfaces:**
- Consumes: `build_pooling` from `src.pooling`
- Produces: `CoralHead(hidden_dim, num_classes, hidden_sizes, dropout)`, `OrdinalScorer(backbone, hidden_dim, num_classes, pooling, head_config)`

- [ ] **Step 1: Write the test script for new CoralHead**

Create `tests/test_head.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_head.py -v`

Expected: FAIL — `CoralHead.__init__` doesn't accept `hidden_sizes`

- [ ] **Step 3: Rewrite CoralHead**

Replace `src/model.py` lines 8-21:

```python
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
        self.output = nn.Linear(in_dim, num_classes - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, hidden_dim)
        if self.blocks is not None:
            for i in range(0, len(self.blocks), 2):
                block = self.blocks[i]       # Sequential: LayerNorm → Linear → GELU → Dropout
                proj = self.blocks[i + 1]    # Projection for residual
                residual = proj(x)
                x = block(x) + residual
        return self.output(x)  # (batch, K-1)
```

- [ ] **Step 4: Refactor OrdinalScorer**

Replace `src/model.py` lines 43-64:

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_head.py -v`

Expected: 9 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/model.py tests/test_head.py
git commit -m "feat: deep CoralHead with configurable depth + OrdinalScorer refactor"
```

---

### Task 4: Train 集成

**Files:**
- Modify: `src/train.py:19-62` (setup_model_and_tokenizer)

**Interfaces:**
- Consumes: `OrdinalScorer(backbone, hidden_dim, num_classes, pooling, head_config)` from Task 3
- Produces: `setup_model_and_tokenizer` passes pooling + head config

- [ ] **Step 1: Update `setup_model_and_tokenizer`**

Replace `src/train.py` lines 56-57:

```python
    hidden_dim = backbone.config.hidden_size
    # Old: model = OrdinalScorer(backbone, hidden_dim, num_classes)
    # New:
    from src.pooling import build_pooling
    pooling = build_pooling(config.pooling, hidden_dim)
    head_config = {
        "hidden_sizes": config.head_hidden_sizes,
        "dropout": config.head_dropout,
    }
    model = OrdinalScorer(backbone, hidden_dim, num_classes,
                          pooling=pooling, head_config=head_config)
```

- [ ] **Step 2: Verify train module imports**

Run: `python3 -c "from src.train import setup_model_and_tokenizer, train; print('OK')"`

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/train.py
git commit -m "feat: wire pooling and head config into training pipeline"
```

---

### Task 5: Infer 向后兼容

**Files:**
- Modify: `src/infer.py:9-56` (load_scorer)

**Interfaces:**
- Consumes: `OrdinalScorer` with new signature from Task 3
- Produces: `load_scorer` rebuilds correct arch from config, handles old checkpoints

- [ ] **Step 1: Update `load_scorer` for backward compat**

Replace `src/infer.py` lines 50-51:

```python
    hidden_dim = backbone.config.hidden_size
    # Old: model = OrdinalScorer(backbone, hidden_dim, len(score_points))
    # New: detect arch from config, fall back for old checkpoints
    from src.pooling import build_pooling
    pooling_strategy = getattr(config, "pooling", "mean")
    pooling = build_pooling(pooling_strategy, hidden_dim)
    head_config = {
        "hidden_sizes": getattr(config, "head_hidden_sizes", []),
        "dropout": getattr(config, "head_dropout", 0.1),
    }
    model = OrdinalScorer(backbone, hidden_dim, len(score_points),
                          pooling=pooling, head_config=head_config)
```

- [ ] **Step 2: Verify infer module imports**

Run: `python3 -c "from src.infer import predict, predict_batch, load_scorer; print('OK')"`

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/infer.py
git commit -m "feat: backward-compatible load_scorer with arch detection"
```

---

### Task 6: Config YAMLs + Smoke Test 更新

**Files:**
- Modify: `configs/q8_config.yaml`
- Modify: `configs/q9_config.yaml`
- Modify: `scripts/smoke_test.sh`

**Interfaces:**
- Consumes: all previous tasks
- Produces: updated configs with new architecture defaults, smoke test covers new paths

- [ ] **Step 1: Update Q8 config**

Add to `configs/q8_config.yaml` after line 20 (`tolerance: 0.67`):

```yaml
# Architecture
pooling: "last"
head_hidden_sizes: [512, 128]
head_dropout: 0.1
```

- [ ] **Step 2: Update Q9 config**

Add to `configs/q9_config.yaml` after line 20 (`tolerance: 6.67`):

```yaml
# Architecture
pooling: "last"
head_hidden_sizes: [512, 128]
head_dropout: 0.1
```

- [ ] **Step 3: Update smoke test**

Replace `scripts/smoke_test.sh` Step [4/5] (lines 15-28):

```bash
echo "[4/5] Model + Pooling + Head shapes..."
python3 -c "
import torch
from src.model import CoralHead, coral_loss, prediction_to_score, DummyBackbone, OrdinalScorer
from src.pooling import build_pooling

# Test old-style head (empty hidden_sizes)
head_old = CoralHead(2560, 9, hidden_sizes=[])
out_old = head_old(torch.randn(4, 2560))
print(f'OK: old head shape={out_old.shape}')

# Test new deep head
head_new = CoralHead(2560, 9, hidden_sizes=[512, 128], dropout=0.1)
out_new = head_new(torch.randn(4, 2560))
loss = coral_loss(out_new, torch.tensor([0,3,5,8]), 9)
print(f'OK: new head shape={out_new.shape}, loss={loss.item():.4f}')

# Test pooling strategies
for s in ['mean', 'last', 'attention']:
    p = build_pooling(s, 2560)
    out = p(torch.randn(4, 128, 2560), torch.ones(4, 128))
    print(f'OK: {s} pooling shape={out.shape}')

# Test OrdinalScorer with last pooling + deep head
backbone = DummyBackbone(2560)
pooling = build_pooling('last', 2560)
scorer = OrdinalScorer(backbone, 2560, 9, pooling=pooling, head_config={'hidden_sizes': [512, 128]})
out2 = scorer(torch.randint(0,1000,(4,128)))
probs = torch.sigmoid(out2)
scores = prediction_to_score(probs, [0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0])
print(f'OK: scorer_out={out2.shape}, scores={scores}')
"
```

- [ ] **Step 4: Run smoke test**

Run: `bash scripts/smoke_test.sh`

Expected: All 5 steps PASS with `=== All smoke tests passed! ===`

- [ ] **Step 5: Commit**

```bash
git add configs/q8_config.yaml configs/q9_config.yaml scripts/smoke_test.sh
git commit -m "feat: set new architecture defaults in configs, update smoke test"
```
