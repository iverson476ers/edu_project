# ASAG Scoring — 模型架构升级设计

**Date**: 2026-07-26
**Status**: Draft

## Overview

对 ASAG 评分模型的架构进行升级，将原来简单的 `backbone → mean pool → Linear head` 架构重构为可配置的 **Pooling Layer + Deep CoralHead** 结构，提升模型的表征利用率和评分精度。

手工特征入口在设计上预留，但**第一阶段不实现**。

## Goals & Success Criteria

- **核心目标**：不改变 backbone（Qwen 4B + LoRA）的前提下，通过优化 pooling 策略和 head 结构提升精确一致率
- **向后兼容**：旧 checkpoint 可正常加载推理
- **可配置**：pooling 策略和 head 深度通过 config 切换，方便 ablation 实验
- **本地可测试**：DummyBackbone 模式不受影响

## Architecture

```
                          ┌─────────────────────────────┐
                          │     ASAGDataset             │
                          │  text → tokenizer           │
                          │  保留原始 text 给特征提取    │
                          └─────────────┬───────────────┘
                                        │
                          ┌─────────────┴───────────────┐
                          │                             │
                    input_ids + mask            raw text strings
                          │                             │
              ┌───────────▼───────────┐                 │
              │  Qwen 4B + LoRA       │                 │
              │  output_hidden_states  │                 │
              │  保留 last 4 layers    │                 │
              └───────────┬───────────┘                 │
                          │                             │
              ┌───────────▼───────────┐                 │
              │  Pooling Layer        │                 │
              │  (可配置策略)          │                 │
              │  → sentence vector    │                 │
              └───────────┬───────────┘                 │
                          │                             │
                          │        ┌────────────────────┘
                          │        │  (预留：手工特征)
                          │        │
                          ▼        ▼
              ┌───────────────────────────────┐
              │       Feature Fusion           │
              │  concat(semantic, handcrafted) │
              │  第一阶段：仅 semantic          │
              └───────────────┬───────────────┘
                              │
              ┌───────────────▼───────────────┐
              │       Deep CoralHead          │
              │  LayerNorm → Linear → GELU    │
              │     → Dropout → Residual      │
              │     → Linear → GELU           │
              │     → Dropout → Residual      │
              │     → Linear → logits(K-1)    │
              └───────────────┬───────────────┘
                              │
                              ▼
                        coral_loss / prediction_to_score
```

### 与旧架构对比

| 组件 | 旧架构 | 新架构 |
|------|--------|--------|
| Pooling | 硬编码 mean pooling | 可配置（mean/last/attention），独立模块 |
| Head | `Linear(2560, K-1)` | 可选深度的 MLP + GELU + LayerNorm + Residual |
| 手工特征 | 无 | 预留 concat 入口，暂不启用 |
| 配置 | 无 pooling/head 配置项 | `pooling`, `head_hidden_sizes`, `head_dropout` |

## Component Design

### 1. Pooling Layer (`src/pooling.py`)

新增独立模块，三种策略：

| 策略 | 配置值 | 额外参数 | 说明 |
|------|--------|---------|------|
| Mean | `"mean"` | 0 | mask 加权平均，当前行为 |
| Last Token | `"last"` | 0 | 取 EOS token 的 hidden state |
| Attention | `"attention"` | 2560 | 学一个 query vector，softmax 加权 |

接口：
```python
class BasePooling(nn.Module):
    def forward(self, hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        # (batch, seq_len, hidden_dim) → (batch, hidden_dim)
```

factory 函数 `build_pooling(strategy: str, hidden_dim: int) → BasePooling`

### 2. Deep CoralHead (`src/model.py`)

`CoralHead` 重写为可配置深度 MLP：

```
输入 (hidden_dim)
  → LayerNorm
  → Linear(hidden_dim, h0) → GELU → Dropout
  → [Residual: 如果 h0 ≠ hidden_dim，用投影层对齐]
  → LayerNorm
  → Linear(h0, h1) → GELU → Dropout
  → [Residual]
  → LayerNorm
  → Linear(h_last, K-1)  ← 输出层
```

- **激活函数**: GELU（Transformer 标配，与 backbone 输出分布兼容）
- **归一化**: Pre-LayerNorm（进 Linear 前归一化）
- **正则化**: Dropout(p=0.1 可配置)
- **残差连接**: 每个 MLP 块后有残差，维度不匹配时用投影层对齐
- **可配置深度**: `head_hidden_sizes` 列表控制

`head_hidden_sizes` 示例：

| 配置 | 结构 | Head 参数量 |
|------|------|-----------|
| `[]` | `Linear(2560, K-1)`（旧行为） | ~20K / ~179K |
| `[256]` | `2560→256→K-1` | ~0.7M |
| `[512, 128]` | `2560→512→128→K-1`（推荐） | ~1.4M |
| `[1024, 256]` | `2560→1024→256→K-1` | ~3.3M |

### 3. OrdinalScorer 重构 (`src/model.py`)

`OrdinalScorer` 的 `forward` 不再硬编码 mean pooling，改为调用注入的 pooling 模块：

```python
class OrdinalScorer(nn.Module):
    def __init__(self, backbone, hidden_dim, num_classes, pooling=None, head_config=None):
        self.backbone = backbone
        self.pooling = pooling or MeanPooling()
        self.head = CoralHead(hidden_dim, num_classes, **head_config)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                                output_hidden_states=True)
        hidden = outputs.hidden_states[-1]
        pooled = self.pooling(hidden, attention_mask)
        return self.head(pooled)
```

### 4. 手工特征预留

`OrdinalScorer.forward` 的参数列表中预留 `handcrafted_features: Optional[Tensor] = None`，forward 中预留 concat 逻辑。**第一阶段不启用**。

`ASAGDataset.__getitem__` 新增返回 `"text"` 字段（原始文本字符串），供未来特征提取使用。训练/推理 DataLoader collate 时自然忽略此字段，不影响现有流程。

## Configuration Changes

`TrainingConfig` 新增字段：

```python
# Pooling
pooling: str = "mean"              # "mean" | "last" | "attention"

# Head
head_hidden_sizes: list[int] = field(default_factory=lambda: [512, 128])
head_dropout: float = 0.1
```

Config YAML 示例（Q8）：
```yaml
pooling: "last"
head_hidden_sizes: [512, 128]
head_dropout: 0.1
```

向后兼容：旧 config 文件没有这些字段 → 使用默认值 `pooling="mean"`, `head_hidden_sizes=[]` → 退化为旧架构行为。

## Checkpoint Compatibility

### 新 checkpoint 格式

```python
{
    "model_state_dict": ...,
    "config": TrainingConfig,        # 含 pooling + head_hidden_sizes
    "score_points": [...],
}
```

### 旧 checkpoint 加载

`load_scorer` 检测 config 是否包含 `pooling` 字段：
- **有** → 按新架构重建对应的 pooling + head
- **无** → 使用 `MeanPooling` + `head_hidden_sizes=[]`（旧行为）

`strict=False` 的 `load_state_dict` 保证权重加载不受影响。

### 推理兼容

`infer.py` 的 `load_scorer` 从 `config.pooling` 和 `config.head_hidden_sizes` 重建架构，自动匹配 checkpoint。用户无需手动指定。

## File Changes Summary

| 文件 | 操作 | 说明 |
|------|------|------|
| `src/pooling.py` | **新增** | BasePooling + 三种策略 + factory |
| `src/model.py` | **修改** | CoralHead 重写、OrdinalScorer 重构 |
| `src/config.py` | **修改** | 新增 pooling/head 配置项 |
| `src/train.py` | **修改** | setup 时传入 pooling + head 配置 |
| `src/infer.py` | **修改** | load_scorer 兼容新旧 checkpoint |
| `src/data_pipeline.py` | **修改** | ASAGDataset 返回 raw text |
| `src/evaluate.py` | 不改 | 评估逻辑不受影响 |
| `configs/q8_config.yaml` | **修改** | 加 pooling/head 配置 |
| `configs/q9_config.yaml` | **修改** | 加 pooling/head 配置 |
| `tests/test_pooling.py` | **新增** | Pooling 形状测试 + DummyBackbone 集成 |
| `tests/test_head.py` | **修改** | 测试新旧 CoralHead 配置 |

## Testing Strategy

1. **Pooling 形状测试**：三种策略输入 `(4, 128, 2560)` → 输出 `(4, 2560)`，本地无 GPU 可跑
2. **CoralHead 配置测试**：三种 hidden_sizes 配置 → 输出 `(batch, K-1)`
3. **向后兼容**：用旧 config 创建的模型可正常 forward
4. **Smoke test**：`scripts/smoke_test.sh` 更新，覆盖新配置路径
5. **训练集成**：DummyBackbone 模式跑 1 epoch，验证 loss 下降
