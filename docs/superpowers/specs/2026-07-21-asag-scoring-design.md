# Automated Short-Answer / Essay Scoring — Design Spec

**Date**: 2026-07-21
**Status**: Draft

## Overview

为 101 科目（语文）的 Q8（简答题，满分4，精度0.5）和 Q9（作文，满分40，精度1）分别训练自动评分模型。模型输入学生作答 OCR 文本，输出预测分数。

## Goals & Success Criteria

- **核心指标**：精确一致率（预测分 == 标注分的比例）
- **次要指标**：MAE、QWK（Quadratic Weighted Kappa）
- **训练目标**：精确一致率最大化，MAE/MAE 总体最小

## Architecture

```
                    ┌──────────────────────────┐
                    │   DataPipeline           │
                    │   answer.xlsx + calib.csv│
                    │   ──merge by BMH──▶      │
                    │   (text, score) pairs    │
                    └───────────┬──────────────┘
                                │ 80/20 split
                ┌───────────────┼───────────────┐
                ▼                               ▼
        ┌───────────────┐               ┌───────────────┐
        │  Train Set    │               │  Test Set     │
        └───────┬───────┘               └───────┬───────┘
                │                               │
                ▼                               ▼
        ┌───────────────┐               ┌───────────────┐
        │  Qwen4B+LoRA  │──────▶────────│  Evaluate     │
        │  + CORAL Head │   checkpoint  │  • Acc        │
        │               │               │  • MAE        │
        └───────────────┘               │  • QWK        │
                                        └───────────────┘
```

## Data Flow

1. **Load & Clean**：读 answer.xlsx → 提取 `BMH` + `OCR文字结果`；读 calibration CSV（只取 `calibration*.csv`）→ 提取 `BMH` + `ZZCJ`
2. **Merge**：按 `BMH` inner join，得到 `(text, label)` 对
3. **Split**：`sklearn.model_selection.train_test_split(test_size=0.2, random_state=42)`
4. **Tokenize**：Qwen tokenizer（`qwen2.5-4b-instruct` 或 `qwen3-4b`），max_length=2048（Q9 作文可适当拉长），padding+truncation
5. **Train**：QLoRA（4bit）+ CORAL ordinal classification head，每个 epoch 后 eval
6. **Eval**：计算精确一致率、MAE、QWK

## Model Design

### Base: Qwen 4B

- `AutoModelForCausalLM` 获取 last hidden state
- QLoRA 微调（nf4, rank=16, alpha=32）
- 不加载完整模型权重到本地开发环境

### Head: CORAL Ordinal Regression

- 将 K 分类转化为 K-1 个二分类：对每个阈值 k，"score > k" 的概率
- 最后一层：`Linear(hidden_dim, K-1)`，逐位置 sigmoid
- 推理时：预测分 = argmax over cumulative probability distribution

### Loss: CORAL Loss

```
L = -Σ [y_k * log(p_k) + (1-y_k) * log(1-p_k)]   # K-1 个二分类 BCE 之和
```

### Discrete Mapping

- **Q8**：9 个分值点 → 8 个阈值（0.5, 1.0, ..., 3.5）
- **Q9**：71 个分值点 → 70 个阈值
- 推理时将 K-1 个概率聚合成最终分值
- `score_points` 从 calibration 数据中自动提取 `sorted(df['ZZCJ'].unique())`，不硬编码

## Project Structure

```
edu_project/
├── data/                          # 原始数据（只读）
│   └── 101/
├── src/
│   ├── config.py                  # 全局配置（yaml→dataclass）
│   ├── data_pipeline.py           # 数据加载、清洗、merge、split
│   ├── model.py                   # 模型定义（架构+loss，不加载权重）
│   ├── train.py                   # 训练循环
│   ├── evaluate.py                # 评估：精确一致率、MAE、QWK
│   └── infer.py                   # 推理接口
├── configs/
│   ├── q8_config.yaml             # Q8 超参
│   └── q9_config.yaml             # Q9 超参
├── scripts/
│   ├── train_q8.sh
│   └── train_q9.sh
├── requirements.txt
└── README.md
```

## Module Details

### `config.py`
- `TrainingConfig` dataclass：lr, epochs, batch_size, lora_rank, lora_alpha, max_length, test_size, seed, model_name, output_dir
- `load_config(yaml_path)` → `TrainingConfig`

### `data_pipeline.py`
- `load_answer_data(xlsx_path) → pd.DataFrame`：只读 BMH + OCR文字结果
- `load_calibration_data(csv_path) → pd.DataFrame`：只读 calibration*.csv
- `merge_data(answer_df, calib_df) → DataFrame[text, label]`
- `split_data(df, test_size, seed) → (train_df, test_df)`
- `ASAGDataset(torch.utils.data.Dataset)`：延迟 tokenize，接受 texts + labels + tokenizer
- `build_dataloaders(config) → (train_loader, test_loader)`：一站式工厂方法

### `model.py`
- `CoralHead(nn.Module)`：hidden_dim → K-1 维输出
- `OrdinalScorer(nn.Module)`：Qwen backbone + CoralHead 的组合
- `coral_loss(logits, labels, num_classes) → Tensor`：CORAL loss 实现
- `build_model(config) → OrdinalScorer`：工厂方法，本地 dev 不加载权重，通过 `load_in_4bit` 控制
- `prediction_to_score(probs, score_points) → int/float`：概率→分值
- **本地开发模式**：`model.py` 可独立 import，用 `torch.randn` 做形状测试；`DummyBackbone(nn.Module)` 返回随机 embedding，用于验证 CoralHead + loss 逻辑正确

### `train.py`
- `train_epoch(model, loader, optimizer, scheduler)`：单 epoch 训练
- `validate(model, loader, score_points)`：返回 acc, mae, qwk
- `run_training(config)`：完整训练流程，含 checkpoint 保存

### `evaluate.py`
- `exact_accuracy(preds, labels)`：精确一致率
- `mae(preds, labels)`：平均绝对误差
- `qwk(preds, labels)`：QWK（sklearn cohen_kappa_score, weights='quadratic'）
- `run_eval(checkpoint_path, test_data)`：独立评估入口

### `infer.py`
- `load_model(checkpoint_path)`：加载 checkpoint
- `predict(text, model, tokenizer)`：单条推理
- `predict_batch(texts, model, tokenizer)`：批量推理

## Configuration

### Q8
```yaml
model_name: "qwen/Qwen2.5-4B-Instruct"  # or Qwen3-4B
max_length: 1024
batch_size: 16
lora_r: 16
lora_alpha: 32
learning_rate: 2e-4
epochs: 5
test_size: 0.2
seed: 42
score_points: [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
```

### Q9
```yaml
model_name: "qwen/Qwen2.5-4B-Instruct"
max_length: 2048       # 作文更长
batch_size: 8           # 作文文本长，batch 减小
lora_r: 16
lora_alpha: 32
learning_rate: 2e-4
epochs: 5
test_size: 0.2
seed: 42
score_points: [0, 1, 2, ..., 40]  # 71个分值
```

## Development Mode

- 本地（无 GPU）：
  - 所有代码正常 import 和类型检查
  - `data_pipeline.py` 可完整运行（纯 pandas + sklearn）
  - `model.py` 定义结构和 loss，单元测试用 `torch.randn` 验证形状和 loss 计算正确
  - 不调用 `from_pretrained`，不加载 tokenizer 权重

- GPU 机器：
  - 安装 requirements.txt
  - 运行 `python src/train.py --config configs/q8_config.yaml`
  - 一键训练

## Implementation Plan

详见 writing-plans skill 产出。
