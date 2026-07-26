# Automated Short-Answer / Essay Scoring (ASAG/AES)

基于 Qwen 4B + QLoRA + CORAL 序数回归的自动评分系统。

## 项目结构

```
src/
├── config.py          # 配置管理 (YAML -> TrainingConfig)
├── data_pipeline.py   # 数据加载、清洗、merge、split
├── model.py           # CoralHead, OrdinalScorer, CORAL loss
├── evaluate.py        # 指标: 精确一致率, MAE, QWK
├── train.py           # 训练循环
└── infer.py           # 推理
configs/
├── q8_config.yaml     # 简答题 (满分4, 精度0.5)
└── q9_config.yaml     # 作文 (满分40, 精度1)
scripts/
├── train_q8.sh
├── train_q9.sh
└── smoke_test.sh      # 本地无GPU冒烟测试
```

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 本地开发测试（无需 GPU）

```bash
bash scripts/smoke_test.sh
```

### GPU 训练

```bash
# Q8 简答题
bash scripts/train_q8.sh

# Q9 作文
bash scripts/train_q9.sh
```

### 推理

```python
from src.infer import load_scorer, predict

model, tokenizer, score_points = load_scorer("output/q8/best_model.pt")
score = predict("学生的答案文本...", model, tokenizer, score_points)
print(score)
```

## 数据格式

- `data/101/answer/answer101_{qid}.xlsx`: 学生作答 OCR 文本
- `data/101/calibration/calibration*_{qid}.csv`: 标注分数 (BMH -> ZZCJ)

## 输出目录

训练完成后，模型权重和运行日志保存在 `output/` 下：

```
output/
├── q8/
│   ├── best_model.pt       # 最优 checkpoints
│   ├── final_model.pt      # 最终 checkpoints
│   └── metrics.json        # 训练/验证指标
└── q9/
    ├── best_model.pt
    ├── final_model.pt
    └── metrics.json
```

## 优化角度
- config参数，batch_size, seq_max_length
- config 参数，lora rank参数 8,16
