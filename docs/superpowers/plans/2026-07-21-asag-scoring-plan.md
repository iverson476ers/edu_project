# Automated Short-Answer / Essay Scoring — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build modular training/inference pipeline for Q8 (short-answer) and Q9 (essay) scoring models using Qwen 4B + QLoRA + CORAL ordinal regression, local-dev-safe (no GPU required for development).

**Architecture:** Data pipeline merges answer xlsx + calibration csv by BMH → 80/20 split → Qwen 4B backbone (4bit QLoRA) → CoralHead (K-1 binary classifiers) → CORAL loss. No model weights loaded in local dev; use DummyBackbone for shape/flow testing.

**Tech Stack:** Python 3.10+, PyTorch 2.x, transformers (Qwen), peft (LoRA), bitsandbytes (4bit), pandas, scikit-learn, PyYAML

## Global Constraints

- Local dev: no GPU, no `from_pretrained`, no tokenizer weight download. Code must import and type-check cleanly.
- GPU training: single `python src/train.py --config configs/q8_config.yaml` entry point.
- `score_points` extracted automatically from calibration data, not hardcoded.
- Q8 max_length=1024, Q9 max_length=2048.
- Metrics: exact accuracy (primary), MAE, QWK (quadratic weighted kappa).
- Random seed: 42, test_size: 0.2.

---

### Task 1: Project scaffold and requirements

**Files:**
- Create: `src/__init__.py`
- Create: `configs/__init__.py` (empty)
- Create: `scripts/__init__.py` (empty)
- Create: `requirements.txt`

**Interfaces:**
- Produces: installable project structure, dependency list

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p src configs scripts
touch src/__init__.py
```

- [ ] **Step 2: Write requirements.txt**

Write `requirements.txt`:
```
torch>=2.1.0
transformers>=4.40.0
peft>=0.10.0
bitsandbytes>=0.43.0
accelerate>=0.28.0
pandas>=2.0.0
openpyxl>=3.1.0
scikit-learn>=1.3.0
pyyaml>=6.0
```

- [ ] **Step 3: Verify project is importable**

```bash
python3 -c "import sys; sys.path.insert(0, '.'); from src import *; print('OK')"
```
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add src/__init__.py configs/ scripts/ requirements.txt
git commit -m "feat: scaffold project structure and requirements"
```

---

### Task 2: Configuration management (`config.py`)

**Files:**
- Create: `src/config.py`
- Create: `configs/q8_config.yaml`
- Create: `configs/q9_config.yaml`

**Interfaces:**
- Produces:
  - `TrainingConfig` dataclass with fields: `model_name: str`, `max_length: int`, `batch_size: int`, `lora_r: int`, `lora_alpha: int`, `learning_rate: float`, `epochs: int`, `test_size: float`, `seed: int`, `output_dir: str`, `data_dir: str`, `question_id: str`
  - `load_config(yaml_path: str) -> TrainingConfig`

- [ ] **Step 1: Write `src/config.py`**

```python
from dataclasses import dataclass, field
from pathlib import Path
import yaml


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


def load_config(yaml_path: str) -> TrainingConfig:
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return TrainingConfig(**{k: v for k, v in data.items() if v is not None})
```

- [ ] **Step 2: Write `configs/q8_config.yaml`**

```yaml
model_name: "qwen/Qwen2.5-4B-Instruct"
max_length: 1024
batch_size: 16
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
learning_rate: 2.0e-4
epochs: 5
test_size: 0.2
seed: 42
output_dir: "./output/q8"
data_dir: "./data/101"
question_id: "8"
warmup_ratio: 0.1
weight_decay: 0.01
max_grad_norm: 1.0
logging_steps: 50
eval_steps: 200
save_steps: 500
```

- [ ] **Step 3: Write `configs/q9_config.yaml`**

```yaml
model_name: "qwen/Qwen2.5-4B-Instruct"
max_length: 2048
batch_size: 8
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
learning_rate: 2.0e-4
epochs: 5
test_size: 0.2
seed: 42
output_dir: "./output/q9"
data_dir: "./data/101"
question_id: "9"
warmup_ratio: 0.1
weight_decay: 0.01
max_grad_norm: 1.0
logging_steps: 50
eval_steps: 200
save_steps: 500
```

- [ ] **Step 4: Verify config loading works**

```bash
python3 -c "
from src.config import load_config
cfg = load_config('configs/q8_config.yaml')
print(f'model={cfg.model_name}, max_len={cfg.max_length}, qid={cfg.question_id}')
"
```
Expected: `model=qwen/Qwen2.5-4B-Instruct, max_len=1024, qid=8`

- [ ] **Step 5: Verify Q9 config**

```bash
python3 -c "
from src.config import load_config
cfg = load_config('configs/q9_config.yaml')
print(f'batch_size={cfg.batch_size}, max_len={cfg.max_length}, output={cfg.output_dir}')
"
```
Expected: `batch_size=8, max_len=2048, output=./output/q9`

- [ ] **Step 6: Commit**

```bash
git add src/config.py configs/q8_config.yaml configs/q9_config.yaml
git commit -m "feat: add TrainingConfig and Q8/Q9 config yamls"
```

---

### Task 3: Data pipeline (`data_pipeline.py`)

**Files:**
- Create: `src/data_pipeline.py`

**Interfaces:**
- Consumes: `TrainingConfig` from Task 2
- Produces:
  - `load_answer_data(xlsx_path: str) -> pd.DataFrame`  # columns: [BMH, text]
  - `load_calibration_data(data_dir: str, question_id: str) -> pd.DataFrame`  # columns: [BMH, label], only calibration*.csv
  - `merge_data(answer_df: pd.DataFrame, calib_df: pd.DataFrame) -> pd.DataFrame`  # columns: [text, label, score_idx]
  - `split_data(df: pd.DataFrame, test_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]`
  - `get_score_points(df: pd.DataFrame) -> list[float]`  # sorted unique labels
  - `build_dataloaders(config: TrainingConfig) -> tuple[DataLoader, DataLoader, list[float]]`  # one-stop factory, returns train_loader, test_loader, score_points. Uses a dummy tokenizer placeholder — tokenizer injection happens in train.py

- [ ] **Step 1: Write `src/data_pipeline.py`**

```python
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


def load_answer_data(xlsx_path: str) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path)
    df = df[["BMH", "OCR文字结果"]].copy()
    df = df.dropna(subset=["OCR文字结果"])
    df.columns = ["BMH", "text"]
    df["text"] = df["text"].astype(str)
    return df.reset_index(drop=True)


def load_calibration_data(data_dir: str, question_id: str) -> pd.DataFrame:
    pattern = f"{data_dir}/calibration/calibration*_{question_id}.csv"
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No calibration file found matching: {pattern}")
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)
    calib = pd.concat(dfs, ignore_index=True)
    calib = calib[["BMH", "ZZCJ"]].copy()
    calib.columns = ["BMH", "label"]
    calib["label"] = calib["label"].astype(float)
    return calib.dropna()


def get_score_points(df: pd.DataFrame) -> list[float]:
    return sorted(df["label"].unique().tolist())


def merge_data(answer_df: pd.DataFrame, calib_df: pd.DataFrame) -> pd.DataFrame:
    merged = answer_df.merge(calib_df, on="BMH", how="inner")
    score_points = get_score_points(merged)
    score_to_idx = {s: i for i, s in enumerate(score_points)}
    merged["label_idx"] = merged["label"].map(score_to_idx)
    return merged[["BMH", "text", "label", "label_idx"]].reset_index(drop=True)


def split_data(
    df: pd.DataFrame, test_size: float = 0.2, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=None
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


class ASAGDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 1024):
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.label_indices = df["label_idx"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        if self.tokenizer is not None:
            encoded = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].squeeze(0)
            attention_mask = encoded["attention_mask"].squeeze(0)
        else:
            # Dummy mode: return text as-is for pipeline testing
            input_ids = text
            attention_mask = None
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": self.labels[idx],
            "label_idx": self.label_indices[idx],
        }


def collate_fn(batch):
    """Placeholder collate — real batching needs tokenizer padding handled by DataLoader.
    In train.py, the DataLoader is set up with the tokenizer's pad_token_id."""
    return batch


def build_dataloaders(
    config: "TrainingConfig",
) -> tuple:
    """Load, merge, split data and return DataLoaders + score_points.

    Returns (train_loader, test_loader, score_points).
    DataLoaders are returned as (train_df, test_df) — actual torch DataLoader
    construction happens in train.py when tokenizer is available.
    """
    from src.config import TrainingConfig

    answer_path = f"{config.data_dir}/answer/answer101_{config.question_id}.xlsx"

    answer_df = load_answer_data(answer_path)
    calib_df = load_calibration_data(config.data_dir, config.question_id)
    merged_df = merge_data(answer_df, calib_df)
    score_points = get_score_points(merged_df)
    train_df, test_df = split_data(merged_df, config.test_size, config.seed)

    print(f"Total labeled samples: {len(merged_df)}")
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    print(f"Score points ({len(score_points)}): {score_points[:5]}...{score_points[-3:]}")

    return train_df, test_df, score_points
```

- [ ] **Step 2: Verify pipeline runs end-to-end locally**

```bash
python3 -c "
from src.data_pipeline import build_dataloaders
from src.config import load_config
cfg = load_config('configs/q8_config.yaml')
train_df, test_df, score_points = build_dataloaders(cfg)
print(f'Train shape: {train_df.shape}')
print(f'Test shape: {test_df.shape}')
print(f'Score points: {score_points}')
print(f'Sample text length: {len(train_df.iloc[0][\"text\"])} chars')
"
```
Expected: output showing dataset sizes and score points, no errors.

- [ ] **Step 3: Verify Q9 pipeline**

```bash
python3 -c "
from src.data_pipeline import build_dataloaders
from src.config import load_config
cfg = load_config('configs/q9_config.yaml')
train_df, test_df, score_points = build_dataloaders(cfg)
print(f'Train shape: {train_df.shape}')
print(f'Score points count: {len(score_points)}')
"
```
Expected: ~6399 train, ~1600 test, 71 score points.

- [ ] **Step 4: Commit**

```bash
git add src/data_pipeline.py
git commit -m "feat: add data pipeline - load, merge, split, ASAGDataset"
```

---

### Task 4: Evaluation metrics (`evaluate.py`)

**Files:**
- Create: `src/evaluate.py`

**Interfaces:**
- Produces:
  - `exact_accuracy(preds: list[float], labels: list[float]) -> float`
  - `mae(preds: list[float], labels: list[float]) -> float`
  - `qwk(preds: list[float], labels: list[float]) -> float`
  - `compute_metrics(preds: list[float], labels: list[float]) -> dict[str, float]`

- [ ] **Step 1: Write `src/evaluate.py`**

```python
from sklearn.metrics import cohen_kappa_score


def exact_accuracy(preds: list[float], labels: list[float]) -> float:
    correct = sum(1 for p, l in zip(preds, labels) if p == l)
    return correct / len(labels)


def mae(preds: list[float], labels: list[float]) -> float:
    return sum(abs(p - l) for p, l in zip(preds, labels)) / len(labels)


def qwk(preds: list[float], labels: list[float]) -> float:
    return cohen_kappa_score(labels, preds, weights="quadratic")


def compute_metrics(preds: list[float], labels: list[float]) -> dict[str, float]:
    return {
        "exact_accuracy": exact_accuracy(preds, labels),
        "mae": mae(preds, labels),
        "qwk": qwk(preds, labels),
    }
```

- [ ] **Step 2: Verify metrics with sample data**

```bash
python3 -c "
from src.evaluate import compute_metrics
preds = [0, 2, 4, 4, 1]
labels = [0, 2, 3, 4, 1]
print(compute_metrics(preds, labels))
"
```
Expected: `{'exact_accuracy': 0.8, 'mae': 0.2, 'qwk': ...}`

- [ ] **Step 3: Commit**

```bash
git add src/evaluate.py
git commit -m "feat: add evaluation metrics - exact accuracy, MAE, QWK"
```

---

### Task 5: Model definition (`model.py`)

**Files:**
- Create: `src/model.py`

**Interfaces:**
- Produces:
  - `CoralHead(hidden_dim: int, num_classes: int) -> nn.Module` # output K-1 logits
  - `DummyBackbone(hidden_dim: int)` → `nn.Module` # returns random embeddings, local dev only
  - `OrdinalScorer(backbone, hidden_dim, num_classes) -> nn.Module` # backbone + CoralHead
  - `coral_loss(logits: Tensor, label_indices: Tensor, num_classes: int) -> Tensor`
  - `prediction_to_score(probs: Tensor, score_points: list[float]) -> list[float]`

- [ ] **Step 1: Write `src/model.py`**

```python
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
```

- [ ] **Step 2: Verify CoralHead shapes with dummy input**

```bash
python3 -c "
import torch
from src.model import CoralHead, coral_loss, prediction_to_score, DummyBackbone, OrdinalScorer

# Test CoralHead
head = CoralHead(hidden_dim=2560, num_classes=9)
x = torch.randn(4, 2560)
out = head(x)
print(f'CoralHead output shape: {out.shape}')  # Expected: (4, 8)

# Test CORAL loss
labels = torch.tensor([0, 3, 5, 8])
loss = coral_loss(out, labels, num_classes=9)
print(f'CORAL loss: {loss.item():.4f}')

# Test prediction_to_score
probs = torch.sigmoid(out)
score_points = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
scores = prediction_to_score(probs, score_points)
print(f'Predicted scores: {scores}')

# Test DummyBackbone + OrdinalScorer
backbone = DummyBackbone(hidden_dim=2560)
scorer = OrdinalScorer(backbone, hidden_dim=2560, num_classes=9)
out2 = scorer(torch.randint(0, 1000, (4, 128)))
print(f'OrdinalScorer output shape: {out2.shape}')  # Expected: (4, 8)
print('All shape checks passed!')
"
```
Expected: all shapes correct, loss is a finite number.

- [ ] **Step 3: Commit**

```bash
git add src/model.py
git commit -m "feat: add CoralHead, OrdinalScorer, CORAL loss, DummyBackbone"
```

---

### Task 6: Training loop (`train.py`)

**Files:**
- Create: `src/train.py`

**Interfaces:**
- Consumes: `TrainingConfig` (Task 2), `build_dataloaders` + `ASAGDataset` (Task 3), `compute_metrics` (Task 4), `OrdinalScorer`, `coral_loss`, `prediction_to_score` (Task 5)
- Produces:
  - `setup_model_and_tokenizer(config: TrainingConfig, num_classes: int) -> tuple[OrdinalScorer, tokenizer]`  # loads Qwen + LoRA, wraps with CoralHead
  - `validate(model, dataloader, score_points, device) -> dict[str, float]`
  - `train(config: TrainingConfig) -> None`  # full training loop

- [ ] **Step 1: Write `src/train.py`**

```python
import os
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType

from src.config import TrainingConfig
from src.data_pipeline import build_dataloaders, ASAGDataset
from src.evaluate import compute_metrics
from src.model import OrdinalScorer, coral_loss, prediction_to_score


def setup_model_and_tokenizer(
    config: TrainingConfig, num_classes: int
):
    """Load Qwen 4B with 4-bit QLoRA + CoralHead. Returns (model, tokenizer)."""
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    backbone = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        output_hidden_states=True,
    )

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    backbone = get_peft_model(backbone, lora_config)
    backbone.config.use_cache = False  # disable KV cache for training

    hidden_dim = backbone.config.hidden_size
    model = OrdinalScorer(backbone, hidden_dim, num_classes)

    # Enable gradient checkpointing
    backbone.gradient_checkpointing_enable()

    return model, tokenizer


def get_dataloader(df, tokenizer, config, shuffle=True):
    ds = ASAGDataset(df, tokenizer, config.max_length)
    return DataLoader(ds, batch_size=config.batch_size, shuffle=shuffle)


def validate(model, dataloader, score_points, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits)
            preds = prediction_to_score(probs.cpu(), score_points)
            all_preds.extend(preds)
            all_labels.extend(batch["label"].tolist())
    return compute_metrics(all_preds, all_labels)


def train(config: TrainingConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config.seed)

    # 1. Load data
    train_df, test_df, score_points = build_dataloaders(config)
    num_classes = len(score_points)

    # 2. Setup model
    model, tokenizer = setup_model_and_tokenizer(config, num_classes)
    model.to(device)

    # 3. Build dataloaders
    train_loader = get_dataloader(train_df, tokenizer, config, shuffle=True)
    test_loader = get_dataloader(test_df, tokenizer, config, shuffle=False)

    # 4. Optimizer & scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    total_steps = len(train_loader) * config.epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # 5. Training loop
    os.makedirs(config.output_dir, exist_ok=True)
    global_step = 0
    best_acc = 0.0

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label_indices = batch["label_idx"].to(device)

            logits = model(input_ids, attention_mask)
            loss = coral_loss(logits, label_indices, num_classes)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            if global_step % config.logging_steps == 0:
                print(f"Step {global_step}: loss={loss.item():.4f}, lr={scheduler.get_last_lr()[0]:.2e}")

            if global_step % config.eval_steps == 0:
                metrics = validate(model, test_loader, score_points, device)
                print(f"Eval @ step {global_step}: {metrics}")
                if metrics["exact_accuracy"] > best_acc:
                    best_acc = metrics["exact_accuracy"]
                    torch.save(
                        {"model_state_dict": model.state_dict(), "config": config, "score_points": score_points},
                        os.path.join(config.output_dir, "best_model.pt"),
                    )
                    print(f"New best model saved! acc={best_acc:.4f}")
                model.train()

        # End of epoch eval
        metrics = validate(model, test_loader, score_points, device)
        print(f"Epoch {epoch+1}/{config.epochs}: {metrics}")
        if metrics["exact_accuracy"] > best_acc:
            best_acc = metrics["exact_accuracy"]
            torch.save(
                {"model_state_dict": model.state_dict(), "config": config, "score_points": score_points},
                os.path.join(config.output_dir, "best_model.pt"),
            )

        epoch_avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} avg loss: {epoch_avg_loss:.4f}")

    print(f"Training complete. Best accuracy: {best_acc:.4f}")
    return model, tokenizer


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    from src.config import load_config
    cfg = load_config(args.config)
    train(cfg)
```

- [ ] **Step 2: Verify train.py imports cleanly (no GPU required)**

```bash
python3 -c "
# Import only — should NOT trigger model loading
from src.train import setup_model_and_tokenizer, validate, train
print('train.py imports OK')
"
```
Expected: `train.py imports OK` (no CUDA errors, no model downloads).

- [ ] **Step 3: Commit**

```bash
git add src/train.py
git commit -m "feat: add training loop with QLoRA + CORAL + eval checkpoints"
```

---

### Task 7: Inference interface (`infer.py`)

**Files:**
- Create: `src/infer.py`

**Interfaces:**
- Consumes: `OrdinalScorer`, `prediction_to_score` (Task 5)
- Produces:
  - `load_scorer(checkpoint_path: str, device: str) -> tuple[OrdinalScorer, tokenizer, list[float]]`
  - `predict(text: str, model, tokenizer, score_points, device) -> float`
  - `predict_batch(texts: list[str], model, tokenizer, score_points, device) -> list[float]`

- [ ] **Step 1: Write `src/infer.py`**

```python
import torch
from transformers import AutoTokenizer

from src.model import OrdinalScorer, prediction_to_score


def load_scorer(checkpoint_path: str, device: str = "cuda"):
    """Load a saved OrdinalScorer checkpoint. Returns (model, tokenizer, score_points)."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt["config"]
    score_points = ckpt["score_points"]

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Rebuild model structure and load weights
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    backbone = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        output_hidden_states=True,
    )
    from peft import PeftModel
    # Note: backbone loaded as base; LoRA weights are in the saved state_dict
    hidden_dim = backbone.config.hidden_size
    model = OrdinalScorer(backbone, hidden_dim, len(score_points))
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device)
    model.eval()

    return model, tokenizer, score_points


def predict(
    text: str, model: OrdinalScorer, tokenizer, score_points: list[float], device: str = "cuda"
) -> float:
    encoded = tokenizer(
        text, max_length=2048, padding="max_length", truncation=True, return_tensors="pt"
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = torch.sigmoid(logits)
        scores = prediction_to_score(probs.cpu(), score_points)
    return scores[0]


def predict_batch(
    texts: list[str], model: OrdinalScorer, tokenizer, score_points: list[float], device: str = "cuda"
) -> list[float]:
    results = []
    for text in texts:
        results.append(predict(text, model, tokenizer, score_points, device))
    return results
```

- [ ] **Step 2: Verify infer.py imports cleanly (no GPU)**

```bash
python3 -c "
from src.infer import predict, predict_batch, load_scorer
print('infer.py imports OK')
"
```
Expected: `infer.py imports OK`

- [ ] **Step 3: Commit**

```bash
git add src/infer.py
git commit -m "feat: add inference interface for checkpoint loading and prediction"
```

---

### Task 8: Training shell scripts

**Files:**
- Create: `scripts/train_q8.sh`
- Create: `scripts/train_q9.sh`

**Interfaces:**
- Produces: executable shell scripts wrapping `python src/train.py --config`

- [ ] **Step 1: Write `scripts/train_q8.sh`**

```bash
#!/bin/bash
set -e
echo "Training Q8 (short-answer scoring)..."
python src/train.py --config configs/q8_config.yaml
echo "Q8 training complete."
```

```bash
chmod +x scripts/train_q8.sh
```

- [ ] **Step 2: Write `scripts/train_q9.sh`**

```bash
#!/bin/bash
set -e
echo "Training Q9 (essay scoring)..."
python src/train.py --config configs/q9_config.yaml
echo "Q9 training complete."
```

```bash
chmod +x scripts/train_q9.sh
```

- [ ] **Step 3: Commit**

```bash
git add scripts/train_q8.sh scripts/train_q9.sh
git commit -m "feat: add train_q8.sh and train_q9.sh launcher scripts"
```

---

### Task 9: Module smoke test and integration check

**Files:**
- Create: `scripts/smoke_test.sh`

- [ ] **Step 1: Write `scripts/smoke_test.sh`**

```bash
#!/bin/bash
# Smoke test: verify all modules import and data pipeline works end-to-end locally
set -e
echo "=== Smoke Test (No GPU) ==="

echo "[1/5] Config loading..."
python3 -c "from src.config import load_config; cfg=load_config('configs/q8_config.yaml'); print('OK:', cfg.question_id)"

echo "[2/5] Data pipeline Q8..."
python3 -c "from src.data_pipeline import build_dataloaders; from src.config import load_config; cfg=load_config('configs/q8_config.yaml'); t, v, sp = build_dataloaders(cfg); print(f'OK: train={len(t)}, test={len(v)}, scores={len(sp)}')"

echo "[3/5] Data pipeline Q9..."
python3 -c "from src.data_pipeline import build_dataloaders; from src.config import load_config; cfg=load_config('configs/q9_config.yaml'); t, v, sp = build_dataloaders(cfg); print(f'OK: train={len(t)}, test={len(v)}, scores={len(sp)}')"

echo "[4/5] Model + loss shapes..."
python3 -c "
import torch
from src.model import CoralHead, coral_loss, prediction_to_score, DummyBackbone, OrdinalScorer
head = CoralHead(2560, 9)
out = head(torch.randn(4, 2560))
loss = coral_loss(out, torch.tensor([0,3,5,8]), 9)
probs = torch.sigmoid(out)
scores = prediction_to_score(probs, [0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0])
backbone = DummyBackbone(2560)
scorer = OrdinalScorer(backbone, 2560, 9)
out2 = scorer(torch.randint(0,1000,(4,128)))
print(f'OK: loss={loss.item():.4f}, scores={scores}, head_out={out.shape}, scorer_out={out2.shape}')
"

echo "[5/5] All imports..."
python3 -c "from src.train import setup_model_and_tokenizer, validate, train; from src.infer import predict, predict_batch, load_scorer; from src.evaluate import compute_metrics; print('OK: all modules importable')"

echo "=== All smoke tests passed! ==="
```

```bash
chmod +x scripts/smoke_test.sh
```

- [ ] **Step 2: Run smoke test**

```bash
bash scripts/smoke_test.sh
```
Expected: all 5 checks pass.

- [ ] **Step 3: Commit**

```bash
git add scripts/smoke_test.sh
git commit -m "test: add smoke test script for no-GPU validation"
```

---

### Task 10: README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write `README.md`**

```markdown
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
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README with project overview and usage"
```
