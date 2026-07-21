#!/bin/bash
# Smoke test: verify all modules import and data pipeline works end-to-end locally
set -e
echo "=== Smoke Test (No GPU) ==="

PYTHON=/usr/local/bin/python3.11
echo "[1/5] Config loading..."
$PYTHON -c "from src.config import load_config; cfg=load_config('configs/q8_config.yaml'); print('OK:', cfg.question_id)"

echo "[2/5] Data pipeline Q8..."
$PYTHON -c "from src.data_pipeline import build_dataloaders; from src.config import load_config; cfg=load_config('configs/q8_config.yaml'); t, v, sp = build_dataloaders(cfg); print(f'OK: train={len(t)}, test={len(v)}, scores={len(sp)}')"

echo "[3/5] Data pipeline Q9..."
$PYTHON -c "from src.data_pipeline import build_dataloaders; from src.config import load_config; cfg=load_config('configs/q9_config.yaml'); t, v, sp = build_dataloaders(cfg); print(f'OK: train={len(t)}, test={len(v)}, scores={len(sp)}')"

echo "[4/5] Model + loss shapes..."
$PYTHON -c "
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
$PYTHON -c "from src.train import setup_model_and_tokenizer, validate, train; from src.infer import predict, predict_batch, load_scorer; from src.evaluate import compute_metrics; print('OK: all modules importable')"

echo "=== All smoke tests passed! ==="
