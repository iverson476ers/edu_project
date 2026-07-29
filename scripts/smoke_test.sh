#!/bin/bash
# Smoke test: verify all modules import and data pipeline works end-to-end locally
set -e
PY="python3.11"
echo "=== Smoke Test (No GPU) ==="

echo "[1/5] Config loading..."
$PY -c "from src.config import load_config; cfg=load_config('configs/q8_config.yaml'); print('OK:', cfg.question_id)"

echo "[2/5] Data pipeline Q8..."
$PY -c "from src.data_pipeline import load_and_split_data; from src.config import load_config; cfg=load_config('configs/q8_config.yaml'); t, v, sp, qc = load_and_split_data(cfg); print(f'OK: train={len(t)}, test={len(v)}, scores={len(sp)}, full_score={qc[\"full_score\"]}')"

echo "[3/5] Data pipeline Q9..."
$PY -c "from src.data_pipeline import load_and_split_data; from src.config import load_config; cfg=load_config('configs/q9_config.yaml'); t, v, sp, qc = load_and_split_data(cfg); print(f'OK: train={len(t)}, test={len(v)}, scores={len(sp)}, full_score={qc[\"full_score\"]}')"

echo "[4/5] Model + Pooling + Head shapes..."
$PY -c "
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

echo "[5/5] All imports..."
$PY -c "from src.train import setup_model_and_tokenizer, validate, train; from src.infer import predict, predict_batch, load_scorer; from src.evaluate import compute_metrics; print('OK: all modules importable')"

echo "=== All smoke tests passed! ==="
