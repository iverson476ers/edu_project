#!/bin/bash
set -e
cd "$(dirname "$0")/.."
echo "Training Q8 (short-answer scoring)..."
python3.11 src/train.py --config configs/q8_config.yaml
echo "Q8 training complete."
