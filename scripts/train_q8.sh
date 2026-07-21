#!/bin/bash
set -e
echo "Training Q8 (short-answer scoring)..."
python src/train.py --config configs/q8_config.yaml
echo "Q8 training complete."
