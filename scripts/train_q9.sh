#!/bin/bash
set -e
echo "Training Q9 (essay scoring)..."
python src/train.py --config configs/q9_config.yaml
echo "Q9 training complete."
