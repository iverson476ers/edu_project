#!/bin/bash
set -e
cd "$(dirname "$0")/.."
echo "Training Q9 (essay scoring)..."
python3.11 src/train.py --config configs/q9_config.yaml
echo "Q9 training complete."
