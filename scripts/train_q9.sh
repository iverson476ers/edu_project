#!/bin/bash
set -e
export PYTHONPATH=$PYTHONPATH:.
cd "$(dirname "$0")/.."
echo "Training Q9 (essay scoring)..."
python src/train.py --config configs/q9_config.yaml
echo "Q9 training complete."
