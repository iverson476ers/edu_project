#!/bin/bash

set -e
export PYTHONPATH=$PYTHONPATH:.
cd "$(dirname "$0")/.."
echo "Training Q8 (short-answer scoring)... $PWD"
python src/train.py --config configs/q8_config.yaml
echo "Q8 training complete."
