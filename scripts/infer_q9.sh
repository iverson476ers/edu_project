#!/bin/bash
set -e
cd "$(dirname "$0")/.."

CHECKPOINT="${1:-output/q9/best_model.pt}"
INPUT="${2:-data/101/answer/answer101_9.xlsx}"
OUTPUT="${3:-output/q9/result.xlsx}"

echo "Q9 Inference"
echo "  Checkpoint: $CHECKPOINT"
echo "  Input:      $INPUT"
echo "  Output:     $OUTPUT"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m src.infer \
  --checkpoint "$CHECKPOINT" \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --batch_size 32

echo "Done."
