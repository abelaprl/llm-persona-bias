#!/bin/bash
# Run bert-nebulon-alpha for woman_implicits
# This runs in parallel with man_implicits

echo "=========================================="
echo "BERT-NEBULON-ALPHA - WOMAN_IMPLICITS"
echo "=========================================="
echo "Model: openrouter/bert-nebulon-alpha"
echo "Dataset: GSM8K (~1300 questions)"
echo "Persona: woman_implicits"
echo "Started: $(date)"
echo "=========================================="
echo ""

python3 src/run_gsm8k_batch.py \
  --input data/gsm8k_test.json \
  --results-dir results/bert_nebulon_alpha/woman_implicits/gsm8k_raw \
  --chunk-size 100 \
  --start-index 300 \
  --limit 1300 \
  --persona-file inputs/implicits_woman_promt.json \
  --persona-id 1 \
  --model-name "openrouter/bert-nebulon-alpha" \
  --no-warmup

echo ""
echo "=========================================="
echo "BERT-NEBULON-ALPHA - WOMAN_IMPLICITS COMPLETED"
echo "Finished: $(date)"
echo "=========================================="
