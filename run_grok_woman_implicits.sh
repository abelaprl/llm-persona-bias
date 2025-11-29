#!/bin/bash
# Run Grok 4.1 Fast for woman_implicits
# This runs in parallel with man_implicits

echo "=========================================="
echo "GROK 4.1 FAST - WOMAN_IMPLICITS"
echo "=========================================="
echo "Model: x-ai: Grok 4.1 Fast"
echo "Dataset: GSM8K (~1300 questions)"
echo "Persona: woman_implicits"
echo "Started: $(date)"
echo "=========================================="
echo ""

python3 src/run_gsm8k_batch.py \
  --input data/gsm8k_test.json \
  --results-dir results/grok_4_1_fast/woman_implicits/gsm8k_raw \
  --chunk-size 100 \
  --start-index 300 \
  --limit 1300 \
  --persona-file inputs/implicits_woman_promt.json \
  --persona-id 1 \
  --model-name "x-ai: Grok 4.1 Fast" \
  --no-warmup

echo ""
echo "=========================================="
echo "GROK 4.1 FAST - WOMAN_IMPLICITS COMPLETED"
echo "Finished: $(date)"
echo "=========================================="
