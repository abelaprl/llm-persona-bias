#!/bin/bash
# Continue Nemotron GSM8K from index 953

echo "=========================================="
echo "NEMOTRON CONTINUATION - GSM8K"
echo "=========================================="
echo "Model: Nvidia - Free (Nemotron)"
echo "Dataset: GSM8K from index 953"
echo "Persona: man_implicits"
echo "Started: $(date)"
echo "=========================================="
echo ""

python3 src/run_gsm8k_batch.py \
  --input data/gsm8k_test.json \
  --results-dir results/man_implicits/gsm8k_raw \
  --chunk-size 100 \
  --start-index 953 \
  --limit 100 \
  --persona-file inputs/implicits_man_prompts.json \
  --persona-id 1 \
  --model-name "Nvidia - Free" \
  --no-warmup

echo ""
echo "=========================================="
echo "NEMOTRON CONTINUATION COMPLETED"
echo "Finished: $(date)"
echo "=========================================="
