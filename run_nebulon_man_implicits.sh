#!/bin/bash
# Nebulon man_implicits continuation

echo "🚀 Starting Nebulon (openrouter/bert-nebulon-alpha) - man_implicits"
python3 src/run_gsm8k_batch.py \
    --input data/gsm8k_test.json \
    --results-dir results/bert_nebulon_alpha/man_implicits/gsm8k_raw \
    --chunk-size 100 \
    --start-index 300 \
    --limit 800 \
    --persona-file inputs/implicits_man_prompts.json \
    --persona-id 0 \
    --model-name openrouter/bert-nebulon-alpha \
    --no-warmup
