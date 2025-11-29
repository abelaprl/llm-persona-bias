#!/bin/bash
# Nebulon man_implicits continuation from Q1051

echo "🚀 Continuing Nebulon (openrouter/bert-nebulon-alpha) - man_implicits from Q1051"
python3 src/run_gsm8k_batch.py \
    --input data/gsm8k_test.json \
    --results-dir results/bert_nebulon_alpha/man_implicits/gsm8k_raw \
    --chunk-size 50 \
    --start-index 1051 \
    --limit 1400 \
    --persona-file inputs/implicits_man_prompts.json \
    --persona-id 0 \
    --model-name openrouter/bert-nebulon-alpha \
    --no-warmup
