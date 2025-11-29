#!/bin/bash
# Nemotron man_implicits continuation

echo "🚀 Starting Nemotron (nvidia/nemotron-nano-12b-v2-vl:free) - man_implicits"
python3 src/run_gsm8k_batch.py \
    --input data/gsm8k_test.json \
    --results-dir results/man_implicits/nemotron/gsm8k_raw \
    --chunk-size 100 \
    --start-index 0 \
    --limit 1400 \
    --persona-file inputs/implicits_man_prompts.json \
    --persona-id 0 \
    --model-name nvidia/nemotron-nano-12b-v2-vl:free \
    --no-warmup
