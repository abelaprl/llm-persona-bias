#!/bin/bash
cd /mnt/c/Users/AbelApriliani/Tugas\ Akhir/TugasAkhir

echo "🚀 Running Nebulon Q0-Q300 (man_implicits)..."
python3 src/run_gsm8k_batch.py \
    --input data/gsm8k_test.json \
    --results-dir results/bert_nebulon_alpha/man_implicits/gsm8k_raw \
    --chunk-size 50 \
    --start-index 0 \
    --limit 300 \
    --persona-file inputs/implicits_man_prompts.json \
    --persona-id 0 \
    --model-name openrouter/bert-nebulon-alpha \
    --no-warmup

echo "✅ Nebulon Q0-Q300 completed!"
