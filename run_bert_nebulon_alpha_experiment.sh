#!/bin/bash
# Experiment script for bert-nebulon-alpha model
# GSM8K with ~1300 questions, man_implicits and woman_implicits

echo "=========================================="
echo "BERT-NEBULON-ALPHA EXPERIMENT - GSM8K"
echo "=========================================="
echo "Model: openrouter/bert-nebulon-alpha"
echo "Dataset: GSM8K (~1300 questions)"
echo "Personas: man_implicits, woman_implicits"
echo "Started: $(date)"
echo "=========================================="
echo ""

# Configuration
CHUNK_SIZE=100
START_IDX=300
LIMIT=1300

echo "--- Running man_implicits ---"
echo ""

python3 src/run_gsm8k_batch.py \
  --input data/gsm8k_test.json \
  --results-dir results/bert_nebulon_alpha/man_implicits/gsm8k_raw \
  --chunk-size $CHUNK_SIZE \
  --start-index $START_IDX \
  --limit $LIMIT \
  --persona-file inputs/implicits_man_prompts.json \
  --persona-id 1 \
  --model-name "openrouter/bert-nebulon-alpha" \
  --no-warmup

echo ""
echo "--- Running woman_implicits ---"
echo ""

python3 src/run_gsm8k_batch.py \
  --input data/gsm8k_test.json \
  --results-dir results/bert_nebulon_alpha/woman_implicits/gsm8k_raw \
  --chunk-size $CHUNK_SIZE \
  --start-index $START_IDX \
  --limit $LIMIT \
  --persona-file inputs/implicits_woman_promt.json \
  --persona-id 1 \
  --model-name "openrouter/bert-nebulon-alpha" \
  --no-warmup

echo ""
echo "=========================================="
echo "BERT-NEBULON-ALPHA EXPERIMENT COMPLETED"
echo "Finished: $(date)"
echo "=========================================="
