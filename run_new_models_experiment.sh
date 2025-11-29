#!/bin/bash
# Experiment script for new models: Grok 4.1 Fast and bert-nebulon-alpha
# GSM8K with ~1300 questions, man_implicits and woman_implicits

echo "=========================================="
echo "NEW MODELS EXPERIMENT - GSM8K"
echo "=========================================="
echo "Models: Grok 4.1 Fast, bert-nebulon-alpha"
echo "Dataset: GSM8K (~1300 questions)"
echo "Personas: man_implicits, woman_implicits"
echo "Started: $(date)"
echo "=========================================="
echo ""

# Configuration
CHUNK_SIZE=100
START_IDX=300  # Start from index 300 for ~1300 samples
LIMIT=1300

# Model 1: Grok 4.1 Fast
echo "--- MODEL 1: Grok 4.1 Fast ---"
echo ""

echo "Running man_implicits..."
python3 src/run_gsm8k_batch.py \
  --input data/gsm8k_test.json \
  --results-dir results/grok_4_1_fast/man_implicits/gsm8k_raw \
  --chunk-size $CHUNK_SIZE \
  --start-index $START_IDX \
  --limit $LIMIT \
  --persona-file inputs/implicits_man_prompts.json \
  --persona-id 1 \
  --model-name "x-ai: Grok 4.1 Fast" \
  --no-warmup

echo ""
echo "Running woman_implicits..."
python3 src/run_gsm8k_batch.py \
  --input data/gsm8k_test.json \
  --results-dir results/grok_4_1_fast/woman_implicits/gsm8k_raw \
  --chunk-size $CHUNK_SIZE \
  --start-index $START_IDX \
  --limit $LIMIT \
  --persona-file inputs/implicits_woman_promt.json \
  --persona-id 1 \
  --model-name "x-ai: Grok 4.1 Fast" \
  --no-warmup

echo ""
echo "--- MODEL 2: bert-nebulon-alpha ---"
echo ""

echo "Running man_implicits..."
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
echo "Running woman_implicits..."
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
echo "EXPERIMENT COMPLETED"
echo "Finished: $(date)"
echo "=========================================="
