#!/bin/bash

# Script to run warmup (default) + experiments for Grok 4.1 and Nebulon
# With persona ID 1 for both woman_implicits and man_implicits

set -e

WORKSPACE_DIR="/mnt/c/Users/AbelApriliani/Tugas Akhir/TugasAkhir"
cd "$WORKSPACE_DIR"

echo "=========================================="
echo "Warmup + GSM8K Experiments"
echo "Persona ID: 1 (all models)"
echo "=========================================="

# ============================================
# BERT NEBULON - WOMAN IMPLICITS
# ============================================
echo ""
echo ">>> BERT NEBULON - WOMAN IMPLICITS <<<"
echo ""

MODEL_NAME="openrouter/bert-nebulon-alpha"
RESULTS_DIR="results/bert_nebulon/woman_implicits"
LOGS_DIR="$RESULTS_DIR/logs"

echo "1. Running warmup (1 sample) for Nebulon (woman)..."
python3 src/run_gsm8k_batch.py \
  --input data/gsm8k_test.json \
  --results-dir "$RESULTS_DIR/persona_warmup" \
  --persona-file inputs/implicits_woman_promt.json \
  --persona-id 1 \
  --model-name "$MODEL_NAME" \
  --chunk-size 1 \
  --limit 1 > "$LOGS_DIR/logs_warmup.txt" 2>&1

echo "✓ Warmup completed"
echo "2. Running GSM8K experiment from Q0 for Nebulon (woman)..."

nohup python3 src/run_gsm8k_batch.py \
  --input data/gsm8k_test.json \
  --results-dir "$RESULTS_DIR/gsm8k_raw" \
  --persona-file inputs/implicits_woman_promt.json \
  --persona-id 1 \
  --model-name "$MODEL_NAME" \
  --chunk-size 50 \
  --start-index 0 \
  --limit 1400 > "$LOGS_DIR/logs_main_experiment.txt" 2>&1 &

NEBULON_WOMAN_PID=$!
echo "$NEBULON_WOMAN_PID" > .nebulon_woman_implicits_pid
echo "✓ Nebulon (woman) experiment started (PID: $NEBULON_WOMAN_PID)"

# ============================================
# BERT NEBULON - MAN IMPLICITS
# ============================================
echo ""
echo ">>> BERT NEBULON - MAN IMPLICITS <<<"
echo ""

RESULTS_DIR="results/bert_nebulon/man_implicits"
LOGS_DIR="$RESULTS_DIR/logs"

echo "1. Running warmup (1 sample) for Nebulon (man)..."
python3 src/run_gsm8k_batch.py \
  --input data/gsm8k_test.json \
  --results-dir "$RESULTS_DIR/persona_warmup" \
  --persona-file inputs/implicits_man_prompts.json \
  --persona-id 1 \
  --model-name "$MODEL_NAME" \
  --chunk-size 1 \
  --limit 1 > "$LOGS_DIR/logs_warmup.txt" 2>&1

echo "✓ Warmup completed"
echo "2. Running GSM8K experiment from Q0 for Nebulon (man)..."

nohup python3 src/run_gsm8k_batch.py \
  --input data/gsm8k_test.json \
  --results-dir "$RESULTS_DIR/gsm8k_raw" \
  --persona-file inputs/implicits_man_prompts.json \
  --persona-id 1 \
  --model-name "$MODEL_NAME" \
  --chunk-size 50 \
  --start-index 0 \
  --limit 1400 > "$LOGS_DIR/logs_main_experiment.txt" 2>&1 &

NEBULON_MAN_PID=$!
echo "$NEBULON_MAN_PID" > .nebulon_man_implicits_pid
echo "✓ Nebulon (man) experiment started (PID: $NEBULON_MAN_PID)"

# ============================================
# GROK 4.1 - WOMAN IMPLICITS
# ============================================
echo ""
echo ">>> GROK 4.1 - WOMAN IMPLICITS <<<"
echo ""

MODEL_NAME="x-ai/grok-4.1-fast"
RESULTS_DIR="results/grok_4_1/woman_implicits"
LOGS_DIR="$RESULTS_DIR/logs"

echo "1. Running warmup (1 sample) for Grok (woman)..."
python3 src/run_gsm8k_batch.py \
  --input data/gsm8k_test.json \
  --results-dir "$RESULTS_DIR/persona_warmup" \
  --persona-file inputs/implicits_woman_promt.json \
  --persona-id 1 \
  --model-name "$MODEL_NAME" \
  --chunk-size 1 \
  --limit 1 > "$LOGS_DIR/logs_warmup.txt" 2>&1

echo "✓ Warmup completed"
echo "2. Running GSM8K experiment from Q0 for Grok (woman)..."

nohup python3 src/run_gsm8k_batch.py \
  --input data/gsm8k_test.json \
  --results-dir "$RESULTS_DIR/gsm8k_raw" \
  --persona-file inputs/implicits_woman_promt.json \
  --persona-id 1 \
  --model-name "$MODEL_NAME" \
  --chunk-size 50 \
  --start-index 0 \
  --limit 1400 > "$LOGS_DIR/logs_main_experiment.txt" 2>&1 &

GROK_WOMAN_PID=$!
echo "$GROK_WOMAN_PID" > .grok_woman_implicits_pid
echo "✓ Grok (woman) experiment started (PID: $GROK_WOMAN_PID)"

# ============================================
# GROK 4.1 - MAN IMPLICITS
# ============================================
echo ""
echo ">>> GROK 4.1 - MAN IMPLICITS <<<"
echo ""

RESULTS_DIR="results/grok_4_1/man_implicits"
LOGS_DIR="$RESULTS_DIR/logs"

echo "1. Running warmup (1 sample) for Grok (man)..."
python3 src/run_gsm8k_batch.py \
  --input data/gsm8k_test.json \
  --results-dir "$RESULTS_DIR/persona_warmup" \
  --persona-file inputs/implicits_man_prompts.json \
  --persona-id 1 \
  --model-name "$MODEL_NAME" \
  --chunk-size 1 \
  --limit 1 > "$LOGS_DIR/logs_warmup.txt" 2>&1

echo "✓ Warmup completed"
echo "2. Running GSM8K experiment from Q0 for Grok (man)..."

nohup python3 src/run_gsm8k_batch.py \
  --input data/gsm8k_test.json \
  --results-dir "$RESULTS_DIR/gsm8k_raw" \
  --persona-file inputs/implicits_man_prompts.json \
  --persona-id 1 \
  --model-name "$MODEL_NAME" \
  --chunk-size 50 \
  --start-index 0 \
  --limit 1400 > "$LOGS_DIR/logs_main_experiment.txt" 2>&1 &

GROK_MAN_PID=$!
echo "$GROK_MAN_PID" > .grok_man_implicits_pid
echo "✓ Grok (man) experiment started (PID: $GROK_MAN_PID)"

echo ""
echo "=========================================="
echo "All experiments started!"
echo "Nebulon (woman) PID: $NEBULON_WOMAN_PID"
echo "Nebulon (man) PID: $NEBULON_MAN_PID"
echo "Grok (woman) PID: $GROK_WOMAN_PID"
echo "Grok (man) PID: $GROK_MAN_PID"
echo "=========================================="
