#!/bin/bash

# Script to run warmup + experiment for Grok 4.1 and Nebulon on woman_implicits
# Starting from question ID 0

set -e

WORKSPACE_DIR="/mnt/c/Users/AbelApriliani/Tugas Akhir/TugasAkhir"
cd "$WORKSPACE_DIR"

echo "=========================================="
echo "Woman Implicits Warmup + Experiment"
echo "=========================================="

# ============================================
# GROK 4.1 FAST
# ============================================
echo ""
echo ">>> GROK 4.1 FAST - WOMAN IMPLICITS <<<"
echo ""

MODEL_NAME="x-ai/grok-4.1-fast"
RESULTS_DIR="results/grok_4_1_fast/woman_implicits"
LOGS_DIR="$RESULTS_DIR/logs"
mkdir -p "$LOGS_DIR"

# Persona echo
echo "1. Running persona echo for Grok..."
python3 src/run_gsm8k_batch.py \
  --input data/gsm8k_test.json \
  --results-dir "$RESULTS_DIR/persona_echo" \
  --persona-file inputs/implicits_woman_promt.json \
  --persona-id 0 \
  --model-name "$MODEL_NAME" \
  --chunk-size 1 \
  --limit 1 \
  --persona-echo > "$LOGS_DIR/logs_persona_echo.txt" 2>&1

echo "✓ Persona echo completed"

# Warmup
echo "2. Running warmup for Grok..."
python3 src/run_gsm8k_batch.py \
  --input data/gsm8k_test.json \
  --results-dir "$RESULTS_DIR/persona_warmup" \
  --persona-file inputs/implicits_woman_promt.json \
  --persona-id 0 \
  --model-name "$MODEL_NAME" \
  --chunk-size 1 \
  --limit 5 \
  --warmup > "$LOGS_DIR/logs_warmup.txt" 2>&1

echo "✓ Warmup completed"

# Main experiment from Q0
echo "3. Running main experiment from Q0..."
nohup python3 src/run_gsm8k_batch.py \
  --input data/gsm8k_test.json \
  --results-dir "$RESULTS_DIR/gsm8k_raw" \
  --persona-file inputs/implicits_woman_promt.json \
  --persona-id 0 \
  --model-name "$MODEL_NAME" \
  --chunk-size 50 \
  --start-index 0 \
  --limit 1400 > "$LOGS_DIR/logs_main_experiment.txt" 2>&1 &

GROK_PID=$!
echo "$GROK_PID" > .grok_woman_implicits_pid
echo "✓ Grok experiment started (PID: $GROK_PID)"

# ============================================
# NEBULON ALPHA
# ============================================
echo ""
echo ">>> BERT NEBULON ALPHA - WOMAN IMPLICITS <<<"
echo ""

MODEL_NAME="openrouter/bert-nebulon-alpha"
RESULTS_DIR="results/bert_nebulon_alpha/woman_implicits"
LOGS_DIR="$RESULTS_DIR/logs"
mkdir -p "$LOGS_DIR"

# Persona echo
echo "1. Running persona echo for Nebulon..."
python3 src/run_gsm8k_batch.py \
  --input data/gsm8k_test.json \
  --results-dir "$RESULTS_DIR/persona_echo" \
  --persona-file inputs/implicits_woman_promt.json \
  --persona-id 0 \
  --model-name "$MODEL_NAME" \
  --chunk-size 1 \
  --limit 1 \
  --persona-echo > "$LOGS_DIR/logs_persona_echo.txt" 2>&1

echo "✓ Persona echo completed"

# Warmup
echo "2. Running warmup for Nebulon..."
python3 src/run_gsm8k_batch.py \
  --input data/gsm8k_test.json \
  --results-dir "$RESULTS_DIR/persona_warmup" \
  --persona-file inputs/implicits_woman_promt.json \
  --persona-id 0 \
  --model-name "$MODEL_NAME" \
  --chunk-size 1 \
  --limit 5 \
  --warmup > "$LOGS_DIR/logs_warmup.txt" 2>&1

echo "✓ Warmup completed"

# Main experiment from Q0
echo "3. Running main experiment from Q0..."
nohup python3 src/run_gsm8k_batch.py \
  --input data/gsm8k_test.json \
  --results-dir "$RESULTS_DIR/gsm8k_raw" \
  --persona-file inputs/implicits_woman_promt.json \
  --persona-id 0 \
  --model-name "$MODEL_NAME" \
  --chunk-size 50 \
  --start-index 0 \
  --limit 1400 > "$LOGS_DIR/logs_main_experiment.txt" 2>&1 &

NEBULON_PID=$!
echo "$NEBULON_PID" > .nebulon_woman_implicits_pid
echo "✓ Nebulon experiment started (PID: $NEBULON_PID)"

echo ""
echo "=========================================="
echo "Both experiments started!"
echo "Grok PID: $GROK_PID"
echo "Nebulon PID: $NEBULON_PID"
echo "=========================================="
