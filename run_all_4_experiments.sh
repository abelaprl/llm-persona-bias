#!/bin/bash

# Run all 4 experiments with warmup
# Grok 4.1 + Nebulon on woman_implicits and man_implicits
# Persona ID: 1

cd /mnt/c/Users/AbelApriliani/Tugas\ Akhir/TugasAkhir

echo "Starting all experiments..."

# GROK WOMAN
nohup python3 src/run_gsm8k_batch.py --input data/gsm8k_test.json --results-dir results/grok_4_1/woman_implicits/gsm8k_raw --persona-file inputs/implicits_woman_promt.json --persona-id 1 --model-name x-ai/grok-4.1-fast --chunk-size 50 --start-index 0 --limit 1400 > results/grok_4_1/woman_implicits/logs/run.log 2>&1 &
GROK_W_PID=$!
echo "Grok (woman) started: $GROK_W_PID"

# GROK MAN
nohup python3 src/run_gsm8k_batch.py --input data/gsm8k_test.json --results-dir results/grok_4_1/man_implicits/gsm8k_raw --persona-file inputs/implicits_man_prompts.json --persona-id 1 --model-name x-ai/grok-4.1-fast --chunk-size 50 --start-index 0 --limit 1400 > results/grok_4_1/man_implicits/logs/run.log 2>&1 &
GROK_M_PID=$!
echo "Grok (man) started: $GROK_M_PID"

# NEBULON WOMAN
nohup python3 src/run_gsm8k_batch.py --input data/gsm8k_test.json --results-dir results/bert_nebulon/woman_implicits/gsm8k_raw --persona-file inputs/implicits_woman_promt.json --persona-id 1 --model-name openrouter/bert-nebulon-alpha --chunk-size 50 --start-index 0 --limit 1400 > results/bert_nebulon/woman_implicits/logs/run.log 2>&1 &
NEBULON_W_PID=$!
echo "Nebulon (woman) started: $NEBULON_W_PID"

# NEBULON MAN
nohup python3 src/run_gsm8k_batch.py --input data/gsm8k_test.json --results-dir results/bert_nebulon/man_implicits/gsm8k_raw --persona-file inputs/implicits_man_prompts.json --persona-id 1 --model-name openrouter/bert-nebulon-alpha --chunk-size 50 --start-index 0 --limit 1400 > results/bert_nebulon/man_implicits/logs/run.log 2>&1 &
NEBULON_M_PID=$!
echo "Nebulon (man) started: $NEBULON_M_PID"

echo ""
echo "All 4 experiments started!"
echo "Grok (woman): $GROK_W_PID"
echo "Grok (man): $GROK_M_PID"
echo "Nebulon (woman): $NEBULON_W_PID"
echo "Nebulon (man): $NEBULON_M_PID"

# Save PIDs
echo "$GROK_W_PID" > .pids_grok_w
echo "$GROK_M_PID" > .pids_grok_m
echo "$NEBULON_W_PID" > .pids_nebulon_w
echo "$NEBULON_M_PID" > .pids_nebulon_m
