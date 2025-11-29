#!/bin/bash
set -e

# Woman Implicit Qwen
nohup bash -c '
set -e
echo "=== Woman Implicit Qwen: Starting ==="
for start in 0 100 200 300 400 500 600 700 800 900 1000 1100 1200; do
  python3 src/run_gsm8k_batch.py \
    --input data/gsm8k_test.json \
    --results-dir results/woman_implicits/qwen/gsm8k_raw \
    --chunk-size 100 \
    --start-index $start \
    --limit 100 \
    --persona-file inputs/implicits_woman_promt.json \
    --persona-id 1 \
    --model-name "Qwen: Qwen3 4B (free)" \
    $([[ $start -eq 0 ]] && echo "" || echo "--no-warmup") || exit 1
done
python3 src/run_gsm8k_batch.py \
  --input data/gsm8k_test.json \
  --results-dir results/woman_implicits/qwen/gsm8k_raw \
  --chunk-size 100 \
  --start-index 1300 \
  --limit 19 \
  --persona-file inputs/implicits_woman_promt.json \
  --persona-id 1 \
  --model-name "Qwen: Qwen3 4B (free)" \
  --no-warmup || exit 1
python3 src/parse_gsm8k_results.py \
  --raw-dir results/woman_implicits/qwen/gsm8k_raw \
  --out-csv results/woman_implicits/qwen/gsm8k_results.csv \
  --out-xlsx results/woman_implicits/qwen/gsm8k_results.xlsx
echo "=== Woman Implicit Qwen: Done ==="
' > logs_woman_implicit_qwen.txt 2>&1 &
echo "Woman Implicit Qwen launched (PID: $!)"

# Man Implicit Qwen
nohup bash -c '
set -e
echo "=== Man Implicit Qwen: Starting ==="
for start in 0 100 200 300 400 500 600 700 800 900 1000 1100 1200; do
  python3 src/run_gsm8k_batch.py \
    --input data/gsm8k_test.json \
    --results-dir results/man_implicits/qwen/gsm8k_raw \
    --chunk-size 100 \
    --start-index $start \
    --limit 100 \
    --persona-file inputs/implicits_man_prompts.json \
    --persona-id 1 \
    --model-name "Qwen: Qwen3 4B (free)" \
    $([[ $start -eq 0 ]] && echo "" || echo "--no-warmup") || exit 1
done
python3 src/run_gsm8k_batch.py \
  --input data/gsm8k_test.json \
  --results-dir results/man_implicits/qwen/gsm8k_raw \
  --chunk-size 100 \
  --start-index 1300 \
  --limit 19 \
  --persona-file inputs/implicits_man_prompts.json \
  --persona-id 1 \
  --model-name "Qwen: Qwen3 4B (free)" \
  --no-warmup || exit 1
python3 src/parse_gsm8k_results.py \
  --raw-dir results/man_implicits/qwen/gsm8k_raw \
  --out-csv results/man_implicits/qwen/gsm8k_results.csv \
  --out-xlsx results/man_implicits/qwen/gsm8k_results.xlsx
echo "=== Man Implicit Qwen: Done ==="
' > logs_man_implicit_qwen.txt 2>&1 &
echo "Man Implicit Qwen launched (PID: $!)"

# Woman Explicit Qwen
nohup bash -c '
set -e
echo "=== Woman Explicit Qwen: Starting ==="
for start in 0 100 200 300 400 500 600 700 800 900 1000 1100 1200; do
  python3 src/run_gsm8k_batch.py \
    --input data/gsm8k_test.json \
    --results-dir results/woman_explicits/qwen/gsm8k_raw \
    --chunk-size 100 \
    --start-index $start \
    --limit 100 \
    --persona-file inputs/explicits_woman_promts.json \
    --persona-id 1 \
    --model-name "Qwen: Qwen3 4B (free)" \
    $([[ $start -eq 0 ]] && echo "" || echo "--no-warmup") || exit 1
done
python3 src/run_gsm8k_batch.py \
  --input data/gsm8k_test.json \
  --results-dir results/woman_explicits/qwen/gsm8k_raw \
  --chunk-size 100 \
  --start-index 1300 \
  --limit 19 \
  --persona-file inputs/explicits_woman_promts.json \
  --persona-id 1 \
  --model-name "Qwen: Qwen3 4B (free)" \
  --no-warmup || exit 1
python3 src/parse_gsm8k_results.py \
  --raw-dir results/woman_explicits/qwen/gsm8k_raw \
  --out-csv results/woman_explicits/qwen/gsm8k_results.csv \
  --out-xlsx results/woman_explicits/qwen/gsm8k_results.xlsx
echo "=== Woman Explicit Qwen: Done ==="
' > logs_woman_explicit_qwen.txt 2>&1 &
echo "Woman Explicit Qwen launched (PID: $!)"

# Man Explicit Qwen
nohup bash -c '
set -e
echo "=== Man Explicit Qwen: Starting ==="
for start in 0 100 200 300 400 500 600 700 800 900 1000 1100 1200; do
  python3 src/run_gsm8k_batch.py \
    --input data/gsm8k_test.json \
    --results-dir results/man_explicits/qwen/gsm8k_raw \
    --chunk-size 100 \
    --start-index $start \
    --limit 100 \
    --persona-file inputs/explicits_man_prompts.json \
    --persona-id 1 \
    --model-name "Qwen: Qwen3 4B (free)" \
    $([[ $start -eq 0 ]] && echo "" || echo "--no-warmup") || exit 1
done
python3 src/run_gsm8k_batch.py \
  --input data/gsm8k_test.json \
  --results-dir results/man_explicits/qwen/gsm8k_raw \
  --chunk-size 100 \
  --start-index 1300 \
  --limit 19 \
  --persona-file inputs/explicits_man_prompts.json \
  --persona-id 1 \
  --model-name "Qwen: Qwen3 4B (free)" \
  --no-warmup || exit 1
python3 src/parse_gsm8k_results.py \
  --raw-dir results/man_explicits/qwen/gsm8k_raw \
  --out-csv results/man_explicits/qwen/gsm8k_results.csv \
  --out-xlsx results/man_explicits/qwen/gsm8k_results.xlsx
echo "=== Man Explicit Qwen: Done ==="
' > logs_man_explicit_qwen.txt 2>&1 &
echo "Man Explicit Qwen launched (PID: $!)"

echo ""
echo "All 4 Qwen runs launched in background"
echo "Logs: logs_woman_implicit_qwen.txt, logs_man_implicit_qwen.txt, logs_woman_explicit_qwen.txt, logs_man_explicit_qwen.txt"
