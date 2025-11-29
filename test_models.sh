#!/bin/bash
# Test models dengan 20 requests untuk cek rate limit

models=(
  "polaris-alpha"
  "DeepSeek V3.1 (free)"
  "OpenAI: gpt-oss-20b (free)"
  "LongCat Flash Chat (free)"
  "Google: Gemma 3n 2B (free)"
  "Google: Gemini 2.0 Flash (free)"
  "alibaba/tongyi-deepresearch-30b-a3b:free"
  "mistralai/mistral-small-3.1-24b-instruct:free"
)

for model in "${models[@]}"; do
  echo "=== Testing: $model ==="
  python3 src/run_gsm8k_batch.py \
    --input data/gsm8k_test.json \
    --results-dir results/test_model_tmp \
    --chunk-size 100 \
    --start-index 0 \
    --limit 20 \
    --persona-file inputs/implicits_woman_promt.json \
    --persona-id 1 \
    --model-name "$model" \
    --no-warmup 2>&1 | grep -E "successes=|failures=|429|404|error" | tail -5
  rm -rf results/test_model_tmp
  echo ""
  sleep 2
done
