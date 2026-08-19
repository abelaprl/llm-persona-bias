#!/usr/bin/env bash
# HARI 3 — Regional Cultural Persona: qwen3-next-80b + mistral ministral-8b (partial)
# Jalankan dari root repo:  bash scripts/commit_ethics_day3.sh
set -e
cd "$(git rev-parse --show-toplevel)"

Q="result/ETHICS/v9_pilot/Regional Cultural Persona/qwen_qwen3-next-80b-a3b-thinking"
M="result/ETHICS/v9_pilot/Regional Cultural Persona/mistralai_ministral-8b-2512"

# --- commit 1: qwen ---
git add -- "$Q"
echo "qwen staged: $(git diff --cached --name-only -- "$Q" | wc -l) (CSV: $(git diff --cached --name-only -- "$Q" | grep -c '\.csv$'))"
git commit -q -m "data: v9_pilot Regional Cultural Persona ETHICS -- qwen3-next-80b (partial)

Adds ~13,726 per-question JSON results for the Regional Cultural Persona
condition, model qwen3-next-80b. Partial run (~800-900/persona of 1,514).

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>" -- "$Q"
echo "  committed qwen -> $(git log --oneline -1)"

# --- commit 2: mistral ---
git add -- "$M"
echo "mistral staged: $(git diff --cached --name-only -- "$M" | wc -l) (CSV: $(git diff --cached --name-only -- "$M" | grep -c '\.csv$'))"
git commit -q -m "data: v9_pilot Regional Cultural Persona ETHICS -- mistral ministral-8b (partial, south_asian missing)

Adds ~8,300 per-question JSON results for the Regional Cultural Persona
condition, model mistral ministral-8b. Partial run; south_asian personas
(explicit + implicit_1/2/3) are entirely missing and NOT included.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>" -- "$M"
echo "  committed mistral -> $(git log --oneline -1)"

echo "DONE hari 3 (2 commit)."
