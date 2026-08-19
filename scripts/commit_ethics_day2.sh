#!/usr/bin/env bash
# HARI 2 — Regional Cultural Persona: openai_gpt-oss-20b (near-complete)
# Jalankan dari root repo:  bash scripts/commit_ethics_day2.sh
set -e
cd "$(git rev-parse --show-toplevel)"

P="result/ETHICS/v9_pilot/Regional Cultural Persona/openai_gpt-oss-20b"
git add -- "$P"

echo "staged: $(git diff --cached --name-only -- "$P" | wc -l) file"
echo "CSV keikut: $(git diff --cached --name-only -- "$P" | grep -c '\.csv$')"
echo "deletion keikut: $(git diff --cached --name-status -- "$P" | grep -c '^D')"

git commit -q -m "data: v9_pilot Regional Cultural Persona ETHICS -- openai gpt-oss-20b

Adds ~22,011 per-question JSON results for the Regional Cultural Persona
condition, model openai gpt-oss-20b. Near-complete (~1,400-1,514/persona).

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>" -- "$P"

echo "DONE hari 2 ->"; git log --oneline -1
