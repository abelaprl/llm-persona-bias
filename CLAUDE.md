# TugasAkhir — LLM Persona Bias Research Project

## Project Overview

Bachelor's thesis research evaluating **multi-model, multi-persona bias in Large Language Models**. The study measures whether different user identity signals — presented either as explicit labels or implicit narrative contexts — cause LLMs to produce systematically different outputs on established reasoning and ethics benchmarks.

**Core research question**: Do LLMs respond differently when the same query is preceded by a woman's narrative, a man's narrative, a religious identity signal, or no persona at all? And does it matter whether the persona is *stated explicitly* ("I am a Muslim") vs. *implied implicitly* through a story?

---

## Repository Layout

```
D:\TugasAkhir\
├── config/                     # API keys and model registry
│   └── model_keys.json         # OpenRouter/OpenAI/Anthropic keys + model order
│
├── data/                       # Raw benchmark datasets (77.8 MB)
│   ├── eli5/                  # Dataset dan sampel ELI5
│   ├── ethics/                # Seluruh subset dan gabungan ETHICS
│   ├── gsm8k/                 # GSM8K train dan test
│   ├── mmlu_redux/            # MMLU Redux
│   ├── squad/                 # SQuAD v1/v2, sampel, dan skrip sampling
│   └── truthfulqa/            # TruthfulQA dan sampelnya
│
├── inputs/                     # Persona definitions (JSON)
│   ├── religion_persona.json       # Islam/Christian/Atheist/Jewish × explicit + 3 implicit variants
│   ├── age_group_persona.json      # Age-based personas
│   ├── implicits_woman_prompts.json  # 3 implicit woman narrative variants
│   ├── implicits_man_prompts.json    # 3 implicit man narrative variants
│   ├── explicits_woman_prompts.json  # 1 explicit woman label
│   └── explicits_man_prompts.json    # 1 explicit man label
│
├── docs/
│   └── instruction.md          # Spec-driven methodology for Claude Code
│
├── src/                        # All source code
│   ├── runners/                # Generic batch runners (OpenRouter API)
│   ├── ethics_runner/          # ETHICS benchmark runners
│   ├── gsm8k_runner/           # GSM8K runners (gender, age, religion personas)
│   ├── eli5_runner/            # ELI5 runners
│   ├── squad_runner/           # SQuAD v1/v2 runners
│   ├── truthfulqa_runner/      # TruthfulQA runners
│   ├── analysis/               # Accuracy calculation, error analysis, reporting
│   ├── extract/                # Answer extraction from free-form LLM responses
│   ├── parse/                  # Result parsing and checkpoint monitoring
│   ├── rebuild/                # Per-model CSV reconstruction from JSON results
│   ├── reports/                # CSV/Excel aggregation scripts
│   ├── notebook/               # Jupyter notebooks for analysis and visualization
│   ├── data_builder/           # Stratified sampling and dataset preparation
│   ├── debug/                  # Debugging utilities for extraction issues
│   ├── inspect/                # Result inspection and audit tools
│   ├── test/                   # Unit tests for extraction logic
│   └── rerun/                  # Re-execution handlers for failed runs
│
├── result/                     # All experimental outputs (main output directory)
│   ├── ETHICS/                 # Ethics benchmark results
│   ├── GMS8K/                  # Grade School Math results
│   ├── SQuAD_v2/               # SQuAD v2 results
│   ├── SQuAD_1/                # SQuAD v1 results
│   ├── ELI5/                   # ELI5 results
│   ├── TRUTHFULQA/             # TruthfulQA results
│   └── reports/                # Aggregated cross-model reports
│
└── rerun_results/              # Results from rerun experiments
```

---

## Persona Taxonomy

### Gender Dimension
| Persona ID | Type | Description |
|---|---|---|
| `man_explicit_1` | Explicit | Label: "Your user is a man" |
| `man_implicit_1` | Implicit | Narrative: professional/career-focused context |
| `man_implicit_2` | Implicit | Narrative: different lifestyle context |
| `man_implicit_3` | Implicit | Narrative: third lifestyle variant |
| `woman_explicit_1` | Explicit | Label: "Your user is a woman" |
| `woman_implicit_1` | Implicit | Narrative: mother of two, family logistics |
| `woman_implicit_2` | Implicit | Narrative: different lifestyle context |
| `woman_implicit_3` | Implicit | Narrative: third lifestyle variant |
| `neutral` | Control | No persona injected |

### Religion Dimension
Each religion has 1 explicit + 3 implicit variants (A=kitab/scripture, B=holiday/fasting, C=place of worship):
- `islam_explicit` / `islam_implicit_1/2/3`
- `christian_explicit` / `christian_implicit_1/2/3`
- `atheist_explicit` / `atheist_implicit_1/2/3`
- `jewish_explicit` / `jewish_implicit_1/2/3`

### Age Dimension
Age-group personas defined in `inputs/age_group_persona.json`.

---

## Benchmarks

| Benchmark | Task Type | Questions | Metric |
|---|---|---|---|
| **ETHICS** | Moral judgment (binary: acceptable/not) | ~200 per subset × 5 subsets | Accuracy |
| **GSM8K** | Grade school math (numeric answer) | 8,819 (full test split) | Accuracy |
| **MMLU Redux** | Multiple choice knowledge | 1,540 | Accuracy |
| **SQuAD v1/v2** | Reading comprehension (extractive QA) | 20 samples | F1 / EM |
| **ELI5** | Open-ended explanation generation | 20–650 samples | Qualitative |
| **TruthfulQA** | Factual accuracy | 20–100 samples | Accuracy |

**ETHICS subsets**: `commonsense`, `deontology`, `justice`, `virtue`, `utilitarianism`

---

## Models Tested (18+ via OpenRouter API)

| Provider | Model Slug |
|---|---|
| Google | `google_gemini-2.5-flash`, `google_gemma-2-9b-it`, `google_gemma-3-27b-it` |
| Meta | `meta-llama_llama-3.1-8b-instruct`, `meta-llama_llama-3.3-70b-instruct` |
| DeepSeek | `deepseek_deepseek-v3.2`, `deepseek_deepseek-r1-distill-qwen-32b` |
| Qwen/Alibaba | `qwen_qwen3-8b`, `qwen_qwen3-235b-a22b-2507`, `qwen_qwen3-next-80b-a3b-thinking` |
| OpenAI | `openai_gpt-oss-20b`, `openai_gpt-oss-120b` |
| Mistral | `mistralai_ministral-8b-2512` |
| Moonshot | `moonshotai_kimi-k2-thinking` |
| Arcee | `arcee-ai_trinity-large-thinking` |
| Allen AI | `allenai_olmo-3.1-32b-instruct`, `allenai_olmo-3-32b-think` |
| Nvidia | `nvidia_nemotron-nano-9b-v2` |

---

## Result Folder Structure

Results are organized hierarchically: **Benchmark → Model → Persona**

```
result/ETHICS/
└── google_gemini-2.5-flash/
    ├── result_ethics_google_gemini-2.5-flash.csv   # All personas aggregated
    ├── reports_accuracy_google_gemini-2.5-flash.csv # Accuracy summary
    ├── man_explicit_1/
    │   ├── ethics_commonsense_0001.json
    │   ├── ethics_commonsense_0002.json
    │   └── ... (one JSON per question)
    ├── woman_implicit_1/
    ├── neutral/
    ├── islam_explicit/
    └── ... (all persona variants)
```

Same pattern applies under `GMS8K/Gender_Persona/`, `GMS8K/Religion_Persona/`, `GMS8K/Age_Group_Persona/`, `SQuAD_v2/`, etc.

---

## Per-Question JSON Schema

Each individual question result is saved as a JSON file:

```json
{
  "subset": "commonsense",
  "sample_id": 1,
  "id": "ethics_commonsense_0001",
  "persona": "woman_implicit_1",
  "scenario": "The scenario text from the dataset",
  "request": {
    "messages": [
      {"role": "system", "content": "You are a helpful AI assistant."},
      {"role": "user", "content": "[implicit persona narrative]"},
      {"role": "user", "content": "Evaluate the following scenario...\nAnswer: <0 or 1>\nExcuse: <>"}
    ],
    "parameters": {"temperature": 0.0, "max_tokens": 2048, "model": "google/gemini-2.5-flash"}
  },
  "prediction": {
    "raw": "Answer: 1\nExcuse: ...",
    "parsed_answer": "1",
    "excuse": "..."
  },
  "response": {
    "success": true,
    "raw_api_response": {},
    "latency_sec": 2.62,
    "error": null
  },
  "metadata": {
    "temperature": 0.0,
    "model_id": "google/gemini-2.5-flash",
    "model_name": "Google: Gemini 2.5 Flash",
    "timestamp": "2026-04-21T02:40:24.733015"
  },
  "analysis": {
    "GT": 1,
    "GT_label_text": "acceptable",
    "ground_truth_label": 1,
    "prediction_label": 1,
    "is_correct": true
  }
}
```

---

## Aggregated CSV Schema

`result_ethics_<model>.csv` columns:

| Column | Description |
|---|---|
| `id` | Question ID (e.g. `ethics_commonsense_0001`) |
| `subset` | ETHICS subset name |
| `persona` | Persona ID |
| `ground_truth_label` | Ground truth (0 or 1) |
| `prediction_label` | Model prediction (0, 1, or NaN if extraction failed) |
| `is_correct` | Boolean |
| `parsed_answer` | Raw extracted answer string |
| `explanation` | Model's excuse/reasoning text |
| `success` | API call succeeded |
| `error` | Error message if failed |
| `latency_sec` | Response latency |
| `retry_count` | Number of retries needed |
| `prompt_tokens` | Input token count |
| `completion_tokens` | Output token count |
| `reasoning_tokens` | Reasoning/thinking tokens (for CoT models) |

---

## Critical Experiment Parameters

- **Temperature**: Always `0.0` (deterministic evaluation)
- **Max tokens**: `2048`
- **API**: OpenRouter (`https://openrouter.ai/api/v1`)
- **Answer format (ETHICS)**: `Answer: <0 or 1>\nExcuse: <reasoning>`
- **Answer format (GSM8K)**: `Final answer: <number>`
- **Persona injection**: Injected as the first `user` message before the task

---

## Key Scripts Reference

| Script | Location | Purpose |
|---|---|---|
| Main ETHICS runner | `src/ethics_runner/run_ethics_5subsets_persona.py` | Run all 5 ETHICS subsets × all personas |
| ETHICS age group | `src/ethics_runner/run_ethics_age_group.py` | Age persona variant |
| GSM8K gender runner | `src/gsm8k_runner/run_gsm8k_9_personas.py` | 9-persona gender experiment |
| GSM8K age runner | `src/gsm8k_runner/run_gsm8k_age_group.py` | Age group variant |
| Answer extractor | `src/extract/extractor34.py` | Parse answers from raw LLM responses |
| CSV rebuilder | `src/extract/rebuild_results_csv_root.py` | Rebuild CSV from JSON results |
| Accuracy calculator | `src/analysis/calc_accuracy.py` | Compute per-persona accuracy |
| Report generator | `src/reports/generate_results_csv.py` | Aggregate to Excel/CSV |
| CoT analysis | `src/notebook/CoT_linguistic_marker_GSM8K.ipynb` | Linguistic marker analysis |

---

## Methodology Summary

1. **Persona Definition** — personas are crafted as either explicit labels ("I am a Muslim") or implicit narratives (a story that reveals identity without naming it)
2. **Prompt Construction** — persona is prepended as the first user message; system prompt stays neutral ("You are a helpful AI assistant")
3. **Inference** — each question is run once per persona at temperature=0 via OpenRouter API; per-question JSON is saved immediately
4. **Answer Extraction** — regex + fallback heuristics parse the structured answer format from free-form LLM output
5. **Accuracy Aggregation** — per-persona accuracy is computed and stored in CSV; cross-model comparisons are exported to Excel
6. **Linguistic Analysis** — Jupyter notebooks analyze Chain-of-Thought reasoning patterns, discourse markers, and tone shifts across persona conditions

---

## Naming Conventions

- **Model folders**: `{provider}_{model-name}` — e.g., `google_gemini-2.5-flash`
- **Persona IDs**: `{group}_{mode}_{variant}` — e.g., `woman_implicit_1`, `islam_explicit`
- **Question IDs**: `{dataset}_{category}_{zero-padded-number}` — e.g., `ethics_commonsense_0001`, `gsm8k_00001`
- **Benchmark folders**: uppercase — `ETHICS`, `GMS8K`, `SQuAD_v2`, `TRUTHFULQA`, `ELI5`

---

## Workflow for Adding a New Model or Persona

1. Add model to `config/model_keys.json`
2. Add persona to the relevant file in `inputs/`
3. Run the appropriate runner script in `src/ethics_runner/` or `src/gsm8k_runner/`
4. Results auto-save to `result/{BENCHMARK}/{model_slug}/{persona}/`
5. Run `src/extract/rebuild_results_csv_root.py` to regenerate the aggregated CSV
6. Run `src/reports/generate_results_csv.py` to update the master Excel report

---

## Python Environment

- **Runtime**: Python 3.x with `.venv/` virtual environment
- **Key dependencies**: `openai` (OpenRouter-compatible), `pandas`, `numpy`, `jupyter`, `openpyxl`
- **Activate venv**: `.venv\Scripts\activate` (Windows PowerShell)
