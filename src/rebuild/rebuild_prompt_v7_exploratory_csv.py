"""
Rebuild a single CSV from all per-question JSON files under
result/ETHICS/deepseek_deepseek-r1-distill-qwen-32b/prompt_v7_exploratory/
"""
import json
import pathlib
import pandas as pd

ROOT = pathlib.Path(r"D:\TugasAkhir\result\ETHICS\deepseek_deepseek-r1-distill-qwen-32b\prompt_v7_exploratory")
OUTPUT = ROOT / "result_ethics_deepseek_r1_distill_qwen_32b_prompt_v7_exploratory.csv"

rows = []
for json_file in sorted(ROOT.rglob("*.json")):
    try:
        with open(json_file, encoding="utf-8") as f:
            d = json.load(f)
    except Exception as e:
        print(f"[SKIP] {json_file}: {e}")
        continue

    usage = (d.get("response", {}).get("raw_api_response") or {}).get("usage", {})
    completion_details = usage.get("completion_tokens_details", {})

    rows.append({
        "id":                   d.get("id"),
        "subset":               d.get("subset"),
        "persona":              d.get("persona"),
        "persona_category":     d.get("metadata", {}).get("persona_category"),
        "scenario":             d.get("scenario"),
        "ground_truth_label":   d.get("analysis", {}).get("ground_truth_label"),
        "prediction_label":     d.get("analysis", {}).get("prediction_label"),
        "is_correct":           d.get("analysis", {}).get("is_correct"),
        "parsed_answer":        d.get("prediction", {}).get("parsed_answer"),
        "explanation":          d.get("prediction", {}).get("explanation") or d.get("prediction", {}).get("excuse"),
        "success":              d.get("response", {}).get("success"),
        "error":                d.get("response", {}).get("error"),
        "latency_sec":          d.get("response", {}).get("latency_sec"),
        "retry_count":          d.get("response", {}).get("retry_count"),
        "prompt_tokens":        usage.get("prompt_tokens"),
        "completion_tokens":    usage.get("completion_tokens"),
        "reasoning_tokens":     completion_details.get("reasoning_tokens"),
        "timestamp":            d.get("metadata", {}).get("timestamp"),
        "source_file":          str(json_file.relative_to(ROOT)),
    })

df = pd.DataFrame(rows)
df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
print(f"Saved {len(df)} rows -> {OUTPUT}")
print(df["persona"].value_counts().to_string())
