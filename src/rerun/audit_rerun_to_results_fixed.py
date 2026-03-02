import os
import json
import pandas as pd
from glob import glob

def extract_answer_from_response(response_content):
    # Simple rule: ambil angka terakhir di string (bisa diimprove)
    import re
    matches = re.findall(r"[-+]?[0-9]*\.?[0-9]+", response_content)
    if matches:
        return float(matches[-1])
    return None

def audit_rerun_results(rerun_dir, persona, original_csv, out_csv):
    # Load original CSV untuk schema dan info
    df_orig = pd.read_csv(original_csv)
    # Index by question_id
    df_orig = df_orig.set_index("question_id")
    # Prepare new rows
    rows = []
    rerun_files = glob(os.path.join(rerun_dir, persona, "gsm8k_*.json"))
    for f in rerun_files:
        with open(f, encoding="utf-8") as jf:
            data = json.load(jf)
        qid = data["input"]["question_id"]
        gt = data["input"].get("ground_truth")
        response_content = data["response"].get("content", "")
        extracted = extract_answer_from_response(response_content)
        is_correct = None
        try:
            is_correct = abs(float(extracted) - float(gt)) < 1e-2 if extracted is not None and gt is not None else None
        except Exception:
            pass
        # Ambil info lain dari rerun JSON
        row = df_orig.loc[qid].to_dict() if qid in df_orig.index else {}
        row.update({
            "question_id": qid,
            "persona_folder": persona,
            "persona_type": data["input"]["persona"]["type"],
            "success": data["response"].get("success", False),
            "is_correct": is_correct,
            "ground_truth_answer": gt,
            "extracted_answer": extracted,
            "response_content": response_content,
            "prompt_tokens": data["response"].get("raw_api_response", {}).get("usage", {}).get("prompt_tokens", None),
            "completion_tokens": data["response"].get("raw_api_response", {}).get("usage", {}).get("completion_tokens", None),
            "reasoning_tokens": data["response"].get("raw_api_response", {}).get("usage", {}).get("reasoning_tokens", None),
            "total_tokens": data["response"].get("raw_api_response", {}).get("usage", {}).get("total_tokens", None),
            "latency_sec": data["response"].get("latency", None),
            "retry_count": data.get("retry_count", 0),
        })
        rows.append(row)
    df_new = pd.DataFrame(rows)
    # Urutkan dan pastikan kolom sama seperti original
    df_new = df_new[df_orig.columns.intersection(df_new.columns).tolist() + [c for c in df_new.columns if c not in df_orig.columns]]
    df_new = df_new.sort_values("question_id")
    df_new.to_csv(out_csv, index=False)
    print(f"Wrote fixed results: {out_csv}")

if __name__ == "__main__":
    # Contoh: audit_rerun_results('rerun_results', 'man_explicit_1', 'result/reports/allenai_olmo-3-32b-think/man_explicit_1/results.csv', 'result/reports/allenai_olmo-3-32b-think/man_explicit_1/results_fixed.csv')
    personas = [
        'man_explicit_1', 'man_implicit_1', 'man_implicit_2', 'man_implicit_3',
        'woman_explicit_1', 'woman_implicit_1', 'woman_implicit_2', 'woman_implicit_3'
    ]
    for persona in personas:
        orig = f'result/reports/allenai_olmo-3-32b-think/{persona}/results.csv'
        out = f'result/reports/allenai_olmo-3-32b-think/{persona}/results_fixed.csv'
        if os.path.exists(orig):
            audit_rerun_results('rerun_results', persona, orig, out)
