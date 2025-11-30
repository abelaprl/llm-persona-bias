#!/usr/bin/env python3
"""Quick accuracy summary - no CSV generation, just summary."""

import json
import re
from pathlib import Path

def extract_answer(text):
    if not text:
        return None
    patterns = [
        r"[Ff]inal [Aa]nswer:?\s*\*\*\$?([-]?[\d,]+(?:\.\d+)?)\*\*",
        r"\*\*[Ff]inal [Aa]nswer:?\*\*\s*\$?([-]?[\d,]+(?:\.\d+)?)",
        r"\*\*[Ff]inal [Aa]nswer:?\s*\$?([-]?[\d,]+(?:\.\d+)?)\*\*",
        r"[Ff]inal [Aa]nswer:?\s*\$?\s*([-]?[\d,]+(?:\.\d+)?)",
        r"\\boxed\{([-]?[\d,]+(?:\.\d+)?)\}",
        r"boxed\{([-]?[\d,]+(?:\.\d+)?)\}",
        r"####\s*([-]?[\d,]+(?:\.\d+)?)",
    ]
    for p in patterns:
        matches = list(re.finditer(p, text))
        if matches:
            try:
                return float(matches[-1].group(1).replace(",", ""))
            except:
                pass
    return None

def extract_gold(text):
    if not text:
        return None
    match = re.search(r"####\s*([\d,]+(?:\.\d+)?)", text)
    if match:
        try:
            return float(match.group(1).replace(",", ""))
        except:
            pass
    return None

models = [
    ("bert_nebulon", "man_implicits"),
    ("bert_nebulon", "woman_implicits"),
    ("grok_4_1", "man_implicits"),
    ("grok_4_1", "woman_implicits"),
    ("nemotron-nano", "man_implicits"),
    ("nemotron-nano", "woman_implicits"),
]

print("="*90)
print("CORRECTED ACCURACY SUMMARY (with bold/boxed patterns)")
print("="*90)

results = []

for model, persona in models:
    gsm8k_dir = Path("results") / model / persona / "gsm8k_raw"
    files = sorted(gsm8k_dir.glob("gsm8k_*.json"))
    
    correct = wrong = missing = 0
    tp = tc = tt = tr = 0  # tokens
    
    for f in files:
        try:
            data = json.load(open(f, encoding="utf-8"))
        except:
            continue
        
        resp = data.get("response_text", "")
        if not resp:
            ch = data.get("response", {}).get("choices", [])
            if ch:
                resp = ch[0].get("message", {}).get("content", "")
        
        gold = extract_gold(data.get("input", {}).get("gold_answer", ""))
        model_ans = extract_answer(resp)
        
        u = data.get("response", {}).get("usage", {})
        tp += u.get("prompt_tokens", 0)
        tc += u.get("completion_tokens", 0)
        tt += u.get("total_tokens", 0)
        tr += u.get("completion_tokens_details", {}).get("reasoning_tokens", 0)
        
        if model_ans is None:
            missing += 1
        elif gold is not None and abs(model_ans - gold) > 0.01:
            wrong += 1
        elif gold is not None:
            correct += 1
    
    total = len(files)
    valid = correct + wrong
    acc = (correct / valid * 100) if valid > 0 else 0
    
    results.append({
        "model": model, "persona": persona, "total": total,
        "correct": correct, "wrong": wrong, "missing": missing,
        "accuracy": acc,
        "avg_prompt": tp/total if total else 0,
        "avg_completion": tc/total if total else 0,
        "avg_total": tt/total if total else 0,
        "avg_reasoning": tr/total if total else 0,
    })

# Print summary table
print(f"\n{'Model':<18} {'Persona':<18} {'Total':<7} {'Correct':<8} {'Wrong':<6} {'Miss':<6} {'Accuracy':<10}")
print("-"*90)
for r in results:
    print(f"{r['model']:<18} {r['persona']:<18} {r['total']:<7} {r['correct']:<8} {r['wrong']:<6} {r['missing']:<6} {r['accuracy']:<10.2f}")

print("\n" + "="*90)
print("TOKEN USAGE SUMMARY")
print("="*90)
print(f"{'Model':<18} {'Persona':<18} {'AvgPrompt':<12} {'AvgCompletion':<14} {'AvgTotal':<12} {'AvgReasoning':<12}")
print("-"*90)
for r in results:
    print(f"{r['model']:<18} {r['persona']:<18} {r['avg_prompt']:<12.1f} {r['avg_completion']:<14.1f} {r['avg_total']:<12.1f} {r['avg_reasoning']:<12.1f}")

# Save to CSV
import csv
with open("results/reports/summary_all_models_corrected.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=results[0].keys())
    w.writeheader()
    w.writerows(results)
print(f"\nSaved: results/reports/summary_all_models_corrected.csv")
