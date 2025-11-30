#!/usr/bin/env python3
"""
Deep analysis of answer extraction accuracy.
Check all patterns and debug why some answers might be missed.
"""

import json
import os
import re
from pathlib import Path
from collections import defaultdict

RESULTS_BASE = "results"

MODELS = {
    "bert_nebulon": ["man_implicits", "woman_implicits"],
    "grok_4_1": ["man_implicits", "woman_implicits"],
    "nemotron-nano": ["man_implicits", "woman_implicits"]
}


def extract_gold_answer(gold_text: str):
    """Extract gold answer from GSM8K format (#### X)."""
    if not gold_text:
        return None, "empty"
    
    match = re.search(r"####\s*([\d,]+(?:\.\d+)?)", gold_text)
    if match:
        try:
            return float(match.group(1).replace(",", "")), "success"
        except:
            return None, "parse_error"
    return None, "no_match"


def extract_model_answer(response_text: str):
    """
    Extract model answer with multiple patterns.
    Returns (answer, pattern_used, debug_info)
    """
    if not response_text:
        return None, "empty", ""
    
    # Clean up response text
    text = response_text.strip()
    
    # Pattern 1: "Final answer: X" (with optional $ and bold markers)
    patterns = [
        (r"[Ff]inal\s*[Aa]nswer[:\s]*\*?\*?\s*\$?\s*([-]?[\d,]+(?:\.\d+)?)", "final_answer_standard"),
        (r"\*\*[Ff]inal\s*[Aa]nswer[:\s]*\$?\s*([-]?[\d,]+(?:\.\d+)?)\*\*", "final_answer_bold"),
        (r"[Ff]inal\s*[Aa]nswer[:\s]*\*\*([-]?[\d,]+(?:\.\d+)?)\*\*", "final_answer_bold2"),
        (r"[Ff]inal [Aa]nswer is[:\s]*\$?\s*([-]?[\d,]+(?:\.\d+)?)", "final_answer_is"),
        (r"[Tt]he [Ff]inal [Aa]nswer is[:\s]*\$?\s*([-]?[\d,]+(?:\.\d+)?)", "the_final_answer_is"),
        (r"####\s*([-]?[\d,]+(?:\.\d+)?)", "hash_format"),
        (r"[Aa]nswer[:\s]*\$?\s*([-]?[\d,]+(?:\.\d+)?)\s*$", "answer_at_end"),
        (r"= \$?([-]?[\d,]+(?:\.\d+)?)\s*$", "equals_at_end"),
        (r"\$\s*([-]?[\d,]+(?:\.\d+)?)\s*(?:dollars?)?\s*$", "dollar_at_end"),
    ]
    
    for pattern, name in patterns:
        matches = list(re.finditer(pattern, text, re.MULTILINE))
        if matches:
            # Use the LAST match (most likely to be the final answer)
            match = matches[-1]
            try:
                value = float(match.group(1).replace(",", ""))
                return value, name, match.group(0)[:50]
            except:
                pass
    
    # Debug: show last 200 chars if no match
    return None, "no_pattern", text[-200:] if len(text) > 200 else text


def analyze_model_persona(model_key: str, persona: str):
    """Analyze all results for a model+persona combination."""
    gsm8k_dir = Path(RESULTS_BASE) / model_key / persona / "gsm8k_raw"
    
    if not gsm8k_dir.exists():
        print(f"  Directory not found: {gsm8k_dir}")
        return {}
    
    files = sorted([f for f in gsm8k_dir.glob("gsm8k_*.json")])
    
    stats = {
        "total": 0,
        "correct": 0,
        "incorrect": 0,
        "gold_missing": 0,
        "model_missing": 0,
        "both_missing": 0,
        "pattern_counts": defaultdict(int),
        "incorrect_samples": [],
        "missing_samples": []
    }
    
    for filepath in files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except:
            continue
        
        stats["total"] += 1
        
        # Get response text
        response_text = data.get("response_text", "")
        if not response_text:
            choices = data.get("response", {}).get("choices", [])
            if choices:
                response_text = choices[0].get("message", {}).get("content", "")
        
        # Get gold answer
        gold_text = data.get("input", {}).get("gold_answer", "")
        gold_answer, gold_status = extract_gold_answer(gold_text)
        
        # Get model answer
        model_answer, pattern_used, debug_info = extract_model_answer(response_text)
        stats["pattern_counts"][pattern_used] += 1
        
        # Check correctness
        if gold_answer is None and model_answer is None:
            stats["both_missing"] += 1
        elif gold_answer is None:
            stats["gold_missing"] += 1
        elif model_answer is None:
            stats["model_missing"] += 1
            if len(stats["missing_samples"]) < 5:
                stats["missing_samples"].append({
                    "file": filepath.name,
                    "response_preview": response_text[-300:] if len(response_text) > 300 else response_text
                })
        else:
            if abs(model_answer - gold_answer) < 0.01:
                stats["correct"] += 1
            else:
                stats["incorrect"] += 1
                if len(stats["incorrect_samples"]) < 5:
                    stats["incorrect_samples"].append({
                        "file": filepath.name,
                        "gold": gold_answer,
                        "model": model_answer,
                        "pattern": pattern_used,
                        "debug": debug_info
                    })
    
    return stats


def main():
    print("="*80)
    print("DEEP ANALYSIS: Answer Extraction Accuracy")
    print("="*80)
    
    all_stats = {}
    
    for model_key, personas in MODELS.items():
        print(f"\n{'='*60}")
        print(f"MODEL: {model_key}")
        print(f"{'='*60}")
        
        for persona in personas:
            print(f"\n  --- {persona} ---")
            stats = analyze_model_persona(model_key, persona)
            all_stats[(model_key, persona)] = stats
            
            if not stats:
                continue
            
            total = stats["total"]
            correct = stats["correct"]
            incorrect = stats["incorrect"]
            model_missing = stats["model_missing"]
            gold_missing = stats["gold_missing"]
            
            # Calculate accuracy only on valid comparisons
            valid_comparisons = correct + incorrect
            accuracy = (correct / valid_comparisons * 100) if valid_comparisons > 0 else 0
            
            print(f"  Total files: {total}")
            print(f"  Valid comparisons: {valid_comparisons}")
            print(f"  Correct: {correct} ({correct/total*100:.2f}%)")
            print(f"  Incorrect: {incorrect} ({incorrect/total*100:.2f}%)")
            print(f"  Model answer missing: {model_missing}")
            print(f"  Gold answer missing: {gold_missing}")
            print(f"  ACCURACY (valid only): {accuracy:.2f}%")
            
            print(f"\n  Pattern usage:")
            for pattern, count in sorted(stats["pattern_counts"].items(), key=lambda x: -x[1]):
                print(f"    {pattern}: {count}")
            
            if stats["incorrect_samples"]:
                print(f"\n  Sample INCORRECT answers:")
                for sample in stats["incorrect_samples"][:3]:
                    print(f"    {sample['file']}: gold={sample['gold']}, model={sample['model']} (pattern: {sample['pattern']})")
            
            if stats["missing_samples"]:
                print(f"\n  Sample MISSING model answers:")
                for sample in stats["missing_samples"][:2]:
                    print(f"    {sample['file']}:")
                    print(f"      Response end: ...{sample['response_preview'][-100:]}")
    
    # Summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Model':<20} {'Persona':<18} {'Total':<8} {'Valid':<8} {'Correct':<8} {'Accuracy':<10} {'Missing':<8}")
    print("-"*80)
    
    for (model_key, persona), stats in all_stats.items():
        if not stats:
            continue
        total = stats["total"]
        valid = stats["correct"] + stats["incorrect"]
        correct = stats["correct"]
        missing = stats["model_missing"]
        accuracy = (correct / valid * 100) if valid > 0 else 0
        print(f"{model_key:<20} {persona:<18} {total:<8} {valid:<8} {correct:<8} {accuracy:<10.2f} {missing:<8}")


if __name__ == "__main__":
    main()
