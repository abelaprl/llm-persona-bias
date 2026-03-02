#!/usr/bin/env python3
"""
Generate CSV reports for GSM8K experiment results.
- Per-model CSV with all questions
- Summary CSV comparing models and personas
- Token usage analysis (excluding warmup)
"""

import json
import os
import re
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional

# Configuration
RESULTS_BASE = "results"
OUTPUT_DIR = "results/reports"

MODELS = {
    "bert_nebulon": {
        "name": "Bert Nebulon",
        "personas": ["man_implicits", "woman_implicits"]
    },
    "grok_4_1": {
        "name": "Grok 4.1 Fast",
        "personas": ["man_implicits", "woman_implicits"]
    },
    "nemotron-nano": {
        "name": "Nemotron Nano",
        "personas": ["man_implicits", "woman_implicits"]
    }
}


def extract_final_answer(text: str) -> Optional[float]:
    """Extract numeric answer from response text with comprehensive patterns."""
    if not text:
        return None
    
    # Order matters - more specific patterns first
    patterns = [
        # Standard patterns
        (r'[Ff]inal\s*[Aa]nswer[:\s]*\*?\*?\s*\$?\s*([-]?[\d,]+(?:\.\d+)?)', 'std'),
        (r'\*\*[Ff]inal\s*[Aa]nswer[:\s]*\$?\s*([-]?[\d,]+(?:\.\d+)?)\*\*', 'bold1'),
        (r'[Ff]inal\s*[Aa]nswer[:\s]*\*\*([-]?[\d,]+(?:\.\d+)?)\*\*', 'bold2'),
        (r'\*\*[Ff]inal\s*[Aa]nswer:?\*\*[:\s]*\$?\s*([-]?[\d,]+(?:\.\d+)?)', 'bold3'),
        # Bold label with colon: **Final answer**: X
        (r'\*\*[Ff]inal\s*[Aa]nswer\*\*[:\s]*\$?€?\s*([-]?[\d,]+(?:\.\d+)?)', 'bold_label'),
        # LaTeX formats
        (r'\\boxed\{([-]?[\d,]+(?:\.\d+)?)\}', 'boxed'),
        (r'\\boxed\{\\dfrac\{(\d+)\}\{\d+\}\}', 'boxed_frac'),
        (r'\\mathbf\{([-]?[\d,]+(?:\.\d+)?)\}', 'mathbf'),
        # LaTeX with comma separator: \(14,\!400\)
        (r'\\\(([-]?[\d,\\!]+(?:\.\d+)?)\\\)', 'latex_paren'),
        # Euro/currency: €77 or **€77**
        (r'[:\s]\*?\*?€\s*([-]?[\d,]+(?:\.\d+)?)\*?\*?', 'euro'),
        # <number> placeholder: Final answer: <number> 48
        (r'[Ff]inal\s*[Aa]nswer[:\s]*<number>\s*([-]?[\d,]+(?:\.\d+)?)', 'placeholder'),
        # Hash format
        (r'####\s*([-]?[\d,]+(?:\.\d+)?)', 'hash'),
        # Pattern: = $XXX at end of text (like "= $144")
        (r'=\s*\$?([-]?[\d,]+(?:\.\d+)?)\s*\.?\s*$', 'equals_end'),
        # Pattern: calculation result in LaTeX like \( 40 - 10 = 30 \)
        (r'=\s*([-]?[\d,]+(?:\.\d+)?)\s*\\\)', 'latex_result'),
        # Pattern: totals X or cost is X at end
        (r'(?:total|cost|is|equals?)\s+(?:is\s+)?\$?([-]?[\d,]+(?:\.\d+)?)\s*\.?\s*$', 'total_end'),
    ]
    
    for pattern, name in patterns:
        matches = list(re.finditer(pattern, text))
        if matches:
            try:
                # Use last match (most likely final answer)
                val = matches[-1].group(1).replace(",", "").replace("\\!", "")
                return float(val)
            except:
                pass
    # Fallback: take last numeric token in the entire text and strip fractional part
    num_matches = re.findall(r'([+-]?\d{1,3}(?:[,\.\s]\d{3})*(?:\.\d+)?|[+-]?\d+(?:\.\d+)?)', text)
    if num_matches:
        s = num_matches[-1]
        # Normalize by handling thousands separators and decimal/fractional part
        try:
            if "." in s and "," in s:
                # assume comma thousands, dot decimal
                s = s.replace(",", "")
                s = s.split(".")[0]
            elif "." in s:
                s = s.split(".")[0]
            elif "," in s:
                # decide if comma is thousands sep or decimal sep
                parts = s.split(",")
                if len(parts[-1]) == 3:
                    # likely thousands separator
                    s = s.replace(",", "")
                else:
                    # likely decimal separator - drop fractional part
                    s = ",".join(parts[:-1]) if len(parts) > 1 else parts[0]
                    s = s.replace(",", "")

            s = re.sub(r"[^0-9+-]", "", s)
            return float(s)
        except:
            return None
    return None
    
        # Prefer last "= NUMBER" occurrence in whole text (handles lines like "... = 160 minutes")
        eq_matches = re.findall(r'=\s*\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)', text)
        if eq_matches:
            s = eq_matches[-1]
            # drop fractional part similar to fallback
            if "." in s and "," in s:
                s = s.replace(',', '')
                s = s.split('.')[0]
            elif "." in s:
            # Prefer last "= NUMBER" occurrence in whole text (handles lines like "... = 160 minutes")
            eq_matches = re.findall(r'=\s*\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)', text)
            if eq_matches:
                s = eq_matches[-1]
                # drop fractional part similar to fallback
                if "." in s and "," in s:
                    s = s.replace(',', '')
                    s = s.split('.')[0]
                elif "." in s:
                    s = s.split('.')[0]
                elif "," in s:
                    parts = s.split(',')
                    if len(parts[-1]) == 3:
                        s = s.replace(',', '')
                    else:
                        s = ','.join(parts[:-1]) if len(parts) > 1 else parts[0]
                        s = s.replace(',', '')

                s = re.sub(r'[^0-9+-]', '', s)
                try:
                    return float(s)
                except:
                    pass
            return float(match.group(1).replace(",", ""))
        except:
            pass
    return None


def load_result_file(filepath: Path) -> Optional[Dict]:
    """Load a single result JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        return None


def process_model_persona(model_key: str, persona: str) -> List[Dict]:
    """Process all GSM8K results for a model+persona combination."""
    results = []
    gsm8k_dir = Path(RESULTS_BASE) / model_key / persona / "gsm8k_raw"
    
    if not gsm8k_dir.exists():
        print(f"  Directory not found: {gsm8k_dir}")
        return results
    
    # Get all gsm8k_XXXXX.json files (exclude warmup, summary, etc.)
    files = sorted([f for f in gsm8k_dir.glob("gsm8k_*.json") if not f.name.startswith("persona")])
    
    for filepath in files:
        data = load_result_file(filepath)
        if not data:
            continue
        
        # Extract fields
        run = data.get("run", {})
        response = data.get("response", {})
        usage = response.get("usage", {})
        input_data = data.get("input", {})
        
        question_id = run.get("question_id", filepath.stem)
        question_index = run.get("question_index", 0)
        
        # Get response text
        response_text = data.get("response_text", "")
        if not response_text:
            choices = response.get("choices", [])
            if choices:
                response_text = choices[0].get("message", {}).get("content", "")
        
        # Extract answers
        model_answer = extract_final_answer(response_text)
        gold_answer = extract_gold_answer(input_data.get("gold_answer", ""))
        
        # Check correctness
        is_correct = False
        if model_answer is not None and gold_answer is not None:
            is_correct = abs(model_answer - gold_answer) < 0.01
        
        # Token usage
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)
        reasoning_tokens = usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0)
        
        results.append({
            "question_id": question_id,
            "question_index": question_index,
            "question": input_data.get("question", "")[:100],  # truncate for CSV
            "response_text": response_text,
            "gold_answer": gold_answer,
            "model_answer": model_answer,
            "is_correct": 1 if is_correct else 0,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "reasoning_tokens": reasoning_tokens,
            "latency_sec": run.get("latency_sec", 0),
            "response_length": len(response_text)
        })
    
    return results


def generate_model_csv(model_key: str, model_info: Dict):
    """Generate CSV for a single model with all personas."""
    model_name = model_info["name"]
    print(f"\nProcessing {model_name}...")
    
    all_results = []
    
    for persona in model_info["personas"]:
        print(f"  Processing {persona}...")
        results = process_model_persona(model_key, persona)
        
        # Add persona column
        for r in results:
            r["persona"] = persona
            r["model"] = model_name
        
        all_results.extend(results)
        print(f"    Found {len(results)} questions")
    
    if not all_results:
        print(f"  No results found for {model_name}")
        return
    
    # Write model CSV
    output_path = Path(OUTPUT_DIR) / f"{model_key}_results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = ["model", "persona", "question_id", "question_index", "question",
                  "gold_answer", "model_answer", "response_text", "is_correct",
                  "prompt_tokens", "completion_tokens", "total_tokens", "reasoning_tokens",
                  "latency_sec", "response_length"]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    
    print(f"  Saved: {output_path}")
    return all_results


def generate_summary_csv(all_model_results: Dict[str, List[Dict]]):
    """Generate summary CSV comparing models and personas."""
    
    summary_rows = []
    
    for model_key, results in all_model_results.items():
        if not results:
            continue
        
        model_name = results[0]["model"] if results else model_key
        
        # Group by persona
        personas = {}
        for r in results:
            persona = r["persona"]
            if persona not in personas:
                personas[persona] = []
            personas[persona].append(r)
        
        for persona, persona_results in personas.items():
            total_q = len(persona_results)
            correct = sum(r["is_correct"] for r in persona_results)
            accuracy = (correct / total_q * 100) if total_q > 0 else 0
            
            # Token stats
            total_prompt = sum(r["prompt_tokens"] for r in persona_results)
            total_completion = sum(r["completion_tokens"] for r in persona_results)
            total_tokens = sum(r["total_tokens"] for r in persona_results)
            total_reasoning = sum(r["reasoning_tokens"] for r in persona_results)
            
            avg_prompt = total_prompt / total_q if total_q > 0 else 0
            avg_completion = total_completion / total_q if total_q > 0 else 0
            avg_total = total_tokens / total_q if total_q > 0 else 0
            avg_reasoning = total_reasoning / total_q if total_q > 0 else 0
            avg_latency = sum(r["latency_sec"] for r in persona_results) / total_q if total_q > 0 else 0
            avg_response_len = sum(r["response_length"] for r in persona_results) / total_q if total_q > 0 else 0
            
            summary_rows.append({
                "model": model_name,
                "model_key": model_key,
                "persona": persona,
                "total_questions": total_q,
                "correct_answers": correct,
                "accuracy_pct": round(accuracy, 2),
                "total_prompt_tokens": total_prompt,
                "total_completion_tokens": total_completion,
                "total_tokens": total_tokens,
                "total_reasoning_tokens": total_reasoning,
                "avg_prompt_tokens": round(avg_prompt, 2),
                "avg_completion_tokens": round(avg_completion, 2),
                "avg_total_tokens": round(avg_total, 2),
                "avg_reasoning_tokens": round(avg_reasoning, 2),
                "avg_latency_sec": round(avg_latency, 3),
                "avg_response_length": round(avg_response_len, 1)
            })
    
    # Write summary CSV
    output_path = Path(OUTPUT_DIR) / "summary_all_models.csv"
    
    fieldnames = ["model", "model_key", "persona", "total_questions", "correct_answers", "accuracy_pct",
                  "total_prompt_tokens", "total_completion_tokens", "total_tokens", "total_reasoning_tokens",
                  "avg_prompt_tokens", "avg_completion_tokens", "avg_total_tokens", "avg_reasoning_tokens",
                  "avg_latency_sec", "avg_response_length"]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    
    print(f"\nSaved summary: {output_path}")
    
    # Print summary to console
    print("\n" + "="*80)
    print("SUMMARY - Accuracy & Token Usage per Model per Persona")
    print("="*80)
    print(f"{'Model':<20} {'Persona':<18} {'Q':<6} {'Correct':<8} {'Acc%':<8} {'AvgPrompt':<10} {'AvgCompl':<10} {'AvgTotal':<10} {'AvgReason':<10}")
    print("-"*80)
    
    for row in summary_rows:
        print(f"{row['model']:<20} {row['persona']:<18} {row['total_questions']:<6} {row['correct_answers']:<8} {row['accuracy_pct']:<8} {row['avg_prompt_tokens']:<10} {row['avg_completion_tokens']:<10} {row['avg_total_tokens']:<10} {row['avg_reasoning_tokens']:<10}")


def main():
    print("="*60)
    print("GSM8K Results CSV Generator")
    print("="*60)
    
    # Ensure output directory exists
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    all_model_results = {}
    
    # Process each model
    for model_key, model_info in MODELS.items():
        results = generate_model_csv(model_key, model_info)
        if results:
            all_model_results[model_key] = results
    
    # Generate summary
    if all_model_results:
        generate_summary_csv(all_model_results)
    
    print("\n" + "="*60)
    print("Done! CSV files saved to:", OUTPUT_DIR)
    print("="*60)


if __name__ == "__main__":
    main()
