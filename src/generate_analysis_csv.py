#!/usr/bin/env python3
"""
Generate analysis CSV files for GSM8K bias study results.
- CSV per model (all personas combined)
- CSV per persona (all models combined)  
- CSV all (comparison of accuracy and token usage across models/personas)
"""

import json
import os
import re
import csv
from datetime import datetime
from collections import defaultdict

# Results directory
RESULTS_DIR = "results"
OUTPUT_DIR = "results/reports"

# Models to analyze
MODELS = [
    "groq-llama-3.1-8b-instant",
    "openai_gpt-oss-20b",
    "openai_gpt-oss-120b",
    "qwen_qwen3-next-80b-a3b-thinking",
    "qwen_qwen3-next-80b-a3b-instruct",
    "meta-llama_llama-3.3-70b-instruct",
]

# Personas to analyze
PERSONAS = [
    "man_explicit",
    "man_explicits",
    "man_explicits_v2",
    "man_implicits",
    "woman_explicit",
    "woman_explicits",
    "woman_implicits",
]

def extract_answer_from_ground_truth(ground_truth):
    """Extract numerical answer from ground_truth string (after ####)"""
    if not ground_truth:
        return None
    
    # Look for #### followed by number
    match = re.search(r'####\s*([-+]?\d*\.?\d+)', ground_truth)
    if match:
        try:
            return float(match.group(1))
        except:
            return None
    return None

def extract_answer_from_response(content):
    """Extract numerical answer from model response - improved for multiple formats"""
    if not content:
        return None
    
    # Clean content - normalize whitespace and remove LaTeX comma formatting {,}
    content_clean = content.strip()
    # Remove LaTeX comma formatting: $70{,}000 → $70000
    content_clean = re.sub(r'\{,\}', '', content_clean)
    # Also handle \, for thousands separator
    content_clean = re.sub(r'\\,', '', content_clean)
    # Remove escaped dollar signs: \$18 → $18 but carefully in boxed
    content_clean = re.sub(r'\\boxed\{\\?\$', r'\\boxed{', content_clean)
    
    # PRIORITY -1: Check LAST 5 LINES specifically for Answer patterns (highest priority)
    last_lines = '\n'.join(content_clean.split('\n')[-5:])
    last_line_answer_patterns = [
        # ✅ Final Answer: **Siobhan has 23 jewels.**
        r'(?:✅\s*)?(?:Final\s+)?Answer:?\s*\*?\*?[^0-9]*?([-+]?\d+(?:,\d{3})*(?:\.\d+)?)',
        # **Answer: 540** or The answer is **540**
        r'(?:the\s+)?answer\s+is:?\s*\*?\*?\\?\$?([-+]?\d+(?:,\d{3})*(?:\.\d+)?)',
    ]
    for pattern in last_line_answer_patterns:
        matches = re.findall(pattern, last_lines, re.IGNORECASE)
        if matches:
            try:
                return float(matches[-1].replace(',', ''))
            except:
                pass
    
    # Priority 0: Look for explicit "Answer:" patterns anywhere in text
    answer_line_patterns = [
        # **Answer:** 540 or **Answer:** 540 meters or **Answer:** The answer is 540
        r'\*\*Answer:?\*\*:?\s*[^0-9$*]*?\\?\$?([-+]?\d+(?:,\d{3})*(?:\.\d+)?)',
        # Answer: **540** or Answer: 540
        r'Answer:?\s*\*?\*?\\?\$?([-+]?\d+(?:,\d{3})*(?:\.\d+)?)',
        # ✅ Answer: **3 bolts** or ✅ Final Answer: **540**
        r'✅?\s*(?:Final\s+)?Answer:?\s*[^0-9$*]*?\\?\$?([-+]?\d+(?:,\d{3})*(?:\.\d+)?)',
        # The answer is 540 or answer is **540**
        r'(?:the\s+)?answer\s+is:?\s*\*?\*?\\?\$?([-+]?\d+(?:,\d{3})*(?:\.\d+)?)',
        # Final answer: 540
        r'final\s+answer:?\s*\*?\*?\\?\$?([-+]?\d+(?:,\d{3})*(?:\.\d+)?)',
    ]
    for pattern in answer_line_patterns:
        matches = re.findall(pattern, content_clean, re.IGNORECASE)
        if matches:
            try:
                return float(matches[-1].replace(',', ''))
            except:
                pass
    
    # Priority 1: \boxed{...= X} - handle equations inside boxed FIRST
    # Look for boxed content that has "= X" at the end (the result)
    boxed_eq_match = re.findall(r'\\boxed\{[^}]*=\s*\\?\$?([-+]?\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:\\text\{[^}]*\})?[^}]*\}', content_clean)
    if boxed_eq_match:
        try:
            return float(boxed_eq_match[-1].replace(',', ''))
        except:
            pass
    
    # Priority 1b: \boxed{X} - simple number in boxed
    boxed_patterns = [
        r'\\boxed\{([-+]?\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:\\text\{[^}]*\})?[^}]*\}',  # \boxed{18} or \boxed{18\text{ cups}}
        r'\\boxed\{([-+]?\d+(?:,\d{3})*(?:\.\d+)?)',  # \boxed{18 with stuff after
    ]
    for pattern in boxed_patterns:
        matches = re.findall(pattern, content_clean)
        if matches:
            try:
                return float(matches[-1].replace(',', ''))
            except:
                pass
    
    # Priority 2: Bold answer patterns **$X** or **X sheep** etc (anywhere)
    # Using [^*]* to match any text between number and **
    bold_patterns = [
        # **3 bolts in total** or **762 M&Ms** - number followed by any non-* chars, then **
        r'\*\*\\?\$?([-+]?\d+(?:,\d{3})*(?:\.\d+)?)[^*]*\*\*',
        # **$18** or **18** - simple number
        r'\*\*\\?\$?([-+]?\d+(?:,\d{3})*(?:\.\d+)?)\*\*',
    ]
    for pattern in bold_patterns:
        matches = re.findall(pattern, content_clean, re.IGNORECASE)
        if matches:
            try:
                return float(matches[-1].replace(',', ''))
            except:
                pass
    
    # Priority 3: LaTeX inline math \( ... = X \) or \[ ... = X \]
    latex_math_patterns = [
        r'\\[\(\[].*?=\s*\\?\$?([-+]?\d+(?:,\d{3})*(?:\.\d+)?)\s*\\?[\)\]]',
        r'\\[\(\[].*?\\?\$?([-+]?\d+(?:,\d{3})*(?:\.\d+)?)\s*\\?[\)\]]',
    ]
    for pattern in latex_math_patterns:
        matches = re.findall(pattern, content_clean, re.DOTALL)
        if matches:
            try:
                return float(matches[-1].replace(',', ''))
            except:
                pass
    
    # Priority 4: "= X" calculations - but only at end of line or before period
    calc_patterns = [
        r'=\s*\\?\$?([-+]?\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:dollars?|cups?|eggs?|hours?|minutes?|days?|weeks?|months?|years?|miles?|km|meters?|feet|pounds?|kg|grams?|times?|bags?|boxes?|lollipops?|bolts?)?\s*[\.\n]?\s*$',
        r'=\s*\\?\$?([-+]?\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:dollars?|cups?|eggs?|hours?|minutes?|days?|weeks?|months?|years?|miles?|km|meters?|feet|pounds?|kg|grams?|times?|bags?|boxes?|lollipops?|bolts?)?[^\d]',
    ]
    for pattern in calc_patterns:
        matches = re.findall(pattern, content_clean, re.IGNORECASE | re.MULTILINE)
        if matches:
            try:
                return float(matches[-1].replace(',', ''))
            except:
                pass
    
    # Priority 5: "makes/earns/costs $X" patterns - plain sentences
    value_patterns = [
        # "Janet makes $18 every day" - capture from end of sentence patterns
        r'(?:makes?|earns?|costs?|is|gets?|pays?|receives?|spends?|ordered?|bought?|sold?|has|have|needs?|requires?|can\s+(?:go|buy|make|fill|take)|will\s+(?:have|need|be|take)|would\s+(?:have|need|be|take))\s+\\?\$?\s*([-+]?\d+(?:,\d{3})*(?:\.\d+)?)',
        # "Total bolts = 3 bolts" or "Total = 3"
        r'[Tt]otal\s*(?:[\w\s]+)?\s*[=:]\s*([-+]?\d+(?:,\d{3})*(?:\.\d+)?)',
        # "There are 3 bolts" or "She has 18 dollars"
        r'(?:There\s+(?:are|is|will\s+be)|She\s+(?:has|earns?|makes?|will\s+have)|He\s+(?:has|earns?|makes?|will\s+have))\s+\\?\$?([-+]?\d+(?:,\d{3})*(?:\.\d+)?)',
    ]
    for pattern in value_patterns:
        matches = re.findall(pattern, content_clean, re.IGNORECASE)
        if matches:
            try:
                return float(matches[-1].replace(',', ''))
            except:
                pass
    
    # Priority 6: Dollar amounts $X (last one in text)
    dollar_matches = re.findall(r'\\?\$\s*([-+]?\d+(?:,\d{3})*(?:\.\d+)?)', content_clean)
    if dollar_matches:
        try:
            return float(dollar_matches[-1].replace(',', ''))
        except:
            pass
    
    # Priority 7: Look at the LAST FEW SENTENCES with smart context matching
    sentences = re.split(r'[.\n]+', content_clean)
    # Get last 3 non-empty sentences
    last_sentences = [s.strip() for s in reversed(sentences) if s.strip()][:3]
    
    for sentence in last_sentences:
        # Look for contextual patterns in order of reliability
        
        # Pattern 1: "is X miles/dollars/etc" - most reliable for conclusions
        is_match = re.findall(r'\bis\s+\\?\$?([-+]?\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:miles?|dollars?|sheep|bolts?|eggs?|downloads?|jewels?|loaves?)?', sentence, re.IGNORECASE)
        if is_match:
            try:
                return float(is_match[-1].replace(',', ''))
            except:
                pass
        
        # Pattern 2: "X unit" at the end of sentence - e.g., "total is 366 downloads"
        end_match = re.findall(r'([-+]?\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:miles?|dollars?|sheep|bolts?|eggs?|downloads?|jewels?|days?|hours?|loaves?|cups?|in total|total)?\s*$', sentence, re.IGNORECASE)
        if end_match:
            try:
                num = float(end_match[-1].replace(',', ''))
                if 0 < num < 10000000:
                    return num
            except:
                pass
        
        # Pattern 3: Dollar amount in sentence
        dollar_match = re.findall(r'\\?\$\s*([-+]?\d+(?:,\d{3})*(?:\.\d+)?)', sentence)
        if dollar_match:
            try:
                return float(dollar_match[-1].replace(',', ''))
            except:
                pass
        
        # Pattern 4: "= X" calculation result in sentence
        eq_match = re.findall(r'=\s*\\?\$?([-+]?\d+(?:,\d{3})*(?:\.\d+)?)', sentence)
        if eq_match:
            try:
                return float(eq_match[-1].replace(',', ''))
            except:
                pass
        
        # Pattern 5: Ordinal-like "Xth place" 
        ordinal_match = re.findall(r'([-+]?\d+)(?:st|nd|rd|th)\s+place', sentence, re.IGNORECASE)
        if ordinal_match:
            try:
                return float(ordinal_match[-1])
            except:
                pass
    
    # Priority 8: Last standalone number (ultimate fallback)
    last_num_matches = re.findall(r'([-+]?\d+(?:,\d{3})*(?:\.\d+)?)\s*\.?\s*$', content_clean, re.MULTILINE)
    if last_num_matches:
        try:
            num = float(last_num_matches[-1].replace(',', ''))
            if 0 < num < 10000000:  # Reasonable range
                return num
        except:
            pass
    
    return None

def check_answer_correct(extracted, ground_truth_answer):
    """Check if extracted answer matches ground truth (with tolerance)"""
    if extracted is None or ground_truth_answer is None:
        return False
    
    # Allow small floating point tolerance
    return abs(extracted - ground_truth_answer) < 0.01

def analyze_json_file(filepath):
    """Analyze a single JSON result file"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return None
    
    result = {
        "question_id": data.get("input", {}).get("question_id"),
        "success": data.get("response", {}).get("success", False),
        "content": data.get("response", {}).get("content"),
        "ground_truth": data.get("input", {}).get("ground_truth"),
        "persona_type": data.get("input", {}).get("persona", {}).get("type"),
        "persona_id": data.get("input", {}).get("persona", {}).get("id"),
        "model_id": data.get("metadata", {}).get("model_info", {}).get("model_id"),
        "latency_sec": data.get("response", {}).get("performance", {}).get("latency_sec", 0),
        "retry_count": data.get("response", {}).get("performance", {}).get("retry_count", 0),
    }
    
    # Extract tokens
    tokens = data.get("response", {}).get("performance", {}).get("tokens", {})
    if not tokens:
        # Try from raw_api_response
        raw_api = data.get("response", {}).get("raw_api_response")
        if raw_api and isinstance(raw_api, dict):
            usage = raw_api.get("usage", {})
            tokens = usage if usage else {}
    
    result["prompt_tokens"] = (tokens.get("prompt_tokens") or 0) if tokens else 0
    result["completion_tokens"] = (tokens.get("completion_tokens") or 0) if tokens else 0
    result["total_tokens"] = (tokens.get("total_tokens") or 0) if tokens else 0
    result["reasoning_tokens"] = (tokens.get("reasoning_tokens") or 0) if tokens else 0
    
    # If reasoning_tokens not in tokens, check completion_tokens_details
    if result["reasoning_tokens"] == 0 and tokens:
        completion_details = tokens.get("completion_tokens_details", {})
        if completion_details:
            result["reasoning_tokens"] = completion_details.get("reasoning_tokens", 0) or 0
    
    # Extract answers and check correctness
    result["ground_truth_answer"] = extract_answer_from_ground_truth(result["ground_truth"])
    result["extracted_answer"] = extract_answer_from_response(result["content"])
    result["is_correct"] = check_answer_correct(result["extracted_answer"], result["ground_truth_answer"])
    
    return result

def scan_results():
    """Scan all results folders and collect data"""
    all_data = []
    
    for model in MODELS:
        model_path = os.path.join(RESULTS_DIR, model)
        if not os.path.isdir(model_path):
            print(f"  ⚠️ Model folder not found: {model}")
            continue
        
        # Get all persona folders
        for persona_folder in os.listdir(model_path):
            persona_path = os.path.join(model_path, persona_folder)
            if not os.path.isdir(persona_path):
                continue
            
            # Normalize persona name
            persona_normalized = persona_folder.replace("_v2", "").rstrip("s")
            if persona_normalized.endswith("_explicit"):
                persona_normalized = persona_normalized
            elif persona_normalized.endswith("_implicit"):
                persona_normalized = persona_normalized
            
            # Scan JSON files
            json_files = [f for f in os.listdir(persona_path) if f.endswith(".json")]
            print(f"  📁 {model}/{persona_folder}: {len(json_files)} files")
            
            for json_file in json_files:
                filepath = os.path.join(persona_path, json_file)
                result = analyze_json_file(filepath)
                
                if result:
                    result["model_folder"] = model
                    result["persona_folder"] = persona_folder
                    result["persona_normalized"] = persona_normalized
                    all_data.append(result)
    
    return all_data

def generate_model_csv(all_data):
    """Generate CSV per model"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Group by model
    by_model = defaultdict(list)
    for item in all_data:
        by_model[item["model_folder"]].append(item)
    
    for model, items in by_model.items():
        filename = f"{OUTPUT_DIR}/model_{model.replace('/', '_')}.csv"
        
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "question_id", "persona_folder", "persona_type", "success", "is_correct",
                "ground_truth_answer", "extracted_answer", 
                "prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens",
                "latency_sec", "retry_count"
            ])
            
            for item in sorted(items, key=lambda x: (x["persona_folder"], x["question_id"] or 0)):
                writer.writerow([
                    item["question_id"],
                    item["persona_folder"],
                    item["persona_type"],
                    item["success"],
                    item["is_correct"],
                    item["ground_truth_answer"],
                    item["extracted_answer"],
                    item["prompt_tokens"],
                    item["completion_tokens"],
                    item["reasoning_tokens"],
                    item["total_tokens"],
                    round(item["latency_sec"], 3) if item["latency_sec"] else 0,
                    item["retry_count"]
                ])
        
        print(f"  ✅ {filename}")

def generate_persona_csv(all_data):
    """Generate CSV per persona"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Group by persona folder
    by_persona = defaultdict(list)
    for item in all_data:
        by_persona[item["persona_folder"]].append(item)
    
    for persona, items in by_persona.items():
        filename = f"{OUTPUT_DIR}/persona_{persona}.csv"
        
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "question_id", "model_folder", "model_id", "success", "is_correct",
                "ground_truth_answer", "extracted_answer",
                "prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens",
                "latency_sec", "retry_count"
            ])
            
            for item in sorted(items, key=lambda x: (x["model_folder"], x["question_id"] or 0)):
                writer.writerow([
                    item["question_id"],
                    item["model_folder"],
                    item["model_id"],
                    item["success"],
                    item["is_correct"],
                    item["ground_truth_answer"],
                    item["extracted_answer"],
                    item["prompt_tokens"],
                    item["completion_tokens"],
                    item["reasoning_tokens"],
                    item["total_tokens"],
                    round(item["latency_sec"], 3) if item["latency_sec"] else 0,
                    item["retry_count"]
                ])
        
        print(f"  ✅ {filename}")

def generate_summary_csv(all_data):
    """Generate summary CSV comparing all models and personas"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Group by model + persona
    summary = defaultdict(lambda: {
        "total": 0,
        "success": 0,
        "correct": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
        "latency_sum": 0,
        "retry_count": 0,
    })
    
    for item in all_data:
        key = (item["model_folder"], item["persona_folder"])
        summary[key]["total"] += 1
        if item["success"]:
            summary[key]["success"] += 1
        if item["is_correct"]:
            summary[key]["correct"] += 1
        summary[key]["prompt_tokens"] += item["prompt_tokens"] or 0
        summary[key]["completion_tokens"] += item["completion_tokens"] or 0
        summary[key]["reasoning_tokens"] += item["reasoning_tokens"] or 0
        summary[key]["total_tokens"] += item["total_tokens"] or 0
        summary[key]["latency_sum"] += item["latency_sec"] or 0
        summary[key]["retry_count"] += item["retry_count"] or 0
    
    # Write summary CSV
    filename = f"{OUTPUT_DIR}/summary_all.csv"
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "persona", "total_questions", 
            "success_count", "success_rate",
            "correct_count", "accuracy",
            "avg_prompt_tokens", "avg_completion_tokens", "avg_reasoning_tokens", "avg_total_tokens",
            "total_prompt_tokens", "total_completion_tokens", "total_reasoning_tokens", "total_all_tokens",
            "avg_latency_sec", "total_retries"
        ])
        
        for (model, persona), stats in sorted(summary.items()):
            total = stats["total"]
            if total == 0:
                continue
                
            writer.writerow([
                model,
                persona,
                total,
                stats["success"],
                round(stats["success"] / total * 100, 2),
                stats["correct"],
                round(stats["correct"] / total * 100, 2),
                round(stats["prompt_tokens"] / total, 2),
                round(stats["completion_tokens"] / total, 2),
                round(stats["reasoning_tokens"] / total, 2),
                round(stats["total_tokens"] / total, 2),
                stats["prompt_tokens"],
                stats["completion_tokens"],
                stats["reasoning_tokens"],
                stats["total_tokens"],
                round(stats["latency_sum"] / total, 3),
                stats["retry_count"]
            ])
    
    print(f"  ✅ {filename}")
    
    # Also create model-only summary
    model_summary = defaultdict(lambda: {
        "total": 0, "success": 0, "correct": 0,
        "prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "total_tokens": 0,
        "latency_sum": 0, "retry_count": 0,
    })
    
    for item in all_data:
        key = item["model_folder"]
        model_summary[key]["total"] += 1
        if item["success"]:
            model_summary[key]["success"] += 1
        if item["is_correct"]:
            model_summary[key]["correct"] += 1
        model_summary[key]["prompt_tokens"] += item["prompt_tokens"] or 0
        model_summary[key]["completion_tokens"] += item["completion_tokens"] or 0
        model_summary[key]["reasoning_tokens"] += item["reasoning_tokens"] or 0
        model_summary[key]["total_tokens"] += item["total_tokens"] or 0
        model_summary[key]["latency_sum"] += item["latency_sec"] or 0
        model_summary[key]["retry_count"] += item["retry_count"] or 0
    
    filename = f"{OUTPUT_DIR}/summary_by_model.csv"
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "total_questions",
            "success_count", "success_rate",
            "correct_count", "accuracy",
            "avg_prompt_tokens", "avg_completion_tokens", "avg_reasoning_tokens", "avg_total_tokens",
            "total_prompt_tokens", "total_completion_tokens", "total_reasoning_tokens", "total_all_tokens",
            "avg_latency_sec", "total_retries"
        ])
        
        for model, stats in sorted(model_summary.items()):
            total = stats["total"]
            if total == 0:
                continue
            writer.writerow([
                model, total,
                stats["success"], round(stats["success"] / total * 100, 2),
                stats["correct"], round(stats["correct"] / total * 100, 2),
                round(stats["prompt_tokens"] / total, 2),
                round(stats["completion_tokens"] / total, 2),
                round(stats["reasoning_tokens"] / total, 2),
                round(stats["total_tokens"] / total, 2),
                stats["prompt_tokens"], stats["completion_tokens"],
                stats["reasoning_tokens"], stats["total_tokens"],
                round(stats["latency_sum"] / total, 3),
                stats["retry_count"]
            ])
    
    print(f"  ✅ {filename}")
    
    # Persona-only summary
    persona_summary = defaultdict(lambda: {
        "total": 0, "success": 0, "correct": 0,
        "prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "total_tokens": 0,
        "latency_sum": 0, "retry_count": 0,
    })
    
    for item in all_data:
        key = item["persona_folder"]
        persona_summary[key]["total"] += 1
        if item["success"]:
            persona_summary[key]["success"] += 1
        if item["is_correct"]:
            persona_summary[key]["correct"] += 1
        persona_summary[key]["prompt_tokens"] += item["prompt_tokens"] or 0
        persona_summary[key]["completion_tokens"] += item["completion_tokens"] or 0
        persona_summary[key]["reasoning_tokens"] += item["reasoning_tokens"] or 0
        persona_summary[key]["total_tokens"] += item["total_tokens"] or 0
        persona_summary[key]["latency_sum"] += item["latency_sec"] or 0
        persona_summary[key]["retry_count"] += item["retry_count"] or 0
    
    filename = f"{OUTPUT_DIR}/summary_by_persona.csv"
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "persona", "total_questions",
            "success_count", "success_rate",
            "correct_count", "accuracy",
            "avg_prompt_tokens", "avg_completion_tokens", "avg_reasoning_tokens", "avg_total_tokens",
            "total_prompt_tokens", "total_completion_tokens", "total_reasoning_tokens", "total_all_tokens",
            "avg_latency_sec", "total_retries"
        ])
        
        for persona, stats in sorted(persona_summary.items()):
            total = stats["total"]
            if total == 0:
                continue
            writer.writerow([
                persona, total,
                stats["success"], round(stats["success"] / total * 100, 2),
                stats["correct"], round(stats["correct"] / total * 100, 2),
                round(stats["prompt_tokens"] / total, 2),
                round(stats["completion_tokens"] / total, 2),
                round(stats["reasoning_tokens"] / total, 2),
                round(stats["total_tokens"] / total, 2),
                stats["prompt_tokens"], stats["completion_tokens"],
                stats["reasoning_tokens"], stats["total_tokens"],
                round(stats["latency_sum"] / total, 3),
                stats["retry_count"]
            ])
    
    print(f"  ✅ {filename}")

def main():
    print("📊 GENERATING ANALYSIS CSV FILES")
    print("=" * 60)
    print(f"📅 Generated at: {datetime.now().isoformat()}")
    print()
    
    # Scan all results
    print("📂 Scanning results...")
    all_data = scan_results()
    print(f"\n📊 Total records: {len(all_data)}")
    print()
    
    # Generate CSVs
    print("📝 Generating model CSVs...")
    generate_model_csv(all_data)
    print()
    
    print("📝 Generating persona CSVs...")
    generate_persona_csv(all_data)
    print()
    
    print("📝 Generating summary CSVs...")
    generate_summary_csv(all_data)
    print()
    
    print("=" * 60)
    print("✅ ANALYSIS COMPLETE!")
    print(f"📁 Output directory: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
