from __future__ import annotations

import csv
import json
import re
from pathlib import Path
import argparse
from typing import Any, Dict, List, Optional


def load_raw_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_final_answer(text: str) -> Optional[str]:
    """
    Extract numeric answer from 'Final answer: <number>' or 'Final answer: number' paragraph.
    Handles markdown formatting like **number** and trailing punctuation.
    Returns the number as string, or None if not found.
    """
    # Look for pattern: Final answer: (optional formatting)number(optional formatting/punctuation)
    # Remove common markdown like ** and trailing . or ,
    pattern = r"Final answer:\s*\*{0,2}<?([0-9,\.]+)>?\*{0,2}[.,]?"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        # Remove commas and return
        return match.group(1).replace(",", "")
    return None


def normalize_answer(ans: str) -> str:
    """
    Normalize answer for comparison (strip whitespace, lowercase, remove commas).
    For GSM8K gold answers, extract final number after #### marker.
    """
    # GSM8K gold format often ends with #### number
    if "####" in ans:
        # Extract number after ####
        parts = ans.split("####")
        if len(parts) > 1:
            ans = parts[-1]
    return ans.strip().replace(",", "").lower()


def try_write_xlsx(rows, out_xlsx: Path) -> None:
    try:
        import pandas as pd  # type: ignore
    except Exception:
        print(f"Pandas not available; skipping XLSX: {out_xlsx}")
        return
    df = pd.DataFrame(rows)
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_excel(out_xlsx, index=False)
        print(f"XLSX written to: {out_xlsx}")
    except Exception as e:
        print(f"Failed to write XLSX: {e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse GSM8K raw outputs to CSV/XLSX")
    parser.add_argument("--raw-dir", default="results/woman_implicits/gsm8k_raw", help="Directory containing per-question JSON files")
    parser.add_argument("--out-csv", default="results/woman_implicits/gsm8k_results.csv", help="Path for output CSV")
    parser.add_argument("--out-xlsx", default="results/woman_implicits/gsm8k_results.xlsx", help="Path for output XLSX")
    return parser.parse_args()


def main():
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    output_csv = Path(args.out_csv)
    output_xlsx = Path(args.out_xlsx)
    
    if not raw_dir.exists():
        print(f"Error: {raw_dir} does not exist. Run batch script first.")
        return
    
    # Collect all question JSON files (skip summary/warmup/echo)
    files = sorted([f for f in raw_dir.glob("*.json") if f.name not in {"run_summary.json", "persona_warmup.json", "persona_echo.json"}])
    
    if not files:
        print(f"No gsm8k_*.json files found in {raw_dir}")
        return
    
    print(f"Parsing {len(files)} GSM8K result files...")
    
    rows = []
    parse_errors = 0
    
    for i, fpath in enumerate(files, start=1):
        print(f"Parsing: {i}/{len(files)} - {fpath.name}")
        data = load_raw_json(fpath)
        
        run = data.get("run", {})
        model_id = run.get("model_id", "")
        question_id = run.get("question_id", "")
        latency = run.get("latency_sec", 0.0)
        
        inp = data.get("input", {})
        gold = inp.get("gold_answer", "")
        question = inp.get("question", "")
        
        response_text = data.get("response_text", "")
        
        # Extract predicted answer
        predicted = extract_final_answer(response_text)
        
        # Determine correctness
        parse_error_flag = 0
        is_correct = 0
        
        if predicted is None:
            parse_error_flag = 1
            predicted_str = ""
        else:
            predicted_str = predicted
            # Normalize for comparison
            if normalize_answer(predicted_str) == normalize_answer(gold):
                is_correct = 1
        
        if parse_error_flag:
            parse_errors += 1
        
        rows.append({
            "model_id": model_id,
            "question_id": question_id,
            "question": question,
            "gold_answer": gold,
            "predicted_answer": predicted_str,
            "is_correct": is_correct,
            "reasoning_text": response_text,
            "latency_sec": latency,
            "parse_error_flag": parse_error_flag,
        })
    
    # Write CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "model_id", "question_id", "question", "gold_answer", "predicted_answer",
            "is_correct", "reasoning_text", "latency_sec", "parse_error_flag"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"\nDone: {len(files)} questions parsed")
    print(f"Parse errors: {parse_errors}")
    print(f"CSV written to: {output_csv}")
    # Try XLSX
    try_write_xlsx(rows, output_xlsx)
    
    # Summary stats
    correct = sum(r["is_correct"] for r in rows)
    total = len(rows)
    accuracy = (correct / total * 100) if total > 0 else 0
    print(f"Accuracy: {correct}/{total} = {accuracy:.1f}%")


if __name__ == "__main__":
    main()
