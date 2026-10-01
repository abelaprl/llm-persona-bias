"""Rebuild results.csv + accuracy.csv for the GSM8K Regional Cultural Persona experiment.

Reuses the battle-tested GSM8K answer extractor from ``rebuild_results_csv_ministrail.py``
(extract_final_answer, normalize_number, get_response_text, truncation/marker helpers) so the
extraction logic stays identical to the rest of the project. This script only adds the
regional persona list, a per-persona accuracy summary, and sensible default paths.

Usage (from D:\\TugasAkhir):
    python src/pipelines/rebuild/rebuild_results_csv_regional_cultural.py
    python src/pipelines/rebuild/rebuild_results_csv_regional_cultural.py --input_root <model_dir> --expected 10
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import sys
from collections import Counter

# --- Reuse the proven extractor from the ministral rebuild script -------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MIN_PATH = os.path.join(_THIS_DIR, "rebuild_results_csv_ministrail.py")
_spec = importlib.util.spec_from_file_location("_rebuild_ministrail", _MIN_PATH)
_min = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_min)

normalize_number = _min.normalize_number
get_response_text = _min.get_response_text
extract_final_answer = _min.extract_final_answer
is_truly_truncated = _min.is_truly_truncated
has_final_marker = _min.has_final_marker

# --- Regional persona taxonomy (4 regions x [1 explicit + 3 implicit]) ---------------
REGIONAL_PERSONA_LIST = [
    "western_explicit", "western_implicit_1", "western_implicit_2", "western_implicit_3",
    "east_asian_explicit", "east_asian_implicit_1", "east_asian_implicit_2", "east_asian_implicit_3",
    "south_asian_explicit", "south_asian_implicit_1", "south_asian_implicit_2", "south_asian_implicit_3",
    "african_explicit", "african_implicit_1", "african_implicit_2", "african_implicit_3",
]

DEFAULT_MODEL_DIR = os.path.join(
    "result", "GMS8K", "regional_cultural_persona", "deepseek_deepseek-r1-distill-qwen-32b"
)

RESULTS_HEADER = [
    "persona_folder", "file", "question_id", "dataset", "ground_truth", "ground_truth_answer",
    "response_content_full", "extracted_answer", "is_correct", "has_boxed", "has_hashes",
    "boxed_match", "hashes_match", "extraction_issue", "error_category", "extraction_reason",
    "finish_reason", "prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens",
    "truncated_by_metadata", "truncated_by_content", "has_final_marker", "text_tail",
]


def safe_get(d, key, default=None):
    return d[key] if isinstance(d, dict) and key in d else default


def parse_args():
    p = argparse.ArgumentParser(description="Rebuild regional cultural persona GSM8K CSVs")
    p.add_argument("--input_root", default=DEFAULT_MODEL_DIR,
                   help="Model folder containing <persona>/gsm8k_*.json")
    p.add_argument("--report_dir", default=None,
                   help="Where to write results.csv/accuracy.csv (default: same as input_root)")
    p.add_argument("--expected", type=int, default=0,
                   help="Expected files per persona; warns on mismatch (0 = skip check)")
    return p.parse_args()


def process_file(persona, fpath, input_root):
    """Return a results row + (is_correct, no_extract, truncated) flags for one JSON file."""
    with open(fpath, encoding="utf-8") as jf:
        data = json.load(jf)

    inp = data.get("input", {})
    resp = data.get("response", {})
    req = data.get("request", {})

    gt = inp.get("ground_truth", "")
    gt_num = None
    m = re.search(r"####\s*([-+]?[$]?[\d,]+(?:\.\d+)?)", gt)
    if m:
        gt_num = normalize_number(m.group(1))
    qid = inp.get("question_id", "")
    dataset = inp.get("dataset", "")

    content = get_response_text(resp)
    extracted, extraction_reason = extract_final_answer(content)
    is_correct = (extracted == gt_num) if (extracted is not None and gt_num is not None) else False

    has_boxed = bool(re.search(r"\\boxed\{[^}]+\}", content))
    has_hashes = bool(re.search(r"####\s*-?[$]?[\d,]+", content))
    boxed_match = has_boxed and (extracted == gt_num) and extraction_reason == "boxed"
    hashes_match = has_hashes and (extracted == gt_num) and extraction_reason == "hashes"

    raw_api = safe_get(resp, "raw_api_response", {})
    choices = safe_get(raw_api, "choices", [{}])
    finish_reason = None
    if choices and isinstance(choices, list):
        finish_reason = safe_get(choices[0], "finish_reason", None)
    max_tokens = safe_get(safe_get(req, "parameters", {}), "max_tokens", None)
    usage = safe_get(raw_api, "usage", {})
    completion_tokens = safe_get(usage, "completion_tokens", None)
    prompt_tokens = safe_get(usage, "prompt_tokens", None)
    total_tokens = safe_get(usage, "total_tokens", None)
    reasoning_tokens = (safe_get(usage, "reasoning_tokens", None)
                        or safe_get(safe_get(usage, "completion_tokens_details", {}) or {}, "reasoning_tokens", None))

    truncated_by_metadata = (finish_reason == "length")
    truncated_by_content = is_truly_truncated(content, finish_reason, completion_tokens, max_tokens)
    has_final = has_final_marker(content)
    text_tail = content.strip()[-120:] if content else ""
    extraction_issue = (extracted is None)

    if extracted is not None:
        error_category = "ok"
    elif truncated_by_content:
        error_category = "truncated_output"
    else:
        error_category = "no_number_found"

    row = [persona, os.path.relpath(fpath, input_root), qid, dataset, gt, gt_num, content,
           extracted, is_correct, has_boxed, has_hashes, boxed_match, hashes_match,
           extraction_issue, error_category, extraction_reason, finish_reason, prompt_tokens,
           completion_tokens, reasoning_tokens, total_tokens, truncated_by_metadata,
           truncated_by_content, has_final, text_tail]
    no_extract = 1 if extracted is None else 0
    truncated = 1 if truncated_by_content else 0
    return row, error_category, is_correct, no_extract, truncated


def main():
    args = parse_args()
    report_dir = args.report_dir or args.input_root
    os.makedirs(report_dir, exist_ok=True)

    if not os.path.isdir(args.input_root):
        print(f"ERROR: input_root not found: {args.input_root}", file=sys.stderr)
        sys.exit(1)

    persona_dirs = sorted(
        p for p in os.listdir(args.input_root)
        if os.path.isdir(os.path.join(args.input_root, p)) and p != "warmup"
    )

    all_rows = []
    error_counter = Counter()
    # persona -> [correct, total, no_extract, truncated]
    acc = {}

    for persona in persona_dirs:
        folder = os.path.join(args.input_root, persona)
        files = sorted(f for f in os.listdir(folder) if f.startswith("gsm8k_") and f.endswith(".json"))
        if args.expected and len(files) != args.expected:
            print(f"WARNING: persona {persona} has {len(files)} files (expected {args.expected})")
        acc.setdefault(persona, [0, 0, 0, 0])
        for fname in files:
            fpath = os.path.join(folder, fname)
            try:
                row, cat, is_correct, no_extract, truncated = process_file(persona, fpath, args.input_root)
            except Exception as e:
                print(f"SKIP: {persona}/{fname} (invalid/empty JSON) -- {e}", file=sys.stderr)
                continue
            all_rows.append(row)
            error_counter[(persona, cat)] += 1
            acc[persona][0] += 1 if is_correct else 0
            acc[persona][1] += 1
            acc[persona][2] += no_extract
            acc[persona][3] += truncated

    # --- results.csv -----------------------------------------------------------------
    out_csv = os.path.join(report_dir, "results.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL, escapechar="\\", doublequote=True)
        writer.writerow(RESULTS_HEADER)
        writer.writerows(all_rows)
    print(f"Wrote {len(all_rows)} rows to {out_csv}")

    # --- accuracy.csv ----------------------------------------------------------------
    acc_csv = os.path.join(report_dir, "accuracy.csv")
    tot_correct = tot_total = tot_no = tot_trunc = 0
    # Order: known regional personas first (as defined), then any extras found on disk
    ordered = [p for p in REGIONAL_PERSONA_LIST if p in acc] + [p for p in persona_dirs if p not in REGIONAL_PERSONA_LIST]
    with open(acc_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["persona", "correct", "total", "accuracy", "no_extract", "truncated"])
        for persona in ordered:
            correct, total, no_ex, trunc = acc[persona]
            accuracy = round(100.0 * correct / total, 4) if total else 0.0
            writer.writerow([persona, correct, total, accuracy, no_ex, trunc])
            tot_correct += correct
            tot_total += total
            tot_no += no_ex
            tot_trunc += trunc
        overall = round(100.0 * tot_correct / tot_total, 4) if tot_total else 0.0
        writer.writerow(["OVERALL", tot_correct, tot_total, overall, tot_no, tot_trunc])
    print(f"Wrote accuracy summary to {acc_csv}")

    # --- console summary -------------------------------------------------------------
    print("\nPer-persona accuracy:")
    for persona in ordered:
        correct, total, no_ex, trunc = acc[persona]
        accuracy = round(100.0 * correct / total, 2) if total else 0.0
        print(f"  {persona:<24} {correct:>3}/{total:<3}  {accuracy:6.2f}%  no_extract={no_ex} truncated={trunc}")
    print(f"  {'OVERALL':<24} {tot_correct:>3}/{tot_total:<3}  {overall:6.2f}%")


if __name__ == "__main__":
    main()
