#!/usr/bin/env python3
"""
Benchmark all GSM8K rebuild pipelines against the Religion Persona dataset
(deepseek_deepseek-r1-distill-qwen-32b).

For each rebuild script:
  1. Run it with --input_root / --report_dir to produce results.csv
  2. Run calc_accuracy.py to produce accuracy.csv
  3. Copy outputs to result/reports/GSM8K/ with the required filename format

Outputs:
  deepseek_deepseek-r1-distill-qwen-32b_religion_persona_<rebuild_name>.csv
  deepseek_deepseek-r1-distill-qwen-32b_religion_persona_<rebuild_name>_accuracy.csv
  benchmark_comparison.csv
"""

import subprocess
import sys
import os
import csv
import shutil
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────────
REPO_ROOT   = Path(r"D:\TugasAkhir")
DATA_ROOT   = REPO_ROOT / "result" / "GMS8K" / "Religion_Persona" / "deepseek_deepseek-r1-distill-qwen-32b"
OUTPUT_ROOT = REPO_ROOT / "result" / "reports" / "GSM8K"
SRC_REBUILD = REPO_ROOT / "src" / "rebuild"
CALC_ACC    = REPO_ROOT / "src" / "analysis" / "calc_accuracy.py"
VENV_PY     = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
MODEL_PREFIX = "deepseek_deepseek-r1-distill-qwen-32b_religion_persona"

# Distinct rebuild scripts to benchmark (excludes copies, ETHICS scripts,
# age-group-only scripts, and failure-repair scripts)
REBUILD_SCRIPTS = [
    "rebuild_results_csv_gemini",
    "rebuild_results_csv_llama",
    "rebuild_results_csv_qwen",
    "rebuild_results_csv_gemma",
    "rebuild_results_csv_gemma9b",
    "rebuild_results_csv_gpt",
    "rebuild_results_csv_ministrail",
    "rebuild_results_csv_olmo",
]

# ── Helpers ──────────────────────────────────────────────────────────────────
def run(cmd, label=""):
    print(f"\n{'='*72}")
    print(f"  {label}")
    print(f"  CMD: {' '.join(str(c) for c in cmd)}")
    print(f"{'='*72}")
    result = subprocess.run(
        [str(c) for c in cmd],
        capture_output=False,   # stream stdout/stderr live
        text=True,
    )
    return result.returncode

def read_accuracy_csv(path):
    """Return the OVERALL row from accuracy.csv as a dict, or None."""
    try:
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        for row in rows:
            if row.get("persona", "").strip().upper() == "OVERALL":
                return row
        # fall back to last row
        return rows[-1] if rows else None
    except Exception as e:
        print(f"  [WARN] Could not read {path}: {e}")
        return None

# ── Main benchmark loop ──────────────────────────────────────────────────────
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

summary_rows = []

for script_name in REBUILD_SCRIPTS:
    script_path = SRC_REBUILD / f"{script_name}.py"
    if not script_path.exists():
        print(f"\n[SKIP] {script_name}: script not found at {script_path}")
        continue

    tmp_dir = OUTPUT_ROOT / f"_tmp_{script_name}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Run rebuild script ───────────────────────────────────────────
    rebuild_rc = run(
        [VENV_PY, script_path, "--input_root", DATA_ROOT, "--report_dir", tmp_dir],
        label=f"REBUILD: {script_name}"
    )

    src_results = tmp_dir / "results.csv"
    if not src_results.exists():
        print(f"  [ERROR] {script_name}: results.csv not produced (rc={rebuild_rc})")
        summary_rows.append({
            "rebuild_script": script_name,
            "status": "FAILED_REBUILD",
            "total_rows": 0,
            "overall_accuracy": "",
            "correct": "",
            "total": "",
            "no_extract": "",
            "truncated": "",
        })
        continue

    # ── Step 2: Copy results CSV ─────────────────────────────────────────────
    dst_results = OUTPUT_ROOT / f"{MODEL_PREFIX}_{script_name}.csv"
    shutil.copy2(src_results, dst_results)
    print(f"  Copied results → {dst_results.name}")

    # Count rows in results CSV
    total_rows = 0
    try:
        with open(src_results, encoding="utf-8") as f:
            total_rows = sum(1 for _ in csv.DictReader(f))
    except Exception:
        pass

    # ── Step 3: Run calc_accuracy.py ─────────────────────────────────────────
    acc_rc = run(
        [VENV_PY, CALC_ACC, "--csv_path", src_results, "--out_dir", tmp_dir, "--max_rows", "1320"],
        label=f"ACCURACY: {script_name}"
    )

    src_acc = tmp_dir / "accuracy.csv"
    if not src_acc.exists():
        print(f"  [ERROR] {script_name}: accuracy.csv not produced (rc={acc_rc})")
        summary_rows.append({
            "rebuild_script": script_name,
            "status": "FAILED_ACCURACY",
            "total_rows": total_rows,
            "overall_accuracy": "",
            "correct": "",
            "total": "",
            "no_extract": "",
            "truncated": "",
        })
        continue

    # ── Step 4: Copy accuracy CSV ────────────────────────────────────────────
    dst_acc = OUTPUT_ROOT / f"{MODEL_PREFIX}_{script_name}_accuracy.csv"
    shutil.copy2(src_acc, dst_acc)
    print(f"  Copied accuracy → {dst_acc.name}")

    # ── Step 5: Read OVERALL row ─────────────────────────────────────────────
    overall = read_accuracy_csv(src_acc)
    if overall:
        summary_rows.append({
            "rebuild_script": script_name,
            "status": "OK",
            "total_rows": total_rows,
            "overall_accuracy": overall.get("accuracy", ""),
            "correct": overall.get("correct", ""),
            "total": overall.get("total", ""),
            "no_extract": overall.get("no_extract", ""),
            "truncated": overall.get("truncated", ""),
            "avg_prompt_tokens": overall.get("avg_prompt_tokens", ""),
            "avg_completion_tokens": overall.get("avg_completion_tokens", ""),
            "avg_reasoning_tokens": overall.get("avg_reasoning_tokens", ""),
        })
    else:
        summary_rows.append({
            "rebuild_script": script_name,
            "status": "NO_OVERALL",
            "total_rows": total_rows,
        })

# ── Produce comparison table ─────────────────────────────────────────────────
print("\n\n" + "="*80)
print("  BENCHMARK COMPARISON TABLE")
print("="*80)
print(f"  Dataset : {DATA_ROOT}")
print(f"  Model   : deepseek_deepseek-r1-distill-qwen-32b")
print(f"  Personas: 16 religion personas (4 religions × explicit + 3 implicit)")
print("="*80)

# Sort by accuracy descending
def sort_key(r):
    try:
        return -float(r.get("overall_accuracy", 0) or 0)
    except Exception:
        return 0

summary_rows_ok = [r for r in summary_rows if r.get("status") == "OK"]
summary_rows_ok.sort(key=sort_key)
summary_rows_fail = [r for r in summary_rows if r.get("status") != "OK"]

print(f"\n{'Rank':<6} {'Rebuild Script':<35} {'Accuracy':>9} {'Correct':>8} {'Total':>7} {'No-Extract':>11} {'Truncated':>10}")
print("-"*90)
for rank, row in enumerate(summary_rows_ok, 1):
    acc_pct = float(row.get("overall_accuracy", 0) or 0)
    print(
        f"{rank:<6} {row['rebuild_script']:<35} {acc_pct:>8.4f}% "
        f"{row.get('correct',''):>8} {row.get('total',''):>7} "
        f"{row.get('no_extract',''):>11} {row.get('truncated',''):>10}"
    )
if summary_rows_fail:
    print("\nFailed / incomplete:")
    for row in summary_rows_fail:
        print(f"  {row['rebuild_script']}: {row.get('status')}")

print("\n--- Analysis ---")
if summary_rows_ok:
    best  = summary_rows_ok[0]
    worst = summary_rows_ok[-1]
    best_acc  = float(best.get("overall_accuracy", 0) or 0)
    worst_acc = float(worst.get("overall_accuracy", 0) or 0)
    print(f"  Best pipeline  : {best['rebuild_script']} ({best_acc:.4f}%)")
    print(f"  Worst pipeline : {worst['rebuild_script']} ({worst_acc:.4f}%)")
    print(f"  Spread         : {best_acc - worst_acc:.4f} pp")

    # Detect ties
    accs = {}
    for r in summary_rows_ok:
        a = round(float(r.get("overall_accuracy", 0) or 0), 4)
        accs.setdefault(a, []).append(r["rebuild_script"])
    ties = {a: names for a, names in accs.items() if len(names) > 1}
    if ties:
        print("  Ties:")
        for a, names in sorted(ties.items(), reverse=True):
            print(f"    {a:.4f}%  →  {', '.join(names)}")
    else:
        print("  Ties: none")

    # Discrepancies
    if best_acc - worst_acc > 0.5:
        print(f"  Discrepancy    : significant ({best_acc - worst_acc:.4f} pp spread)")
    else:
        print(f"  Discrepancy    : small ({best_acc - worst_acc:.4f} pp spread)")

print("\n--- Recommendation ---")
if summary_rows_ok:
    best_name = best['rebuild_script']
    best_no_extract = int(best.get("no_extract", 0) or 0)
    best_total = int(best.get("total", 0) or 0)
    no_extract_pct = 100.0 * best_no_extract / best_total if best_total else 0
    print(f"  Use: {best_name}")
    print(f"  Reason: highest overall accuracy ({best_acc:.4f}%), "
          f"no-extract rate {no_extract_pct:.2f}% ({best_no_extract}/{best_total})")

# ── Write comparison CSV ─────────────────────────────────────────────────────
all_sorted = summary_rows_ok + summary_rows_fail
comp_csv = OUTPUT_ROOT / f"{MODEL_PREFIX}_comparison.csv"
fieldnames = [
    "rank", "rebuild_script", "status", "total_rows", "overall_accuracy",
    "correct", "total", "no_extract", "truncated",
    "avg_prompt_tokens", "avg_completion_tokens", "avg_reasoning_tokens"
]
with open(comp_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for rank, row in enumerate(summary_rows_ok, 1):
        writer.writerow({"rank": rank, **row})
    for row in summary_rows_fail:
        writer.writerow({"rank": "N/A", **row})

print(f"\n  Comparison table written → {comp_csv}")
print("  Done.")
