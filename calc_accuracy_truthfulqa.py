import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


PERSONA_ORDER = [
    "man_explicit_1",
    "man_implicit_1",
    "man_implicit_2",
    "man_implicit_3",
    "woman_explicit_1",
    "woman_implicit_1",
    "woman_implicit_2",
    "woman_implicit_3",
    "neutral",
]


def default_model_dir():
    root = Path("result") / "TRUTHFULQA"
    if not root.exists():
        return ""
    candidates = [p for p in root.iterdir() if p.is_dir() and p.name != "testing-20samples"]
    if not candidates:
        return ""
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return str(latest)


def numeric_value(value):
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return None


def load_json_safely(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        obj, _ = decoder.raw_decode(text.lstrip("\ufeff \t\r\n"))
        return obj


def gather_persona_stats(persona_dir):
    sums = defaultdict(float)
    total = 0
    truthful_count = 0
    skipped_files = 0
    token_sums = {
        "sum_prompt_tokens": 0.0,
        "sum_completion_tokens": 0.0,
        "sum_reasoning_tokens": 0.0,
        "sum_total_tokens": 0.0,
    }

    for fp in sorted(persona_dir.glob("*.json")):
        try:
            obj = load_json_safely(fp)
        except Exception as exc:
            skipped_files += 1
            print(f"[WARN] skip invalid JSON: {fp} ({exc})")
            continue

        analysis = obj.get("analysis", {}) or {}
        total += 1

        if bool(analysis.get("is_truthful", False)):
            truthful_count += 1

        for key, value in analysis.items():
            num = numeric_value(value)
            if num is not None:
                sums[key] += num

        perf_tokens = obj.get("response", {}).get("performance", {}).get("tokens", {})
        token_sums["sum_prompt_tokens"] += float(perf_tokens.get("prompt_tokens", 0) or 0)
        token_sums["sum_completion_tokens"] += float(perf_tokens.get("completion_tokens", 0) or 0)
        token_sums["sum_total_tokens"] += float(perf_tokens.get("total_tokens", 0) or 0)
        token_sums["sum_reasoning_tokens"] += float(
            perf_tokens.get("completion_tokens_details", {}).get("reasoning_tokens", 0) or 0
        )

    means = {}
    if total > 0:
        for key, val in sums.items():
            means[key] = val / total

    token_avgs = {
        "avg_prompt_tokens": (token_sums["sum_prompt_tokens"] / total) if total else 0.0,
        "avg_completion_tokens": (token_sums["sum_completion_tokens"] / total) if total else 0.0,
        "avg_reasoning_tokens": (token_sums["sum_reasoning_tokens"] / total) if total else 0.0,
        "avg_total_tokens": (token_sums["sum_total_tokens"] / total) if total else 0.0,
    }

    accuracy = (truthful_count / total) if total else 0.0
    return total, truthful_count, accuracy, sums, means, token_sums, token_avgs, skipped_files


def merge_overall(persona_results):
    total_all = sum(item["total"] for item in persona_results)
    truthful_all = sum(item["truthful_count"] for item in persona_results)
    sums_all = defaultdict(float)
    token_sums_all = {
        "sum_prompt_tokens": 0.0,
        "sum_completion_tokens": 0.0,
        "sum_reasoning_tokens": 0.0,
        "sum_total_tokens": 0.0,
    }

    for item in persona_results:
        for key, val in item["sums"].items():
            sums_all[key] += val
        for key in token_sums_all:
            token_sums_all[key] += item["token_sums"].get(key, 0.0)

    means_all = {}
    if total_all > 0:
        for key, val in sums_all.items():
            means_all[key] = val / total_all

    token_avgs_all = {
        "avg_prompt_tokens": (token_sums_all["sum_prompt_tokens"] / total_all) if total_all else 0.0,
        "avg_completion_tokens": (token_sums_all["sum_completion_tokens"] / total_all) if total_all else 0.0,
        "avg_reasoning_tokens": (token_sums_all["sum_reasoning_tokens"] / total_all) if total_all else 0.0,
        "avg_total_tokens": (token_sums_all["sum_total_tokens"] / total_all) if total_all else 0.0,
    }

    accuracy_all = (truthful_all / total_all) if total_all else 0.0
    return total_all, truthful_all, accuracy_all, sums_all, means_all, token_sums_all, token_avgs_all


def build_analysis_column_order(persona_results):
    keys = set()
    for item in persona_results:
        keys.update(item["means"].keys())

    priority = [
        "rouge_correct",
        "rouge_incorrect",
        "truth_score_rouge",
        "bleu_1_correct",
        "bleu_2_correct",
        "bleu_3_correct",
        "bleu_4_correct",
        "bleu_correct",
        "bleu_1_incorrect",
        "bleu_2_incorrect",
        "bleu_3_incorrect",
        "bleu_4_incorrect",
        "bleu_incorrect",
        "truth_score_bleu",
        "is_truthful",
        "prediction_length",
        "total_tokens",
        "reasoning_tokens",
        "latency_sec",
    ]

    ordered = [k for k in priority if k in keys]
    ordered += sorted(k for k in keys if k not in ordered)
    return ordered


def print_table(persona_results, overall, model_dir):
    header = (
        f"{'Persona':<20} {'Truthful':>8} {'Total':>6} {'Accuracy':>9} "
        f"{'truth_score_rouge':>17} {'truth_score_bleu':>16} {'latency_sec':>12}"
    )
    print(f"\nTRUTHFULQA Accuracy Summary From JSON ({model_dir})")
    print(header)
    print("-" * len(header))

    for item in persona_results:
        print(
            f"{item['persona']:<20} {item['truthful_count']:>8} {item['total']:>6} "
            f"{item['accuracy'] * 100:>8.2f}% {item['means'].get('truth_score_rouge', 0.0):>17.4f} "
            f"{item['means'].get('truth_score_bleu', 0.0):>16.4f} {item['means'].get('latency_sec', 0.0):>12.4f}"
        )

    print("-" * len(header))
    print(
        f"{'OVERALL':<20} {overall['truthful_count']:>8} {overall['total']:>6} "
        f"{overall['accuracy'] * 100:>8.2f}% {overall['means'].get('truth_score_rouge', 0.0):>17.4f} "
        f"{overall['means'].get('truth_score_bleu', 0.0):>16.4f} {overall['means'].get('latency_sec', 0.0):>12.4f}"
    )


def save_summary_csv(out_path, persona_results, overall, analysis_keys):
    prefix = [
        "persona_name",
        "truthful_count",
        "total",
        "accuracy",
        "sum_prompt_tokens",
        "sum_completion_tokens",
        "sum_reasoning_tokens",
        "sum_total_tokens",
        "avg_prompt_tokens",
        "avg_completion_tokens",
        "avg_reasoning_tokens",
        "avg_total_tokens",
    ]
    fieldnames = prefix + analysis_keys

    rows = []
    for item in persona_results:
        row = {
            "persona_name": item["persona"],
            "truthful_count": item["truthful_count"],
            "total": item["total"],
            "accuracy": round(item["accuracy"], 6),
            "sum_prompt_tokens": round(item["token_sums"].get("sum_prompt_tokens", 0.0), 6),
            "sum_completion_tokens": round(item["token_sums"].get("sum_completion_tokens", 0.0), 6),
            "sum_reasoning_tokens": round(item["token_sums"].get("sum_reasoning_tokens", 0.0), 6),
            "sum_total_tokens": round(item["token_sums"].get("sum_total_tokens", 0.0), 6),
            "avg_prompt_tokens": round(item["token_avgs"].get("avg_prompt_tokens", 0.0), 6),
            "avg_completion_tokens": round(item["token_avgs"].get("avg_completion_tokens", 0.0), 6),
            "avg_reasoning_tokens": round(item["token_avgs"].get("avg_reasoning_tokens", 0.0), 6),
            "avg_total_tokens": round(item["token_avgs"].get("avg_total_tokens", 0.0), 6),
        }
        for key in analysis_keys:
            row[key] = round(item["means"].get(key, 0.0), 6)
        rows.append(row)

    row = {
        "persona_name": "OVERALL",
        "truthful_count": overall["truthful_count"],
        "total": overall["total"],
        "accuracy": round(overall["accuracy"], 6),
        "sum_prompt_tokens": round(overall["token_sums"].get("sum_prompt_tokens", 0.0), 6),
        "sum_completion_tokens": round(overall["token_sums"].get("sum_completion_tokens", 0.0), 6),
        "sum_reasoning_tokens": round(overall["token_sums"].get("sum_reasoning_tokens", 0.0), 6),
        "sum_total_tokens": round(overall["token_sums"].get("sum_total_tokens", 0.0), 6),
        "avg_prompt_tokens": round(overall["token_avgs"].get("avg_prompt_tokens", 0.0), 6),
        "avg_completion_tokens": round(overall["token_avgs"].get("avg_completion_tokens", 0.0), 6),
        "avg_reasoning_tokens": round(overall["token_avgs"].get("avg_reasoning_tokens", 0.0), 6),
        "avg_total_tokens": round(overall["token_avgs"].get("avg_total_tokens", 0.0), 6),
    }
    for key in analysis_keys:
        row[key] = round(overall["means"].get(key, 0.0), 6)
    rows.append(row)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Calculate TRUTHFULQA accuracy and aggregate analysis metrics from JSON results."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=default_model_dir(),
        help="TRUTHFULQA model result folder, e.g. result/TRUTHFULQA/allenai_olmo-3.1-32b-instruct",
    )
    parser.add_argument(
        "--save_csv",
        action="store_true",
        help="Write summary CSV in model folder",
    )
    args = parser.parse_args()

    if not args.model_dir:
        raise SystemExit("No model_dir found. Use --model_dir to set TRUTHFULQA model folder.")

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise SystemExit(f"Model dir not found: {model_dir}")

    persona_results = []
    total_skipped = 0
    for persona in PERSONA_ORDER:
        pdir = model_dir / persona
        if not pdir.exists() or not pdir.is_dir():
            continue

        total, truthful_count, accuracy, sums, means, token_sums, token_avgs, skipped_files = gather_persona_stats(pdir)
        if total == 0:
            continue

        total_skipped += skipped_files
        persona_results.append({
            "persona": persona,
            "total": total,
            "truthful_count": truthful_count,
            "accuracy": accuracy,
            "sums": sums,
            "means": means,
            "token_sums": token_sums,
            "token_avgs": token_avgs,
        })

    if not persona_results:
        raise SystemExit(f"No persona JSON files found under: {model_dir}")

    total_all, truthful_all, acc_all, sums_all, means_all, token_sums_all, token_avgs_all = merge_overall(persona_results)
    overall = {
        "total": total_all,
        "truthful_count": truthful_all,
        "accuracy": acc_all,
        "sums": sums_all,
        "means": means_all,
        "token_sums": token_sums_all,
        "token_avgs": token_avgs_all,
    }

    analysis_keys = build_analysis_column_order(persona_results)
    print_table(persona_results, overall, model_dir)

    if total_skipped > 0:
        print(f"\n[WARN] skipped invalid JSON files: {total_skipped}")

    if args.save_csv:
        out_path = model_dir / "accuracy_truthfulqa_analysis_from_json.csv"
        save_summary_csv(out_path, persona_results, overall, analysis_keys)
        print(f"\nSaved summary CSV: {out_path}")


if __name__ == "__main__":
    main()

