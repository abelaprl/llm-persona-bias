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
    root = Path("result") / "ELI5"
    if not root.exists():
        return ""
    candidates = [p for p in root.iterdir() if p.is_dir()]
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


def gather_persona_stats(persona_dir):
    sums = defaultdict(float)
    total = 0
    token_sums = {
        "sum_prompt_tokens": 0.0,
        "sum_completion_tokens": 0.0,
        "sum_reasoning_tokens": 0.0,
        "sum_total_tokens": 0.0,
    }

    for fp in sorted(persona_dir.glob("*.json")):
        with open(fp, "r", encoding="utf-8") as f:
            obj = json.load(f)

        analysis = obj.get("analysis", {}) or {}
        total += 1
        for key, value in analysis.items():
            num = numeric_value(value)
            if num is not None:
                sums[key] += num

        perf_tokens = (
            obj.get("response", {})
            .get("performance", {})
            .get("tokens", {})
        )
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

    return total, sums, means, token_sums, token_avgs


def merge_overall(persona_results):
    total_all = sum(item["total"] for item in persona_results)
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

    return total_all, sums_all, means_all, token_sums_all, token_avgs_all


def build_column_order(persona_results):
    keys = set()
    for item in persona_results:
        keys.update(item["means"].keys())

    priority = [
        "rouge1",
        "rouge2",
        "rougeL",
        "rouge_l",
        "jaccard_similarity",
        "verbosity_ratio",
        "prediction_length",
        "reference_length",
        "ground_truth_length",
        "answered",
        "temperature",
    ]

    ordered = [k for k in priority if k in keys]
    ordered += sorted(k for k in keys if k not in ordered)
    return ordered


def print_table(persona_results, overall, metric_keys, model_dir):
    display_metrics = metric_keys[:6]
    header = ["Persona", "Total"] + display_metrics
    widths = [20, 7] + [12] * len(display_metrics)

    def row_line(values):
        out = []
        for i, val in enumerate(values):
            if i == 0:
                out.append(f"{val:<{widths[i]}}")
            elif i == 1:
                out.append(f"{val:>{widths[i]}}")
            else:
                out.append(f"{val:>{widths[i]}.4f}")
        return " ".join(out)

    print(f"\nELI5 Analysis Summary From JSON ({model_dir})")
    print(" ".join([
        f"{header[0]:<{widths[0]}}",
        f"{header[1]:>{widths[1]}}",
        *[f"{h:>{w}}" for h, w in zip(header[2:], widths[2:])],
    ]))
    print("-" * (sum(widths) + len(widths) - 1))

    for item in persona_results:
        values = [item["persona"], item["total"]]
        values += [item["means"].get(k, 0.0) for k in display_metrics]
        print(row_line(values))

    values = ["OVERALL", overall["total"]]
    values += [overall["means"].get(k, 0.0) for k in display_metrics]
    print("-" * (sum(widths) + len(widths) - 1))
    print(row_line(values))

    if len(metric_keys) > len(display_metrics):
        print("\nAdditional metrics included in output CSV:")
        print(", ".join(metric_keys[len(display_metrics):]))


def save_summary_csv(out_path, persona_results, overall, metric_keys):
    requested_prefix = [
        "persona_name",
        "prediction_length",
        "ground_truth_length",
        "verbosity_ratio",
        "verbosity_Ratio_avg",
        "total",
        "sum_prompt_tokens",
        "sum_completion_tokens",
        "sum_reasoning_tokens",
        "sum_total_tokens",
        "avg_prompt_tokens",
        "avg_completion_tokens",
        "avg_reasoning_tokens",
        "avg_total_tokens",
    ]
    remaining_metrics = [m for m in metric_keys if m not in requested_prefix]
    fieldnames = requested_prefix + remaining_metrics
    rows = []

    for item in persona_results:
        row = {
            "persona_name": item["persona"],
            "prediction_length": round(item["means"].get("prediction_length", 0.0), 6),
            "ground_truth_length": round(item["means"].get("ground_truth_length", 0.0), 6),
            "verbosity_ratio": round(item["means"].get("verbosity_ratio", 0.0), 6),
            "verbosity_Ratio_avg": round(item["means"].get("verbosity_ratio", 0.0), 6),
            "total": item["total"],
            "sum_prompt_tokens": round(item["token_sums"].get("sum_prompt_tokens", 0.0), 6),
            "sum_completion_tokens": round(item["token_sums"].get("sum_completion_tokens", 0.0), 6),
            "sum_reasoning_tokens": round(item["token_sums"].get("sum_reasoning_tokens", 0.0), 6),
            "sum_total_tokens": round(item["token_sums"].get("sum_total_tokens", 0.0), 6),
            "avg_prompt_tokens": round(item["token_avgs"].get("avg_prompt_tokens", 0.0), 6),
            "avg_completion_tokens": round(item["token_avgs"].get("avg_completion_tokens", 0.0), 6),
            "avg_reasoning_tokens": round(item["token_avgs"].get("avg_reasoning_tokens", 0.0), 6),
            "avg_total_tokens": round(item["token_avgs"].get("avg_total_tokens", 0.0), 6),
        }
        for k in metric_keys:
            row[k] = round(item["means"].get(k, 0.0), 6)
        rows.append(row)

    row = {
        "persona_name": "OVERALL",
        "prediction_length": round(overall["means"].get("prediction_length", 0.0), 6),
        "ground_truth_length": round(overall["means"].get("ground_truth_length", 0.0), 6),
        "verbosity_ratio": round(overall["means"].get("verbosity_ratio", 0.0), 6),
        "verbosity_Ratio_avg": round(overall["means"].get("verbosity_ratio", 0.0), 6),
        "total": overall["total"],
        "sum_prompt_tokens": round(overall["token_sums"].get("sum_prompt_tokens", 0.0), 6),
        "sum_completion_tokens": round(overall["token_sums"].get("sum_completion_tokens", 0.0), 6),
        "sum_reasoning_tokens": round(overall["token_sums"].get("sum_reasoning_tokens", 0.0), 6),
        "sum_total_tokens": round(overall["token_sums"].get("sum_total_tokens", 0.0), 6),
        "avg_prompt_tokens": round(overall["token_avgs"].get("avg_prompt_tokens", 0.0), 6),
        "avg_completion_tokens": round(overall["token_avgs"].get("avg_completion_tokens", 0.0), 6),
        "avg_reasoning_tokens": round(overall["token_avgs"].get("avg_reasoning_tokens", 0.0), 6),
        "avg_total_tokens": round(overall["token_avgs"].get("avg_total_tokens", 0.0), 6),
    }
    for k in metric_keys:
        row[k] = round(overall["means"].get(k, 0.0), 6)
    rows.append(row)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate ELI5 metrics directly from JSON analysis fields."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=default_model_dir(),
        help="ELI5 model result folder, e.g. result/benchmarks/eli5/allenai_olmo-3.1-32b-instruct",
    )
    parser.add_argument(
        "--save_csv",
        action="store_true",
        help="Write summary CSV in model folder",
    )
    args = parser.parse_args()

    if not args.model_dir:
        raise SystemExit("No model_dir found. Use --model_dir to set ELI5 model folder.")

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise SystemExit(f"Model dir not found: {model_dir}")

    persona_results = []

    for persona in PERSONA_ORDER:
        pdir = model_dir / persona
        if not pdir.exists() or not pdir.is_dir():
            continue
        total, sums, means, token_sums, token_avgs = gather_persona_stats(pdir)
        if total == 0:
            continue
        persona_results.append({
            "persona": persona,
            "total": total,
            "sums": sums,
            "means": means,
            "token_sums": token_sums,
            "token_avgs": token_avgs,
        })

    if not persona_results:
        raise SystemExit(f"No persona JSON files found under: {model_dir}")

    overall_total, overall_sums, overall_means, overall_token_sums, overall_token_avgs = merge_overall(persona_results)
    overall = {
        "total": overall_total,
        "sums": overall_sums,
        "means": overall_means,
        "token_sums": overall_token_sums,
        "token_avgs": overall_token_avgs,
    }

    metric_keys = build_column_order(persona_results)
    print_table(persona_results, overall, metric_keys, model_dir)

    if args.save_csv:
        out_path = model_dir / "accuracy_eli5_analysis_from_json.csv"
        save_summary_csv(out_path, persona_results, overall, metric_keys)
        print(f"\nSaved summary CSV: {out_path}")


if __name__ == "__main__":
    main()
