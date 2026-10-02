import csv
import json
from pathlib import Path

ROOT = Path("result")
REPORTS_ROOT = ROOT / "reports"
DATASETS = ["SQuAD", "ETHICS", "ELI5", "TRUTHFULQA"]


def get_nested(d, keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def to_number(value):
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def is_truncated(rec):
    pred = (get_nested(rec, ["response", "prediction"], "") or "").strip().lower()
    err = (get_nested(rec, ["response", "error"], "") or "").lower()
    max_tokens = get_nested(rec, ["request", "parameters", "max_tokens"], 0) or 0
    completion_tokens = get_nested(rec, ["response", "performance", "tokens", "completion_tokens"], 0) or 0

    if "truncat" in err:
        return 1
    if pred.endswith("..."):
        return 1
    if max_tokens and completion_tokens and completion_tokens >= max_tokens:
        return 1
    return 0


def is_no_extract(rec):
    success = bool(get_nested(rec, ["response", "success"], False))
    pred = (get_nested(rec, ["response", "prediction"], "") or "").strip()
    if not success:
        return 1
    if pred == "":
        return 1
    return 0


def correct_flag(rec):
    analysis = rec.get("analysis", {}) or {}

    if "correct" in analysis:
        return 1 if bool(analysis.get("correct")) else 0
    if "exact_match" in analysis:
        return 1 if float(analysis.get("exact_match", 0.0) or 0.0) >= 1.0 else 0
    if "semantic_similarity" in analysis:
        return 1 if float(analysis.get("semantic_similarity", 0.0) or 0.0) >= 0.5 else 0
    return 0


def collect_persona_stats(dataset, persona_dir, gt_filter=None):
    files = sorted(persona_dir.glob("*.json"))
    records = []

    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            rec = json.load(f)

        if gt_filter is not None:
            gt_label = get_nested(rec, ["analysis", "ground_truth_label"], None)
            if gt_label is None:
                gt_label = get_nested(rec, ["input", "ground_truth_label"], None)
            if gt_label != gt_filter:
                continue

        records.append(rec)

    total = len(records)

    stats = {
        "persona": persona_dir.name,
        "total": total,
        "no_extract": 0,
        "truncated": 0,
        "sum_prompt_tokens": 0,
        "sum_completion_tokens": 0,
        "sum_reasoning_tokens": 0,
        "sum_total_tokens": 0,
        "avg_prompt_tokens": 0.0,
        "avg_completion_tokens": 0.0,
        "avg_reasoning_tokens": 0.0,
        "avg_total_tokens": 0.0,
    }

    # Preserve the first-seen order of analysis keys.
    analysis_key_order = []
    analysis_sums = {}

    for rec in records:

        no_extract = is_no_extract(rec)
        truncated = is_truncated(rec)

        stats["no_extract"] += no_extract
        stats["truncated"] += truncated

        prompt_tokens = get_nested(rec, ["response", "performance", "tokens", "prompt_tokens"], 0) or 0
        completion_tokens = get_nested(rec, ["response", "performance", "tokens", "completion_tokens"], 0) or 0
        total_tokens = get_nested(rec, ["response", "performance", "tokens", "total_tokens"], 0) or 0
        reasoning_tokens = get_nested(
            rec,
            ["response", "performance", "tokens", "completion_tokens_details", "reasoning_tokens"],
            get_nested(rec, ["analysis", "reasoning_tokens"], 0),
        ) or 0

        stats["sum_prompt_tokens"] += int(prompt_tokens)
        stats["sum_completion_tokens"] += int(completion_tokens)
        stats["sum_reasoning_tokens"] += int(reasoning_tokens)
        stats["sum_total_tokens"] += int(total_tokens)

        analysis = rec.get("analysis", {}) or {}
        for key, value in analysis.items():
            if dataset == "ETHICS" and key == "ground_truth_label":
                continue
            num = to_number(value)
            if num is None:
                continue
            if key not in analysis_sums:
                analysis_sums[key] = 0.0
                analysis_key_order.append(key)
            analysis_sums[key] += num

    if stats["total"] > 0:
        stats["avg_prompt_tokens"] = round(stats["sum_prompt_tokens"] / stats["total"], 2)
        stats["avg_completion_tokens"] = round(stats["sum_completion_tokens"] / stats["total"], 2)
        stats["avg_reasoning_tokens"] = round(stats["sum_reasoning_tokens"] / stats["total"], 2)
        stats["avg_total_tokens"] = round(stats["sum_total_tokens"] / stats["total"], 2)

    for key in analysis_key_order:
        if dataset == "ETHICS" and key in {"correct", "predicted_label"}:
            # Represent match-to-GT as accuracy percentage.
            correct_sum = analysis_sums.get("correct", 0.0)
            stats[key] = round((correct_sum / stats["total"]) * 100, 4) if stats["total"] > 0 else 0.0
        else:
            stats[key] = round(analysis_sums[key] / stats["total"], 4) if stats["total"] > 0 else 0.0

    # ETHICS-specific request: predicted_label should mirror GT-match accuracy.
    if dataset == "ETHICS" and "correct" in stats:
        stats["predicted_label"] = stats["correct"]

    return stats, analysis_key_order


def add_overall_row(dataset, rows, analysis_keys):
    if not rows:
        return rows

    total = sum(r["total"] for r in rows)

    overall = {
        "persona": "OVERALL",
        "total": total,
        "no_extract": sum(r["no_extract"] for r in rows),
        "truncated": sum(r["truncated"] for r in rows),
        "sum_prompt_tokens": sum(r["sum_prompt_tokens"] for r in rows),
        "sum_completion_tokens": sum(r["sum_completion_tokens"] for r in rows),
        "sum_reasoning_tokens": sum(r["sum_reasoning_tokens"] for r in rows),
        "sum_total_tokens": sum(r["sum_total_tokens"] for r in rows),
        "avg_prompt_tokens": 0.0,
        "avg_completion_tokens": 0.0,
        "avg_reasoning_tokens": 0.0,
        "avg_total_tokens": 0.0,
    }

    if total > 0:
        overall["avg_prompt_tokens"] = round(overall["sum_prompt_tokens"] / total, 2)
        overall["avg_completion_tokens"] = round(overall["sum_completion_tokens"] / total, 2)
        overall["avg_reasoning_tokens"] = round(overall["sum_reasoning_tokens"] / total, 2)
        overall["avg_total_tokens"] = round(overall["sum_total_tokens"] / total, 2)

    for key in analysis_keys:
        weighted_sum = sum((r.get(key, 0.0) or 0.0) * r["total"] for r in rows)
        overall[key] = round(weighted_sum / total, 4) if total > 0 else 0.0

    if dataset == "ETHICS" and "correct" in overall:
        overall["predicted_label"] = overall["correct"]

    rows.append(overall)
    return rows


def columns_for_dataset(analysis_keys):
    base_cols = [
        "persona",
        "total",
        "no_extract",
        "truncated",
        "sum_prompt_tokens",
        "sum_completion_tokens",
        "sum_reasoning_tokens",
        "sum_total_tokens",
        "avg_prompt_tokens",
        "avg_completion_tokens",
        "avg_reasoning_tokens",
        "avg_total_tokens",
    ]
    return base_cols + analysis_keys


def build_reports():
    created = []

    for dataset in DATASETS:
        dataset_root = ROOT / dataset
        if not dataset_root.exists():
            continue

        model_dirs = [p for p in dataset_root.iterdir() if p.is_dir()]
        for model_dir in model_dirs:
            persona_dirs = [p for p in model_dir.iterdir() if p.is_dir() and p.name != "warmup"]
            rows = []
            analysis_keys = []

            for persona_dir in sorted(persona_dirs):
                json_files = list(persona_dir.glob("*.json"))
                if not json_files:
                    continue
                stats, found_keys = collect_persona_stats(dataset, persona_dir)
                rows.append(stats)
                for key in found_keys:
                    if key not in analysis_keys:
                        analysis_keys.append(key)

            if not rows:
                continue

            rows = add_overall_row(dataset, rows, analysis_keys)
            report_dir = REPORTS_ROOT / model_dir.name
            report_dir.mkdir(parents=True, exist_ok=True)

            out_file = report_dir / f"accuracy_{dataset.lower()}.csv"
            cols = columns_for_dataset(analysis_keys)

            with open(out_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=cols)
                writer.writeheader()
                writer.writerows(rows)

            created.append(out_file)

            if dataset == "ETHICS":
                for gt_value, suffix in [(0, "wrong"), (1, "correct")]:
                    split_rows = []
                    split_keys = []

                    for persona_dir in sorted(persona_dirs):
                        json_files = list(persona_dir.glob("*.json"))
                        if not json_files:
                            continue
                        split_stats, found_keys = collect_persona_stats(dataset, persona_dir, gt_filter=gt_value)
                        if split_stats["total"] == 0:
                            continue
                        split_rows.append(split_stats)
                        for key in found_keys:
                            if key not in split_keys:
                                split_keys.append(key)

                    if not split_rows:
                        continue

                    split_rows = add_overall_row(dataset, split_rows, split_keys)
                    split_cols = columns_for_dataset(split_keys)
                    split_file = report_dir / f"accuracy_ethics_{suffix}.csv"
                    with open(split_file, "w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=split_cols)
                        writer.writeheader()
                        writer.writerows(split_rows)
                    created.append(split_file)

    return created


if __name__ == "__main__":
    files = build_reports()
    print("Created reports:")
    for fp in files:
        print(fp)
