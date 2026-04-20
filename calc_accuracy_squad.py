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
    root = Path("result") / "SQuAD_v2"
    if not root.exists():
        return ""
    candidates = [p for p in root.iterdir() if p.is_dir()]
    if not candidates:
        return ""
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return str(latest)


def to_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_json_safely(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        obj, _ = decoder.raw_decode(text.lstrip("\ufeff \t\r\n"))
        return obj


def aggregate_json_model(model_dir):
    persona_rows = []

    for persona in PERSONA_ORDER:
        pdir = model_dir / persona
        if not pdir.exists() or not pdir.is_dir():
            continue

        stats = {
            "persona_name": persona,
            "total": 0,
            "success_count": 0,
            "exact_match_sum": 0.0,
            "f1_score_sum": 0.0,
            "sum_prompt_tokens": 0.0,
            "sum_completion_tokens": 0.0,
            "sum_reasoning_tokens": 0.0,
            "sum_total_tokens": 0.0,
        }

        for fp in sorted(pdir.glob("*.json")):
            try:
                obj = load_json_safely(fp)
            except Exception:
                continue

            stats["total"] += 1

            if bool(obj.get("response", {}).get("success", False)):
                stats["success_count"] += 1

            analysis = obj.get("analysis", {}) or {}
            stats["exact_match_sum"] += to_float(analysis.get("exact_match"), 0.0)
            stats["f1_score_sum"] += to_float(analysis.get("f1_score"), 0.0)

            tokens = obj.get("response", {}).get("performance", {}).get("tokens", {})
            stats["sum_prompt_tokens"] += to_float(tokens.get("prompt_tokens"), 0.0)
            stats["sum_completion_tokens"] += to_float(tokens.get("completion_tokens"), 0.0)
            stats["sum_total_tokens"] += to_float(tokens.get("total_tokens"), 0.0)
            stats["sum_reasoning_tokens"] += to_float(
                tokens.get("completion_tokens_details", {}).get("reasoning_tokens"), 0.0
            )

        if stats["total"] == 0:
            continue

        total = stats["total"]
        persona_rows.append({
            "persona_name": persona,
            "total": total,
            "success_rate": stats["success_count"] / total,
            "exact_match": stats["exact_match_sum"] / total,
            "f1_score": stats["f1_score_sum"] / total,
            "sum_prompt_tokens": stats["sum_prompt_tokens"],
            "sum_completion_tokens": stats["sum_completion_tokens"],
            "sum_reasoning_tokens": stats["sum_reasoning_tokens"],
            "sum_total_tokens": stats["sum_total_tokens"],
            "avg_prompt_tokens": stats["sum_prompt_tokens"] / total,
            "avg_completion_tokens": stats["sum_completion_tokens"] / total,
            "avg_reasoning_tokens": stats["sum_reasoning_tokens"] / total,
            "avg_total_tokens": stats["sum_total_tokens"] / total,
        })

    if not persona_rows:
        return []

    overall_total = sum(r["total"] for r in persona_rows)
    overall = {
        "persona_name": "OVERALL",
        "total": overall_total,
        "success_rate": sum(r["success_rate"] * r["total"] for r in persona_rows) / overall_total,
        "exact_match": sum(r["exact_match"] * r["total"] for r in persona_rows) / overall_total,
        "f1_score": sum(r["f1_score"] * r["total"] for r in persona_rows) / overall_total,
        "sum_prompt_tokens": sum(r["sum_prompt_tokens"] for r in persona_rows),
        "sum_completion_tokens": sum(r["sum_completion_tokens"] for r in persona_rows),
        "sum_reasoning_tokens": sum(r["sum_reasoning_tokens"] for r in persona_rows),
        "sum_total_tokens": sum(r["sum_total_tokens"] for r in persona_rows),
        "avg_prompt_tokens": 0.0,
        "avg_completion_tokens": 0.0,
        "avg_reasoning_tokens": 0.0,
        "avg_total_tokens": 0.0,
    }
    overall["avg_prompt_tokens"] = overall["sum_prompt_tokens"] / overall_total
    overall["avg_completion_tokens"] = overall["sum_completion_tokens"] / overall_total
    overall["avg_reasoning_tokens"] = overall["sum_reasoning_tokens"] / overall_total
    overall["avg_total_tokens"] = overall["sum_total_tokens"] / overall_total

    persona_rows.append(overall)
    return persona_rows


def aggregate_csv(csv_path):
    by_persona = defaultdict(lambda: {
        "total": 0,
        "success_count": 0,
        "exact_match_sum": 0.0,
        "f1_score_sum": 0.0,
    })

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            persona = row.get("persona_name", "unknown")
            st = by_persona[persona]
            st["total"] += 1
            if str(row.get("status", "")).strip().upper() == "SUCCESS":
                st["success_count"] += 1
            st["exact_match_sum"] += to_float(row.get("exact_match"), 0.0)
            st["f1_score_sum"] += to_float(row.get("f1_score"), 0.0)

    rows = []
    for persona in PERSONA_ORDER:
        if persona not in by_persona:
            continue
        st = by_persona[persona]
        total = st["total"]
        rows.append({
            "persona_name": persona,
            "total": total,
            "success_rate": st["success_count"] / total if total else 0.0,
            "exact_match": st["exact_match_sum"] / total if total else 0.0,
            "f1_score": st["f1_score_sum"] / total if total else 0.0,
            "sum_prompt_tokens": 0.0,
            "sum_completion_tokens": 0.0,
            "sum_reasoning_tokens": 0.0,
            "sum_total_tokens": 0.0,
            "avg_prompt_tokens": 0.0,
            "avg_completion_tokens": 0.0,
            "avg_reasoning_tokens": 0.0,
            "avg_total_tokens": 0.0,
        })

    if not rows:
        return []

    overall_total = sum(r["total"] for r in rows)
    rows.append({
        "persona_name": "OVERALL",
        "total": overall_total,
        "success_rate": sum(r["success_rate"] * r["total"] for r in rows) / overall_total,
        "exact_match": sum(r["exact_match"] * r["total"] for r in rows) / overall_total,
        "f1_score": sum(r["f1_score"] * r["total"] for r in rows) / overall_total,
        "sum_prompt_tokens": 0.0,
        "sum_completion_tokens": 0.0,
        "sum_reasoning_tokens": 0.0,
        "sum_total_tokens": 0.0,
        "avg_prompt_tokens": 0.0,
        "avg_completion_tokens": 0.0,
        "avg_reasoning_tokens": 0.0,
        "avg_total_tokens": 0.0,
    })
    return rows


def print_summary(rows, title):
    print(f"\n{title}")
    print(f"{'Persona':<20} {'Total':>6} {'EM':>8} {'F1':>8} {'Success':>9}")
    print("-" * 58)
    for r in rows:
        print(
            f"{r['persona_name']:<20} {r['total']:>6} "
            f"{r['exact_match'] * 100:>7.2f}% {r['f1_score'] * 100:>7.2f}% "
            f"{r['success_rate'] * 100:>8.2f}%"
        )


def save_csv(rows, out_path):
    fieldnames = [
        "persona_name",
        "total",
        "success_rate",
        "exact_match",
        "f1_score",
        "sum_prompt_tokens",
        "sum_completion_tokens",
        "sum_reasoning_tokens",
        "sum_total_tokens",
        "avg_prompt_tokens",
        "avg_completion_tokens",
        "avg_reasoning_tokens",
        "avg_total_tokens",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Calculate SQuAD accuracy summary with exact_match and f1_score")
    parser.add_argument("--csv_path", type=str, default="", help="Path to result_squad_*.csv")
    parser.add_argument("--model_dir", type=str, default=default_model_dir(), help="Path to model folder containing persona JSON files")
    args = parser.parse_args()

    if args.csv_path:
        csv_path = Path(args.csv_path)
        if not csv_path.exists():
            raise SystemExit(f"CSV not found: {csv_path}")
        rows = aggregate_csv(csv_path)
        if not rows:
            raise SystemExit("No rows found in CSV.")
        print_summary(rows, f"SQuAD Summary From CSV ({csv_path})")
        out_path = csv_path.parent / "accuracy_squad_from_csv.csv"
        save_csv(rows, out_path)
        print(f"\nSaved summary CSV: {out_path}")
        return

    if not args.model_dir:
        raise SystemExit("No model_dir found. Provide --model_dir or --csv_path.")

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise SystemExit(f"Model dir not found: {model_dir}")

    rows = aggregate_json_model(model_dir)
    if not rows:
        raise SystemExit(f"No persona JSON files found under: {model_dir}")

    print_summary(rows, f"SQuAD Summary From JSON ({model_dir})")
    out_path = model_dir / "accuracy_squad_analysis_from_json.csv"
    save_csv(rows, out_path)
    print(f"\nSaved summary CSV: {out_path}")


if __name__ == "__main__":
    main()
