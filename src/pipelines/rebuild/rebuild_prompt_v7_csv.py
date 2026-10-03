"""Rebuild CSV from prompt_v7 JSON results.

Reads all per-question JSON files from result/benchmarks/ethics/runs/<model_slug>/prompt_v7/
and writes a CSV with the standard V7 schema.

Usage:
  python src/pipelines/rebuild/rebuild_prompt_v7_csv.py
  python src/pipelines/rebuild/rebuild_prompt_v7_csv.py --model_slug deepseek_deepseek-r1-distill-qwen-32b
  python src/pipelines/rebuild/rebuild_prompt_v7_csv.py --all
"""

import argparse
import csv
import json
from pathlib import Path

RESULT_ROOT = Path("result/benchmarks/ethics/runs")
OUTPUT_NAME = "prompt_v7"

CSV_FIELDNAMES = [
    "prompt_variant", "category", "persona", "sub_folder",
    "id", "subset", "ground_truth_label", "prediction_label",
    "is_correct", "parsed_answer", "explanation",
    "success", "error", "latency_sec", "retry_count",
]


def rebuild_model(model_slug: str) -> int:
    variant_dir = RESULT_ROOT / model_slug / OUTPUT_NAME
    if not variant_dir.exists():
        print(f"  [SKIP] {variant_dir} does not exist")
        return 0

    rows = []
    for json_file in sorted(variant_dir.rglob("*.json")):
        if "warmup" in json_file.parts:
            continue
        try:
            with open(json_file, encoding="utf-8") as f:
                d = json.load(f)
        except Exception as e:
            print(f"  [WARN] {json_file.name}: {e}")
            continue

        meta = d.get("metadata", {})
        pred = d.get("prediction", {})
        resp = d.get("response", {})
        analysis = d.get("analysis", {})

        # category and sub_folder are stored in metadata by the runner
        category = meta.get("persona_category", "")
        persona = d.get("persona", "")

        # derive sub_folder from the folder structure if not in metadata
        # path: prompt_v7/<category>/<sub_folder>/<id>.json  or  prompt_v7/<category>/<id>.json
        rel_parts = json_file.relative_to(variant_dir).parts
        sub_folder = rel_parts[1] if len(rel_parts) >= 3 else ""

        rows.append({
            "prompt_variant":     meta.get("prompt_variant", OUTPUT_NAME),
            "category":           category,
            "persona":            persona,
            "sub_folder":         sub_folder,
            "id":                 d.get("id"),
            "subset":             d.get("subset"),
            "ground_truth_label": analysis.get("ground_truth_label"),
            "prediction_label":   analysis.get("prediction_label"),
            "is_correct":         analysis.get("is_correct"),
            "parsed_answer":      pred.get("parsed_answer", ""),
            "explanation":        pred.get("explanation") or pred.get("excuse", ""),
            "success":            resp.get("success"),
            "error":              resp.get("error") or "",
            "latency_sec":        resp.get("latency_sec"),
            "retry_count":        resp.get("retry_count"),
        })

    if not rows:
        print(f"  [SKIP] No rows found for {model_slug}")
        return 0

    csv_path = RESULT_ROOT / model_slug / f"result_ethics_{OUTPUT_NAME}_{model_slug}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Saved {len(rows)} rows -> {csv_path}")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_slug", type=str, default="deepseek_deepseek-r1-distill-qwen-32b")
    parser.add_argument("--all", action="store_true", help="Rebuild all model slugs that have prompt_v7/")
    args = parser.parse_args()

    if args.all:
        slugs = [d.name for d in RESULT_ROOT.iterdir()
                 if d.is_dir() and (d / OUTPUT_NAME).exists()]
        if not slugs:
            print(f"No model folders with {OUTPUT_NAME}/ found under {RESULT_ROOT}")
            return
        print(f"Rebuilding {len(slugs)} model(s)...")
        total = 0
        for slug in sorted(slugs):
            print(f"\n[{slug}]")
            total += rebuild_model(slug)
        print(f"\nTotal rows written: {total}")
    else:
        print(f"Rebuilding {args.model_slug}...")
        rebuild_model(args.model_slug)


if __name__ == "__main__":
    main()
