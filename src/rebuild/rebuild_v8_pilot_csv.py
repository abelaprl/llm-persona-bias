"""Rebuild V7-compatible CSV from existing v8_pilot JSON files.

Reads all per-question JSON files from result/ETHICS/<model_slug>/v8_pilot/
and writes a CSV with the V7 schema:
  prompt_variant, category, persona, sub_folder, id, subset,
  ground_truth_label, prediction_label, is_correct,
  parsed_answer, explanation, success, error, latency_sec, retry_count

Usage:
  python src/rebuild/rebuild_v8_pilot_csv.py
  python src/rebuild/rebuild_v8_pilot_csv.py --model_slug deepseek_deepseek-r1-distill-qwen-32b
  python src/rebuild/rebuild_v8_pilot_csv.py --all
"""

import argparse
import csv
import json
from pathlib import Path

RESULT_ROOT = Path("result/ETHICS")
OUTPUT_NAME = "v8_pilot"

PERSONA_CATEGORY: dict[str, tuple[str, str]] = {
    "neutral":              ("neutral",            ""),
    "man_explicit_1":       ("gender_explicit",    "man"),
    "man_implicit_1":       ("gender_implicit_1",  "man"),
    "man_implicit_2":       ("gender_implicit_2",  "man"),
    "man_implicit_3":       ("gender_implicit_3",  "man"),
    "woman_explicit_1":     ("gender_explicit",    "woman"),
    "woman_implicit_1":     ("gender_implicit_1",  "woman"),
    "woman_implicit_2":     ("gender_implicit_2",  "woman"),
    "woman_implicit_3":     ("gender_implicit_3",  "woman"),
    "islam_explicit":       ("religion_explicit",  "islam"),
    "islam_implicit_1":     ("religion_implicit_1","islam"),
    "islam_implicit_2":     ("religion_implicit_2","islam"),
    "islam_implicit_3":     ("religion_implicit_3","islam"),
    "christian_explicit":   ("religion_explicit",  "christian"),
    "christian_implicit_1": ("religion_implicit_1","christian"),
    "christian_implicit_2": ("religion_implicit_2","christian"),
    "christian_implicit_3": ("religion_implicit_3","christian"),
    "jewish_explicit":      ("religion_explicit",  "jewish"),
    "jewish_implicit_1":    ("religion_implicit_1","jewish"),
    "jewish_implicit_2":    ("religion_implicit_2","jewish"),
    "jewish_implicit_3":    ("religion_implicit_3","jewish"),
    "atheist_explicit":      ("religion_explicit",   "atheist"),
    "atheist_implicit_1":    ("religion_implicit_1", "atheist"),
    "atheist_implicit_2":    ("religion_implicit_2", "atheist"),
    "atheist_implicit_3":    ("religion_implicit_3", "atheist"),
    "western_explicit":      ("regional_explicit",   "western"),
    "western_implicit_1":    ("regional_implicit_1", "western"),
    "western_implicit_2":    ("regional_implicit_2", "western"),
    "western_implicit_3":    ("regional_implicit_3", "western"),
    "east_asian_explicit":   ("regional_explicit",   "east_asian"),
    "east_asian_implicit_1": ("regional_implicit_1", "east_asian"),
    "east_asian_implicit_2": ("regional_implicit_2", "east_asian"),
    "east_asian_implicit_3": ("regional_implicit_3", "east_asian"),
    "south_asian_explicit":  ("regional_explicit",   "south_asian"),
    "south_asian_implicit_1":("regional_implicit_1", "south_asian"),
    "south_asian_implicit_2":("regional_implicit_2", "south_asian"),
    "south_asian_implicit_3":("regional_implicit_3", "south_asian"),
    "african_explicit":      ("regional_explicit",   "african"),
    "african_implicit_1":    ("regional_implicit_1", "african"),
    "african_implicit_2":    ("regional_implicit_2", "african"),
    "african_implicit_3":    ("regional_implicit_3", "african"),
}

CSV_FIELDNAMES = [
    "prompt_variant", "category", "persona", "sub_folder",
    "id", "subset", "ground_truth_label", "prediction_label",
    "is_correct", "parsed_answer", "explanation",
    "success", "error", "latency_sec", "retry_count",
]


def rebuild_model(model_slug: str) -> int:
    pilot_dir = RESULT_ROOT / model_slug / OUTPUT_NAME
    if not pilot_dir.exists():
        print(f"  [SKIP] {pilot_dir} does not exist")
        return 0

    rows = []
    for json_file in sorted(pilot_dir.rglob("*.json")):
        # skip warmup files
        if "warmup" in json_file.parts:
            continue
        try:
            with open(json_file, encoding="utf-8") as f:
                d = json.load(f)
        except Exception as e:
            print(f"  [WARN] {json_file.name}: {e}")
            continue

        persona = d.get("persona", "")
        category, sub_folder = PERSONA_CATEGORY.get(persona, ("unknown", ""))
        pred = d.get("prediction", {})
        resp = d.get("response", {})
        analysis = d.get("analysis", {})

        rows.append({
            "prompt_variant":     d.get("metadata", {}).get("prompt_variant", "prompt_v8"),
            "category":           category,
            "persona":            persona,
            "sub_folder":         sub_folder,
            "id":                 d.get("id"),
            "subset":             d.get("subset"),
            "ground_truth_label": analysis.get("ground_truth_label"),
            "prediction_label":   analysis.get("prediction_label"),
            "is_correct":         analysis.get("is_correct"),
            "parsed_answer":      pred.get("parsed_answer", ""),
            "explanation":        pred.get("reasoning_text", ""),
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
    parser.add_argument("--all", action="store_true", help="Rebuild all model slugs that have v8_pilot/")
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
