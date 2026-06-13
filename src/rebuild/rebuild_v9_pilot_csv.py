"""Rebuild V7-compatible CSV from existing v9_pilot JSON files.

Reads all per-question JSON files from result/ETHICS/<model_slug>/v9_pilot/
and writes TWO CSVs with the V7 schema — one for gender personas and one for
religion personas (each includes the neutral control rows).

Output files per model:
  result_ethics_v9_pilot_gender_{model_slug}.csv   — man/woman personas + neutral
  result_ethics_v9_pilot_religion_{model_slug}.csv — islam/christian/jewish/atheist + neutral

Usage:
  python src/rebuild/rebuild_v9_pilot_csv.py
  python src/rebuild/rebuild_v9_pilot_csv.py --model_slug deepseek_deepseek-r1-distill-qwen-32b
  python src/rebuild/rebuild_v9_pilot_csv.py --all
"""

import argparse
import csv
import json
from pathlib import Path

RESULT_ROOT = Path("result/ETHICS")
OUTPUT_NAME = "v9_pilot"

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
    "atheist_explicit":     ("religion_explicit",  "atheist"),
    "atheist_implicit_1":   ("religion_implicit_1","atheist"),
    "atheist_implicit_2":   ("religion_implicit_2","atheist"),
    "atheist_implicit_3":   ("religion_implicit_3","atheist"),
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

# Personas belonging to each split (neutral is shared by both)
GENDER_PERSONAS = {
    "neutral",
    "man_explicit_1", "man_implicit_1", "man_implicit_2", "man_implicit_3",
    "woman_explicit_1", "woman_implicit_1", "woman_implicit_2", "woman_implicit_3",
}

RELIGION_PERSONAS = {
    "neutral",
    "islam_explicit", "islam_implicit_1", "islam_implicit_2", "islam_implicit_3",
    "christian_explicit", "christian_implicit_1", "christian_implicit_2", "christian_implicit_3",
    "jewish_explicit", "jewish_implicit_1", "jewish_implicit_2", "jewish_implicit_3",
    "atheist_explicit", "atheist_implicit_1", "atheist_implicit_2", "atheist_implicit_3",
}

CSV_FIELDNAMES = [
    "prompt_variant", "category", "persona", "sub_folder",
    "id", "subset", "ground_truth_label", "prediction_label",
    "is_correct", "parsed_answer", "explanation",
    "success", "error", "latency_sec", "retry_count",
]


def _sanitize(value: object) -> object:
    """Replace newlines in string values so pandas can read the CSV reliably."""
    if isinstance(value, str):
        return value.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    return value


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction="ignore",
                                lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _sanitize(v) for k, v in row.items()})


def rebuild_model(model_slug: str) -> int:
    pilot_dir = RESULT_ROOT / model_slug / OUTPUT_NAME
    if not pilot_dir.exists():
        print(f"  [SKIP] {pilot_dir} does not exist")
        return 0

    all_rows: list[dict] = []
    for json_file in sorted(pilot_dir.rglob("*.json")):
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

        all_rows.append({
            "prompt_variant":     d.get("metadata", {}).get("prompt_variant", "prompt_v9"),
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

    if not all_rows:
        print(f"  [SKIP] No rows found for {model_slug}")
        return 0

    gender_rows   = [r for r in all_rows if r["persona"] in GENDER_PERSONAS]
    religion_rows = [r for r in all_rows if r["persona"] in RELIGION_PERSONAS]

    base = RESULT_ROOT / model_slug

    gender_path   = base / f"result_ethics_{OUTPUT_NAME}_gender_{model_slug}.csv"
    religion_path = base / f"result_ethics_{OUTPUT_NAME}_religion_{model_slug}.csv"

    _write_csv(gender_path,   gender_rows)
    _write_csv(religion_path, religion_rows)

    print(f"  Gender   ({len(gender_rows):4d} rows) -> {gender_path.name}")
    print(f"  Religion ({len(religion_rows):4d} rows) -> {religion_path.name}")
    return len(gender_rows) + len(religion_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_slug", type=str, default="deepseek_deepseek-r1-distill-qwen-32b")
    parser.add_argument("--all", action="store_true", help="Rebuild all model slugs that have v9_pilot/")
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
