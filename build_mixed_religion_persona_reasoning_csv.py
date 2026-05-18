import argparse
import csv
import json
import re
from pathlib import Path


def iter_persona_dirs(base_dir: Path):
    persona_re = re.compile(r"^(islam|christian|jewish|atheist)_(explicit|implicit_[123])$")
    for entry in sorted(base_dir.iterdir()):
        if entry.is_dir() and persona_re.match(entry.name):
            yield entry


def safe_get(dct, *keys, default=None):
    cur = dct
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def extract_reasoning(data: dict):
    choices = safe_get(data, "response", "raw_api_response", "choices", default=[])
    if not choices:
        return None
    message = choices[0].get("message", {})
    reasoning = message.get("reasoning")
    if reasoning:
        return reasoning
    details = message.get("reasoning_details")
    if isinstance(details, list):
        parts = [d.get("text") for d in details if isinstance(d, dict) and d.get("text")]
        if parts:
            return "".join(parts)
    return None


def extract_row(file_path: Path, persona: str):
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    analysis = data.get("analysis", {})
    is_correct = analysis.get("is_correct")

    ground_truth = analysis.get("ground_truth_label")
    prediction_label = analysis.get("prediction_label")

    prediction = data.get("prediction", {})
    prediction_answer = prediction.get("parsed_answer")
    prediction_excuse = prediction.get("excuse")

    return {
        "id": data.get("id"),
        "subset": data.get("subset"),
        "scenario": data.get("scenario"),
        "persona": persona,
        "is_correct": is_correct,
        "ground_truth_label": ground_truth,
        "prediction_label": prediction_label,
        "prediction_answer": prediction_answer,
        "prediction_excuse": prediction_excuse,
        "reasoning": extract_reasoning(data),
        "file_path": file_path.as_posix(),
    }


def build_rows(base_dir: Path):
    rows_by_id = {}
    for persona_dir in iter_persona_dirs(base_dir):
        persona = persona_dir.name
        for file_path in persona_dir.rglob("ethics_*.json"):
            row = extract_row(file_path, persona)
            if not row or not row.get("id"):
                continue
            rows_by_id.setdefault(row["id"], []).append(row)
    return rows_by_id


def filter_mixed(rows_by_id: dict):
    mixed = {}
    for item_id, rows in rows_by_id.items():
        correctness = [r.get("is_correct") for r in rows if r.get("is_correct") is not None]
        if not correctness:
            continue
        if any(correctness) and not all(correctness):
            mixed[item_id] = rows
    return mixed


def write_csv(rows_by_id: dict, output_path: Path):
    fieldnames = [
        "id",
        "subset",
        "scenario",
        "persona",
        "is_correct",
        "ground_truth_label",
        "prediction_label",
        "prediction_answer",
        "prediction_excuse",
        "reasoning",
        "file_path",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item_id in sorted(rows_by_id.keys()):
            for row in sorted(rows_by_id[item_id], key=lambda r: r.get("persona", "")):
                writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="Build CSV of mixed-correctness religion personas with reasoning."
    )
    parser.add_argument(
        "--base_dir",
        default="result/ETHICS/deepseek_deepseek-r1-distill-qwen-32b",
        help="Base directory containing persona folders.",
    )
    parser.add_argument(
        "--output",
        default="result/ETHICS/deepseek_deepseek-r1-distill-qwen-32b/mixed_religion_persona_reasoning.csv",
        help="CSV output path.",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    rows_by_id = build_rows(base_dir)
    mixed_rows = filter_mixed(rows_by_id)
    write_csv(mixed_rows, Path(args.output))

    total_ids = len(rows_by_id)
    mixed_ids = len(mixed_rows)
    total_rows = sum(len(rows) for rows in mixed_rows.values())
    print(f"Total IDs scanned: {total_ids}")
    print(f"Mixed IDs: {mixed_ids}")
    print(f"Rows written: {total_rows}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
