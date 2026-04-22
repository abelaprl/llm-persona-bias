"""
Generate rerun plan CSV for ETHICS rows with empty prediction_label.

Main output includes metadata for auditing and manual reruns.
Optional root rerun_plan.csv (json_path only) is created for compatibility
with src/rerun/rerun_truncated.py, which already uses MAX_TOKENS=4096.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def is_empty_prediction_label(value: object) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    if text == "":
        return True
    return text.lower() in {"nan", "none", "null"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build ETHICS rerun plan for empty prediction_label rows"
    )
    parser.add_argument(
        "--result_csv",
        type=str,
        required=True,
        help="Path to result_ethics_<model>.csv",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="ETHICS model directory containing persona JSON folders",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="",
        help="Optional output path; default is <model_dir>/rerun_plan_empty_prediction_label_4096.csv",
    )
    parser.add_argument(
        "--write_root_rerun_plan",
        action="store_true",
        help="Also write root rerun_plan.csv (json_path only) for rerun_truncated.py",
    )
    args = parser.parse_args()

    result_csv = Path(args.result_csv)
    model_dir = Path(args.model_dir)

    if not result_csv.exists():
        raise SystemExit(f"Result CSV not found: {result_csv}")
    if not model_dir.exists():
        raise SystemExit(f"Model dir not found: {model_dir}")

    output_csv = (
        Path(args.output_csv)
        if args.output_csv
        else model_dir / "rerun_plan_empty_prediction_label_4096.csv"
    )

    selected_rows: list[dict] = []
    missing_json_count = 0

    with open(result_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not is_empty_prediction_label(row.get("prediction_label")):
                continue

            persona = (row.get("persona") or "").strip()
            sample_id = (row.get("id") or "").strip()
            subset = (row.get("subset") or "").strip()

            if not persona or not sample_id:
                continue

            json_path = model_dir / persona / f"{sample_id}.json"
            exists = json_path.exists()
            if not exists:
                missing_json_count += 1

            selected_rows.append(
                {
                    "id": sample_id,
                    "subset": subset,
                    "persona": persona,
                    "ground_truth_label": (row.get("ground_truth_label") or "").strip(),
                    "prediction_label": "",
                    "reason": "empty_prediction_label",
                    "new_max_tokens": 4096,
                    "json_path": json_path.as_posix(),
                    "json_exists": exists,
                }
            )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "subset",
        "persona",
        "ground_truth_label",
        "prediction_label",
        "reason",
        "new_max_tokens",
        "json_path",
        "json_exists",
    ]

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected_rows)

    print(f"Wrote: {output_csv}")
    print(f"Rows selected: {len(selected_rows)}")
    print(f"Missing JSON files: {missing_json_count}")

    if args.write_root_rerun_plan:
        root_plan = Path("rerun_plan.csv")
        with open(root_plan, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["json_path"])
            writer.writeheader()
            for row in selected_rows:
                writer.writerow({"json_path": row["json_path"]})
        print(f"Also wrote: {root_plan}")


if __name__ == "__main__":
    main()
