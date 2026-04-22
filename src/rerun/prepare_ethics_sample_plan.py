import csv
from collections import defaultdict
from pathlib import Path

BASE = Path("result/ETHICS/qwen_qwen3-next-80b-a3b-thinking")
SOURCE = BASE / "rerun_plan_empty_prediction_label_4096.csv"
OUT_DETAILED = BASE / "rerun_plan_sample10_per_persona.csv"
OUT_ROOT = Path("rerun_plan.csv")
MAX_PER_PERSONA = 10


def main() -> None:
    if not SOURCE.exists():
        raise SystemExit(f"Source file not found: {SOURCE}")

    picked = []
    per_persona = defaultdict(int)

    with open(SOURCE, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        persona = (row.get("persona") or "").strip()
        if not persona:
            continue
        if per_persona[persona] >= MAX_PER_PERSONA:
            continue
        picked.append(row)
        per_persona[persona] += 1

    OUT_DETAILED.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_DETAILED, "w", encoding="utf-8", newline="") as f:
        fields = [
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
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(picked)

    with open(OUT_ROOT, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["json_path"])
        writer.writeheader()
        for row in picked:
            writer.writerow({"json_path": row["json_path"]})

    print(f"Source rows: {len(rows)}")
    print(f"Sample rows: {len(picked)}")
    for persona in sorted(per_persona.keys()):
        print(f"{persona}: {per_persona[persona]}")
    print(f"Wrote: {OUT_DETAILED}")
    print(f"Wrote: {OUT_ROOT}")


if __name__ == "__main__":
    main()
