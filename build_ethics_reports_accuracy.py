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
    "neutral",
    "woman_explicit_1",
    "woman_implicit_1",
    "woman_implicit_2",
    "woman_implicit_3",
]


def to_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def to_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_json_safely(file_path: Path):
    text = file_path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        obj, _ = decoder.raw_decode(text.lstrip("\ufeff \t\r\n"))
        return obj


def normalize_rows_from_model_dir(model_dir: Path):
    rows = []

    for persona in PERSONA_ORDER:
        pdir = model_dir / persona
        if not pdir.exists() or not pdir.is_dir():
            continue

        for fp in sorted(pdir.glob("*.json")):
            try:
                obj = load_json_safely(fp)
            except Exception:
                continue

            analysis = obj.get("analysis", {}) or {}
            prediction = obj.get("prediction", {}) or {}
            response = obj.get("response", {}) or {}
            usage = (response.get("raw_api_response", {}) or {}).get("usage", {}) or {}
            completion_details = usage.get("completion_tokens_details", {}) or {}

            row = {
                "id": obj.get("id", fp.stem),
                "subset": obj.get("subset", ""),
                "persona": persona,
                "ground_truth_label": analysis.get("ground_truth_label", analysis.get("GT", "")),
                "prediction_label": analysis.get("prediction_label", prediction.get("parsed_answer", "")),
                "is_correct": analysis.get("is_correct", False),
                "parsed_answer": prediction.get("parsed_answer", ""),
                "explanation": prediction.get("explanation", ""),
                "success": response.get("success", False),
                "error": response.get("error", ""),
                "latency_sec": response.get("latency_sec", 0.0),
                "retry_count": response.get("retry_count", 0),
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "reasoning_tokens": completion_details.get("reasoning_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }
            rows.append(row)

    return rows


def write_result_csv(rows, model_dir: Path):
    model_name = model_dir.name
    result_path = model_dir / f"result_ethics_{model_name}.csv"
    fields = [
        "id",
        "subset",
        "persona",
        "ground_truth_label",
        "prediction_label",
        "is_correct",
        "parsed_answer",
        "explanation",
        "success",
        "error",
        "latency_sec",
        "retry_count",
        "prompt_tokens",
        "completion_tokens",
        "reasoning_tokens",
        "total_tokens",
    ]

    with open(result_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    return result_path


def build_reports_from_rows(rows, model_dir: Path):
    model_name = model_dir.name


    stats = defaultdict(lambda: {
        "total": 0,
        "predicted_label_count": 0,
        "correct_count": 0,
        "sum_explanation_length": 0.0,
        "sum_latency_sec": 0.0,
        "sum_prompt_tokens": 0,
        "sum_completion_tokens": 0,
        "sum_reasoning_tokens": 0,
        "sum_total_tokens": 0,
    })

    for row in rows:
        persona = row.get("persona", "unknown")
        st = stats[persona]

        st["total"] += 1
        st["predicted_label_count"] += to_int(row.get("prediction_label"), 0)
        st["correct_count"] += 1 if to_bool(row.get("is_correct")) else 0
        st["sum_explanation_length"] += len((row.get("explanation") or "").strip())
        st["sum_latency_sec"] += to_float(row.get("latency_sec"), 0.0)
        st["sum_prompt_tokens"] += to_int(row.get("prompt_tokens"), 0)
        st["sum_completion_tokens"] += to_int(row.get("completion_tokens"), 0)
        st["sum_reasoning_tokens"] += to_int(row.get("reasoning_tokens"), 0)
        st["sum_total_tokens"] += to_int(row.get("total_tokens"), 0)

    out_rows = []
    implicit_man_rows = []
    implicit_woman_rows = []

    for persona in PERSONA_ORDER:
        if persona not in stats:
            continue
        st = stats[persona]
        total = st["total"]
        if total == 0:
            continue

        row = {
            "persona_folder": persona,
            "predicted_label_count": st["predicted_label_count"],
            "correct_accuracy": st["correct_count"] / total,
            "explanation_length": st["sum_explanation_length"] / total,
            "latency_sec": st["sum_latency_sec"] / total,
            "total": total,
            "sum_prompt_tokens": st["sum_prompt_tokens"],
            "sum_completion_tokens": st["sum_completion_tokens"],
            "sum_reasoning_tokens": st["sum_reasoning_tokens"],
            "sum_total_tokens": st["sum_total_tokens"],
            "avg_prompt_tokens": st["sum_prompt_tokens"] / total,
            "avg_completion_tokens": st["sum_completion_tokens"] / total,
            "avg_reasoning_tokens": st["sum_reasoning_tokens"] / total,
            "avg_total_tokens": st["sum_total_tokens"] / total,
        }

        if persona.startswith("man_implicit"):
            implicit_man_rows.append(row)
        elif persona.startswith("woman_implicit"):
            implicit_woman_rows.append(row)

        row["correct_accuracy_avg"] = row["correct_accuracy"]
        row["explanation_length_avg"] = row["explanation_length"]
        row["latency_sec_avg"] = row["latency_sec"]
        out_rows.append(row)

    def fill_group_avg(rows_group):
        if not rows_group:
            return
        c = sum(r["correct_accuracy"] for r in rows_group) / len(rows_group)
        e = sum(r["explanation_length"] for r in rows_group) / len(rows_group)
        l = sum(r["latency_sec"] for r in rows_group) / len(rows_group)
        for r in rows_group:
            r["correct_accuracy_avg"] = c
            r["explanation_length_avg"] = e
            r["latency_sec_avg"] = l

    fill_group_avg(implicit_man_rows)
    fill_group_avg(implicit_woman_rows)

    if out_rows:
        all_total = sum(r["total"] for r in out_rows)
        all_row = {
            "persona_folder": "ALL",
            "predicted_label_count": sum(r["predicted_label_count"] for r in out_rows),
            "correct_accuracy": sum(r["correct_accuracy"] for r in out_rows) / len(out_rows),
            "explanation_length": sum(r["explanation_length"] for r in out_rows) / len(out_rows),
            "latency_sec": sum(r["latency_sec"] for r in out_rows) / len(out_rows),
            "total": all_total,
            "sum_prompt_tokens": sum(r["sum_prompt_tokens"] for r in out_rows),
            "sum_completion_tokens": sum(r["sum_completion_tokens"] for r in out_rows),
            "sum_reasoning_tokens": sum(r["sum_reasoning_tokens"] for r in out_rows),
            "sum_total_tokens": sum(r["sum_total_tokens"] for r in out_rows),
            "avg_prompt_tokens": sum(r["sum_prompt_tokens"] for r in out_rows) / all_total,
            "avg_completion_tokens": sum(r["sum_completion_tokens"] for r in out_rows) / all_total,
            "avg_reasoning_tokens": sum(r["sum_reasoning_tokens"] for r in out_rows) / all_total,
            "avg_total_tokens": sum(r["sum_total_tokens"] for r in out_rows) / all_total,
            "correct_accuracy_avg": sum(r["correct_accuracy"] for r in out_rows) / len(out_rows),
            "explanation_length_avg": sum(r["explanation_length"] for r in out_rows) / len(out_rows),
            "latency_sec_avg": sum(r["latency_sec"] for r in out_rows) / len(out_rows),
        }
        out_rows.append(all_row)

    report_path = model_dir / f"reports_accuracy_{model_name}.csv"
    fields = [
        "persona_folder",
        "predicted_label_count",
        "correct_accuracy",
        "explanation_length",
        "latency_sec",
        "total",
        "sum_prompt_tokens",
        "sum_completion_tokens",
        "sum_reasoning_tokens",
        "sum_total_tokens",
        "avg_prompt_tokens",
        "avg_completion_tokens",
        "avg_reasoning_tokens",
        "avg_total_tokens",
        "correct_accuracy_avg",
        "explanation_length_avg",
        "latency_sec_avg",
    ]

    with open(report_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)

    return report_path


def read_rows_from_result_csv(result_csv_path: Path):
    with open(result_csv_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description="Build ETHICS report CSVs for a model folder")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="",
        help="Path to ETHICS model dir containing persona folders with JSON files",
    )
    parser.add_argument(
        "--result_csv",
        type=str,
        default=r"result\ETHICS\deepseek_deepseek-v3.2\result_ethics_deepseek_deepseek-v3.2.csv",
        help="Path to source result_ethics_<model>.csv",
    )
    args = parser.parse_args()

    if args.model_dir:
        model_dir = Path(args.model_dir)
        if not model_dir.exists():
            raise SystemExit(f"Model dir not found: {model_dir}")

        rows = normalize_rows_from_model_dir(model_dir)
        if not rows:
            raise SystemExit(f"No JSON rows found under: {model_dir}")

        result_out = write_result_csv(rows, model_dir)
        report_out = build_reports_from_rows(rows, model_dir)
        print(f"Result CSV ready: {result_out}")
        print(f"Report CSV ready: {report_out}")
        return

    result_path = Path(args.result_csv)
    if not result_path.exists():
        raise SystemExit(f"Result CSV not found: {result_path}")

    rows = read_rows_from_result_csv(result_path)
    report_out = build_reports_from_rows(rows, result_path.parent)
    desired_result = result_path.parent / f"result_ethics_{result_path.parent.name}.csv"
    if result_path.name != desired_result.name:
        desired_result.write_text(result_path.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        desired_result = result_path

    print(f"Result CSV ready: {desired_result}")
    print(f"Report CSV ready: {report_out}")


if __name__ == "__main__":
    main()
