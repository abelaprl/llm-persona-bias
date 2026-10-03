import argparse
import csv
import json
import re
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
    "islam_explicit",
    "islam_implicit_1",
    "islam_implicit_2",
    "islam_implicit_3",
    "christian_explicit",
    "christian_implicit_1",
    "christian_implicit_2",
    "christian_implicit_3",
    "jewish_explicit",
    "jewish_implicit_1",
    "jewish_implicit_2",
    "jewish_implicit_3",
    "atheist_explicit",
    "atheist_implicit_1",
    "atheist_implicit_2",
    "atheist_implicit_3",
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


def normalize_binary_label(value):
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text or text in {"none", "null", "nan", "na", "n/a"}:
        return None
    if text in {"0", "1"}:
        return int(text)
    m = re.search(r"\b([01])\b", text)
    if m:
        return int(m.group(1))
    return None


def compute_is_correct(ground_truth_label, prediction_label):
    gt = normalize_binary_label(ground_truth_label)
    pred = normalize_binary_label(prediction_label)
    if gt is None or pred is None:
        return None
    return gt == pred


def load_json_safely(file_path: Path):
    text = file_path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        obj, _ = decoder.raw_decode(text.lstrip("\ufeff \t\r\n"))
        return obj


def first_non_empty(*values):
    for value in values:
        text = "" if value is None else str(value).strip()
        if text and text.lower() not in {"<>", "<&gt;"}:
            return text
    return ""


def extract_prediction_parts(text: str):
    content = first_non_empty(text)
    if not content:
        return "", ""

    answer = ""
    excuse_lines = []
    in_excuse = False

    for raw_line in content.splitlines():
        line = raw_line.strip()
        low = line.lower()

        if not line:
            if in_excuse:
                excuse_lines.append("")
            continue

        if low.startswith("answer:"):
            if not answer:
                answer = line.split(":", 1)[1].strip()
            in_excuse = False
            continue

        if low.startswith("excuse:"):
            excuse_lines = [line.split(":", 1)[1].strip()]
            in_excuse = True
            continue

        if in_excuse:
            excuse_lines.append(raw_line.rstrip())

    excuse = "\n".join(excuse_lines).strip()
    if excuse in {"<>", "<&gt;"}:
        excuse = ""

    if not answer:
        answer_match = re.search(r"(?im)^\s*answer\s*:\s*(.+?)\s*$", content)
        if answer_match:
            answer = answer_match.group(1).strip()

    if answer:
        answer_match = re.search(r"\b([01])\b", answer)
        if answer_match:
            answer = answer_match.group(1)

    return answer, excuse


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
            raw_response_content = ""
            raw_api_response = response.get("raw_api_response", {}) or {}
            choices = raw_api_response.get("choices") or []
            if choices and isinstance(choices[0], dict):
                msg = choices[0].get("message") or {}
                raw_response_content = first_non_empty(msg.get("content"))
            usage = (response.get("raw_api_response", {}) or {}).get("usage", {}) or {}
            completion_details = usage.get("completion_tokens_details", {}) or {}

            parsed_answer = first_non_empty(prediction.get("parsed_answer"))
            explanation = first_non_empty(prediction.get("explanation"))
            fallback_content = first_non_empty(response.get("content"), raw_response_content)
            fallback_answer, fallback_explanation = extract_prediction_parts(fallback_content)

            if not parsed_answer:
                parsed_answer = fallback_answer
            if not explanation:
                explanation = fallback_explanation

            prediction_label_raw = first_non_empty(
                analysis.get("prediction_label"),
                parsed_answer,
                fallback_answer,
            )
            prediction_label = normalize_binary_label(prediction_label_raw)

            ground_truth_label_raw = first_non_empty(analysis.get("ground_truth_label"), analysis.get("GT", ""))
            ground_truth_label = normalize_binary_label(ground_truth_label_raw)
            if ground_truth_label is None:
                ground_truth_label = ground_truth_label_raw

            is_correct_raw = analysis.get("is_correct", None)
            is_correct_missing = is_correct_raw is None or str(is_correct_raw).strip().lower() in {
                "",
                "none",
                "null",
                "nan",
            }
            is_correct = compute_is_correct(ground_truth_label, prediction_label) if is_correct_missing else to_bool(is_correct_raw)

            row = {
                "id": obj.get("id", fp.stem),
                "subset": obj.get("subset", ""),
                "persona": persona,
                "ground_truth_label": ground_truth_label,
                "prediction_label": prediction_label,
                "is_correct": is_correct,
                "parsed_answer": parsed_answer,
                "explanation": explanation,
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


def write_result_csv(rows, model_dir: Path, out_path: Path | None = None):
    model_name = model_dir.name
    result_path = out_path if out_path is not None else (model_dir / f"result_ethics_{model_name}.csv")
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


def build_reports_from_rows(rows, model_dir: Path, out_path: Path | None = None):
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
        pred_label = normalize_binary_label(row.get("prediction_label"))
        if pred_label is not None:
            st["predicted_label_count"] += 1

        is_correct = row.get("is_correct")
        if is_correct is None or str(is_correct).strip().lower() in {"", "none", "null", "nan"}:
            is_correct = compute_is_correct(row.get("ground_truth_label"), row.get("prediction_label"))
        st["correct_count"] += 1 if is_correct is True else 0
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

    report_path = out_path if out_path is not None else (model_dir / f"reports_accuracy_{model_name}.csv")
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
        default=r"result/benchmarks/ethics/runs\deepseek_deepseek-v3.2\result_ethics_deepseek_deepseek-v3.2.csv",
        help="Path to source result_ethics_<model>.csv",
    )
    parser.add_argument(
        "--result_out",
        type=str,
        default="",
        help="Optional output path for rebuilt result CSV (useful if default file is locked)",
    )
    parser.add_argument(
        "--report_out",
        type=str,
        default="",
        help="Optional output path for report CSV (useful if default file is locked)",
    )
    args = parser.parse_args()

    if args.model_dir:
        model_dir = Path(args.model_dir)
        if not model_dir.exists():
            raise SystemExit(f"Model dir not found: {model_dir}")

        rows = normalize_rows_from_model_dir(model_dir)
        if not rows:
            raise SystemExit(f"No JSON rows found under: {model_dir}")

        result_out_path = Path(args.result_out) if args.result_out else None
        report_out_path = Path(args.report_out) if args.report_out else None

        result_out = write_result_csv(rows, model_dir, result_out_path)
        report_out = build_reports_from_rows(rows, model_dir, report_out_path)
        print(f"Result CSV ready: {result_out}")
        print(f"Report CSV ready: {report_out}")
        return

    result_path = Path(args.result_csv)
    if not result_path.exists():
        raise SystemExit(f"Result CSV not found: {result_path}")

    rows = read_rows_from_result_csv(result_path)
    result_out_path = Path(args.result_out) if args.result_out else None
    report_out_path = Path(args.report_out) if args.report_out else None
    report_out = build_reports_from_rows(rows, result_path.parent, report_out_path)

    if result_out_path is not None:
        desired_result = result_out_path
        desired_result.write_text(result_path.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        desired_result = result_path.parent / f"result_ethics_{result_path.parent.name}.csv"
        if result_path.name != desired_result.name:
            desired_result.write_text(result_path.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            desired_result = result_path

    print(f"Result CSV ready: {desired_result}")
    print(f"Report CSV ready: {report_out}")


if __name__ == "__main__":
    main()
