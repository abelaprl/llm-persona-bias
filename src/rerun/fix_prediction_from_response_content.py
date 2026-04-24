import argparse
import csv
import json
import re
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

EMPTY_TOKENS = {"", "none", "null", "nan", "na", "n/a"}


def is_empty(value):
    if value is None:
        return True
    text = str(value).strip().lower()
    return text in EMPTY_TOKENS or text in {"<>", "<&gt;"}


def first_non_empty(*values):
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def get_response_content(response):
    if not isinstance(response, dict):
        return ""

    direct = first_non_empty(response.get("content"))
    if direct:
        return direct

    raw_api = response.get("raw_api_response")
    if isinstance(raw_api, dict):
        choices = raw_api.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0] if isinstance(choices[0], dict) else {}
            message = first.get("message") if isinstance(first, dict) else {}
            if isinstance(message, dict):
                msg_content = first_non_empty(message.get("content"))
                if msg_content:
                    return msg_content

        for key in ("content", "text", "response"):
            fallback = first_non_empty(raw_api.get(key))
            if fallback:
                return fallback

    if isinstance(raw_api, str):
        return first_non_empty(raw_api)

    return ""


def load_json_safely(file_path: Path):
    text = file_path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        obj, _ = decoder.raw_decode(text.lstrip("\ufeff \t\r\n"))
        return obj


def extract_answer_and_excuse(content):
    text = first_non_empty(content)
    if not text:
        return "", ""

    answer = ""
    excuse_lines = []
    in_excuse = False

    for raw_line in text.splitlines():
        line = raw_line.strip()
        lower_line = line.lower()

        if not line:
            if in_excuse:
                excuse_lines.append("")
            continue

        if lower_line.startswith("answer:"):
            if not answer:
                answer = line.split(":", 1)[1].strip()
            in_excuse = False
            continue

        if lower_line.startswith("excuse:"):
            excuse_lines = [line.split(":", 1)[1].strip()]
            in_excuse = True
            continue

        if in_excuse:
            excuse_lines.append(raw_line.rstrip())

    if not answer:
        m = re.search(r"(?im)^\s*answer\s*:\s*(.+?)\s*$", text)
        if m:
            answer = m.group(1).strip()

    if answer:
        m = re.search(r"\b([01])\b", answer)
        if m:
            answer = m.group(1)

    excuse = "\n".join(excuse_lines).strip()
    if excuse in {"<>", "<&gt;"}:
        excuse = ""
    if not excuse:
        m = re.search(r"(?is)\bexcuse\s*:\s*(.+)$", text)
        if m:
            excuse = m.group(1).strip()
            if excuse in {"<>", "<&gt;"}:
                excuse = ""

    return answer.strip(), excuse.strip()


def process_model(model_dir: Path, dry_run=False):
    fixed_count = 0
    scanned = 0
    details = []

    for persona in PERSONA_ORDER:
        persona_dir = model_dir / persona
        if not persona_dir.exists() or not persona_dir.is_dir():
            continue

        for json_path in sorted(persona_dir.glob("*.json")):
            scanned += 1
            try:
                obj = load_json_safely(json_path)
            except Exception:
                details.append({
                    "json_path": json_path.as_posix(),
                    "status": "read_failed",
                    "fixed": 0,
                    "reason": "json_read_error",
                })
                continue

            prediction = obj.get("prediction") or {}
            response = obj.get("response") or {}
            content = get_response_content(response)

            parsed_answer = prediction.get("parsed_answer")
            excuse = prediction.get("excuse")

            is_missing_prediction = is_empty(parsed_answer) and is_empty(excuse)
            has_content = not is_empty(content)

            if not (is_missing_prediction and has_content):
                continue

            answer_new, excuse_new = extract_answer_and_excuse(content)
            if is_empty(answer_new) and is_empty(excuse_new):
                details.append({
                    "json_path": json_path.as_posix(),
                    "status": "skip_no_extract",
                    "fixed": 0,
                    "reason": "could_not_parse_content",
                })
                continue

            changed = False
            if is_empty(prediction.get("parsed_answer")) and not is_empty(answer_new):
                prediction["parsed_answer"] = answer_new
                changed = True

            if is_empty(prediction.get("excuse")) and not is_empty(excuse_new):
                prediction["excuse"] = excuse_new
                changed = True

            if is_empty(prediction.get("explanation")) and not is_empty(excuse_new):
                prediction["explanation"] = excuse_new
                changed = True

            if changed:
                obj["prediction"] = prediction
                if not dry_run:
                    json_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
                fixed_count += 1
                details.append({
                    "json_path": json_path.as_posix(),
                    "status": "fixed",
                    "fixed": 1,
                    "reason": "parsed_from_response_content",
                })
            else:
                details.append({
                    "json_path": json_path.as_posix(),
                    "status": "skip_no_change",
                    "fixed": 0,
                    "reason": "extracted_empty_or_fields_already_set",
                })

    return scanned, fixed_count, details


def main():
    parser = argparse.ArgumentParser(description="Fix empty prediction fields from response.content")
    parser.add_argument("--base_dir", type=str, default="result/ETHICS")
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Model directories inside base_dir",
    )
    parser.add_argument(
        "--log_csv",
        type=str,
        default="result/ETHICS/fix_prediction_from_response_content_log.csv",
    )
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    all_rows = []

    print("=" * 80)
    print("FIX PREDICTION FROM RESPONSE.CONTENT")
    print(f"Base dir: {base_dir}")
    print(f"Dry run : {args.dry_run}")
    print("=" * 80)

    for model_name in args.models:
        model_dir = base_dir / model_name
        if not model_dir.exists() or not model_dir.is_dir():
            print(f"[SKIP] model not found: {model_dir}")
            continue

        scanned, fixed, details = process_model(model_dir, dry_run=args.dry_run)
        print(f"[MODEL] {model_name}: scanned={scanned}, fixed={fixed}")
        for d in details:
            d["model"] = model_name
        all_rows.extend(details)

    log_path = Path(args.log_csv)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "json_path", "status", "fixed", "reason"],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Log CSV: {log_path}")


if __name__ == "__main__":
    main()
