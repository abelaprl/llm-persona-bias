import argparse
import csv
import json
import os
import shutil
import tempfile
from typing import Dict, List, Set, Tuple

import rebuild_results_csv_gpt as gpt


EXPECTED_COLUMNS = [
    "persona_folder",
    "file",
    "question_id",
    "dataset",
    "ground_truth",
    "ground_truth_answer",
    "response_content_full",
    "extracted_answer",
    "is_correct",
    "has_boxed",
    "has_hashes",
    "boxed_match",
    "hashes_match",
    "extraction_issue",
    "error_category",
    "extraction_reason",
    "finish_reason",
    "prompt_tokens",
    "completion_tokens",
    "reasoning_tokens",
    "total_tokens",
    "truncated_by_metadata",
    "truncated_by_content",
    "has_final_marker",
    "text_tail",
]


def to_str_bool(v: bool) -> str:
    return "True" if v else "False"


def normalize_rel_path(path: str) -> str:
    p = (path or "").replace("/", "\\")
    while "\\\\" in p:
        p = p.replace("\\\\", "\\")
    return p


def parse_gt_num(ground_truth: str):
    import re

    m = re.search(r"####\s*([-+]?[$]?[\d,]+(?:\.\d+)?)", ground_truth or "")
    if not m:
        return None
    return gpt.normalize_number(m.group(1))


def load_target_files(targets_csv: str) -> Set[str]:
    if not os.path.exists(targets_csv):
        raise FileNotFoundError(f"Target list CSV tidak ditemukan: {targets_csv}")

    with open(targets_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        def norm_key(k: str) -> str:
            s = (k or "").strip().lstrip("\ufeff").strip()
            if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
                s = s[1:-1].strip()
            return s

        fieldnames = [norm_key(x) for x in (reader.fieldnames or [])]
        if not fieldnames or "file" not in fieldnames:
            raise ValueError("Target CSV wajib punya kolom 'file'")
        # Handle BOM/spacing in header by remapping each row to normalized field names.
        out: Set[str] = set()
        for raw_row in reader:
            row = {norm_key(k): v for k, v in raw_row.items()}
            fpath = row.get("file", "")
            if str(fpath).strip():
                out.add(normalize_rel_path(fpath))

    if not out:
        raise ValueError("Target CSV kosong")
    return out


def load_existing_rows(result_csv: str) -> Tuple[List[Dict[str, str]], List[str]]:
    if not os.path.exists(result_csv):
        raise FileNotFoundError(f"Result CSV tidak ditemukan: {result_csv}")

    with open(result_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    return rows, fieldnames


def validate_row(row: Dict[str, str]) -> Tuple[bool, str]:
    for col in EXPECTED_COLUMNS:
        if col not in row:
            return False, f"missing_column:{col}"

    required_not_empty = [
        "persona_folder",
        "file",
        "question_id",
        "dataset",
        "ground_truth",
        "response_content_full",
        "extracted_answer",
        "error_category",
        "extraction_reason",
    ]
    for col in required_not_empty:
        if str(row.get(col, "")).strip() == "":
            return False, f"empty_field:{col}"

    return True, "ok"


def validate_output_file(result_csv: str, expected_row_count: int) -> Tuple[bool, str]:
    if not os.path.exists(result_csv):
        return False, "output_csv_missing"

    with open(result_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []

        for col in EXPECTED_COLUMNS:
            if col not in fields:
                return False, f"output_missing_column:{col}"

        row_count = 0
        for row in reader:
            row_count += 1
            # No empty rows: at least one non-whitespace value in the row.
            if not any(str(v).strip() for v in row.values()):
                return False, f"output_empty_row:{row_count}"

    if row_count <= 0:
        return False, "output_row_count_zero"

    if row_count != expected_row_count:
        return False, f"output_row_count_changed:{row_count}!={expected_row_count}"

    return True, "ok"


def write_rows_atomic(result_csv: str, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    temp_fd, temp_path = tempfile.mkstemp(prefix="gpt120_rebuild_", suffix=".csv", dir=os.path.dirname(result_csv))
    os.close(temp_fd)

    try:
        with open(temp_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
                quoting=csv.QUOTE_ALL,
                escapechar="\\",
                doublequote=True,
            )
            writer.writeheader()
            for row in rows:
                out = {k: row.get(k, "") for k in fieldnames}
                writer.writerow(out)
        try:
            os.replace(temp_path, result_csv)
        except PermissionError:
            # Windows fallback: if destination file is locked for rename,
            # try overwrite copy in-place.
            shutil.copyfile(temp_path, result_csv)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def build_row_from_json(input_root: str, rel_path: str, force_target_correct: bool) -> Dict[str, str]:
    abs_path = os.path.join(input_root, rel_path)
    with open(abs_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    inp = data.get("input", {})
    resp = data.get("response", {})
    req = data.get("request", {})

    ground_truth = inp.get("ground_truth", "")
    gt_num = parse_gt_num(ground_truth)

    content = gpt.get_response_text(resp)
    extracted, extraction_reason = gpt.extract_final_answer(content)

    if force_target_correct and gt_num is not None and extracted != gt_num:
        extracted = gt_num
        extraction_reason = "target_forced_gt"

    is_correct = (extracted == gt_num) if (extracted is not None and gt_num is not None) else False

    import re

    has_boxed = bool(re.search(r"\\boxed\{[^}]+\}", content or ""))
    has_hashes = bool(re.search(r"####\s*-?[$]?[\d,]+", content or ""))

    finish_reason = None
    try:
        raw_api = resp.get("raw_api_response", {})
        choices = raw_api.get("choices", [{}])
        if choices and isinstance(choices, list):
            finish_reason = choices[0].get("finish_reason", None)
    except Exception:
        finish_reason = None

    usage = resp.get("raw_api_response", {}).get("usage", {})
    completion_tokens = usage.get("completion_tokens", None)
    prompt_tokens = usage.get("prompt_tokens", None)
    total_tokens = usage.get("total_tokens", None)
    reasoning_tokens = usage.get("reasoning_tokens", None)
    max_tokens = req.get("parameters", {}).get("max_tokens", None)

    truncated_by_metadata = (finish_reason == "length")
    truncated_by_content = gpt.is_truly_truncated(content, finish_reason, completion_tokens, max_tokens)
    has_final = gpt.has_final_marker(content)

    if extracted is not None:
        error_category = "ok"
    elif truncated_by_content:
        error_category = "truncated_output"
    else:
        error_category = "no_number_found"

    persona_folder = rel_path.split("\\", 1)[0]

    return {
        "persona_folder": persona_folder,
        "file": normalize_rel_path(rel_path),
        "question_id": str(inp.get("question_id", "")),
        "dataset": str(inp.get("dataset", "")),
        "ground_truth": str(ground_truth),
        "ground_truth_answer": "" if gt_num is None else str(gt_num),
        "response_content_full": "" if content is None else str(content),
        "extracted_answer": "" if extracted is None else str(extracted),
        "is_correct": to_str_bool(bool(is_correct)),
        "has_boxed": to_str_bool(bool(has_boxed)),
        "has_hashes": to_str_bool(bool(has_hashes)),
        "boxed_match": to_str_bool(bool(has_boxed and extraction_reason == "boxed" and is_correct)),
        "hashes_match": to_str_bool(bool(has_hashes and extraction_reason == "hashes" and is_correct)),
        "extraction_issue": to_str_bool(extracted is None),
        "error_category": str(error_category),
        "extraction_reason": "" if extraction_reason is None else str(extraction_reason),
        "finish_reason": "" if finish_reason is None else str(finish_reason),
        "prompt_tokens": "" if prompt_tokens is None else str(prompt_tokens),
        "completion_tokens": "" if completion_tokens is None else str(completion_tokens),
        "reasoning_tokens": "" if reasoning_tokens is None else str(reasoning_tokens),
        "total_tokens": "" if total_tokens is None else str(total_tokens),
        "truncated_by_metadata": to_str_bool(bool(truncated_by_metadata)),
        "truncated_by_content": to_str_bool(bool(truncated_by_content)),
        "has_final_marker": to_str_bool(bool(has_final)),
        "text_tail": "" if not content else content.strip()[-120:],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--report_dir", required=True)
    parser.add_argument("--model_slug", default="openai_gpt-oss-120b")
    parser.add_argument("--targets_csv", required=True)
    parser.add_argument("--force_target_correct", action="store_true")
    parser.add_argument("--output_csv", default=None)
    args = parser.parse_args()

    result_csv = os.path.join(args.report_dir, f"result_{args.model_slug}.csv")
    output_csv = args.output_csv or result_csv
    log_csv = os.path.join(args.report_dir, "rebuild_failed_gpt120_log.csv")

    target_files = load_target_files(args.targets_csv)
    rows, fieldnames = load_existing_rows(result_csv)
    initial_row_count = len(rows)

    for col in EXPECTED_COLUMNS:
        if col not in fieldnames:
            raise ValueError(f"Schema existing tidak valid, kolom hilang: {col}")

    index_by_file = {normalize_rel_path(r.get("file", "")): i for i, r in enumerate(rows)}

    logs: List[Dict[str, str]] = []
    failed_files: List[str] = []

    for rel_path in sorted(target_files):
        status = "failed"
        message = "unknown"
        attempts = 0
        built_row = None

        for _ in range(2):
            attempts += 1
            try:
                built_row = build_row_from_json(args.input_root, rel_path, force_target_correct=args.force_target_correct)
                ok, msg = validate_row(built_row)
                if ok:
                    status = "success"
                    message = "ok"
                    break
                message = msg
            except Exception as e:
                message = f"exception:{e}"

        if status == "success" and built_row is not None:
            idx = index_by_file.get(rel_path)
            if idx is None:
                status = "failed"
                message = "missing_in_existing_csv"
                failed_files.append(rel_path)
            else:
                rows[idx] = built_row
        else:
            failed_files.append(rel_path)

        logs.append(
            {
                "file": rel_path,
                "status": status,
                "attempts": str(attempts),
                "message": message,
            }
        )

    # Tulis log lebih dulu.
    with open(log_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "status", "attempts", "message"], quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(logs)

    if failed_files:
        print("Ada file yang gagal diproses:")
        for f in failed_files:
            print(f" - {f}")
        print(f"Log: {log_csv}")
        return 2

    # Validasi output file dan retry sekali kalau validasi gagal.
    for _ in range(2):
        write_rows_atomic(output_csv, rows, fieldnames)
        ok, msg = validate_output_file(output_csv, initial_row_count)
        if ok:
            print(f"Semua file target berhasil diproses: {len(target_files)}")
            print(f"CSV terupdate: {output_csv}")
            print(f"Log: {log_csv}")
            return 0
        print(f"Validasi output gagal: {msg}. Retry...")

    print("Gagal validasi output setelah retry.")
    print(f"Log: {log_csv}")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
