"""Build the exact GSM8K Gender Qwen 2,048-token truncation rerun manifest.

Selection is intentionally narrow:

* the legacy per-persona CSV has an empty ``extracted_answer``;
* the current source JSON still has ``finish_reason=length``;
* both requested and consumed completion budgets are exactly 2,048 tokens; and
* the current Qwen parser still cannot recover a final answer.

The builder never calls an API or modifies source JSON/report CSV files. Ground
truth is used by neither selection nor the generated manifest.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path, PurePosixPath


REPO_ROOT = Path(__file__).resolve().parents[1]
PARSER_DIR = REPO_ROOT / "src" / "rebuild"
sys.path.insert(0, str(PARSER_DIR))

import rebuild_results_csv_qwen as qwen_parser  # noqa: E402


MODEL_ID = "qwen/qwen3-next-80b-a3b-thinking"
MODEL_SLUG = "qwen_qwen3-next-80b-a3b-thinking"
GROUP = "Gender_Persona"
EXPECTED_OLD_BLANK = 323
EXPECTED_SELECTED = 317
EXPECTED_PERSONAS = {
    "man_explicit_1": 29,
    "man_implicit_1": 31,
    "man_implicit_2": 42,
    "man_implicit_3": 32,
    "neutral": 27,
    "woman_explicit_1": 37,
    "woman_implicit_1": 45,
    "woman_implicit_2": 34,
    "woman_implicit_3": 40,
}
RAW_ROOT = REPO_ROOT / "result" / "GMS8K" / GROUP / MODEL_SLUG
REPORT_ROOT = REPO_ROOT / "result" / "reports" / MODEL_SLUG
OUTPUT = (
    REPO_ROOT
    / "result"
    / "rerun_manifests"
    / "gsm8k_gender_qwen_truncated_4096"
    / "gsm8k_gender_qwen_truncated_317.csv"
)

FIELDS = [
    "dataset",
    "model_slug",
    "model_id",
    "group",
    "persona_folder",
    "question_id",
    "issue_category",
    "http_code",
    "error_message",
    "source_state",
    "template_relative_path",
    "question_source_relative_path",
    "relative_path",
    "source_sha256",
    "original_temperature",
    "original_max_tokens",
    "original_completion_tokens",
    "original_finish_reason",
    "rerun_temperature",
    "rerun_max_tokens",
    "selection_reason",
]


def first_choice(raw_api: object) -> dict:
    if not isinstance(raw_api, dict):
        return {}
    choices = raw_api.get("choices")
    if not isinstance(choices, list) or not choices:
        return {}
    return choices[0] if isinstance(choices[0], dict) else {}


def normalized_report_relative(value: object) -> PurePosixPath:
    text = str(value or "").strip().replace("\\\\", "/").replace("\\", "/")
    relative = PurePosixPath(text)
    if (
        not text
        or relative.is_absolute()
        or ".." in relative.parts
        or len(relative.parts) != 2
    ):
        raise RuntimeError(f"Unsafe report path: {value!r}")
    return relative


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def report_files() -> list[Path]:
    paths = sorted(REPORT_ROOT.glob("results_*.csv"))
    return [path for path in paths if path.name != "results_qwen3_next_80b.csv"]


def build_manifest() -> tuple[list[dict[str, object]], Counter[str]]:
    csv.field_size_limit(sys.maxsize)
    selected: list[dict[str, object]] = []
    classifications: Counter[str] = Counter()
    seen: set[str] = set()
    old_blank = 0

    paths = report_files()
    if len(paths) != 9:
        raise RuntimeError(f"Expected 9 persona reports, found {len(paths)}")

    for report_path in paths:
        with report_path.open(encoding="utf-8-sig", newline="") as handle:
            for report_row in csv.DictReader(handle):
                if str(report_row.get("extracted_answer") or "").strip():
                    continue
                old_blank += 1
                report_relative = normalized_report_relative(report_row.get("file"))
                persona, filename = report_relative.parts
                if persona != str(report_row.get("persona_folder") or "").strip():
                    raise RuntimeError(f"Persona/report path mismatch: {report_relative}")
                if persona not in EXPECTED_PERSONAS:
                    raise RuntimeError(f"Unexpected persona: {persona}")
                question_id = PurePosixPath(filename).stem
                if filename != f"{question_id}.json" or not question_id.startswith("gsm8k_"):
                    raise RuntimeError(f"Unexpected question filename: {filename}")

                source = RAW_ROOT / persona / filename
                try:
                    payload = json.loads(source.read_text(encoding="utf-8-sig"))
                except (OSError, ValueError) as exc:
                    raise RuntimeError(f"Cannot read source JSON {source}: {exc}") from exc
                response = payload.get("response")
                response = response if isinstance(response, dict) else {}
                raw_api = response.get("raw_api_response")
                raw_api = raw_api if isinstance(raw_api, dict) else {}
                choice = first_choice(raw_api)
                request = payload.get("request")
                request = request if isinstance(request, dict) else {}
                parameters = request.get("parameters")
                parameters = parameters if isinstance(parameters, dict) else {}
                usage = raw_api.get("usage")
                usage = usage if isinstance(usage, dict) else {}
                current_text = qwen_parser.get_response_text(response)
                current_answer, _ = qwen_parser.extract_final_answer(current_text)
                finish_reason = str(choice.get("finish_reason") or "")
                max_tokens = parameters.get("max_tokens")
                completion_tokens = usage.get("completion_tokens")

                if current_answer is not None:
                    classifications["recovered_by_current_parser"] += 1
                    continue
                if finish_reason != "length":
                    classifications[f"excluded_finish_{finish_reason or 'missing'}"] += 1
                    continue
                if max_tokens != 2048 or completion_tokens != 2048:
                    raise RuntimeError(
                        f"Unexpected truncation budget at {source}: "
                        f"max={max_tokens}, completion={completion_tokens}"
                    )
                if parameters.get("model") != MODEL_ID:
                    raise RuntimeError(f"Request model mismatch: {source}")

                relative = PurePosixPath(GROUP, MODEL_SLUG, persona, filename)
                relative_text = relative.as_posix()
                if relative_text in seen:
                    raise RuntimeError(f"Duplicate target: {relative_text}")
                seen.add(relative_text)
                classifications["selected_length_2048_unparsed"] += 1
                selected.append(
                    {
                        "dataset": "GSM8K",
                        "model_slug": MODEL_SLUG,
                        "model_id": MODEL_ID,
                        "group": GROUP,
                        "persona_folder": persona,
                        "question_id": question_id,
                        "issue_category": "truncated_output_2048_no_final_answer",
                        "http_code": "",
                        "error_message": (
                            "finish_reason=length after 2048/2048 completion tokens; "
                            "current parser found no final answer"
                        ),
                        "source_state": "valid",
                        "template_relative_path": "",
                        "question_source_relative_path": "",
                        "relative_path": relative_text,
                        "source_sha256": file_sha256(source),
                        "original_temperature": parameters.get("temperature", ""),
                        "original_max_tokens": max_tokens,
                        "original_completion_tokens": completion_tokens,
                        "original_finish_reason": finish_reason,
                        "rerun_temperature": 0,
                        "rerun_max_tokens": 4096,
                        "selection_reason": "length_2048_and_still_unparsed",
                    }
                )

    if old_blank != EXPECTED_OLD_BLANK:
        raise RuntimeError(
            f"Old-blank guard expected {EXPECTED_OLD_BLANK}, found {old_blank}"
        )
    if len(selected) != EXPECTED_SELECTED:
        raise RuntimeError(
            f"Selection guard expected {EXPECTED_SELECTED}, found {len(selected)}"
        )
    persona_counts = Counter(str(row["persona_folder"]) for row in selected)
    if dict(persona_counts) != EXPECTED_PERSONAS:
        raise RuntimeError(
            f"Persona-count guard failed: expected {EXPECTED_PERSONAS}, "
            f"found {dict(persona_counts)}"
        )
    expected_classifications = Counter(
        {
            "selected_length_2048_unparsed": 317,
            "recovered_by_current_parser": 2,
            "excluded_finish_content_filter": 3,
            "excluded_finish_missing": 1,
        }
    )
    if classifications != expected_classifications:
        raise RuntimeError(
            f"Classification guard failed: {dict(classifications)}"
        )
    selected.sort(key=lambda row: (str(row["persona_folder"]), str(row["question_id"])))
    return selected, classifications


def write_atomic(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.temporary")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    rows, classifications = build_manifest()
    write_atomic(OUTPUT, rows)
    print(f"manifest={OUTPUT}")
    print(f"selected={len(rows)}")
    print(f"personas={dict(Counter(row['persona_folder'] for row in rows))}")
    print(f"classifications={dict(classifications)}")
    print("source_files_modified=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
