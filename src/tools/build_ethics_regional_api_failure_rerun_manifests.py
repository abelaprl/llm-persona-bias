"""Build exact ETHICS Regional manifests for explicit OpenRouter/API failures.

Only GPT-OSS-20B and Qwen3 Next 80B are in scope.  Reasoning-only,
``finish_reason`` failures, content filtering, and truncation are deliberately
excluded unless the stored response also contains an explicit API/transport
error.  The default mode audits without writing; pass ``--write`` to replace
the two generated manifest CSVs.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = (
    REPO_ROOT
    / "result"
    / "ETHICS"
    / "v9 [THIS IS THE MAIN]"
)
GROUP = "Regional Cultural Persona"
OUTPUT_ROOT = REPO_ROOT / "result" / "rerun_manifests" / "ethics_v9"
EXPECTED_PERSONAS = {
    "east_asian_explicit",
    "east_asian_implicit_1",
    "east_asian_implicit_2",
    "east_asian_implicit_3",
    "north_american_explicit",
    "north_american_implicit_1",
    "north_american_implicit_2",
    "north_american_implicit_3",
    "south_asian_explicit",
    "south_asian_implicit_1",
    "south_asian_implicit_2",
    "south_asian_implicit_3",
    "sub_saharan_african_explicit",
    "sub_saharan_african_implicit_1",
    "sub_saharan_african_implicit_2",
    "sub_saharan_african_implicit_3",
}
EXPECTED_FILES_PER_PERSONA = 1514
MODELS = {
    "openai_gpt-oss-20b": {
        "expected": 764,
        "output": "regional_api_failures_openai_gpt-oss-20b.csv",
        "source_report": (
            REPO_ROOT
            / "result"
            / "reports"
            / "v9_main"
            / "regional"
            / "openai_gpt-oss-20b"
            / "results_regional_openai_gpt-oss-20b.csv"
        ),
    },
    "qwen_qwen3-next-80b-a3b-thinking": {
        "expected": 45,
        "output": "regional_api_failures_qwen_qwen3-next-80b-a3b-thinking.csv",
        "source_report": (
            REPO_ROOT
            / "result"
            / "reports"
            / "v9_main"
            / "regional"
            / "qwen_qwen3-next-80b-a3b-thinking"
            / "results_regional_qwen_qwen3-next-80b-a3b-thinking.csv"
        ),
    },
}
FIELDNAMES = (
    "dataset",
    "model_slug",
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
)


@dataclass(frozen=True)
class Candidate:
    model_slug: str
    persona: str
    path: Path


def sanitized_error(value: object) -> str:
    text = str(value or "").strip()
    text = re.sub(r"https?://\S+", "[URL omitted]", text)
    text = re.sub(
        r"(?i)(api[_ -]?key|bearer)\s*[:=]\s*\S+",
        r"\1=[omitted]",
        text,
    )
    return text[:500]


def raw_error(raw_api: object) -> object | None:
    if not isinstance(raw_api, dict):
        return None
    # Choice-level errors are paired with ``finish_reason=error`` in this
    # corpus.  They are intentionally outside the previously reported strict
    # API/transport-failure count and therefore outside this rerun scope.
    return raw_api.get("error")


def error_details(payload: dict[str, Any]) -> tuple[int | None, str] | None:
    response = payload.get("response")
    response = response if isinstance(response, dict) else {}
    response_error = str(response.get("error") or "").strip()
    api_error = raw_error(response.get("raw_api_response"))
    if not response_error and api_error is None:
        return None

    api_message = ""
    code: int | None = None
    if isinstance(api_error, dict):
        api_message = str(api_error.get("message") or "").strip()
        metadata = api_error.get("metadata")
        if not api_message and isinstance(metadata, dict):
            api_message = str(metadata.get("raw") or "").strip()
        try:
            candidate = int(api_error.get("code"))
            code = candidate if 400 <= candidate <= 599 else None
        except (TypeError, ValueError):
            pass
    elif api_error is not None:
        api_message = str(api_error).strip()

    combined = " | ".join(part for part in (response_error, api_message) if part)
    if code is None:
        match = re.search(r"HTTP\s+([45]\d\d)", combined, re.IGNORECASE)
        code = int(match.group(1)) if match else None
    return code, sanitized_error(combined) or "explicit_api_or_transport_error"


def inspect(candidate: Candidate) -> dict[str, object] | None:
    with candidate.path.open(encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Top-level JSON is not an object: {candidate.path}")
    details = error_details(payload)
    if details is None:
        return None
    code, message = details
    relative = candidate.path.relative_to(RESULT_ROOT).as_posix()
    return {
        "dataset": "ETHICS",
        "model_slug": candidate.model_slug,
        "group": GROUP,
        "persona_folder": candidate.persona,
        "question_id": candidate.path.stem,
        "issue_category": f"http_{code}" if code is not None else "network_transport",
        "http_code": code or "",
        "error_message": message,
        "source_state": "valid",
        "template_relative_path": "",
        "question_source_relative_path": "",
        "relative_path": relative,
    }


def collect_candidates() -> list[Candidate]:
    candidates: list[Candidate] = []
    seen: set[Path] = set()
    for model_slug, spec in MODELS.items():
        model_dir = RESULT_ROOT / GROUP / model_slug
        persona_dirs = sorted(path for path in model_dir.iterdir() if path.is_dir())
        persona_names = {path.name for path in persona_dirs}
        if persona_names != EXPECTED_PERSONAS:
            raise RuntimeError(
                f"Unexpected persona set for {model_slug}: "
                f"missing={sorted(EXPECTED_PERSONAS - persona_names)}, "
                f"extra={sorted(persona_names - EXPECTED_PERSONAS)}"
            )
        for persona_dir in persona_dirs:
            paths = sorted(persona_dir.glob("*.json"))
            if len(paths) != EXPECTED_FILES_PER_PERSONA:
                raise RuntimeError(
                    f"{model_slug}/{persona_dir.name} expected "
                    f"{EXPECTED_FILES_PER_PERSONA} JSON, found {len(paths)}"
                )
        source_report = Path(spec["source_report"])
        with source_report.open(newline="", encoding="utf-8-sig") as handle:
            for row_number, row in enumerate(csv.DictReader(handle), start=2):
                if str(row.get("answer_status") or "").strip() != "missing":
                    continue
                persona = str(row.get("persona_folder") or "").strip()
                file_name = str(row.get("file") or "").strip()
                if persona not in EXPECTED_PERSONAS or Path(file_name).name != file_name:
                    raise RuntimeError(
                        f"Unsafe report target at row {row_number}: {persona}/{file_name}"
                    )
                relative = Path(GROUP) / model_slug / persona / file_name
                path = (RESULT_ROOT / relative).resolve()
                if RESULT_ROOT.resolve() not in path.parents or not path.is_file():
                    raise RuntimeError(
                        f"Missing or unsafe target at row {row_number}: {path}"
                    )
                if path in seen:
                    continue
                seen.add(path)
                candidates.append(Candidate(model_slug, persona, path))
    return candidates


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--expected-gpt", type=int, default=764)
    parser.add_argument("--expected-qwen", type=int, default=45)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.expected_gpt < 0 or args.expected_qwen < 0:
        parser.error("expected counts must be zero or greater")

    expected_counts = {
        "openai_gpt-oss-20b": args.expected_gpt,
        "qwen_qwen3-next-80b-a3b-thinking": args.expected_qwen,
    }

    candidates = collect_candidates()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        inspected = list(executor.map(inspect, candidates))
    rows = [row for row in inspected if row is not None]
    rows.sort(key=lambda row: str(row["relative_path"]))

    by_model: dict[str, list[dict[str, object]]] = {
        model_slug: [] for model_slug in MODELS
    }
    for row in rows:
        by_model[str(row["model_slug"])].append(row)

    for model_slug, spec in MODELS.items():
        model_rows = by_model[model_slug]
        expected = expected_counts[model_slug]
        if len(model_rows) != expected:
            raise RuntimeError(
                f"Guard expected {expected} explicit failures for {model_slug}, "
                f"found {len(model_rows)}"
            )
        issues = Counter(str(row["issue_category"]) for row in model_rows)
        output = OUTPUT_ROOT / str(spec["output"])
        print(f"{model_slug}: count={len(model_rows)} issues={dict(sorted(issues.items()))}")
        print(f"  manifest={output}")
        if args.write:
            write_csv(output, model_rows)

    print(f"Scanned={len(candidates)} explicit_failures={len(rows)} write={args.write}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
