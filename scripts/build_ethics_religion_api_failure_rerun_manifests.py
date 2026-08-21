"""Build exact ETHICS Religion manifests for explicit API failures.

Only GPT-OSS-20B and Qwen3 Next 80B are in scope. DeepSeek, Neutral,
``finish_reason`` failures, content filtering, and truncation are excluded.
The default mode validates without writing; pass ``--write`` to create the
two model-specific manifests used by the audited OpenRouter rerun runner.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import build_ethics_regional_api_failure_rerun_manifests as common


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = REPO_ROOT / "result" / "ETHICS" / "v9 [THIS IS THE MAIN]"
GROUP = "Religion Persona"
OUTPUT_ROOT = REPO_ROOT / "result" / "rerun_manifests" / "ethics_v9"
EXPECTED_PERSONAS = {
    "atheist_explicit",
    "atheist_implicit_1",
    "atheist_implicit_2",
    "atheist_implicit_3",
    "christian_explicit",
    "christian_implicit_1",
    "christian_implicit_2",
    "christian_implicit_3",
    "islam_explicit",
    "islam_implicit_1",
    "islam_implicit_2",
    "islam_implicit_3",
    "jewish_explicit",
    "jewish_implicit_1",
    "jewish_implicit_2",
    "jewish_implicit_3",
}
EXPECTED_FILES_PER_PERSONA = 1514
MODELS = {
    "openai_gpt-oss-20b": {
        "expected": 4,
        "output": "religion_api_failures_openai_gpt-oss-20b.csv",
    },
    "qwen_qwen3-next-80b-a3b-thinking": {
        "expected": 86,
        "output": "religion_api_failures_qwen_qwen3-next-80b-a3b-thinking.csv",
    },
}


def inspect(candidate: common.Candidate) -> dict[str, object] | None:
    with candidate.path.open(encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Top-level JSON is not an object: {candidate.path}")
    details = common.error_details(payload)
    if details is None:
        return None
    code, message = details
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
        "relative_path": candidate.path.relative_to(RESULT_ROOT).as_posix(),
    }


def collect_candidates() -> list[common.Candidate]:
    candidates: list[common.Candidate] = []
    seen: set[Path] = set()
    for model_slug in MODELS:
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
            count = len(list(persona_dir.glob("*.json")))
            if count != EXPECTED_FILES_PER_PERSONA:
                raise RuntimeError(
                    f"{model_slug}/{persona_dir.name} expected "
                    f"{EXPECTED_FILES_PER_PERSONA} JSON, found {count}"
                )

        report = (
            REPO_ROOT
            / "result"
            / "reports"
            / "v9_main"
            / "religion"
            / model_slug
            / f"results_religion_{model_slug}.csv"
        )
        with report.open(newline="", encoding="utf-8-sig") as handle:
            for row_number, row in enumerate(csv.DictReader(handle), start=2):
                if str(row.get("answer_status") or "").strip() != "missing":
                    continue
                persona = str(row.get("persona_folder") or "").strip()
                file_name = str(row.get("file") or "").strip()
                if persona not in EXPECTED_PERSONAS or Path(file_name).name != file_name:
                    raise RuntimeError(
                        f"Unsafe report target at row {row_number}: {persona}/{file_name}"
                    )
                path = (model_dir / persona / file_name).resolve()
                if model_dir.resolve() not in path.parents or not path.is_file():
                    raise RuntimeError(
                        f"Missing or unsafe target at row {row_number}: {path}"
                    )
                if path in seen:
                    continue
                seen.add(path)
                candidates.append(common.Candidate(model_slug, persona, path))
    return candidates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--expected-gpt", type=int, default=4)
    parser.add_argument("--expected-qwen", type=int, default=86)
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

    by_model = {model_slug: [] for model_slug in MODELS}
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
            common.write_csv(output, model_rows)

    print(f"Scanned={len(candidates)} explicit_failures={len(rows)} write={args.write}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
