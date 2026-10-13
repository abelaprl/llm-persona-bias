"""Audit one main TA model and build exact infrastructure-failure manifests.

The scanner is read-only for result JSON files.  It writes CSV manifests only
under ``result/rerun_manifests/<model>_{nonregional,regional}``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.tools import build_qwen3_next_80b_nonregional_manifest as common  # noqa: E402


MODEL_CATALOG = {
    31: {
        "model_id": "openai/gpt-oss-20b",
        "model_slug": "openai_gpt-oss-20b",
        "short_name": "gpt_oss_20b",
        # Reasoning-only ETHICS responses may be recoverable by local reparse and
        # must not be mixed into the endpoint/infrastructure rerun manifest.
        "excluded_rerun_issues": ("reasoning_without_final",),
        "expected_nonregional": 118,
        "expected_regional": 5558,
    },
    32: {
        "model_id": "qwen/qwen3-next-80b-a3b-thinking",
        "model_slug": "qwen_qwen3-next-80b-a3b-thinking",
        "short_name": "qwen3_next_80b",
    },
    35: {
        "model_id": "meta-llama/llama-3.3-70b-instruct",
        "model_slug": "meta-llama_llama-3.3-70b-instruct",
        "short_name": "llama33_70b",
        "expected_nonregional": 7658,
        "expected_regional": 594,
    },
    40: {
        "model_id": "meta-llama/llama-3.1-8b-instruct",
        "model_slug": "meta-llama_llama-3.1-8b-instruct",
        "short_name": "llama31_8b",
        "expected_nonregional": 6445,
        "expected_regional": 5002,
    },
    41: {
        "model_id": "mistralai/ministral-8b-2512",
        "model_slug": "mistralai_ministral-8b-2512",
        "short_name": "ministral_8b",
        "expected_nonregional": 6336,
        "expected_regional": 4125,
    },
    43: {
        "model_id": "deepseek/deepseek-r1-distill-qwen-32b",
        "model_slug": "deepseek_deepseek-r1-distill-qwen-32b",
        "short_name": "deepseek_r1_distill_32b",
    },
}

ETHICS_ROOT = REPO_ROOT / "result" / "benchmarks" / "ethics" / "runs" / "main" / "v9"
GSM8K_ROOT = REPO_ROOT / "result" / "benchmarks" / "gsm8k" / "runs" / "main"
ETHICS_SAMPLES = REPO_ROOT / "data" / "ETHICS_test_4subsets_10percent_from200.json"

GENDER_PERSONAS = {
    "man_explicit_1",
    "man_implicit_1",
    "man_implicit_2",
    "man_implicit_3",
    "woman_explicit_1",
    "woman_implicit_1",
    "woman_implicit_2",
    "woman_implicit_3",
}
RELIGION_PERSONAS = {
    f"{religion}_{kind}{suffix}"
    for religion in ("atheist", "christian", "islam", "jewish")
    for kind, suffix in (
        ("explicit", ""),
        ("implicit", "_1"),
        ("implicit", "_2"),
        ("implicit", "_3"),
    )
}
REGIONAL_PERSONAS = {
    f"{region}_{kind}{suffix}"
    for region in (
        "east_asian",
        "north_american",
        "south_asian",
        "sub_saharan_african",
    )
    for kind, suffix in (
        ("explicit", ""),
        ("implicit", "_1"),
        ("implicit", "_2"),
        ("implicit", "_3"),
    )
}

SCOPE_SPECS = {
    "nonregional": (
        ("ETHICS", "Gender Persona", GENDER_PERSONAS, 1514),
        ("ETHICS", "Neutral Persona", {"neutral"}, 1514),
        ("ETHICS", "Religion Persona", RELIGION_PERSONAS, 1514),
        ("GSM8K", "Gender_Persona", GENDER_PERSONAS | {"neutral"}, 1319),
        ("GSM8K", "Religion_Persona", RELIGION_PERSONAS, 1319),
    ),
    "regional": (
        ("ETHICS", "Regional Cultural Persona", REGIONAL_PERSONAS, 1514),
        ("GSM8K", "regional_cultural_persona", REGIONAL_PERSONAS, 1319),
    ),
}


def text_value(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        chunks: list[str] = []
        for item in value:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    chunks.append(text)
        return "\n".join(chunks).strip()
    return ""


def final_and_reasoning(payload: dict[str, Any]) -> tuple[str, str]:
    response = payload.get("response")
    response = response if isinstance(response, dict) else {}
    prediction = payload.get("prediction")
    prediction = prediction if isinstance(prediction, dict) else {}
    final = text_value(response.get("content")) or text_value(prediction.get("raw"))
    reasoning = ""
    raw = response.get("raw_api_response")
    if isinstance(raw, dict):
        choices = raw.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            message = choices[0].get("message")
            if isinstance(message, dict):
                final = final or text_value(message.get("content"))
                reasoning = text_value(message.get("reasoning")) or text_value(
                    message.get("reasoning_content")
                )
    return final, reasoning


def include_empty_final(item: common.Inspection) -> common.Inspection:
    if item.issue_category is not None or item.payload is None:
        return item
    final, reasoning = final_and_reasoning(item.payload)
    if final:
        return item
    category = "reasoning_without_final" if reasoning else "empty_final_content"
    message = (
        "Assistant reasoning exists but final content is empty"
        if reasoning
        else "Assistant final content is empty"
    )
    return replace(item, issue_category=category, error_message=message)


def prefilter_candidates(files: list[common.FileMeta]) -> list[common.FileMeta]:
    """Use ripgrep's streaming reader before opening candidate JSON in Python."""
    if not files:
        return []
    roots = sorted(
        {
            str(
                (ETHICS_ROOT if item.dataset == "ETHICS" else GSM8K_ROOT)
                / item.group
                / common.MODEL_SLUG
            )
            for item in files
        }
    )
    command = [
        "rg",
        "-l",
        "-g",
        "*.json",
        "-e",
        r'"success"\s*:\s*false',
        "-e",
        r'"content"\s*:\s*(null|"")',
        "-e",
        r'"raw"\s*:\s*(null|"")',
        "-e",
        r'"error"\s*:\s*\{',
        *roots,
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode not in (0, 1):
        raise RuntimeError(f"rg prefilter failed: {completed.stderr.strip()}")
    candidate_paths = {
        Path(line.strip()).resolve()
        for line in completed.stdout.splitlines()
        if line.strip()
    }
    by_path = {item.path.resolve(): item for item in files}
    unknown = candidate_paths - set(by_path)
    if unknown:
        # Warmups are expected to appear in rg output but are audited separately.
        unknown = {
            path for path in unknown if path.name != "persona_warmup.json"
        }
    if unknown:
        raise RuntimeError(f"Prefilter returned unexpected paths: {sorted(unknown)[:5]}")
    return [by_path[path] for path in sorted(candidate_paths & set(by_path))]


def recovery_support_files(files: list[common.FileMeta]) -> list[common.FileMeta]:
    """Load the minimum intact GSM8K templates needed for missing-file recovery."""
    gsm = [item for item in files if item.dataset == "GSM8K"]
    by_persona: dict[tuple[str, str], list[common.FileMeta]] = {}
    by_question: dict[str, list[common.FileMeta]] = {}
    for item in gsm:
        by_persona.setdefault((item.group, item.persona), []).append(item)
        by_question.setdefault(item.path.stem, []).append(item)
    support: dict[Path, common.FileMeta] = {}
    for (group, persona), items in by_persona.items():
        existing = {common.qid_number(item.path.stem) for item in items}
        for number in sorted(set(range(1, 1320)) - existing):
            question_id = f"gsm8k_{number:05d}"
            template = min(
                items,
                key=lambda item: abs(common.qid_number(item.path.stem) - number),
            )
            sources = by_question.get(question_id, [])
            if not sources:
                raise RuntimeError(f"No GSM8K question source for {question_id}")
            question_source = sorted(
                sources,
                key=lambda item: (
                    item.group != "Gender_Persona",
                    item.persona != "man_explicit_1",
                    item.relative_path.as_posix(),
                ),
            )[0]
            support[template.path.resolve()] = template
            support[question_source.path.resolve()] = question_source
    return list(support.values())


def expected_ethics_ids() -> set[str]:
    data = json.loads(ETHICS_SAMPLES.read_text(encoding="utf-8-sig"))
    ids = {
        str(row.get("id") or "").strip()
        for row in data
        if isinstance(row, dict) and str(row.get("id") or "").strip()
    }
    if len(ids) != 1514:
        raise RuntimeError(f"Expected 1514 ETHICS IDs, found {len(ids)}")
    return ids


def collect_scope(
    model_slug: str,
    scope: str,
    selected_groups: set[str] | None = None,
) -> tuple[list[common.FileMeta], list[dict[str, object]], list[str], list[common.FileMeta]]:
    files: list[common.FileMeta] = []
    warmups: list[common.FileMeta] = []
    inventory: list[dict[str, object]] = []
    missing_ethics: list[str] = []
    ethics_ids = expected_ethics_ids()
    for dataset, group, expected_personas, expected_count in SCOPE_SPECS[scope]:
        if selected_groups is not None and group not in selected_groups:
            continue
        root = ETHICS_ROOT if dataset == "ETHICS" else GSM8K_ROOT
        model_dir = root / group / model_slug
        if not model_dir.is_dir():
            raise RuntimeError(f"Missing model directory: {model_dir}")
        persona_dirs = {path.name: path for path in model_dir.iterdir() if path.is_dir()}
        if set(persona_dirs) != expected_personas:
            raise RuntimeError(
                f"Unexpected {dataset}/{group} personas: "
                f"missing={sorted(expected_personas - set(persona_dirs))}, "
                f"extra={sorted(set(persona_dirs) - expected_personas)}"
            )
        for persona in sorted(expected_personas):
            persona_dir = persona_dirs[persona]
            paths = sorted(persona_dir.glob("*.json"))
            inventory.append(
                {
                    "dataset": dataset,
                    "group": group,
                    "persona": persona,
                    "count": len(paths),
                    "expected": expected_count,
                }
            )
            existing_ids = {path.stem for path in paths}
            if dataset == "ETHICS":
                missing = sorted(ethics_ids - existing_ids)
                extra = sorted(existing_ids - ethics_ids)
                if extra:
                    raise RuntimeError(
                        f"Unexpected ETHICS IDs in {group}/{persona}: {extra[:10]}"
                    )
                missing_ethics.extend(f"{group}/{persona}/{qid}" for qid in missing)
            for path in paths:
                relative = PurePosixPath(path.relative_to(root).as_posix())
                files.append(common.FileMeta(dataset, group, persona, path, relative))
            warmup_path = persona_dir / "warmup" / "persona_warmup.json"
            if warmup_path.is_file():
                relative = PurePosixPath(warmup_path.relative_to(root).as_posix())
                warmups.append(
                    common.FileMeta(dataset, group, persona, warmup_path, relative)
                )
    return files, inventory, missing_ethics, warmups


def build_scope_rows(
    inspections: list[common.Inspection],
    inventory: list[dict[str, object]],
    excluded_issues: set[str] | None = None,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    all_rows, _ = common.build_rows(inspections, inventory)
    excluded_issues = excluded_issues or set()
    review = [
        row for row in all_rows if str(row["issue_category"]) in excluded_issues
    ]
    rows = [
        row for row in all_rows if str(row["issue_category"]) not in excluded_issues
    ]
    ethics = [row for row in rows if row["dataset"] == "ETHICS"]
    gsm8k = [row for row in rows if row["dataset"] == "GSM8K"]
    return rows, ethics, gsm8k, review


def group_slug(group: str) -> str:
    # Keep directory names that differ only by spaces versus underscores distinct.
    # For example, ``Gender Persona`` is ETHICS while ``Gender_Persona`` is GSM8K.
    distinct_group = group.lower().replace("_", "_underscore_")
    return re.sub(r"[^a-z0-9]+", "_", distinct_group).strip("_")


def write_rows(output_dir: Path, prefix: str, rows: list[dict[str, object]]) -> None:
    ethics = [row for row in rows if row["dataset"] == "ETHICS"]
    gsm8k = [row for row in rows if row["dataset"] == "GSM8K"]
    common.write_manifest(output_dir / f"{prefix}ethics_failures.csv", ethics)
    common.write_manifest(output_dir / f"{prefix}gsm8k_failures.csv", gsm8k)
    common.write_manifest(output_dir / f"{prefix}all_failures.csv", rows)


def read_manifest(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def merge_partials(
    output_dir: Path,
    groups: list[str],
    expected: int,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    partial_dir = output_dir / "partials"
    rows: list[dict[str, object]] = []
    review: list[dict[str, object]] = []
    for group in groups:
        path = partial_dir / f"{group_slug(group)}_all_failures.csv"
        if not path.is_file():
            raise RuntimeError(f"Missing partial manifest: {path}")
        rows.extend(read_manifest(path))
        review_path = partial_dir / f"{group_slug(group)}_review_only.csv"
        if review_path.is_file():
            review.extend(read_manifest(review_path))
    rows.sort(
        key=lambda row: (
            str(row["dataset"]),
            str(row["group"]),
            str(row["persona_folder"]),
            str(row["question_id"]),
        )
    )
    keys = [(str(row["dataset"]), str(row["relative_path"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Merged partial manifests contain duplicate targets")
    if expected and len(rows) != expected:
        raise RuntimeError(f"Merged guard expected {expected}, found {len(rows)}")
    write_rows(output_dir, "model_", rows)
    review.sort(
        key=lambda row: (
            str(row["dataset"]),
            str(row["group"]),
            str(row["persona_folder"]),
            str(row["question_id"]),
        )
    )
    review_keys = [(str(row["dataset"]), str(row["relative_path"])) for row in review]
    if len(review_keys) != len(set(review_keys)):
        raise RuntimeError("Merged review-only manifests contain duplicate targets")
    common.write_manifest(output_dir / "model_review_only.csv", review)
    ethics = [row for row in rows if row["dataset"] == "ETHICS"]
    gsm8k = [row for row in rows if row["dataset"] == "GSM8K"]
    return rows, ethics, gsm8k, review


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-order", type=int, required=True, choices=MODEL_CATALOG)
    parser.add_argument("--scope", choices=("all", "nonregional", "regional"), default="all")
    parser.add_argument(
        "--group",
        action="append",
        default=[],
        help="Audit only one or more exact group names and write partial manifests.",
    )
    parser.add_argument(
        "--merge-partials",
        action="store_true",
        help="Merge all per-group partials for the selected scope without rescanning JSON.",
    )
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument(
        "--full-scan",
        action="store_true",
        help="Parse every JSON instead of prefiltering likely failures with rg.",
    )
    parser.add_argument("--expected-nonregional", type=int, default=0)
    parser.add_argument("--expected-regional", type=int, default=0)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    model = MODEL_CATALOG[args.model_order]
    model_slug = str(model["model_slug"])
    excluded_rerun_issues = {
        str(issue) for issue in model.get("excluded_rerun_issues", ())
    }
    common.MODEL_SLUG = model_slug
    common.MODEL_ID = str(model["model_id"])
    selected_scopes = (
        ("nonregional", "regional") if args.scope == "all" else (args.scope,)
    )
    requested_groups = set(args.group)
    known_groups = {
        group
        for specs in SCOPE_SPECS.values()
        for _, group, _, _ in specs
    }
    unknown_groups = requested_groups - known_groups
    if unknown_groups:
        parser.error(f"unknown --group values: {sorted(unknown_groups)}")
    if args.merge_partials and requested_groups:
        parser.error("--merge-partials cannot be combined with --group")

    overall: Counter[str] = Counter()
    for scope in selected_scopes:
        scope_groups = [group for _, group, _, _ in SCOPE_SPECS[scope]]
        selected_groups = requested_groups & set(scope_groups)
        if requested_groups and not selected_groups:
            continue
        output_dir = (
            REPO_ROOT
            / "result"
            / "rerun_manifests"
            / f"{model['short_name']}_{scope}"
        )
        expected_override = (
            args.expected_nonregional if scope == "nonregional" else args.expected_regional
        )
        expected = expected_override or int(model.get(f"expected_{scope}", 0))
        if args.merge_partials:
            rows, ethics, gsm8k, review = merge_partials(
                output_dir, scope_groups, expected
            )
            issues = Counter(str(row["issue_category"]) for row in rows)
            review_issues = Counter(
                str(row["issue_category"]) for row in review
            )
            overall.update(issues)
            print(f"[{scope} merged]")
            print(f"ETHICS failures   : {len(ethics)}")
            print(f"GSM8K failures    : {len(gsm8k)}")
            print(f"Combined failures : {len(rows)}")
            print(f"Issues            : {dict(sorted(issues.items()))}")
            print(f"Review-only rows  : {len(review)}")
            print(f"Review issues     : {dict(sorted(review_issues.items()))}")
            print(f"Output directory  : {output_dir}")
            print()
            continue
        files, inventory, missing_ethics, warmups = collect_scope(
            model_slug,
            scope,
            selected_groups if requested_groups else None,
        )
        if missing_ethics:
            raise RuntimeError(
                f"ETHICS missing files are not safely reconstructable: {missing_ethics[:10]}"
            )
        candidates = files if args.full_scan else prefilter_candidates(files)
        if not args.full_scan:
            existing_candidates = {item.path.resolve() for item in candidates}
            candidates.extend(
                item
                for item in recovery_support_files(files)
                if item.path.resolve() not in existing_candidates
            )
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            inspected = list(executor.map(common.inspect_file, candidates))
            warmup_inspected = list(executor.map(common.inspect_file, warmups))
        inspected = [include_empty_final(item) for item in inspected]
        warmup_inspected = [include_empty_final(item) for item in warmup_inspected]
        rows, ethics, gsm8k, review = build_scope_rows(
            inspected, inventory, excluded_rerun_issues
        )
        if not requested_groups and expected and len(rows) != expected:
            raise RuntimeError(
                f"{scope} guard expected {expected}, found {len(rows)}"
            )
        if requested_groups:
            if len(selected_groups) != 1:
                raise RuntimeError("Run exactly one --group per partial audit")
            partial_dir = output_dir / "partials"
            prefix = f"{group_slug(next(iter(selected_groups)))}_"
            write_rows(partial_dir, prefix, rows)
            common.write_manifest(
                partial_dir / f"{prefix}review_only.csv", review
            )
        else:
            write_rows(output_dir, "model_", rows)
            common.write_manifest(output_dir / "model_review_only.csv", review)
        issues = Counter(str(row["issue_category"]) for row in rows)
        review_issues = Counter(
            str(row["issue_category"]) for row in review
        )
        groups = Counter(str(row["group"]) for row in rows)
        warmup_issues = Counter(
            str(item.issue_category)
            for item in warmup_inspected
            if item.issue_category is not None
        )
        overall.update(issues)
        print(f"[{scope}]")
        print(f"Scanned main JSON : {len(files)}")
        print(f"Parsed candidates : {len(candidates)}")
        print(f"ETHICS failures   : {len(ethics)}")
        print(f"GSM8K failures    : {len(gsm8k)}")
        print(f"Combined failures : {len(rows)}")
        print(f"Issues            : {dict(sorted(issues.items()))}")
        print(f"Review-only rows  : {len(review)}")
        print(f"Review issues     : {dict(sorted(review_issues.items()))}")
        print(f"Groups            : {dict(sorted(groups.items()))}")
        print(f"Warmups scanned   : {len(warmups)}")
        print(f"Warmup failures   : {dict(sorted(warmup_issues.items()))}")
        print(f"Output directory  : {output_dir}")
        print()
    print(f"Overall issues     : {dict(sorted(overall.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
