"""Build source-locked GSM8K Gender and Religion combined CSVs.

The selected sources and their SHA-256 values are taken from the publication
v5 provenance manifest.  This script never reparses answers and never changes
the designated accuracy.  It only normalizes stable keys, adds source-row
provenance, validates counts, and writes Gender/Religion to separate versioned
directories.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VERSION = "publication_source_aligned_v1_20260817"
REPORT_ROOT = REPO_ROOT / "result" / "GMS8K" / "reports"
V5_TABLE_ROOT = (
    REPORT_ROOT
    / "persona_publication_analysis_v5_paired_significance_six_models"
    / "tables"
)
V5_SOURCE_MANIFEST = V5_TABLE_ROOT / "source_manifest.csv"
V5_ACCURACY = V5_TABLE_ROOT / "source_consistency_audit.csv"
V5_INVENTORY = V5_TABLE_ROOT / "data_inventory_six_models.csv"

MODELS = (
    "qwen_qwen3-next-80b-a3b-thinking",
    "meta-llama_llama-3.1-8b-instruct",
    "mistralai_ministral-8b-2512",
    "deepseek_deepseek-r1-distill-qwen-32b",
    "openai_gpt-oss-20b",
    "meta-llama_llama-3.3-70b-instruct",
)
GENDER_PERSONAS = (
    "neutral",
    "man_explicit_1",
    "man_implicit_1",
    "man_implicit_2",
    "man_implicit_3",
    "woman_explicit_1",
    "woman_implicit_1",
    "woman_implicit_2",
    "woman_implicit_3",
)
RELIGION_PERSONAS = (
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
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def truth(value) -> bool:
    return str(value or "").strip().lower() in {"true", "1", "yes"}


def stable_id(file_value: str) -> str:
    match = re.search(r"gsm8k_(\d+)\.json$", file_value.replace("\\", "/"))
    if not match:
        raise ValueError(f"invalid GSM8K file key: {file_value!r}")
    return f"gsm8k_{int(match.group(1)):05d}"


def atomic_dict_csv(path: Path, fields, row_iterator, quote_all=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    try:
        with open(temporary, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fields,
                extrasaction="raise",
                quoting=csv.QUOTE_ALL if quote_all else csv.QUOTE_MINIMAL,
            )
            writer.writeheader()
            writer.writerows(row_iterator)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_v5_source_rows(dataset: str):
    with V5_SOURCE_MANIFEST.open(encoding="utf-8-sig", newline="") as handle:
        return [
            row
            for row in csv.DictReader(handle)
            if row.get("dataset") == dataset and row.get("model") in MODELS
        ]


def verify_sources(dataset: str):
    rows = load_v5_source_rows(dataset)
    expected = 54 if dataset == "Gender" else 6
    if len(rows) != expected:
        raise RuntimeError(f"v5 {dataset} manifest rows={len(rows)}, expected={expected}")
    verified = []
    hash_cache = {}
    for row in rows:
        path = REPO_ROOT / row["relative_path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = hash_cache.setdefault(path, sha256_file(path))
        expected_hash = row.get("sha256") or ""
        if actual != expected_hash:
            raise RuntimeError(
                f"source changed after publication v5: {relative(path)}\n"
                f"expected {expected_hash}\nactual   {actual}"
            )
        verified.append(
            {
                "dataset": dataset,
                "model": row["model"],
                "source_path": relative(path),
                "bytes": path.stat().st_size,
                "modified_local": datetime.fromtimestamp(path.stat().st_mtime)
                .astimezone()
                .isoformat(timespec="seconds"),
                "expected_sha256_v5": expected_hash,
                "actual_sha256": actual,
                "hash_match": True,
            }
        )
    return rows, verified


def gender_assignments(manifest_rows):
    by_model = {model: [] for model in MODELS}
    for row in manifest_rows:
        by_model[row["model"]].append(REPO_ROOT / row["relative_path"])

    assignments = []
    for model in MODELS:
        paths = by_model[model]
        for persona in GENDER_PERSONAS:
            direct = next(
                (path for path in paths if path.name == f"results_{persona}.csv"),
                None,
            )
            if direct is None and model == "openai_gpt-oss-20b" and persona == "neutral":
                direct = next(
                    path for path in paths if path.name == "results_openai-gpt-oss_20b.csv"
                )
            if direct is None:
                raise RuntimeError(f"no v5 source assignment for {model}/{persona}")
            assignments.append((model, persona, direct))
    return assignments


def gender_header(assignments):
    fields = []
    for _, _, path in assignments:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            header = next(csv.reader(handle))
        for field in header:
            if field and field not in fields:
                fields.append(field)
    prefix = [
        "model",
        "persona_folder",
        "file",
        "question_id",
        "stable_question_id",
        "source_file",
        "source_row",
    ]
    return prefix + [field for field in fields if field not in prefix]


def build_gender(output_dir: Path, manifest_rows):
    assignments = gender_assignments(manifest_rows)
    fields = gender_header(assignments)
    output = output_dir / "all_models_results_full.csv"
    counts = Counter()
    keys = set()
    correct = Counter()

    def rows():
        for model, persona, path in assignments:
            with path.open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if "response_content_full" not in (reader.fieldnames or []):
                    raise RuntimeError(f"full response text missing: {path}")
                for source_row, row in enumerate(reader, 2):
                    row_persona = (row.get("persona_folder") or "").strip()
                    if row_persona != persona:
                        continue
                    qid = stable_id(row.get("file") or "")
                    key = (model, persona, qid)
                    if key in keys:
                        raise RuntimeError(f"duplicate Gender key: {key}")
                    keys.add(key)
                    counts[(model, persona)] += 1
                    correct[model] += int(truth(row.get("is_correct")))
                    normalized = {field: row.get(field, "") for field in fields}
                    normalized.update(
                        {
                            "model": model,
                            "persona_folder": persona,
                            "question_id": qid,
                            "stable_question_id": qid,
                            "source_file": relative(path),
                            "source_row": source_row,
                        }
                    )
                    yield normalized

    atomic_dict_csv(output, fields, rows())
    total = sum(counts.values())
    if total != 71_215 or len(keys) != total:
        raise RuntimeError(f"Gender combined rows/keys invalid: {total}/{len(keys)}")
    if any(not 1309 <= count <= 1319 for count in counts.values()):
        raise RuntimeError(f"Gender condition coverage outside v5 guard: {counts}")
    return output, counts, correct


def religion_header(manifest_rows):
    path = REPO_ROOT / manifest_rows[0]["relative_path"]
    with path.open(encoding="utf-8-sig", newline="") as handle:
        source = next(csv.reader(handle))
    prefix = [
        "model",
        "persona_folder",
        "persona",
        "file",
        "question_id",
        "stable_question_id",
        "source_file",
        "source_row",
    ]
    return prefix + [field for field in source if field not in prefix]


def build_religion(output_dir: Path, manifest_rows):
    paths = {row["model"]: REPO_ROOT / row["relative_path"] for row in manifest_rows}
    fields = religion_header(manifest_rows)
    output = output_dir / "all_models_results_full.csv"
    counts = Counter()
    keys = set()
    correct = Counter()

    def rows():
        for model in MODELS:
            path = paths[model]
            with path.open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                for source_row, row in enumerate(reader, 2):
                    persona = (row.get("persona") or "").strip()
                    if persona not in RELIGION_PERSONAS:
                        raise RuntimeError(f"unexpected Religion persona: {persona!r}")
                    qid = stable_id(row.get("file") or "")
                    key = (model, persona, qid)
                    if key in keys:
                        raise RuntimeError(f"duplicate Religion key: {key}")
                    keys.add(key)
                    counts[(model, persona)] += 1
                    correct[model] += int(truth(row.get("is_correct")))
                    normalized = {field: row.get(field, "") for field in fields}
                    normalized.update(
                        {
                            "model": model,
                            "persona_folder": persona,
                            "persona": persona,
                            "question_id": qid,
                            "stable_question_id": qid,
                            "source_file": relative(path),
                            "source_row": source_row,
                        }
                    )
                    yield normalized

    atomic_dict_csv(output, fields, rows())
    expected = 6 * 16 * 1319
    if sum(counts.values()) != expected or len(keys) != expected:
        raise RuntimeError("Religion combined row/key count is not 126624")
    if any(count != 1319 for count in counts.values()):
        raise RuntimeError(f"Religion condition coverage invalid: {counts}")
    return output, counts, correct


def load_accuracy_rows(dataset):
    with V5_ACCURACY.open(encoding="utf-8-sig", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("dataset") == dataset and row.get("model") in MODELS
        ]
    expected = len(MODELS) * (9 if dataset == "Gender" else 16)
    if len(rows) != expected:
        raise RuntimeError(f"v5 accuracy rows for {dataset}: {len(rows)} != {expected}")
    return rows


def load_inventory_row(dataset):
    with V5_INVENTORY.open(encoding="utf-8-sig", newline="") as handle:
        return [
            row
            for row in csv.DictReader(handle)
            if row.get("dataset") == dataset and row.get("model") in MODELS
        ]


def write_summary_files(output_dir, dataset, counts, correct, source_manifest):
    accuracy = load_accuracy_rows(dataset)
    accuracy_path = output_dir / "accuracy_by_persona_v5.csv"
    if not accuracy_path.exists():
        atomic_dict_csv(
            accuracy_path, list(accuracy[0]), accuracy, quote_all=False
        )
    inventory = load_inventory_row(dataset)
    if len(inventory) != 6:
        raise RuntimeError(f"v5 inventory rows for {dataset}: {len(inventory)}")
    for row in inventory:
        model = row["model"]
        model_rows = sum(
            count for (item_model, _), count in counts.items() if item_model == model
        )
        # Gender source CSVs omit unavailable units entirely. Religion keeps
        # one explicit missing-content row per expected unit, while v5's
        # inferential denominator excludes those unavailable outcomes.
        expected_evaluated = (
            model_rows
            if dataset == "Gender"
            else model_rows - int(float(row["total_missing"]))
        )
        if int(float(row["total_evaluated"])) != expected_evaluated:
            raise RuntimeError(f"v5 evaluated mismatch for {dataset}/{model}")
        if int(float(row["total_correct"])) != correct[model]:
            raise RuntimeError(f"v5 correct mismatch for {dataset}/{model}")
    overall_path = output_dir / "accuracy_overall_v5.csv"
    if not overall_path.exists():
        atomic_dict_csv(
            overall_path, list(inventory[0]), inventory, quote_all=False
        )
    manifest_path = output_dir / "source_manifest.csv"
    if not manifest_path.exists():
        atomic_dict_csv(
            manifest_path,
            list(source_manifest[0]),
            source_manifest,
            quote_all=False,
        )


def audit_existing_combined(path: Path, dataset: str):
    counts = Counter()
    correct = Counter()
    keys = set()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            model = row.get("model") or ""
            persona = row.get("persona_folder") or row.get("persona") or ""
            qid = row.get("stable_question_id") or stable_id(row.get("file") or "")
            key = (model, persona, qid)
            if key in keys:
                raise RuntimeError(f"duplicate key in existing combined: {key}")
            keys.add(key)
            counts[(model, persona)] += 1
            correct[model] += int(truth(row.get("is_correct")))
    expected = 71_215 if dataset == "Gender" else 126_624
    if len(keys) != expected:
        raise RuntimeError(f"existing {dataset} combined keys={len(keys)}, expected={expected}")
    return counts, correct


def build(dataset: str, resume: bool = False):
    group = "Gender_Persona" if dataset == "Gender" else "Religion_Persona"
    output_dir = REPORT_ROOT / group / VERSION / "_combined"
    final_output = output_dir / "all_models_results_full.csv"
    if output_dir.exists():
        if final_output.exists() and not resume:
            raise FileExistsError(f"refusing to overwrite versioned output: {output_dir}")
        if not final_output.exists() and not resume:
            raise FileExistsError(
                f"incomplete output exists; inspect it or use --resume: {output_dir}"
            )
        for temporary in output_dir.glob(".*.tmp"):
            temporary.unlink()
    manifest_rows, verified = verify_sources(dataset)
    if final_output.exists():
        output = final_output
        counts, correct = audit_existing_combined(output, dataset)
    elif dataset == "Gender":
        output, counts, correct = build_gender(output_dir, manifest_rows)
    else:
        output, counts, correct = build_religion(output_dir, manifest_rows)
    write_summary_files(output_dir, dataset, counts, correct, verified)
    payload = {
        "version": VERSION,
        "dataset": dataset,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "selection_authority": relative(V5_SOURCE_MANIFEST),
        "accuracy_authority": relative(V5_ACCURACY),
        "source_policy": "exact publication-v5 paths and SHA-256; no answer reparse",
        "stable_key_policy": "model + persona + filename-derived gsm8k ID",
        "row_count": sum(counts.values()),
        "combined_sha256": sha256_file(output),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"{dataset}: rows={payload['row_count']} -> {output}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=["gender", "religion", "all"], default="all")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="remove only interrupted temp files when no final combined exists",
    )
    args = parser.parse_args(argv)
    if not args.execute:
        parser.error("--execute is required")
    datasets = ["Gender", "Religion"] if args.group == "all" else [args.group.title()]
    csv.field_size_limit(sys.maxsize)
    for dataset in datasets:
        build(dataset, resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
