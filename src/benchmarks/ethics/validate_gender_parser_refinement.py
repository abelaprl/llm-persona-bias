"""Validate the non-destructive Gender parser-refinement report."""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from scipy.stats import chi2_contingency

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_gender_parser_refinement import (  # noqa: E402
    AITA_REVIEWED,
    INPUT,
    OUTPUT,
    OUTPUT_DIR,
    SEMANTIC_CANDIDATES,
    key,
)


INVARIANT_COLUMNS = frozenset(
    {
        "model",
        "persona_folder",
        "persona_group",
        "file",
        "question_id",
        "dataset",
        "subset",
        "ground_truth",
        "gt_stance",
        "response_content_full",
        "finish_reason",
        "prompt_tokens",
        "completion_tokens",
        "reasoning_tokens",
        "total_tokens",
        "truncated",
        "text_tail",
        "construct_valid",
    }
)
ALLOWED_CHANGED_COLUMNS = frozenset(
    {
        "extracted_answer",
        "extracted_stance",
        "answer_status",
        "parse_source",
        "is_correct",
        "digit_convention",
        "candidate_answer",
        "candidate_source",
        "behavior_category",
        "abstain_reason",
        "extraction_confidence",
        "numeric_semantic_conflict",
        "response_flags",
    }
)


def load_keys(path: Path) -> set[tuple[str, str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {key(row) for row in csv.DictReader(handle)}


def rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def chi_square_rows(
    per_model: dict[str, dict[str, Counter]], scope: str
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    statuses = ("committed", "abstain", "missing")
    for model in sorted(per_model):
        personas = sorted(per_model[model])
        table = [[per_model[model][persona][status] for status in statuses] for persona in personas]
        columns = [index for index in range(3) if any(row[index] for row in table)]
        table = [[row[index] for index in columns] for row in table]
        chi2, p_value, dof, _ = chi2_contingency(table)
        n = sum(sum(row) for row in table)
        k = min(len(table), len(table[0])) - 1
        cramer_v = (chi2 / (n * k)) ** 0.5 if n and k else 0.0
        rows.append(
            {
                "scope": scope,
                "model": model,
                "personas": len(personas),
                "n": n,
                "chi2": chi2,
                "dof": dof,
                "p_value": p_value,
                "cramer_v": cramer_v,
                "significant_0_05": p_value < 0.05,
            }
        )
    return rows


def main() -> int:
    aita_keys = load_keys(AITA_REVIEWED)
    semantic_keys = load_keys(SEMANTIC_CANDIDATES)
    allowed_keys = aita_keys | semantic_keys
    if len(aita_keys) != 12 or len(semantic_keys) != 297 or len(allowed_keys) != 309:
        raise RuntimeError("Unexpected refinement target counts")

    csv.field_size_limit(sys.maxsize)
    changed_rows = 0
    changed_columns: Counter = Counter()
    transitions: Counter = Counter()
    status: Counter = Counter()
    status_by_group: dict[str, Counter] = defaultdict(Counter)
    per_model_gender: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    per_model_all: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    seen_keys: set[tuple[str, str, str]] = set()

    with INPUT.open(encoding="utf-8-sig", newline="") as old_handle, OUTPUT.open(
        encoding="utf-8-sig", newline=""
    ) as new_handle:
        old_reader = csv.DictReader(old_handle)
        new_reader = csv.DictReader(new_handle)
        if old_reader.fieldnames != new_reader.fieldnames:
            raise RuntimeError("Headers changed")

        for old, new in zip(old_reader, new_reader, strict=True):
            old_key = key(old)
            new_key = key(new)
            if old_key != new_key:
                raise RuntimeError(f"Row order/key changed: {old_key} != {new_key}")
            seen_keys.add(new_key)
            differences = {field for field in old if old.get(field, "") != new.get(field, "")}
            if differences:
                changed_rows += 1
                if new_key not in allowed_keys:
                    raise RuntimeError(f"Untargeted row changed: {new_key}")
                if differences & INVARIANT_COLUMNS:
                    raise RuntimeError(
                        f"Invariant columns changed for {new_key}: {sorted(differences & INVARIANT_COLUMNS)}"
                    )
                if differences - ALLOWED_CHANGED_COLUMNS:
                    raise RuntimeError(
                        f"Unexpected columns changed for {new_key}: {sorted(differences - ALLOWED_CHANGED_COLUMNS)}"
                    )
                changed_columns.update(differences)

            old_status = old.get("answer_status", "")
            new_status = new.get("answer_status", "")
            transitions[(old_status, new_status)] += 1
            status[new_status] += 1
            group = new.get("persona_group", "")
            status_by_group[group][new_status] += 1
            model = new.get("model", "")
            persona = new.get("persona_folder", "")
            per_model_all[model][persona][new_status] += 1
            if group == "gender":
                per_model_gender[model][persona][new_status] += 1

    if seen_keys != allowed_keys | (seen_keys - allowed_keys):
        raise RuntimeError("Internal key accounting error")
    if changed_rows != 309:
        raise RuntimeError(f"Expected 309 changed rows, found {changed_rows}")
    if transitions[("abstain", "committed")] != 12:
        raise RuntimeError("Expected exactly 12 abstain->committed transitions")
    if any(old == "committed" and new != "committed" for old, new in transitions):
        raise RuntimeError("A committed row was lost")

    chi_rows = chi_square_rows(per_model_gender, "gender_only")
    chi_rows.extend(chi_square_rows(per_model_all, "gender_plus_neutral"))
    chi_path = OUTPUT_DIR / "persona_status_independence.csv"
    with chi_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(chi_rows[0]))
        writer.writeheader()
        writer.writerows(chi_rows)

    group_metrics = {}
    for group, counts in sorted(status_by_group.items()):
        total = sum(counts.values())
        group_metrics[group] = {
            "total": total,
            "committed": counts["committed"],
            "commit_rate": rate(counts["committed"], total),
            "abstain": counts["abstain"],
            "abstain_rate": rate(counts["abstain"], total),
            "missing": counts["missing"],
            "missing_rate": rate(counts["missing"], total),
        }

    payload = {
        "valid": True,
        "rows": len(seen_keys),
        "changed_rows": changed_rows,
        "changed_columns": dict(changed_columns),
        "status": dict(status),
        "status_transitions": {
            f"{old}->{new}": count for (old, new), count in sorted(transitions.items())
        },
        "group_metrics": group_metrics,
        "persona_status_independence_file": str(chi_path),
        "persona_status_significant_models": {
            scope: [
                row["model"]
                for row in chi_rows
                if row["scope"] == scope and row["significant_0_05"]
            ]
            for scope in ("gender_only", "gender_plus_neutral")
        },
        "invariants": {
            "headers_identical": True,
            "row_order_and_keys_identical": True,
            "response_text_identical": True,
            "ground_truth_columns_identical": True,
            "only_309_target_rows_changed": True,
            "committed_to_abstain": 0,
        },
    }
    path = OUTPUT_DIR / "validation.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
