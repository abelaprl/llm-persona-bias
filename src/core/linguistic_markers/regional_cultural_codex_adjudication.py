"""Write-once Codex first-pass adjudication for a regional-cultural audit bundle.

This module deliberately does not fill either human-reviewer field.  Its labels
are an AI-assisted lexical/recall audit and cannot promote the candidate
registry to human-validated status.
"""

from __future__ import annotations

import argparse
from copy import copy
import hashlib
import json
from pathlib import Path
import re

import pandas as pd

from .ethics_regional_registry import (
    REGIONAL_CULTURAL_MARKERS,
    REGIONAL_CULTURAL_REGISTRY_VERSION,
)
from .extractor import extract_markers
from .regional_cultural_dataset_audit import (
    _ANCHOR_CAPTURE_MARKERS,
    _OPEN_ANCHORS,
)


AI_REVIEWER = "OpenAI Codex (AI first-pass; not an independent human reviewer)"
_VALUE_INTENTIONAL_EXCLUSIONS = re.compile(
    r"^(?:collective|collectively|communal|family|families|familial|community|"
    r"communities|autonomy|elder|elders|harmony|hierarchies|hierarchically|"
    r"hospitality|generosity|interconnected)$",
    re.IGNORECASE,
)
_CULTURAL_META_INTENTIONAL_EXCLUSIONS = re.compile(
    r"^(?:norm|norms|custom|customs)$", re.IGNORECASE
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().casefold() in {"1", "true", "yes"}


def adjudicate_precision(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    semantic_overlap: list[bool] = []
    attribution: list[str] = []
    notes: list[str] = []
    for row in result.itertuples(index=False):
        matched = str(row.matched_text or "")
        scenario = str(row.full_scenario or "")
        exact = _as_bool(row.exact_matched_text_in_scenario)
        semantic = bool(
            row.marker_id == "culture_named_concept_reference"
            and re.search(r"\bzen\b", matched, re.IGNORECASE)
            and re.search(r"\b(?:roshi|ens[oō])\b", scenario, re.IGNORECASE)
        )
        semantic_overlap.append(semantic)
        if exact:
            attribution.append("AMBIGUOUS_SCENARIO_EXACT")
            notes.append(
                "Lexical TP under the registered definition; the exact matched span is also "
                "visible in the scenario, so no persona-carryover attribution is allowed."
            )
        elif semantic:
            attribution.append("AMBIGUOUS_SCENARIO_SEMANTIC")
            notes.append(
                "Lexical TP; Zen is a semantic gloss of scenario terms roshi/enso even when "
                "the exact word is absent, so persona attribution remains ambiguous."
            )
        else:
            attribution.append("RESPONSE_ONLY_CANDIDATE")
            notes.append(
                "Lexical TP under the registered bounded definition; this is descriptive "
                "response evidence, not proof that the persona caused the phrase."
            )
    result["codex_review_label"] = "TP"
    result["codex_review_note"] = notes
    result["codex_attribution_label"] = attribution
    result["semantic_scenario_overlap"] = semantic_overlap
    result["codex_reviewer"] = AI_REVIEWER
    return result


def _intentional_exclusion(category: str, forms: list[str]) -> tuple[bool, str, str]:
    if category == "regional_geocultural":
        return (
            True,
            "EXCLUDE_BARE_DEMONYM_OR_GEOGRAPHY",
            "Bare demonym/place, organization name, food, law, or scenario-inherited geography "
            "without a cultural head or accommodation frame is intentionally excluded.",
        )
    if category == "cultural_meta" and all(
        _CULTURAL_META_INTENTIONAL_EXCLUSIONS.fullmatch(form) for form in forms
    ):
        return (
            True,
            "EXCLUDE_GENERIC_NORM_OR_CUSTOM",
            "Bare professional, legal, conversational, or social norm/custom without a "
            "cultural anchor is intentionally excluded.",
        )
    if category == "value_orientation" and all(
        _VALUE_INTENTIONAL_EXCLUSIONS.fullmatch(form) for form in forms
    ):
        return (
            True,
            "EXCLUDE_UNBOUNDED_VALUE_WORD",
            "Bare family/community/autonomy/harmony/hierarchy, unanchored hospitality/"
            "generosity, or generic technical/ecological interconnectedness does not satisfy "
            "a registered cultural-value construction.",
        )
    return (
        False,
        "POTENTIAL_UNREGISTERED_CONSTRUCTION",
        "The open anchor is not covered by a documented bare-word exclusion and requires "
        "specific review before registry promotion.",
    )


def adjudicate_unmatched(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    labels: list[str] = []
    proposals: list[str] = []
    notes: list[str] = []
    for row in result.itertuples(index=False):
        forms = [part.strip().casefold() for part in str(row.anchor_forms).split(";") if part.strip()]
        intentional, proposal, note = _intentional_exclusion(row.anchor_category, forms)
        labels.append("INTENTIONAL_EXCLUSION" if intentional else "POTENTIAL_MISS")
        proposals.append(proposal)
        notes.append(note)
    result["codex_recall_label"] = labels
    result["proposed_marker_or_exclusion"] = proposals
    result["codex_recall_note"] = notes
    result["codex_reviewer"] = AI_REVIEWER
    return result


def adjudicate_anchor_forms(frame: pd.DataFrame) -> pd.DataFrame:
    """Adjudicate every aggregated open-anchor form, not only sampled contexts."""

    result = frame.copy()
    labels: list[str] = []
    proposals: list[str] = []
    notes: list[str] = []
    for row in result.itertuples(index=False):
        if _as_bool(row.captured_by_relevant_regional_marker):
            labels.append("CAPTURED_BY_RELEVANT_MARKER")
            proposals.append("REGISTERED_COVERAGE")
            notes.append(
                "At least one relevant registered marker is present in every response "
                "represented by this aggregated row."
            )
            continue
        form = str(row.anchor_form).strip().casefold()
        intentional, proposal, note = _intentional_exclusion(
            str(row.anchor_category), [form]
        )
        labels.append("INTENTIONAL_EXCLUSION" if intentional else "POTENTIAL_MISS")
        proposals.append(proposal)
        notes.append(note)
    result["codex_recall_label"] = labels
    result["proposed_marker_or_exclusion"] = proposals
    result["codex_recall_note"] = notes
    result["codex_reviewer"] = AI_REVIEWER
    return result


def adjudicate_recall(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    reviewer: list[str] = []
    marker_ids: list[str] = []
    missed_notes: list[str] = []
    audit_notes: list[str] = []
    labels: list[str] = []
    for row in result.itertuples(index=False):
        text = str(row.visible_text or "")
        extracted = extract_markers(
            {"benchmark": "ethics", "response_content": text},
            markers=REGIONAL_CULTURAL_MARKERS,
            pipeline_version=REGIONAL_CULTURAL_REGISTRY_VERSION,
        )
        present = {
            marker.marker_id
            for marker in REGIONAL_CULTURAL_MARKERS
            if extracted[f"{marker.marker_id}__present"] == 1
        }
        uncovered: list[tuple[str, list[str], bool, str, str]] = []
        for category, pattern in _OPEN_ANCHORS.items():
            forms = sorted({match.group(0).casefold() for match in pattern.finditer(text)})
            if not forms or present & _ANCHOR_CAPTURE_MARKERS[category]:
                continue
            intentional, proposal, note = _intentional_exclusion(category, forms)
            uncovered.append((category, forms, intentional, proposal, note))
        unexpected = [item for item in uncovered if not item[2]]
        reviewer.append(AI_REVIEWER)
        marker_ids.append(";".join(sorted(present)))
        if unexpected:
            labels.append("POTENTIAL_MISS")
            missed_notes.append(
                " | ".join(
                    f"{category}:{','.join(forms)}" for category, forms, *_ in unexpected
                )
            )
            audit_notes.append(
                "Open-anchor review found a construction outside documented exclusions; "
                "manual follow-up is required."
            )
        elif uncovered:
            labels.append("REVIEWED_INTENTIONAL_EXCLUSION")
            missed_notes.append("")
            audit_notes.append(
                "All uncovered open anchors are documented bare-word/general-context "
                "exclusions; no missed registered construct was observed."
            )
        else:
            labels.append("NO_MISS_OBSERVED")
            missed_notes.append("")
            audit_notes.append(
                "No uncovered regional-cultural open anchor was observed in the complete "
                "visible response."
            )
    result["reviewer"] = reviewer
    result["manual_marker_ids"] = marker_ids
    result["missed_span_notes"] = missed_notes
    result["audit_note"] = audit_notes
    result["codex_recall_label"] = labels
    return result


def _excel_safe(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.columns:
        if result[column].dtype.kind not in {"O", "U"}:
            continue
        result[column] = result[column].map(
            lambda value: (
                value[:32_000] + "\n[TRUNCATED IN XLSX; SEE CSV FOR FULL TEXT]"
                if isinstance(value, str) and len(value) > 32_000
                else value
            )
        )
    return result


def _write_workbook(
    path: Path,
    *,
    precision: pd.DataFrame,
    recall: pd.DataFrame,
    unmatched: pd.DataFrame,
    anchor_forms: pd.DataFrame,
    summary: pd.DataFrame,
    registry: pd.DataFrame,
    base_manifest: dict[str, object],
) -> None:
    readme = pd.DataFrame(
        [
            {
                "section": "status",
                "note": (
                    "Codex AI first-pass audit only. Human reviewer columns in the base audit "
                    "remain untouched; registry promotion still requires two independent humans."
                ),
            },
            {
                "section": "precision",
                "note": (
                    "TP means the visible span satisfies the registered lexical definition. "
                    "It is not a causal or persona-attribution claim."
                ),
            },
            {
                "section": "recall",
                "note": (
                    "Random recall rows are rescanned with open anchors per relevant marker "
                    "family. Potential misses are never silently converted to exclusions."
                ),
            },
            {
                "section": "all_anchor_forms",
                "note": (
                    "Every aggregated candidate-anchor form is adjudicated, including forms "
                    "that did not enter the sampled unmatched-context sheet."
                ),
            },
            {"section": "base_pipeline", "note": base_manifest["pipeline_version"]},
            {"section": "input_sha256", "note": base_manifest["input_sha256"]},
        ]
    )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        _excel_safe(readme).to_excel(writer, sheet_name="README", index=False)
        _excel_safe(summary).to_excel(writer, sheet_name="audit_summary", index=False)
        _excel_safe(precision).to_excel(writer, sheet_name="precision_codex", index=False)
        _excel_safe(recall).to_excel(writer, sheet_name="recall_codex", index=False)
        _excel_safe(unmatched).to_excel(writer, sheet_name="unmatched_codex", index=False)
        _excel_safe(anchor_forms).to_excel(writer, sheet_name="anchor_forms_codex", index=False)
        _excel_safe(registry).to_excel(writer, sheet_name="marker_registry", index=False)
        for sheet in writer.book.worksheets:
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
            sheet.sheet_view.showGridLines = False
            for column_cells in sheet.columns:
                header = str(column_cells[0].value or "")
                width = 70 if header in {"visible_text", "full_response", "full_scenario"} else 35
                sheet.column_dimensions[column_cells[0].column_letter].width = width
            for row in sheet.iter_rows():
                for cell in row:
                    alignment = copy(cell.alignment)
                    alignment.vertical = "top"
                    alignment.wrap_text = True
                    cell.alignment = alignment


def run(base_output: Path, output_dir: Path) -> dict[str, object]:
    base_output = base_output.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty audit output: {output_dir}")
    required = {
        "manifest": base_output / "manifest.json",
        "precision": base_output / "audit_precision_regional_hits.csv",
        "recall": base_output / "audit_recall_responses.csv",
        "unmatched": base_output / "audit_unmatched_anchor_contexts.csv",
        "anchor_forms": base_output / "candidate_anchor_forms.csv",
        "registry": base_output / "marker_registry.csv",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing base audit files: {missing}")
    base_manifest = json.loads(required["manifest"].read_text(encoding="utf-8"))
    if base_manifest["pipeline_version"] != REGIONAL_CULTURAL_REGISTRY_VERSION:
        raise ValueError(
            "Base output and code registry differ: "
            f"{base_manifest['pipeline_version']} != {REGIONAL_CULTURAL_REGISTRY_VERSION}"
        )

    precision = adjudicate_precision(pd.read_csv(required["precision"], keep_default_na=False))
    recall = adjudicate_recall(pd.read_csv(required["recall"], keep_default_na=False))
    unmatched = adjudicate_unmatched(pd.read_csv(required["unmatched"], keep_default_na=False))
    anchor_forms = adjudicate_anchor_forms(
        pd.read_csv(required["anchor_forms"], keep_default_na=False)
    )
    registry = pd.read_csv(required["registry"], keep_default_na=False)
    summary = pd.DataFrame(
        [
            {"audit": "precision", "label": label, "n": int(count)}
            for label, count in precision["codex_review_label"].value_counts().items()
        ]
        + [
            {"audit": "precision_attribution", "label": label, "n": int(count)}
            for label, count in precision["codex_attribution_label"].value_counts().items()
        ]
        + [
            {"audit": "recall_random", "label": label, "n": int(count)}
            for label, count in recall["codex_recall_label"].value_counts().items()
        ]
        + [
            {"audit": "unmatched_anchor", "label": label, "n": int(count)}
            for label, count in unmatched["codex_recall_label"].value_counts().items()
        ]
        + [
            {"audit": "all_anchor_forms", "label": label, "n": int(count)}
            for label, count in anchor_forms["codex_recall_label"].value_counts().items()
        ]
        + [
            {
                "audit": "all_anchor_form_responses",
                "label": label,
                "n": int(group["n_responses"].astype(int).sum()),
            }
            for label, group in anchor_forms.groupby("codex_recall_label", sort=False)
        ]
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    outputs = {
        "precision": output_dir / "audit_precision_codex_adjudicated.csv",
        "recall": output_dir / "audit_recall_codex_adjudicated.csv",
        "unmatched": output_dir / "audit_unmatched_codex_adjudicated.csv",
        "anchor_forms": output_dir / "audit_anchor_forms_codex_adjudicated.csv",
        "summary": output_dir / "audit_codex_summary.csv",
        "workbook": output_dir / "audit_codex_regional_cultural.xlsx",
    }
    precision.to_csv(outputs["precision"], index=False, encoding="utf-8-sig")
    recall.to_csv(outputs["recall"], index=False, encoding="utf-8-sig")
    unmatched.to_csv(outputs["unmatched"], index=False, encoding="utf-8-sig")
    anchor_forms.to_csv(outputs["anchor_forms"], index=False, encoding="utf-8-sig")
    summary.to_csv(outputs["summary"], index=False, encoding="utf-8-sig")
    _write_workbook(
        outputs["workbook"],
        precision=precision,
        recall=recall,
        unmatched=unmatched,
        anchor_forms=anchor_forms,
        summary=summary,
        registry=registry,
        base_manifest=base_manifest,
    )
    audit_manifest = {
        "audit_type": "codex_ai_first_pass_not_human_validation",
        "reviewer": AI_REVIEWER,
        "base_output": str(base_output),
        "base_manifest_sha256": _sha256(required["manifest"]),
        "pipeline_version": base_manifest["pipeline_version"],
        "input_sha256": base_manifest["input_sha256"],
        "human_review_status": "pending_two_independent_human_reviewers",
        "output_files": {
            path.name: _sha256(path) for path in outputs.values()
        },
    }
    manifest_path = output_dir / "audit_manifest.json"
    manifest_path.write_text(
        json.dumps(audit_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return audit_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    run(args.base_output, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
