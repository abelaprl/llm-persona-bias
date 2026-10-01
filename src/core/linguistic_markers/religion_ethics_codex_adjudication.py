"""Write-once Codex first-pass adjudication for ETHICS Religion Persona.

This is an AI-assisted operational precision/recall review.  It deliberately
does not fill either human-reviewer field and cannot promote the registry from
candidate to human-validated status.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import pandas as pd

from .ethics_religion_registry import (
    ETHICS_RELIGION_REGISTRY_VERSION,
    RELIGION_MARKERS,
)
from .extractor import extract_markers
from .religion_ethics_dataset_audit import (
    _ANCHOR_CAPTURE_MARKERS,
    _OPEN_ANCHORS,
)
from .surface import prepare_text_surface


AI_REVIEWER = "OpenAI Codex (AI first-pass; not an independent human reviewer)"
_RELIGIOUS_ANCHOR = re.compile(
    r"\b(?:religions?|religious|faith|spiritual|islam(?:ic)?|muslims?|"
    r"christian(?:ity)?|jewish|judaism|atheis(?:m|t|tic)|qur['’]?an|bible|"
    r"torah|scriptures?|hadiths?|ahadith|allah|god|jesus|christ|prophet|"
    r"church|synagogue|mosque|ramadan|yom\s+kippur|lent|prayer|worship|"
    r"mitzv(?:ah|ahs|ot)|talmud|midrash|rabbi|imam|priest|bishop|"
    r"world\s+to\s+come|derech\s+eretz|"
    r"taubah|tawbah|teshuvah|yom\s+kippur|religious\s+observances?|"
    r"kibbud\s+av\s+va['’]?eim|lo\s+ta['’]?aseh)\b",
    re.IGNORECASE,
)
_RELIGIOUS_META_BARE = {
    "belief",
    "beliefs",
    "faith",
    "faithful",
    "faithfulness",
    "holy",
    "sacred",
}


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


def _forms(value: object) -> list[str]:
    return [part.strip().casefold() for part in str(value).split(";") if part.strip()]


def _intentional_exclusion(
    category: str, forms: list[str], context: str
) -> tuple[bool, str, str]:
    """Classify documented broad-form exclusions using visible context only."""

    normalized = re.sub(
        r"\s+", " ", re.sub(r"[*_`#>]+", "", context)
    ).casefold()
    if category == "religious_meta" and forms and all(
        form in _RELIGIOUS_META_BARE for form in forms
    ):
        return (
            True,
            "EXCLUDE_UNBOUNDED_RELIGIOUS_META_WORD",
            "Bare belief/faith/faithful/holy/sacred is polysemous; bounded religious "
            "constructions are registered separately.",
        )
    if category == "deity_figure":
        if forms and all(form == "creator" for form in forms):
            divine_creator = re.search(
                r"\b(?:image\s+of\s+(?:its|his|her|the)\s+creator|"
                r"(?:the\s+)?creator\s+(?:guides?|commands?|created|made))\b",
                normalized,
            )
            ordinary_creator = re.search(
                r"\b(?:creator\s+of\s+(?:the|this|a|an|her|his|their)\s+"
                r"(?:app|art|artwork|cake|content|film|painting|picture|recipe|space|"
                r"video|work)|creator['’]s\s+(?:intent|vision|feelings?|rights?)|"
                r"(?:sole|content|app|platform|video|original)\s+creator|"
                r"(?:he|she|you|they)\s+(?:was|is)\s+(?:the\s+)?creator)\b",
                normalized,
            )
            if not divine_creator and (
                ordinary_creator or not _RELIGIOUS_ANCHOR.search(normalized)
            ):
                return (
                    True,
                    "EXCLUDE_NONRELIGIOUS_CREATOR",
                    "Creator denotes an ordinary maker and has no local religious anchor.",
                )
        if forms and all(form == "lord" for form in forms) and re.search(
            r"\blord\s+(?:justice|acton|of\s+the\s+rings)\b", normalized
        ):
            return True, "EXCLUDE_SECULAR_LORD_TITLE", "Lord is a legal/historical title."
        if forms and all(form.startswith("prophet") for form in forms) and not _RELIGIOUS_ANCHOR.search(normalized):
            return True, "EXCLUDE_FIGURATIVE_PROPHET", "Prophet is an unanchored secular metaphor."
    if category == "practice_community":
        if forms and all(form == "lent" for form in forms):
            if not re.search(r"\bLent\b", context) and not re.search(
                r"\b(?:religious|christian|church|easter|observance|fasting|prayer)\b",
                normalized,
            ):
                return True, "EXCLUDE_LEND_VERB", "Lowercase lent is the past tense of lend."
        if forms and all(form == "fasting" for form in forms):
            if re.search(
                r"\b(?:surg(?:ery|ical)|ana?esthe(?:sia|tic)|pre[- ]?operative|"
                r"doctors?|medical|procedure|blood\s+test|rabbit|"
                r"health\s+reasons?\s*,?\s*fasting)\b",
                normalized,
            ):
                return True, "EXCLUDE_MEDICAL_FASTING", "Fasting is medical/pre-operative."
            if not _RELIGIOUS_ANCHOR.search(normalized):
                return True, "EXCLUDE_UNANCHORED_FASTING", "Bare fasting lacks a religious-observance anchor."
    if category == "doctrine":
        if "salvation" in forms and re.search(r"\bsalvation\s+army\b", normalized):
            return True, "EXCLUDE_SALVATION_ARMY", "Salvation is part of an organization name."
        if "hell" in forms and re.search(
            r"\b(?:what|who|why|how|where)\s+the\s+hell\b|\bhell\s+no\b|"
            r"\bas\s+hell\b|\bhell[- ]bent\b|\bhell\s+of\s+(?:a|an)\b",
            normalized,
        ):
            return True, "EXCLUDE_HELL_IDIOM", "Hell is an intensifier/idiom."
        if any(form in {"soul", "souls"} for form in forms) and re.search(
            r"\b(?:dark\s+souls?|no\s+soul|poor\s+soul|old\s+soul|corrupt\s+the\s+soul|"
            r"(?:i|ai|machine|model)\s+(?:do\s+not|don['’]?t|cannot|can['’]?t)\s+"
            r"(?:have|possess)\b[^.!?\n]{0,80}\bsouls?|without\s+(?:a\s+)?souls?|"
            r"soul[- ](?:crushing|food|music|searching)|"
            r"soul\s+surfer|soul\s+(?:is\s+)?tied\s+to\s+(?:the\s+)?ocean|"
            r"(?:greatness|great)\s+of\s+soul|"
            r"(?:feed|nourish(?:es|ed|ing)?|serve[ds]?)\s+(?:my|your|her|his|"
            r"their|our|the\s+woman['’]s|the)\s+soul|"
            r"judge\b[^.!?\n]{0,45}\bsoul|"
            r"soul\s+of\s+(?:learning|a\s+(?:business|company|party|project|team)|"
            r"the\s+(?:business|company|party|project|team|work))|"
            r"tailored\s+to\s+your\s+soul|tool\s+for\s+(?:the\s+)?soul)\b",
            normalized,
        ):
            return True, "EXCLUDE_SOUL_IDIOM_OR_TITLE", "Soul is figurative, an idiom, or a title."
        if "redemption" in forms and re.search(
            r"\bredemption\s*\([^)]*(?:art|recycl|rehabilitat)|"
            r"\bopportunit(?:y|ies)\s+for\s+redemption\b[^.!?\n]{0,100}"
            r"\brehabilitat",
            normalized,
        ):
            return True, "EXCLUDE_SECULAR_REDEMPTION", "Redemption denotes repair, repurposing, or rehabilitation."
        if "salvation" in forms and re.search(
            r"\bleading\s+to\s+salvation\s+for\s+(?:her|his|their)\s+people\b",
            normalized,
        ):
            return True, "EXCLUDE_SALVATION_AS_RESCUE", "Salvation denotes historical rescue rather than soteriology."
        if "heaven" in forms and re.search(
            r"\b(?:heaven\s+forbid|for\s+heaven['’]s\s+sake|match\s+made\s+in\s+heaven)\b",
            normalized,
        ):
            return True, "EXCLUDE_HEAVEN_IDIOM", "Heaven occurs in a conventional idiom."
        if "heaven" in forms and re.search(r"\bfear\s+of\s+heaven\b", normalized):
            return (
                False,
                "POTENTIAL_UNREGISTERED_CONSTRUCTION",
                "Fear of Heaven is an explicit doctrinal construction.",
            )
        if not _RELIGIOUS_ANCHOR.search(normalized):
            return True, "EXCLUDE_UNANCHORED_DOCTRINAL_WORD", "The broad doctrinal word lacks a local religious anchor."
    if category == "scripture_authority":
        if re.search(
            r"\b(?:not|isn['’]?t|wasn['’]?t)\s+(?:a\s+)?(?:sacred\s+)?commandment\b|"
            r"\blike\s+a\s+commandment\b|"
            r"\b(?:moral\s+or\s+practical\s+|rule\s+as\s+(?:a\s+)?)"
            r"[\"“']?commandment[\"”']?\b",
            normalized,
        ):
            return (
                True,
                "EXCLUDE_FIGURATIVE_COMMANDMENT",
                "Commandment is explicitly negated or used as a secular comparison.",
            )
        if not _RELIGIOUS_ANCHOR.search(normalized):
            return True, "EXCLUDE_UNANCHORED_COMMANDMENT_OR_VERSE", "Bare commandment/verse is social, poetic, or otherwise non-scriptural."
    return (
        False,
        "POTENTIAL_UNREGISTERED_CONSTRUCTION",
        "The visible context is not covered by a documented exclusion and needs follow-up.",
    )


def adjudicate_precision(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    labels: list[str] = []
    notes: list[str] = []
    attribution: list[str] = []
    for row in result.itertuples(index=False):
        extracted = extract_markers(
            {"benchmark": "ethics", "response_content": str(row.full_response)},
            markers=RELIGION_MARKERS,
            pipeline_version=ETHICS_RELIGION_REGISTRY_VERSION,
        )
        present = extracted.get(f"{row.marker_id}__present") == 1
        labels.append("TP_OPERATIONAL" if present else "STALE_OR_FALSE_POSITIVE")
        notes.append(
            "Re-extraction with the pinned registry reproduces this bounded lexical hit."
            if present
            else "The pinned registry no longer reproduces this sampled hit."
        )
        attribution.append(
            "AMBIGUOUS_SCENARIO_EXACT"
            if _as_bool(row.exact_matched_text_in_scenario)
            else "RESPONSE_ONLY_CANDIDATE"
        )
    result["codex_review_label"] = labels
    result["codex_review_note"] = notes
    result["codex_attribution_label"] = attribution
    result["codex_reviewer"] = AI_REVIEWER
    return result


def adjudicate_unmatched(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    labels: list[str] = []
    proposals: list[str] = []
    notes: list[str] = []
    for row in result.itertuples(index=False):
        intentional, proposal, note = _intentional_exclusion(
            str(row.anchor_category), _forms(row.anchor_forms), str(row.anchor_context)
        )
        labels.append("INTENTIONAL_EXCLUSION" if intentional else "POTENTIAL_MISS")
        proposals.append(proposal)
        notes.append(note)
    result["codex_recall_label"] = labels
    result["proposed_marker_or_exclusion"] = proposals
    result["codex_recall_note"] = notes
    result["codex_reviewer"] = AI_REVIEWER
    return result


def adjudicate_anchor_forms(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    labels: list[str] = []
    notes: list[str] = []
    for row in result.itertuples(index=False):
        if _as_bool(row.captured_by_relevant_religion_marker):
            labels.append("CAPTURED_BY_RELEVANT_MARKER")
            notes.append("At least one relevant registered marker is present.")
        else:
            labels.append("CONTEXT_GUARDED_UNCAPTURED_FORM")
            notes.append(
                "This aggregated broad form is not automatically promoted; sampled complete "
                "contexts are adjudicated separately."
            )
    result["codex_recall_label"] = labels
    result["codex_recall_note"] = notes
    result["codex_reviewer"] = AI_REVIEWER
    return result


def adjudicate_recall(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    detected: list[str] = []
    labels: list[str] = []
    notes: list[str] = []
    for row in result.itertuples(index=False):
        text = str(row.visible_text)
        extracted = extract_markers(
            {"benchmark": "ethics", "response_content": text},
            markers=RELIGION_MARKERS,
            pipeline_version=ETHICS_RELIGION_REGISTRY_VERSION,
        )
        present = {
            marker.marker_id
            for marker in RELIGION_MARKERS
            if extracted[f"{marker.marker_id}__present"] == 1
        }
        detected.append(";".join(sorted(present)))
        lexical = prepare_text_surface(text, benchmark="ethics").lexical_text
        potential: list[str] = []
        exclusions = 0
        for category, pattern in _OPEN_ANCHORS.items():
            matches = list(pattern.finditer(lexical))
            if not matches or present & _ANCHOR_CAPTURE_MARKERS[category]:
                continue
            forms = sorted({match.group(0).casefold() for match in matches})
            local_context = " ... ".join(
                lexical[max(0, match.start() - 300) : min(len(lexical), match.end() + 300)]
                for match in matches
            )
            intentional, _, _ = _intentional_exclusion(category, forms, local_context)
            if intentional:
                exclusions += 1
            else:
                potential.append(f"{category}:{','.join(forms)}")
        if potential:
            labels.append("POTENTIAL_MISS")
            notes.append(" | ".join(potential))
        elif exclusions:
            labels.append("REVIEWED_INTENTIONAL_EXCLUSION")
            notes.append(f"{exclusions} uncovered anchor category/categories matched documented exclusions.")
        else:
            labels.append("NO_MISS_OBSERVED")
            notes.append("No uncovered religion anchor was observed on the measured lexical surface.")
    result["codex_detected_marker_ids"] = detected
    result["codex_recall_label"] = labels
    result["codex_recall_note"] = notes
    result["codex_reviewer"] = AI_REVIEWER
    return result


def run(base_output: Path, output_dir: Path) -> dict[str, object]:
    base_output = base_output.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")
    required = {
        "manifest": base_output / "manifest.json",
        "precision": base_output / "audit_precision_religion_hits.csv",
        "recall": base_output / "audit_recall_responses.csv",
        "unmatched": base_output / "audit_unmatched_anchor_contexts.csv",
        "forms": base_output / "candidate_anchor_forms.csv",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete base audit: {missing}")
    base_manifest = json.loads(required["manifest"].read_text(encoding="utf-8"))
    if base_manifest["pipeline_version"] != ETHICS_RELIGION_REGISTRY_VERSION:
        raise ValueError("Base output and current Religion registry versions differ.")

    precision = adjudicate_precision(pd.read_csv(required["precision"], keep_default_na=False))
    recall = adjudicate_recall(pd.read_csv(required["recall"], keep_default_na=False))
    unmatched = adjudicate_unmatched(pd.read_csv(required["unmatched"], keep_default_na=False))
    forms = adjudicate_anchor_forms(pd.read_csv(required["forms"], keep_default_na=False))

    output_dir.mkdir(parents=True, exist_ok=False)
    outputs = {
        "precision": output_dir / "audit_precision_codex_adjudicated.csv",
        "recall": output_dir / "audit_recall_codex_adjudicated.csv",
        "unmatched": output_dir / "audit_unmatched_codex_adjudicated.csv",
        "forms": output_dir / "audit_anchor_forms_codex_adjudicated.csv",
    }
    precision.to_csv(outputs["precision"], index=False, encoding="utf-8-sig")
    recall.to_csv(outputs["recall"], index=False, encoding="utf-8-sig")
    unmatched.to_csv(outputs["unmatched"], index=False, encoding="utf-8-sig")
    forms.to_csv(outputs["forms"], index=False, encoding="utf-8-sig")

    workbook = output_dir / "audit_codex_religion.xlsx"
    summary = pd.DataFrame(
        [
            {"audit": "precision", "label": key, "n": int(value)}
            for key, value in precision["codex_review_label"].value_counts().items()
        ]
        + [
            {"audit": "recall", "label": key, "n": int(value)}
            for key, value in recall["codex_recall_label"].value_counts().items()
        ]
        + [
            {"audit": "unmatched", "label": key, "n": int(value)}
            for key, value in unmatched["codex_recall_label"].value_counts().items()
        ]
    )
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        precision.to_excel(writer, sheet_name="precision_codex", index=False)
        recall.to_excel(writer, sheet_name="recall_codex", index=False)
        unmatched.to_excel(writer, sheet_name="unmatched_codex", index=False)
        forms.to_excel(writer, sheet_name="anchor_forms_codex", index=False)

    manifest: dict[str, object] = {
        "reviewer": AI_REVIEWER,
        "pipeline_version": base_manifest["pipeline_version"],
        "input_sha256": base_manifest["input_sha256"],
        "base_output": str(base_output),
        "base_manifest_sha256": _sha256(required["manifest"]),
        "human_review_status": "pending_two_independent_human_reviewers",
        "precision_labels": precision["codex_review_label"].value_counts().to_dict(),
        "recall_labels": recall["codex_recall_label"].value_counts().to_dict(),
        "unmatched_labels": unmatched["codex_recall_label"].value_counts().to_dict(),
        "output_files": {},
    }
    all_outputs = [*outputs.values(), workbook]
    manifest["output_files"] = {path.name: _sha256(path) for path in all_outputs}
    manifest_path = output_dir / "audit_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = run(args.base_output, args.output_dir)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
