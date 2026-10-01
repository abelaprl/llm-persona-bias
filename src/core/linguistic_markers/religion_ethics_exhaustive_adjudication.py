"""Second-pass AI adjudication of exhaustive ETHICS Religion review queues.

This pass is intentionally separate from extraction.  It screens every stored
occurrence for corpus-observed false-context signatures, adjudicates every
response-level open-anchor result, and summarizes all normalized occurrence
clusters.  It is AI review and never fills or replaces human-review fields.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import pandas as pd

from .ethics_religion_registry import ETHICS_RELIGION_REGISTRY_VERSION
from .religion_ethics_codex_adjudication import _intentional_exclusion


AI_REVIEWER = "OpenAI Codex (AI exhaustive second-pass; not a human reviewer)"
ADJUDICATION_VERSION = "ethics-religion-codex-exhaustive-adjudication-v3"


_RELIGIOUS_CONTEXT = re.compile(
    r"\b(?:religions?|religious|spiritual|scriptures?|bible|biblical|qur['’]?an|"
    r"torah|talmud|halach\w*|mitzv\w*|commandments?|church|synagogue|mosque|"
    r"christian(?:ity)?|jewish|judaism|islam(?:ic)?|muslims?|allah|god|prophet|"
    r"ramadan|lent|worship|prayer|theolog\w*)\b"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _focused_occurrence(form: str, normalized_context: str, radius: int = 100) -> str:
    """Return the context around the occurrence closest to the excerpt centre.

    Exhaustive queue excerpts are centred on the reviewed occurrence but can
    contain a second identical form.  Screening the complete excerpt can then
    attribute the second form's false-context signature to the valid centred
    occurrence.  This deterministic focus keeps the negative screen local.
    """

    matches = list(
        re.finditer(
            rf"(?<!\w){re.escape(form.casefold())}(?!\w)", normalized_context
        )
    )
    if not matches:
        return normalized_context
    midpoint = len(normalized_context) / 2
    match = min(matches, key=lambda item: abs((item.start() + item.end()) / 2 - midpoint))
    return normalized_context[
        max(0, match.start() - radius) : min(len(normalized_context), match.end() + radius)
    ]


def _false_context_reason(marker_id: str, form: str, context: str) -> str:
    """Independent negative screen for ambiguity found during corpus reading."""

    normalized = re.sub(r"\s+", " ", context).casefold()
    focused = _focused_occurrence(form, normalized)
    if marker_id == "religion_generic_reference" and form == "faith":
        faith_focused = _focused_occurrence(form, normalized, radius=35)
        if re.search(r"\b(?:good|bad)-faith\b", faith_focused):
            return "GOOD_BAD_FAITH"
        if re.search(
            r"\bfaith\s+in\s+(?:"
            r"(?:other\s+)?people|others?|humanity|human\s+(?:connection|potential)|"
            r"(?:the\s+)?(?:system|justice\s+system|legal\s+system|government|"
            r"institutions?|business|company|organization|money|chance|luck)|"
            r"(?:one['’]s|your|their|his|her|our)\s+(?:abilities|ability|skills|"
            r"judg(?:e)?ment|competence|reliability|resilience))\b",
            faith_focused,
        ):
            return "SECULAR_FAITH_AS_TRUST"
    if marker_id == "religion_deity_invocation" and "creator" in form:
        if re.search(
            r"\bcreator(?:['’]s)?\s+(?:of\s+(?:the|this|a|an)\s+)?"
            r"(?:app|art|artwork|content|film|platform|product|recipe|video|intent|"
            r"vision|rights?|polic(?:y|ies))\b",
            focused,
        ):
            return "ORDINARY_CREATOR"
    if marker_id == "religion_figure_reference" and form.startswith("prophet"):
        if re.search(r"\bprophet\s+of\s+(?:doom|profit|technology|change)\b", focused):
            return "SECULAR_PROPHET_METAPHOR"
    if marker_id == "religion_practice_community":
        if form == "fasting" and re.search(
            r"\b(?:surg(?:ery|ical)|ana?esthe(?:sia|tic)|pre[- ]?operative|"
            r"medical|procedure|blood\s+test|rabbit)\b",
            focused,
        ) and not _RELIGIOUS_CONTEXT.search(normalized):
            return "MEDICAL_FASTING"
        if form == "lent" and re.search(r"\b(?:she|he|they|i|we)\s+lent\b", focused):
            return "LEND_VERB"
    if marker_id == "religion_afterlife_soteriology":
        if form == "salvation" and re.search(r"\bsalvation\s+army\b", focused):
            return "SALVATION_ARMY"
        if form == "hell" and re.search(
            r"\b(?:what|who|why|how|where)\s+the\s+hell\b|\bas\s+hell\b|"
            r"\bhell[- ]bent\b|\bhell\s+of\s+(?:a|an)\b",
            _focused_occurrence(form, normalized, radius=40),
        ):
            return "HELL_IDIOM"
        if form in {"soul", "souls"} and re.search(
            r"\b(?:i|ai|machine|model)\s+(?:do\s+not|don['’]?t|cannot|can['’]?t)\s+"
            r"(?:have|possess)\b[^.!?\n]{0,80}\bsouls?\b|"
            r"\bsoul[- ](?:crushing|food|music|searching)\b|\bsoul\s+surfer\b|"
            r"\bsoul\s+of\s+(?:learning|a\s+(?:business|company|project|team))\b",
            focused,
        ):
            return "SOUL_IDIOM_TITLE_OR_AI_DISCLAIMER"
        if form == "redemption" and re.search(
            r"\bredemption\s+(?:arc|story|narrative)\b|"
            r"\bredemption\s*\([^)]*(?:art|recycl|rehabilitat)",
            focused,
        ):
            return "SECULAR_REDEMPTION"
    if marker_id == "religion_scripture_reference" and form in {
        "verse",
        "verses",
        "commandment",
        "commandments",
    }:
        if re.search(
            r"\b(?:poem|song|chorus|lyrics?)\b[^.!?\n]{0,120}\bverses?\b|"
            r"\bnot\s+(?:a\s+)?commandment\b|\blike\s+a\s+commandment\b",
            focused,
        ) and not _RELIGIOUS_CONTEXT.search(normalized):
            return "NON_SCRIPTURAL_VERSE_OR_COMMANDMENT"
    return ""


def _parse_uncovered(value: str) -> list[tuple[str, list[str]]]:
    parsed: list[tuple[str, list[str]]] = []
    for group in str(value).split(" | "):
        if not group or ":" not in group:
            continue
        category, joined = group.split(":", 1)
        parsed.append((category, [part for part in joined.split(";") if part]))
    return parsed


def run(queue_dir: Path, output_dir: Path) -> dict[str, object]:
    queue_dir = queue_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")
    queue_manifest_path = queue_dir / "review_manifest.json"
    queue_manifest = json.loads(queue_manifest_path.read_text(encoding="utf-8"))
    if queue_manifest["pipeline_version"] != ETHICS_RELIGION_REGISTRY_VERSION:
        raise ValueError("Queue and current Religion registry versions differ.")

    occurrence = pd.read_csv(
        queue_dir / "occurrence_review_queue.csv.gz",
        usecols=[
            "occurrence_id",
            "response_key",
            "marker_id",
            "normalized_matched_text",
            "match_context",
        ],
        keep_default_na=False,
        low_memory=False,
    )
    occurrence["codex_false_context_reason"] = [
        _false_context_reason(row.marker_id, row.normalized_matched_text, row.match_context)
        for row in occurrence.itertuples(index=False)
    ]
    occurrence["codex_second_pass_label"] = occurrence[
        "codex_false_context_reason"
    ].map(lambda value: "POTENTIAL_FALSE_POSITIVE" if value else "NO_KNOWN_FALSE_CONTEXT")
    flagged = occurrence[occurrence["codex_false_context_reason"].ne("")].copy()

    cluster = pd.read_csv(
        queue_dir / "occurrence_context_clusters.csv", keep_default_na=False
    )
    cluster_flags = (
        flagged.groupby(["marker_id", "normalized_matched_text"], as_index=False)
        .agg(
            n_flagged_occurrences=("occurrence_id", "size"),
            false_context_reasons=(
                "codex_false_context_reason",
                lambda values: ";".join(sorted(set(values))),
            ),
        )
        if not flagged.empty
        else pd.DataFrame(
            columns=[
                "marker_id",
                "normalized_matched_text",
                "n_flagged_occurrences",
                "false_context_reasons",
            ]
        )
    )
    cluster = cluster.merge(
        cluster_flags,
        on=["marker_id", "normalized_matched_text"],
        how="left",
    )
    cluster["n_flagged_occurrences"] = (
        pd.to_numeric(cluster["n_flagged_occurrences"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    cluster["false_context_reasons"] = cluster["false_context_reasons"].fillna("")
    cluster["codex_second_pass_label"] = cluster["n_flagged_occurrences"].map(
        lambda value: "POTENTIAL_FALSE_POSITIVE_PRESENT"
        if value
        else "NO_KNOWN_FALSE_CONTEXT_IN_CLUSTER"
    )
    cluster["codex_reviewer"] = AI_REVIEWER

    response = pd.read_csv(
        queue_dir / "response_review_queue.csv.gz", keep_default_na=False, low_memory=False
    )
    response_labels: list[str] = []
    response_notes: list[str] = []
    for row in response.itertuples(index=False):
        groups = _parse_uncovered(row.uncovered_anchor_forms)
        if not groups:
            response_labels.append(row.automated_screen_label)
            response_notes.append("No uncovered registered open-anchor form requires adjudication.")
            continue
        if row.automated_screen_label != "POTENTIAL_MISS_REQUIRES_MANUAL_REVIEW":
            response_labels.append(
                "REVIEWED_INTENTIONAL_EXCLUSION"
                if row.automated_screen_label
                == "NO_HIT_DOCUMENTED_OPEN_ANCHOR_EXCLUSION"
                else row.automated_screen_label
            )
            response_notes.append(
                "The lexical-surface queue matched documented local exclusions; "
                "full-response context does not override that local decision."
            )
            continue
        unresolved: list[str] = []
        exclusions: list[str] = []
        for category, forms in groups:
            intentional, code, _ = _intentional_exclusion(
                category, forms, row.visible_response
            )
            (exclusions if intentional else unresolved).append(f"{category}:{code}")
        if unresolved:
            response_labels.append("POTENTIAL_MISS")
            response_notes.append(" | ".join(unresolved))
        else:
            response_labels.append("REVIEWED_INTENTIONAL_EXCLUSION")
            response_notes.append(" | ".join(exclusions))
    response["codex_second_pass_recall_label"] = response_labels
    response["codex_second_pass_recall_note"] = response_notes
    response["codex_reviewer"] = AI_REVIEWER

    marker_summary = (
        occurrence.groupby("marker_id", as_index=False)
        .agg(
            occurrences=("occurrence_id", "size"),
            responses=("response_key", "nunique"),
            normalized_forms=("normalized_matched_text", "nunique"),
            potential_false_positives=(
                "codex_false_context_reason",
                lambda values: int(sum(bool(value) for value in values)),
            ),
        )
        .merge(
            cluster.groupby("marker_id", as_index=False).agg(
                occurrence_clusters=("marker_id", "size")
            ),
            on="marker_id",
            how="left",
        )
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    outputs = {
        "flagged": output_dir / "potential_false_positive_occurrences.csv",
        "clusters": output_dir / "occurrence_clusters_codex_second_pass.csv",
        "recall": output_dir / "response_recall_codex_second_pass.csv.gz",
        "summary": output_dir / "marker_second_pass_summary.csv",
    }
    flagged.to_csv(outputs["flagged"], index=False, encoding="utf-8-sig")
    cluster.to_csv(outputs["clusters"], index=False, encoding="utf-8-sig")
    response.to_csv(
        outputs["recall"], index=False, encoding="utf-8-sig", compression="gzip"
    )
    marker_summary.to_csv(outputs["summary"], index=False, encoding="utf-8-sig")

    manifest: dict[str, object] = {
        "adjudication_version": ADJUDICATION_VERSION,
        "pipeline_version": ETHICS_RELIGION_REGISTRY_VERSION,
        "reviewer": AI_REVIEWER,
        "review_scope": "all_occurrences_all_clusters_all_responses",
        "human_review_status": "pending_two_independent_human_reviewers",
        "queue_dir": str(queue_dir),
        "queue_manifest_sha256": _sha256(queue_manifest_path),
        "occurrences_reviewed_by_second_pass_screen": len(occurrence),
        "occurrence_clusters_reviewed": len(cluster),
        "responses_reviewed_by_open_anchor_screen": len(response),
        "potential_false_positive_occurrences": len(flagged),
        "recall_labels": response["codex_second_pass_recall_label"].value_counts().to_dict(),
        "output_files": {path.name: _sha256(path) for path in outputs.values()},
    }
    manifest_path = output_dir / "adjudication_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.queue_dir, args.output_dir), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
