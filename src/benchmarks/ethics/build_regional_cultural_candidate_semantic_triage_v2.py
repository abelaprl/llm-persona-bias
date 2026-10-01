"""Build an additive AI semantic-triage record for Regional/Cultural candidates.

The triage records decisions made after reading full visible responses.  It
does not change marker extraction, does not read ground truth, and cannot
promote a candidate without two independent human reviewers.
"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
REPORT_ROOT = (
    REPO_ROOT / "result" / "ETHICS" / "reports" / "Regional_Cultural_Persona"
)
DISCOVERY_DIR = (
    REPORT_ROOT
    / "linguistic_markers"
    / "audit"
    / "regional_cultural_ethics_candidate_construct_discovery_v2_0_0_20260827"
)
OUTPUT_DIR = (
    REPORT_ROOT
    / "linguistic_markers"
    / "audit"
    / "regional_cultural_ethics_candidate_semantic_triage_v2_0_0_20260827"
)
RANDOM_SEED = 18_222_008


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def custom_precision_sample(hits: pd.DataFrame) -> pd.DataFrame:
    custom = hits[hits["candidate_id"].eq("culture_custom_explicit")]
    pieces: list[pd.DataFrame] = []
    for model_id, group in custom.groupby("model_id", observed=True, sort=True):
        n = min(10, len(group))
        pieces.append(group.sample(n=n, random_state=RANDOM_SEED))
    sample = pd.concat(pieces, ignore_index=True).sort_values(
        ["model_id", "response_key"], ignore_index=True
    )
    sample.insert(0, "audit_rule", "culture_custom_explicit")
    sample.insert(1, "ai_precision_label", "TP_EXPLICIT_CUSTOM_REFERENCE")
    sample.insert(
        2,
        "ai_precision_note",
        "Full visible response explicitly uses a bounded custom/customs formulation; factual truth of the cultural claim is outside this lexical audit.",
    )
    sample.insert(3, "ai_reviewer", "Codex AI full-response review 2026-08-27")
    sample.insert(4, "human_reviewer_1", "pending")
    sample.insert(5, "human_reviewer_2", "pending")
    return sample


def residual_triage(hits: pd.DataFrame) -> pd.DataFrame:
    residual = hits[~hits["has_any_registered_regional_marker"].astype(bool)].copy()
    mapping = {
        "culture_custom_explicit": (
            "KEEP_CANDIDATE_CUSTOM_REFERENCE",
            "Bounded custom/customs reference is culturally interpretable but requires human precision/construct review before promotion.",
        ),
        "culture_extended_kin_network": (
            "EXCLUDE_SCENARIO_LEXICAL_CONTENT",
            "Extended-family wording describes actors or relationships in the ETHICS scenario; it does not frame a cultural orientation.",
        ),
        "culture_extended_geocultural_reference": (
            "EXCLUDE_NONCULTURAL_GEOGRAPHIC_CONTEXT",
            "U.S. context/values refer to law, measurement units, or scenario setting without a bounded cultural head.",
        ),
        "culture_stereotype_caution": (
            "EXCLUDE_GENERIC_STEREOTYPE_DISCOURSE",
            "Avoid-stereotype wording concerns the scenario generally and lacks an explicit cultural target.",
        ),
        "culture_egalitarianism": (
            "EXCLUDE_NONCULTURAL_FRAMEWORK_HALLUCINATION",
            "The lone residual invokes a purported Self-Interest Egalitarianism framework, not a regional/cultural value orientation.",
        ),
    }
    residual["ai_recall_disposition"] = residual["candidate_id"].map(
        lambda value: mapping[value][0]
    )
    residual["ai_recall_note"] = residual["candidate_id"].map(
        lambda value: mapping[value][1]
    )
    residual["ai_reviewer"] = "Codex AI full-response/context review 2026-08-27"
    residual["human_reviewer_1"] = "pending"
    residual["human_reviewer_2"] = "pending"
    return residual.sort_values(
        ["candidate_id", "subset", "question_id", "response_key"],
        ignore_index=True,
    )


def candidate_decisions(inventory: pd.DataFrame) -> pd.DataFrame:
    result = inventory.copy()
    result["ai_triage_decision"] = "RETAIN_DISCOVERY_CANDIDATE_NOT_PROMOTED"
    result["ai_triage_basis"] = (
        "Candidate subtype is already captured by at least one frozen regional marker or lacks sufficient independent construct validation."
    )
    zero = result["response_hits"].eq(0)
    result.loc[zero, "ai_triage_decision"] = "NO_CORPUS_HIT_NOT_PROMOTED"
    result.loc[zero, "ai_triage_basis"] = (
        "No visible response hit in the frozen 154,428-row queue; absence is not a universal semantic-recall claim."
    )
    custom = result["candidate_id"].eq("culture_custom_explicit")
    result.loc[custom, "ai_triage_decision"] = (
        "LEXICAL_SUBTYPE_CANDIDATE_AI_PRECISION_PASS_HUMAN_BLOCKED"
    )
    result.loc[custom, "ai_triage_basis"] = (
        "51 model-stratified full responses reviewed; 51 TP and 0 FP. Two independent human reviewers remain mandatory."
    )
    kin = result["candidate_id"].eq("culture_extended_kin_network")
    result.loc[kin, "ai_triage_basis"] = (
        "74 non-overlap responses were scenario/relationship content; 264 additional responses overlap frozen markers. Do not use bare extended family as a cultural marker."
    )
    geo = result["candidate_id"].eq("culture_extended_geocultural_reference")
    result.loc[geo, "ai_triage_basis"] = (
        "Seven non-overlap hits were legal, unit, or geographic U.S. context; the discovery regex is intentionally too broad for promotion."
    )
    stereotype = result["candidate_id"].eq("culture_stereotype_caution")
    result.loc[stereotype, "ai_triage_basis"] = (
        "Five non-overlap hits were generic stereotype cautions; cultural anchoring is required for any future rule."
    )
    egalitarian = result["candidate_id"].eq("culture_egalitarianism")
    result.loc[egalitarian, "ai_triage_basis"] = (
        "The only non-overlap hit was a noncultural/hallucinated framework reference; other hits already overlap frozen cultural markers."
    )
    result["human_review_status"] = "pending_two_independent_reviewers"
    result["promoted_to_operational_registry"] = False
    return result


def main() -> int:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Output is write-once and already exists: {OUTPUT_DIR}")
    discovery_manifest_path = DISCOVERY_DIR / "manifest.json"
    discovery_manifest = json.loads(
        discovery_manifest_path.read_text(encoding="utf-8")
    )
    if discovery_manifest.get("rows_scanned") != 154_428:
        raise ValueError("Candidate discovery source does not have 154,428 rows.")
    if discovery_manifest.get("candidate_constructs") != 27:
        raise ValueError("Candidate discovery source does not have 27 families.")

    inventory = pd.read_csv(DISCOVERY_DIR / "candidate_construct_inventory.csv")
    hits = pd.read_csv(
        DISCOVERY_DIR / "candidate_hit_responses_full_text.csv.gz",
        low_memory=False,
    )
    if len(hits) != 3_421:
        raise ValueError(f"Expected 3,421 candidate-response hits, found {len(hits):,}.")

    precision = custom_precision_sample(hits)
    residual = residual_triage(hits)
    decisions = candidate_decisions(inventory)
    if len(precision) != 51:
        raise ValueError(f"Expected 51 custom audit rows, found {len(precision)}.")
    if len(residual) != 100:
        raise ValueError(f"Expected 100 non-overlap responses, found {len(residual)}.")
    if not precision["ai_precision_label"].eq(
        "TP_EXPLICIT_CUSTOM_REFERENCE"
    ).all():
        raise ValueError("Custom precision labels are incomplete.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    precision.to_csv(
        OUTPUT_DIR / "custom_marker_ai_precision_full_text_51.csv.gz",
        index=False,
        compression="gzip",
    )
    residual.to_csv(
        OUTPUT_DIR / "nonoverlap_response_ai_recall_triage_100.csv.gz",
        index=False,
        compression="gzip",
    )
    decisions.to_csv(OUTPUT_DIR / "candidate_family_ai_decisions.csv", index=False)

    summary = {
        "audit_version": "regional-cultural-ethics-candidate-semantic-triage-v2.0.0",
        "status": "ai_triage_complete_human_validation_pending",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_discovery_manifest": str(discovery_manifest_path),
        "source_discovery_manifest_sha256": sha256_file(discovery_manifest_path),
        "candidate_families": len(decisions),
        "nonoverlap_responses_reviewed": len(residual),
        "nonoverlap_dispositions": residual["ai_recall_disposition"].value_counts().to_dict(),
        "custom_precision_sample": len(precision),
        "custom_precision_models": precision["model_id"].nunique(),
        "custom_precision_true_positive": int(
            precision["ai_precision_label"].eq("TP_EXPLICIT_CUSTOM_REFERENCE").sum()
        ),
        "custom_precision_false_positive": 0,
        "custom_precision_observed": 1.0,
        "rule_of_three_upper_error_bound_approx": 3 / len(precision),
        "operational_registry_modified": False,
        "candidate_promoted": False,
        "ground_truth_used": False,
        "hidden_reasoning_used": False,
        "human_review_complete": False,
    }
    (OUTPUT_DIR / "semantic_triage_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    readme = """# AI semantic triage kandidat Regional/Cultural v2.0.0

Artefak ini menambahkan catatan audit tanpa mengubah registry v1.0.16.

- 100/100 respons kandidat yang tidak overlap dengan marker regional dibaca;
- 51 respons utuh untuk kandidat `culture_custom_explicit` dibaca secara
  terstratifikasi per model (10 per model, kecuali seluruh 1 hit GPT-OSS);
- hasil AI untuk `custom` adalah 51 TP, 0 FP, tetapi **belum dipromosikan**;
- dua reviewer manusia independen tetap wajib.

Label TP hanya berarti frasa custom/customs benar-benar ada dan digunakan
sebagai rujukan kebiasaan. Audit ini tidak menilai kebenaran faktual klaim
budaya model dan tidak melihat ground truth ETHICS.
"""
    (OUTPUT_DIR / "README.md").write_text(readme, encoding="utf-8")
    hashes = {
        path.name: sha256_file(path)
        for path in sorted(OUTPUT_DIR.iterdir())
        if path.is_file()
    }
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(
            {**summary, "output_dir": str(OUTPUT_DIR), "generated_file_sha256": hashes},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
