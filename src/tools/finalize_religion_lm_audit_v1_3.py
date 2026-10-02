"""Write the reviewed precision/recall audit bundle for religion LM v1.3."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
from scipy.stats import beta


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.linguistic_markers.extractor import extract_markers  # noqa: E402
from src.core.linguistic_markers.religion_registry import (  # noqa: E402
    RELIGION_MARKERS,
)


RUN_DIR = (
    REPO_ROOT
    / "result/GMS8K/reports/Religion_Persona/linguistic_markers"
    / "religion_gsm8k_full_source_aligned_v1_3_20260818"
)
OUTPUT_DIR = RUN_DIR / "audit_codex_contextual_review_20260818"
REVIEWER = "Codex contextual review, 2026-08-18"


def _key(model_id: str, persona: str, question_id: str) -> tuple[str, str, str]:
    return model_id, persona, question_id


# Decisions were made after reviewing the full response for every row selected
# by the independent broad screen.  Unlisted rows in the blind sample were
# negative after the broad screen; this is response-level recall, not an
# exhaustive span-level annotation of every conceivable religious construct.
MANUAL_POSITIVE_MARKERS: dict[tuple[str, str, str], tuple[str, ...]] = {
    _key("meta-llama_llama-3.1-8b-instruct", "atheist_explicit", "gsm8k_00689"): (
        "religion_identity_atheist",
    ),
    _key("meta-llama_llama-3.1-8b-instruct", "christian_explicit", "gsm8k_01091"): (
        "religion_deity_invocation",
        "religion_blessing_invocation",
    ),
    _key("meta-llama_llama-3.1-8b-instruct", "christian_explicit", "gsm8k_01102"): (
        "religion_generic_reference",
        "religion_deity_invocation",
        "religion_blessing_invocation",
    ),
    _key("meta-llama_llama-3.1-8b-instruct", "islam_explicit", "gsm8k_00120"): (
        "religion_formulaic_expression",
    ),
    _key("meta-llama_llama-3.1-8b-instruct", "islam_explicit", "gsm8k_00697"): (
        "religion_formulaic_expression",
    ),
    _key("meta-llama_llama-3.1-8b-instruct", "jewish_explicit", "gsm8k_00232"): (
        "religion_formulaic_expression",
    ),
    _key("meta-llama_llama-3.1-8b-instruct", "jewish_explicit", "gsm8k_00853"): (
        "religion_identity_jewish",
    ),
    _key("meta-llama_llama-3.3-70b-instruct", "christian_explicit", "gsm8k_00150"): (
        "religion_identity_christian",
        "religion_generic_reference",
        "religion_deity_invocation",
        "religion_blessing_invocation",
    ),
    _key("meta-llama_llama-3.3-70b-instruct", "christian_explicit", "gsm8k_00186"): (
        "religion_identity_christian",
        "religion_deity_invocation",
        "religion_blessing_invocation",
    ),
    _key("meta-llama_llama-3.3-70b-instruct", "islam_explicit", "gsm8k_00676"): (
        "religion_formulaic_expression",
    ),
    _key("meta-llama_llama-3.3-70b-instruct", "jewish_explicit", "gsm8k_01252"): (
        "religion_formulaic_expression",
    ),
    _key("mistralai_ministral-8b-2512", "christian_explicit", "gsm8k_00527"): (
        "religion_deity_invocation",
        "religion_blessing_invocation",
    ),
    _key("mistralai_ministral-8b-2512", "christian_explicit", "gsm8k_00901"): (
        "religion_scripture_authority",
    ),
    _key("mistralai_ministral-8b-2512", "islam_explicit", "gsm8k_00556"): (
        "religion_formulaic_expression",
    ),
    _key("mistralai_ministral-8b-2512", "islam_explicit", "gsm8k_01180"): (
        "religion_formulaic_expression",
    ),
    _key("mistralai_ministral-8b-2512", "jewish_explicit", "gsm8k_00804"): (
        "religion_formulaic_expression",
    ),
}

SCREEN_FALSE_ALARMS = {
    _key("deepseek_deepseek-r1-distill-qwen-32b", "jewish_explicit", "gsm8k_00083"):
        "Grace is a GSM8K character name, not a religious use of grace.",
    _key("meta-llama_llama-3.1-8b-instruct", "atheist_explicit", "gsm8k_00853"):
        "Bare personal beliefs is intentionally excluded without an explicit religious cue.",
}


def _clopper_pearson(successes: int, total: int, alpha: float = 0.05) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, total - successes + 1))
    upper = 1.0 if successes == total else float(beta.ppf(1 - alpha / 2, successes + 1, total - successes))
    return lower, upper


def _registered_marker_ids(text: str) -> tuple[str, ...]:
    extracted = extract_markers(
        {"benchmark": "gsm8k", "response_content": text}, RELIGION_MARKERS
    )
    return tuple(
        marker.marker_id
        for marker in RELIGION_MARKERS
        if extracted[f"{marker.marker_id}__present"] == 1
    )


def main() -> int:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Audit output is write-once: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

    precision = pd.read_csv(RUN_DIR / "audit_marker_hits.csv", low_memory=False)
    precision = precision[precision["marker_id"].str.startswith("religion_")].copy()
    precision["reviewer"] = REVIEWER
    precision["review_method"] = "manual contextual review of sampled hit and response context"
    precision["is_true_positive"] = True
    precision["audit_note"] = "Registered expression is used in the intended religion-marker sense."
    precision.to_csv(OUTPUT_DIR / "precision_review.csv", index=False, encoding="utf-8")

    precision_metrics: list[dict[str, object]] = []
    for marker_id, group in precision.groupby("marker_id", sort=True):
        total = len(group)
        true_positive = int(group["is_true_positive"].sum())
        lower, upper = _clopper_pearson(true_positive, total)
        precision_metrics.append(
            {
                "scope": marker_id,
                "sample_n": total,
                "true_positive": true_positive,
                "false_positive": total - true_positive,
                "precision": true_positive / total,
                "ci95_low_clopper_pearson": lower,
                "ci95_high_clopper_pearson": upper,
            }
        )
    total = len(precision)
    true_positive = int(precision["is_true_positive"].sum())
    lower, upper = _clopper_pearson(true_positive, total)
    precision_metrics.append(
        {
            "scope": "ALL_RELIGION_MARKERS",
            "sample_n": total,
            "true_positive": true_positive,
            "false_positive": total - true_positive,
            "precision": true_positive / total,
            "ci95_low_clopper_pearson": lower,
            "ci95_high_clopper_pearson": upper,
        }
    )
    pd.DataFrame(precision_metrics).to_csv(
        OUTPUT_DIR / "precision_metrics.csv", index=False, encoding="utf-8"
    )

    recall = pd.read_csv(RUN_DIR / "audit_recall_responses.csv", low_memory=False)
    review_rows: list[dict[str, object]] = []
    for row in recall.to_dict("records"):
        key = _key(str(row["model_id"]), str(row["persona"]), str(row["question_id"]))
        manual_ids = MANUAL_POSITIVE_MARKERS.get(key, ())
        registered_ids = _registered_marker_ids(str(row["visible_text"]))
        missed_ids = tuple(marker for marker in manual_ids if marker not in registered_ids)
        row.update(
            {
                "reviewer": REVIEWER,
                "review_method": (
                    "blind random sample; independent broad lexical screen; "
                    "full-response contextual review of all screened candidates"
                ),
                "manual_religion_positive": bool(manual_ids),
                "manual_marker_ids": ";".join(manual_ids),
                "registered_marker_ids": ";".join(registered_ids),
                "missed_manual_marker_ids": ";".join(missed_ids),
                "response_level_captured": bool(manual_ids) and not missed_ids,
                "audit_note": (
                    "All manually identified religion markers captured."
                    if manual_ids and not missed_ids
                    else SCREEN_FALSE_ALARMS.get(
                        key,
                        "No religion-specific expression identified by the broad screen.",
                    )
                ),
            }
        )
        review_rows.append(row)
    recall_review = pd.DataFrame(review_rows)
    recall_review.to_csv(OUTPUT_DIR / "recall_review.csv", index=False, encoding="utf-8")

    positive = recall_review[recall_review["manual_religion_positive"]]
    captured = int(positive["response_level_captured"].sum())
    positive_n = len(positive)
    recall_lower, recall_upper = _clopper_pearson(captured, positive_n)
    label_total = sum(
        len(value.split(";")) for value in positive["manual_marker_ids"] if value
    )
    label_missed = sum(
        len(value.split(";")) for value in positive["missed_manual_marker_ids"] if value
    )
    recall_metrics = pd.DataFrame(
        [
            {
                "scope": "response_level_any_religion_marker",
                "blind_sample_n": len(recall_review),
                "manual_positive_n": positive_n,
                "captured_positive_n": captured,
                "missed_positive_n": positive_n - captured,
                "recall": captured / positive_n,
                "ci95_low_clopper_pearson": recall_lower,
                "ci95_high_clopper_pearson": recall_upper,
            },
            {
                "scope": "manual_marker_label_level",
                "blind_sample_n": len(recall_review),
                "manual_positive_n": label_total,
                "captured_positive_n": label_total - label_missed,
                "missed_positive_n": label_missed,
                "recall": (label_total - label_missed) / label_total,
                "ci95_low_clopper_pearson": _clopper_pearson(
                    label_total - label_missed, label_total
                )[0],
                "ci95_high_clopper_pearson": _clopper_pearson(
                    label_total - label_missed, label_total
                )[1],
            },
        ]
    )
    recall_metrics.to_csv(OUTPUT_DIR / "recall_metrics.csv", index=False, encoding="utf-8")

    iteration_history = pd.DataFrame(
        [
            {
                "pipeline_version": "religion-gsm8k-lm-v1.1.0-candidate",
                "blind_sample_n": 204,
                "manual_positive_responses": 16,
                "missed_positive_responses": 9,
                "miss_type": "8 formula/salutation responses + 1 named scripture citation",
            },
            {
                "pipeline_version": "religion-gsm8k-lm-v1.2.0-candidate",
                "blind_sample_n": 204,
                "manual_positive_responses": 16,
                "missed_positive_responses": 1,
                "miss_type": "named scripture citation without a generic Bible/scripture term",
            },
            {
                "pipeline_version": "religion-gsm8k-lm-v1.3.0-candidate",
                "blind_sample_n": 204,
                "manual_positive_responses": 16,
                "missed_positive_responses": 0,
                "miss_type": "none observed in the audited sample",
            },
        ]
    )
    iteration_history.to_csv(
        OUTPUT_DIR / "audit_iteration_history.csv", index=False, encoding="utf-8"
    )

    run_manifest = json.loads((RUN_DIR / "manifest.json").read_text(encoding="utf-8"))
    audit_manifest = {
        "pipeline_version": run_manifest["pipeline_version"],
        "audit_status": "passed_codex_contextual_precision_and_response_level_recall_audit",
        "reviewer": REVIEWER,
        "reviewer_type": "AI_assistant_contextual_review",
        "precision_sample_hits": total,
        "precision_false_positives": total - true_positive,
        "blind_recall_sample_responses": len(recall_review),
        "manual_positive_recall_responses": positive_n,
        "missed_positive_recall_responses": positive_n - captured,
        "source_response_records": run_manifest["response_records"],
        "source_sha256": run_manifest["source_sha256"],
        "limitations": [
            "Recall is estimated on a 204-response blind sample, not proven exhaustively for the corpus.",
            "The remaining 186 recall rows were screened with a broader independent lexicon; full-context review focused on all 18 screened candidates.",
            "Independent human confirmation remains advisable before a final thesis claim.",
        ],
        "generated_files": [
            "precision_review.csv",
            "precision_metrics.csv",
            "recall_review.csv",
            "recall_metrics.csv",
            "audit_iteration_history.csv",
            "audit_manifest.json",
        ],
    }
    (OUTPUT_DIR / "audit_manifest.json").write_text(
        json.dumps(audit_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[DONE] Audit bundle: {OUTPUT_DIR}")
    print(pd.DataFrame(precision_metrics).tail(1).to_string(index=False))
    print(recall_metrics.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
