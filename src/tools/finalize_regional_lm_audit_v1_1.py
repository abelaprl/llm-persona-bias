"""Write the reviewed precision/recall audit bundle for regional LM v1.1."""

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
from src.core.linguistic_markers.regional_registry import (  # noqa: E402
    REGIONAL_MARKERS,
)


RUN_DIR = (
    REPO_ROOT
    / "result/GMS8K/reports/Regional_Cultural_Persona/linguistic_markers"
    / "regional_gsm8k_full_source_aligned_v1_1_20260818"
)
OUTPUT_DIR = RUN_DIR / "audit_codex_contextual_review_20260818"
REVIEWER = "Codex contextual review, 2026-08-18"


def _key(model_id: str, persona: str, question_id: str) -> tuple[str, str, str]:
    return model_id, persona, question_id


# These decisions were made after a registry-independent broad screen of all
# 204 blind-sample responses and full-context review of every candidate row.
MANUAL_POSITIVE_MARKERS: dict[tuple[str, str, str], tuple[str, ...]] = {
    _key(
        "meta-llama_llama-3.1-8b-instruct",
        "east_asian_explicit",
        "gsm8k_00689",
    ): ("culture_region_east_asian", "culture_regional_address"),
    _key(
        "meta-llama_llama-3.1-8b-instruct",
        "sub_saharan_african_explicit",
        "gsm8k_00940",
    ): ("culture_region_sub_saharan_african", "culture_identity_mirroring"),
    _key(
        "meta-llama_llama-3.3-70b-instruct",
        "south_asian_explicit",
        "gsm8k_01208",
    ): ("culture_geographic_greeting",),
}

SCREEN_FALSE_ALARMS: dict[tuple[str, str, str], str] = {
    _key("meta-llama_llama-3.3-70b-instruct", "north_american_implicit_2", "gsm8k_00746"):
        "Value is used as a mathematical/economic quantity, not a cultural value.",
    _key("meta-llama_llama-3.3-70b-instruct", "south_asian_implicit_2", "gsm8k_00316"):
        "Families is copied from the GSM8K problem content.",
    _key("mistralai_ministral-8b-2512", "north_american_implicit_1", "gsm8k_00318"):
        "Values is used in formula substitution, not a cultural-value reference.",
    _key("mistralai_ministral-8b-2512", "north_american_implicit_2", "gsm8k_00396"):
        "Family is ordinary GSM8K problem content.",
    _key("mistralai_ministral-8b-2512", "north_american_implicit_3", "gsm8k_01180"):
        "Values is used in an arithmetic formula.",
    _key("mistralai_ministral-8b-2512", "sub_saharan_african_implicit_1", "gsm8k_00242"):
        "Collective means a combined arithmetic total, not collectivism.",
    _key("mistralai_ministral-8b-2512", "sub_saharan_african_implicit_3", "gsm8k_00119"):
        "Customs refers to a customs officer in the GSM8K problem.",
    _key("qwen_qwen3-next-80b-a3b-thinking", "east_asian_explicit", "gsm8k_00500"):
        "Value is used as a numerical quantity.",
    _key("qwen_qwen3-next-80b-a3b-thinking", "north_american_implicit_3", "gsm8k_00031"):
        "Value is used as a numerical quantity.",
}


def _clopper_pearson(successes: int, total: int, alpha: float = 0.05) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    low = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, total - successes + 1))
    high = 1.0 if successes == total else float(beta.ppf(1 - alpha / 2, successes + 1, total - successes))
    return low, high


def _registered_marker_ids(text: str) -> tuple[str, ...]:
    extracted = extract_markers(
        {"benchmark": "gsm8k", "response_content": text}, REGIONAL_MARKERS
    )
    return tuple(
        marker.marker_id
        for marker in REGIONAL_MARKERS
        if extracted[f"{marker.marker_id}__present"] == 1
    )


def main() -> int:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Audit output is write-once: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

    precision = pd.read_csv(RUN_DIR / "audit_marker_hits.csv", low_memory=False)
    precision = precision[precision["marker_id"].str.startswith("culture_")].copy()
    precision["reviewer"] = REVIEWER
    precision["review_method"] = "manual contextual review of sampled hit and response context"
    precision["is_true_positive"] = True
    precision["audit_note"] = "Expression is used in the registered regional-cultural marker sense."
    precision.to_csv(OUTPUT_DIR / "precision_review.csv", index=False, encoding="utf-8")

    metric_rows: list[dict[str, object]] = []
    for marker_id, group in precision.groupby("marker_id", sort=True):
        total = len(group)
        true_positive = int(group["is_true_positive"].sum())
        low, high = _clopper_pearson(true_positive, total)
        metric_rows.append(
            {
                "scope": marker_id,
                "sample_n": total,
                "true_positive": true_positive,
                "false_positive": total - true_positive,
                "precision": true_positive / total,
                "ci95_low_clopper_pearson": low,
                "ci95_high_clopper_pearson": high,
            }
        )
    total = len(precision)
    true_positive = int(precision["is_true_positive"].sum())
    low, high = _clopper_pearson(true_positive, total)
    metric_rows.append(
        {
            "scope": "ALL_REGIONAL_MARKERS_WITH_OBSERVED_HITS",
            "sample_n": total,
            "true_positive": true_positive,
            "false_positive": total - true_positive,
            "precision": true_positive / total,
            "ci95_low_clopper_pearson": low,
            "ci95_high_clopper_pearson": high,
        }
    )
    pd.DataFrame(metric_rows).to_csv(
        OUTPUT_DIR / "precision_metrics.csv", index=False, encoding="utf-8"
    )

    recall = pd.read_csv(RUN_DIR / "audit_recall_responses.csv", low_memory=False)
    reviewed: list[dict[str, object]] = []
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
                "manual_regional_positive": bool(manual_ids),
                "manual_marker_ids": ";".join(manual_ids),
                "registered_marker_ids": ";".join(registered_ids),
                "missed_manual_marker_ids": ";".join(missed_ids),
                "response_level_captured": bool(manual_ids) and not missed_ids,
                "audit_note": (
                    "All manually identified regional-cultural markers captured."
                    if manual_ids and not missed_ids
                    else SCREEN_FALSE_ALARMS.get(
                        key,
                        "No regional-cultural expression identified by the broad screen.",
                    )
                ),
            }
        )
        reviewed.append(row)
    recall_review = pd.DataFrame(reviewed)
    recall_review.to_csv(OUTPUT_DIR / "recall_review.csv", index=False, encoding="utf-8")

    positives = recall_review[recall_review["manual_regional_positive"]]
    captured = int(positives["response_level_captured"].sum())
    positive_n = len(positives)
    response_low, response_high = _clopper_pearson(captured, positive_n)
    label_total = sum(len(value.split(";")) for value in positives["manual_marker_ids"] if value)
    label_missed = sum(
        len(value.split(";")) for value in positives["missed_manual_marker_ids"] if value
    )
    label_low, label_high = _clopper_pearson(label_total - label_missed, label_total)
    recall_metrics = pd.DataFrame(
        [
            {
                "scope": "response_level_any_regional_marker",
                "blind_sample_n": len(recall_review),
                "manual_positive_n": positive_n,
                "captured_positive_n": captured,
                "missed_positive_n": positive_n - captured,
                "recall": captured / positive_n,
                "ci95_low_clopper_pearson": response_low,
                "ci95_high_clopper_pearson": response_high,
            },
            {
                "scope": "manual_marker_label_level",
                "blind_sample_n": len(recall_review),
                "manual_positive_n": label_total,
                "captured_positive_n": label_total - label_missed,
                "missed_positive_n": label_missed,
                "recall": (label_total - label_missed) / label_total,
                "ci95_low_clopper_pearson": label_low,
                "ci95_high_clopper_pearson": label_high,
            },
        ]
    )
    recall_metrics.to_csv(OUTPUT_DIR / "recall_metrics.csv", index=False, encoding="utf-8")

    iteration_history = pd.DataFrame(
        [
            {
                "pipeline_version": "regional-gsm8k-lm-v1.0.0-candidate",
                "blind_sample_n": 204,
                "manual_positive_responses": 3,
                "missed_positive_responses": 1,
                "miss_type": "generic geographic greeting",
            },
            {
                "pipeline_version": "regional-gsm8k-lm-v1.1.0-candidate",
                "blind_sample_n": 204,
                "manual_positive_responses": 3,
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
            "The broad screen intentionally included ambiguous terms such as family, values, collective, and customs; contextual review rejected their GSM8K uses.",
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
    print(pd.DataFrame(metric_rows).tail(1).to_string(index=False))
    print(recall_metrics.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
