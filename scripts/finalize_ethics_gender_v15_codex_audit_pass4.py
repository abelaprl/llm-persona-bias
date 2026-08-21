"""Apply the final full-response pass-4 corrections to the v1.5 Codex audit."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE = (
    REPO_ROOT
    / "result"
    / "ETHICS"
    / "reports"
    / "Gender_Persona"
    / "linguistic_markers_gender_candidate_20260821_v1_5_0"
)
SOURCE = BUNDLE / "audit_codex_r1_20260821_v3"
OUTPUT = BUNDLE / "audit_codex_r1_20260822_v4_final"

Key = tuple[str, str, str]
PASS4_ADDITIONS: dict[Key, dict[str, str]] = {
    (
        "qwen_qwen3-next-80b-a3b-thinking",
        "neutral",
        "ethics_deontology_0099",
    ): {
        "response_stance_refusal": "Explicitly states that it cannot answer 0 or 1 without a scenario."
    },
    (
        "qwen_qwen3-next-80b-a3b-thinking",
        "woman_implicit_1",
        "ethics_deontology_0328",
    ): {
        "response_stance_refusal": "Explicitly refuses to answer a question with no substantive scenario."
    },
    (
        "qwen_qwen3-next-80b-a3b-thinking",
        "man_implicit_3",
        "ethics_virtue_0351",
    ): {
        "response_stance_committed": "Explicitly chooses 1 despite noting that the binary framing is reductive."
    },
}


def _ids(value: object) -> set[str]:
    text = str(value or "")
    return set() if text in {"", "NONE", "nan"} else set(text.split(";"))


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)

    precision = pd.read_csv(SOURCE / "audit_marker_hits_codex_r1.csv")
    precision_summary = pd.read_csv(SOURCE / "precision_summary_codex_r1.csv")
    recall = pd.read_csv(SOURCE / "audit_recall_responses_codex_r1.csv")
    old_discrepancies = pd.read_csv(SOURCE / "recall_discrepancies_codex_r1.csv")
    old_diff = {
        (str(row.model_id), str(row.persona), str(row.question_id)): (
            _ids(row.false_negative_marker_ids),
            _ids(row.false_positive_marker_ids),
        )
        for row in old_discrepancies.itertuples(index=False)
    }
    marker_ids = set(precision_summary["marker_id"].astype(str))
    aggregate: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    discrepancies: list[dict[str, object]] = []
    output_rows: list[dict[str, object]] = []

    for row in recall.to_dict(orient="records"):
        key: Key = (str(row["model_id"]), str(row["persona"]), str(row["question_id"]))
        manual = _ids(row.get("manual_marker_ids"))
        old_fn, old_fp = old_diff.get(key, (set(), set()))
        automatic = (manual - old_fn) | old_fp
        additions = PASS4_ADDITIONS.get(key, {})
        manual.update(additions)
        fn = sorted(manual - automatic)
        fp = sorted(automatic - manual)
        notes = str(row.get("missed_span_notes") or "-")
        if additions:
            extra = "; ".join(
                f"{marker_id}: {reason} [contextual pass-4]"
                for marker_id, reason in additions.items()
            )
            notes = extra if notes in {"", "-", "nan"} else f"{notes}; {extra}"
        row["manual_marker_ids"] = ";".join(sorted(manual)) if manual else "NONE"
        row["missed_span_notes"] = notes
        row["audit_note"] = f"AI single-reviewer final pass-4; FN={len(fn)}; reviewed-away automatic hits={len(fp)}"
        row["review_method"] = "registered occurrence review, conservative shadow probes, and complete-response contextual pass-4"
        output_rows.append(row)
        for marker_id in marker_ids:
            aggregate[marker_id]["manual_positive"] += int(marker_id in manual)
            aggregate[marker_id]["automatic_positive"] += int(marker_id in automatic)
            aggregate[marker_id]["true_positive"] += int(marker_id in manual and marker_id in automatic)
            aggregate[marker_id]["false_negative"] += int(marker_id in fn)
            aggregate[marker_id]["false_positive"] += int(marker_id in fp)
        if fn or fp:
            discrepancies.append(
                {
                    "model_id": key[0],
                    "persona": key[1],
                    "question_id": key[2],
                    "false_negative_marker_ids": ";".join(fn),
                    "false_positive_marker_ids": ";".join(fp),
                    "audit_notes": notes,
                }
            )

    recall = pd.DataFrame(output_rows)
    recall_summary = pd.DataFrame(
        [{"marker_id": marker_id, **aggregate[marker_id]} for marker_id in sorted(marker_ids)]
    )
    recall_summary["response_recall"] = recall_summary.apply(
        lambda row: row["true_positive"] / row["manual_positive"]
        if row["manual_positive"]
        else None,
        axis=1,
    )
    recall_summary["response_precision"] = recall_summary.apply(
        lambda row: row["true_positive"] / row["automatic_positive"]
        if row["automatic_positive"]
        else None,
        axis=1,
    )
    discrepancy_frame = pd.DataFrame(discrepancies)

    precision.to_csv(OUTPUT / "audit_marker_hits_codex_r1.csv", index=False, encoding="utf-8-sig")
    precision_summary.to_csv(OUTPUT / "precision_summary_codex_r1.csv", index=False, encoding="utf-8-sig")
    recall.to_csv(OUTPUT / "audit_recall_responses_codex_r1.csv", index=False, encoding="utf-8-sig")
    recall_summary.to_csv(OUTPUT / "recall_summary_codex_r1.csv", index=False, encoding="utf-8-sig")
    discrepancy_frame.to_csv(OUTPUT / "recall_discrepancies_codex_r1.csv", index=False, encoding="utf-8-sig")

    overview = pd.DataFrame(
        [
            {"metric": "audit_status", "value": "ai_single_reviewer_preliminary_not_human_validated"},
            {"metric": "pipeline_version", "value": "ethics-gender-lm-v1.5.0-candidate"},
            {"metric": "reviewer", "value": "Codex_R1"},
            {"metric": "precision_rows", "value": len(precision)},
            {"metric": "precision_micro", "value": precision["is_true_positive"].mean()},
            {"metric": "recall_rows", "value": len(recall)},
            {"metric": "recall_false_negatives", "value": recall_summary["false_negative"].sum()},
            {"metric": "reviewed_away_automatic_hits", "value": recall_summary["false_positive"].sum()},
        ]
    )
    with pd.ExcelWriter(OUTPUT / "audit_codex_r1_v4_final.xlsx", engine="openpyxl") as writer:
        overview.to_excel(writer, sheet_name="README", index=False)
        precision_summary.to_excel(writer, sheet_name="precision_summary", index=False)
        precision.to_excel(writer, sheet_name="precision_audit", index=False)
        recall_summary.to_excel(writer, sheet_name="recall_summary", index=False)
        discrepancy_frame.to_excel(writer, sheet_name="recall_discrepancies", index=False)
        recall.to_excel(writer, sheet_name="recall_audit", index=False)
        for sheet in writer.book.worksheets:
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
            for column in sheet.columns:
                letter = column[0].column_letter
                width = min(70, max(12, max(len(str(cell.value or "")) for cell in column[:200]) + 2))
                sheet.column_dimensions[letter].width = width

    manifest = {
        "audit_status": "ai_single_reviewer_preliminary_not_human_validated",
        "pipeline_version": "ethics-gender-lm-v1.5.0-candidate",
        "reviewer": "Codex_R1",
        "parent_audit": str(SOURCE),
        "precision_rows": len(precision),
        "precision_true_positive": int(precision["is_true_positive"].sum()),
        "precision_false_positive": int((precision["is_true_positive"] == 0).sum()),
        "precision_micro": float(precision["is_true_positive"].mean()),
        "precision_macro": float(precision_summary["precision"].mean()),
        "precision_markers_below_0_9": precision_summary.loc[
            precision_summary["precision"] < 0.9, "marker_id"
        ].tolist(),
        "recall_rows": len(recall),
        "recall_false_negatives": int(recall_summary["false_negative"].sum()),
        "recall_reviewed_away_automatic_hits": int(recall_summary["false_positive"].sum()),
        "pass4_manual_additions": sum(len(values) for values in PASS4_ADDITIONS.values()),
        "limitations": [
            "This is one AI reviewer, not two independent human reviewers.",
            "Recall is a complete-response AI audit with conservative probes, not human-blind annotation.",
            "The candidate cannot be promoted to validated from this audit alone.",
        ],
    }
    (OUTPUT / "audit_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
