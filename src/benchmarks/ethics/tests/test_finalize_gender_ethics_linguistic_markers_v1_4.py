from __future__ import annotations

import unittest

import pandas as pd

from src.benchmarks.ethics.finalize_gender_ethics_linguistic_markers_v1_4 import (
    _agreement_stats,
    _build_disagreements,
    _marker_summary,
    _paired_reviews,
    _resolve_reviews,
    _validate_against_candidate_template,
)


class GenderEthicsLinguisticMarkerFinalizerTests(unittest.TestCase):
    def _precision_review(self, slot: int, reviewer_id: str, values: list[bool]) -> pd.DataFrame:
        frame = pd.DataFrame(
            {
                "audit_item_id": ["precision_00001", "precision_00002"],
                "model_id": ["m", "m"],
                "persona": ["man_explicit_1", "woman_explicit_1"],
                "question_id": ["q1", "q2"],
                "marker_id": ["stance_hedge", "stance_booster"],
                "matched_text": ["might", "clearly"],
                "review_value": values,
            }
        )
        frame.attrs["reviewer_id"] = reviewer_id
        return frame

    def _recall_review(
        self,
        slot: int,
        reviewer_id: str,
        values: list[frozenset[str]],
    ) -> pd.DataFrame:
        frame = pd.DataFrame(
            {
                "audit_item_id": ["recall_00001", "recall_00002"],
                "model_id": ["m", "m"],
                "persona": ["man_explicit_1", "woman_explicit_1"],
                "question_id": ["q1", "q2"],
                "visible_text": ["text one", "text two"],
                "review_value": values,
            }
        )
        frame.attrs["reviewer_id"] = reviewer_id
        return frame

    def test_agreement_stats_reports_perfect_binary_kappa(self) -> None:
        result = _agreement_stats(["true", "false"], ["true", "false"])
        self.assertEqual(result["observed_agreement"], 1.0)
        self.assertEqual(result["cohen_kappa"], 1.0)
        self.assertEqual(result["kappa_status"], "estimated")

    def test_disagreements_are_stable_and_require_adjudication(self) -> None:
        precision = _paired_reviews(
            self._precision_review(1, "human-a", [True, True]),
            self._precision_review(2, "human-b", [True, False]),
            audit_type="precision",
        )
        recall = _paired_reviews(
            self._recall_review(1, "human-a", [frozenset(), frozenset()]),
            self._recall_review(2, "human-b", [frozenset(), frozenset({"stance_hedge"})]),
            audit_type="recall",
        )
        disagreements = _build_disagreements(precision, recall)
        self.assertEqual(
            set(disagreements["disagreement_id"]),
            {"precision:precision_00002", "recall:recall_00002"},
        )
        precision_final, recall_final, complete = _resolve_reviews(
            precision, recall, disagreements, None
        )
        self.assertFalse(complete)
        self.assertTrue(precision_final["final_is_true_positive"].isna().any())
        self.assertTrue(recall_final["final_manual_marker_ids"].isna().any())

    def test_false_positive_or_recall_miss_closes_rule_gate(self) -> None:
        precision = _paired_reviews(
            self._precision_review(1, "human-a", [True, False]),
            self._precision_review(2, "human-b", [True, False]),
            audit_type="precision",
        )
        recall = _paired_reviews(
            self._recall_review(1, "human-a", [frozenset(), frozenset({"stance_hedge"})]),
            self._recall_review(2, "human-b", [frozenset(), frozenset({"stance_hedge"})]),
            audit_type="recall",
        )
        disagreements = _build_disagreements(precision, recall)
        precision_final, recall_final, complete = _resolve_reviews(
            precision, recall, disagreements, None
        )
        summary = _marker_summary(precision_final, recall_final)
        self.assertTrue(complete)
        self.assertFalse(summary["rule_audit_passed"].all())
        hedge = summary[summary["marker_id"].eq("stance_hedge")].iloc[0]
        booster = summary[summary["marker_id"].eq("stance_booster")].iloc[0]
        self.assertEqual(hedge["recall_miss_items"], 1)
        self.assertEqual(booster["false_positive_items"], 1)

    def test_review_cannot_change_immutable_candidate_text(self) -> None:
        template = pd.DataFrame(
            {
                "audit_item_id": ["recall_00001"],
                "human_review_complete": [""],
                "reviewer_slot": [1],
                "reviewer_id": [""],
                "visible_text": ["original visible response"],
                "manual_marker_ids": [""],
                "missed_span_notes": [""],
                "audit_note": [""],
            }
        )
        review = template.copy()
        review.loc[0, "human_review_complete"] = "true"
        review.loc[0, "reviewer_id"] = "human-a"
        _validate_against_candidate_template(
            review, template, audit_type="recall", reviewer_slot=1
        )
        review.loc[0, "visible_text"] = "changed response"
        with self.assertRaisesRegex(ValueError, "immutable template column"):
            _validate_against_candidate_template(
                review, template, audit_type="recall", reviewer_slot=1
            )


if __name__ == "__main__":
    unittest.main()
