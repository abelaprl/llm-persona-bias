"""Audit sampling and human-evidence gates, without benchmark labels."""
import hashlib
import unittest
from pathlib import Path

import pandas as pd

from src.benchmarks.ethics.gender_lm_v2.audit import (
    assign_split, build_audit_package, evaluate_audit, promotion_decisions,
    sample_audit,
)


REGISTRY = [{"marker_id": "hedge", "definition": "Epistemic uncertainty",
             "patterns": [{"rule_id": "hedge.r1", "pattern": "may"}]}]


def fixtures(n=1000):
    records = pd.DataFrame([
        {"model_id": "model", "persona": "neutral", "subset": "justice",
         "question_id": str(i), "raw_text": "may " + "complete " * 80,
         "visible_text": "may " + "complete " * 80,
         "explanation_text": "may " + "complete " * 80,
         "text_status": "available" if i % 5 else "missing",
         "text_available": bool(i % 5), "word_count": 81 if i % 5 else 0}
        for i in range(n)
    ])
    spans = records.loc[records.text_available, ["model_id", "persona", "subset", "question_id"]].copy()
    spans = spans.assign(marker_id="hedge", rule_id="hedge.r1", start=0,
                         end=3, text="may", surface_name="explanation_text",
                         flags="", rule_status="candidate")
    return records, spans


def annotated(rule="hedge.r1", n=100, fp=False):
    rows = []
    for i in range(n):
        for reviewer in ("alice", "bob"):
            rows.append({"annotation_id": f"{rule}:{i}", "marker_id": "hedge",
                         "rule_id": rule, "annotation_type": "precision",
                         "question_id": str(i), "subset": "justice",
                         "reviewer_type": "human", "reviewer_id": reviewer,
                         "reviewed_at": "2026-09-15", "full_response_read": True,
                         "is_true_positive": not (fp and i == 0),
                         "rule_kind": "semantic", "total_hits": n,
                         "required_sample": n, "split": "development",
                         "sampling_probability": 1.0})
    return pd.DataFrame(rows)


class AuditTests(unittest.TestCase):
    def test_split_matches_contract_and_exclusion(self):
        bucket = int(hashlib.sha256(b"42|justice|q1").hexdigest(), 16) % 100
        expected = "discovery" if bucket < 60 else "development" if bucket < 80 else "held_out"
        self.assertEqual(assign_split("justice", "q1"), expected)
        heldout = next(str(i) for i in range(100) if assign_split("justice", str(i)) == "held_out")
        self.assertEqual(assign_split("justice", heldout, previously_read={("justice", heldout)}), "discovery")

    def test_empty_annotations_do_not_promote(self):
        empty = annotated().drop(columns="is_true_positive")
        self.assertTrue(all(x.status != "accepted" for x in promotion_decisions(empty)))
        summary = evaluate_audit(empty, registry=REGISTRY)
        self.assertEqual(summary.iloc[0].n_reviewed_precision, 0)
        self.assertEqual(summary.iloc[0].target_status, "not_demonstrated")

    def test_one_false_positive_holds_only_its_rule(self):
        rows = pd.concat([annotated(), annotated("hedge.r2", fp=True)], ignore_index=True)
        decisions = {d.rule_id: d for d in promotion_decisions(rows)}
        self.assertEqual(decisions["hedge.r1"].status, "accepted")
        self.assertNotEqual(decisions["hedge.r2"].status, "accepted")

    def test_unsigned_and_ai_labels_cannot_promote(self):
        rows = annotated()
        rows["reviewer_type"] = "agent"
        self.assertTrue(all(d.status != "accepted" for d in promotion_decisions(rows)))
        rows["reviewer_type"] = "human"
        rows["reviewer_id"] = ""
        self.assertTrue(all(d.status != "accepted" for d in promotion_decisions(rows)))

    def test_disagreement_prevents_promotion(self):
        rows = annotated()
        rows.loc[0, "is_true_positive"] = False
        self.assertTrue(all(d.status != "accepted" for d in promotion_decisions(rows)))

    def test_sampling_is_deterministic_blind_and_recall_hit_independent(self):
        records, spans = fixtures()
        first = sample_audit(records, spans, registry=REGISTRY)
        second = sample_audit(records, spans.iloc[:0], registry=REGISTRY)
        pd.testing.assert_frame_equal(first["recall_reviewer_1"], second["recall_reviewer_1"])
        for frame in (first["precision_reviewer_1"], first["recall_reviewer_1"]):
            self.assertFalse(set(frame.columns) & {"model_id", "persona", "ground_truth", "correctness"})
            self.assertTrue(frame.reviewer_id.eq("").all())
            self.assertTrue(frame.full_response_read.eq("").all())
        self.assertGreaterEqual(len(first["precision_reviewer_1"]), 100)
        self.assertNotIn("held_out", set(first["discovery_samples"].split))
        self.assertEqual(first["corpus_lookup"].explanation_text.str.len().max(), len(records.iloc[0].explanation_text))

    def test_recall_sample_contains_unavailable_and_no_hit_audit_info(self):
        records, spans = fixtures()
        package = sample_audit(records, spans.iloc[:1], registry=REGISTRY)
        self.assertIn(False, set(package["recall_selection"].text_available))
        self.assertIn(False, set(package["recall_selection"].has_detector_hit))
        self.assertNotIn("has_detector_hit", package["recall_reviewer_1"].columns)

    def test_positive_reference_is_recall_denominator(self):
        rows = annotated(n=50)
        rows["annotation_type"] = "recall"
        rows["reference_count"] = 2
        rows["detected_count"] = 1
        rows["matched_count"] = 1
        rows["reference_complete"] = True
        summary = evaluate_audit(rows, registry=REGISTRY).iloc[0]
        self.assertEqual(summary.reference_positive_count, 100)
        self.assertEqual(summary.recall, 0.5)
        self.assertEqual(summary.target_status, "not_demonstrated")

    def test_zero_errors_has_nontrivial_uncertainty(self):
        summary = evaluate_audit(annotated(), registry=REGISTRY).iloc[0]
        self.assertEqual(summary.precision, 1.0)
        self.assertLess(summary.precision_ci_low, 0.99)
        self.assertEqual(summary.target_status, "not_demonstrated")

    def test_duplicate_reviewer_is_not_independent(self):
        rows = annotated()
        rows["reviewer_id"] = "alice"
        self.assertTrue(all(d.status != "accepted" for d in promotion_decisions(rows)))

    def test_output_collision_is_rejected(self):
        records, spans = fixtures(25)
        with self.assertRaises(FileExistsError):
            build_audit_package(records, spans, REGISTRY, Path(__file__).parent)

    def test_unavailable_text_does_not_become_recall_negative(self):
        rows = annotated(n=10)
        rows["annotation_type"] = "recall"
        rows["reference_count"] = 0
        rows["detected_count"] = 0
        rows["matched_count"] = 0
        rows["reference_complete"] = True
        rows["text_available"] = False
        self.assertEqual(evaluate_audit(rows, registry=REGISTRY).iloc[0].n_reviewed_recall, 0)

    def test_many_spans_one_question_never_certifies_99_percent(self):
        rows = annotated(n=400)
        rows["question_id"] = "one_question"
        summary = evaluate_audit(rows, registry=REGISTRY).iloc[0]
        self.assertEqual(summary.precision_question_clusters, 1)
        self.assertLess(summary.precision_ci_low, 0.99)
        self.assertEqual(summary.target_status, "not_demonstrated")

    def test_rule_revocation_and_mixed_revision_are_local(self):
        rows = pd.concat([annotated(), annotated("hedge.r2")], ignore_index=True)
        rows["rule_status"] = "candidate"
        rows.loc[rows.rule_id.eq("hedge.r2"), "rule_status"] = "revoked"
        decisions = {d.rule_id: d.status for d in promotion_decisions(rows)}
        self.assertEqual(decisions, {"hedge.r1": "accepted", "hedge.r2": "revoked"})

    def test_fake_summary_is_not_human_evidence(self):
        summary = pd.DataFrame([{"rule_id": "hedge.r1", "marker_id": "hedge", "precision": 1,
                                 "recall": 1, "target_status": "demonstrated", "n_reviewed": 1000}])
        self.assertNotEqual(promotion_decisions(summary)[0].status, "accepted")

    def test_foreign_spans_rejected(self):
        records, spans = fixtures(5)
        spans.loc[spans.index[0], "question_id"] = "not_in_corpus"
        with self.assertRaises(ValueError):
            sample_audit(records, spans, registry=REGISTRY)

    def test_structure_spans_have_their_full_surface_in_lookup(self):
        records, spans = fixtures()
        records["structure_text"] = "**may** " + "complete " * 80
        records["visible_text"] = records.structure_text
        records["raw_text"] = records.structure_text
        records["explanation_text"] = "  may   " + "complete " * 80
        spans["surface_name"] = "structure_text"
        spans["start"], spans["end"] = 2, 5
        package = sample_audit(records, spans, registry=REGISTRY)
        lookup = package["corpus_lookup"].set_index("response_id")
        self.assertIn("structure_text", lookup.columns)
        self.assertTrue(lookup.structure_text.eq(records.structure_text.iloc[0]).all())
        precision = package["precision_reviewer_1"]
        self.assertGreater(len(precision), 0)
        for row in precision.itertuples(index=False):
            self.assertEqual(lookup.loc[row.response_id, row.surface_name][row.start:row.end], row.text)


if __name__ == "__main__":
    unittest.main()
