"""End-to-end extraction invariants without filesystem side effects."""
import unittest


class PipelineTests(unittest.TestCase):
    def test_all_rows_and_unavailable_are_preserved(self):
        from src.benchmarks.ethics.gender_lm_v2.run import extract_batch
        rows = [{"model_id": "model", "persona": "neutral", "subset": "justice", "question_id": str(i), "file": "f.json", "response_content_full": text} for i, text in enumerate(["1", "", "Perhaps this is fair."])]
        extracted = extract_batch(rows)
        self.assertEqual(len(extracted), 3)
        self.assertEqual([item[0]["text_status"] for item in extracted], ["verdict_only", "missing", "substantive_explanation"])
        self.assertIsNone(extracted[0][0]["stance_hedge__count"])
        self.assertEqual(extracted[2][0]["stance_hedge__count"], 1)
        self.assertEqual(extracted[2][1][0]["rule_status"], "candidate")

    def test_metadata_does_not_change_markers(self):
        from src.benchmarks.ethics.gender_lm_v2.run import extract_batch
        first = {"model_id": "m", "persona": "neutral", "subset": "justice", "question_id": "q", "file": "f", "response_content_full": "Perhaps this is fair."}
        second = dict(first, model_id="different", persona="woman_implicit_2")
        a, b = extract_batch([first, second])
        for key in a[0]:
            if key.endswith(("__count", "__density", "__present")):
                self.assertEqual(a[0][key], b[0][key])

    def test_validated_requires_independent_audit(self):
        from src.benchmarks.ethics.gender_lm_v2.run import check_mode
        with self.assertRaises(ValueError):
            check_mode("validated", None)


if __name__ == "__main__":
    unittest.main()
