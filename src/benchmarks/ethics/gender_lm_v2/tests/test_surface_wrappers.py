"""Observed binary-wrapper boundaries without interpreting or scoring verdicts."""
import json
from pathlib import Path
import unittest

from src.benchmarks.ethics.gender_lm_v2.surfaces import build_surfaces


class ObservedWrapperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture = Path(__file__).parent / "fixtures" / "surface_wrapper_responses.json"
        cls.rows = json.loads(fixture.read_text(encoding="utf-8"))["rows"]

    def test_all_twelve_observed_pure_wrappers_have_no_explanation(self):
        rows = [row for row in self.rows if row["expected_status"] == "verdict_only"]
        self.assertEqual(len(rows), 12)
        for row in rows:
            with self.subTest(model=row["model_id"], persona=row["persona"], question=row["question_id"]):
                surface = build_surfaces(row["raw_text"])
                self.assertEqual(surface.status, "verdict_only")
                self.assertEqual(surface.word_count, 0)
                self.assertEqual(surface.explanation_text, "")
                self.assertEqual(surface.raw_text, row["raw_text"])
                self.assertEqual(surface.visible_text, row["raw_text"])

    def test_all_eighteen_observed_natural_continuations_remain_available(self):
        rows = [row for row in self.rows if row["expected_status"] == "substantive_explanation"]
        self.assertEqual(len(rows), 18)
        for row in rows:
            with self.subTest(model=row["model_id"], persona=row["persona"], question=row["question_id"]):
                surface = build_surfaces(row["raw_text"])
                self.assertEqual(surface.status, "substantive_explanation")
                self.assertEqual(surface.word_count, row["word_count"])
                self.assertEqual(surface.raw_text, row["raw_text"])
                self.assertEqual(len(surface.explanation_text), len(row["raw_text"]))

    def test_fullmatch_does_not_strip_natural_text_after_a_wrapper(self):
        for wrapper in ("The answer is 0.", "The answer is 1.", "**Answer: 0**", "**Answer: 1**", "**Answer: <0>**"):
            for separator in (" ", "\n"):
                text = wrapper + separator + "There are additional considerations."
                with self.subTest(text=text):
                    surface = build_surfaces(text)
                    self.assertEqual(surface.status, "substantive_explanation")
                    self.assertIn("There are additional considerations.", surface.explanation_text)
                    self.assertEqual(surface.raw_text, text)


if __name__ == "__main__":
    unittest.main()
