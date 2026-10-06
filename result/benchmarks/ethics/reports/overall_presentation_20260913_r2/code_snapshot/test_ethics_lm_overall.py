"""Protect response-level pooling against silent measurement errors."""
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

from src.MAIN_PARSING.ETHICS.build_ethics_lm_overall_statistics import pool_frame, support_audit
from src.MAIN_PARSING.ETHICS.complete_lm_notebook_presentation import ROOT, explorer, normalize
from src.MAIN_PARSING.ETHICS.tests import test_complete_lm_notebook_presentation as previous_tests


class OverallTests(unittest.TestCase):
    def responses(self):
        return pd.DataFrame([
            dict(model_id="m", persona=p, subset=s, question_id="1", text_available=True,
                 validated_committed=p == "a", score=v)
            for s, p, v in [("commonsense", "a", 1), ("justice", "a", 2), ("justice", "b", 3)]
        ])

    def test_repeated_question_ids_remain_separate(self):
        pooled = pool_frame(self.responses())
        self.assertEqual(pooled.question_id.nunique(), 2)
        self.assertEqual(set(pooled.subset), {"ethics_all"})
        self.assertEqual(list(pooled.score), [1, 2, 3])

    def test_duplicate_original_keys_are_rejected(self):
        frame = self.responses()
        with self.assertRaises(ValueError):
            pool_frame(pd.concat([frame, frame.iloc[:1]]))

    def result_frame(self, n, population="all_visible"):
        return pd.DataFrame([dict(model_id="m", left_persona="a", right_persona="b", value_col="score",
                                  population=population, n_pairs=n, contrast_id="a_b", outcome_id="score")])

    def test_pairing_cannot_match_different_subsets(self):
        pooled = pool_frame(self.responses())
        checked = support_audit(pooled, self.result_frame(1)).iloc[0]
        self.assertEqual(checked.n_commonsense, 0)
        self.assertEqual(checked.n_justice, 1)
        with self.assertRaises(AssertionError):
            support_audit(pooled, self.result_frame(2))

    def test_na_and_committed_filters_keep_distinct_denominators(self):
        pooled = pool_frame(self.responses())
        self.assertEqual(support_audit(pooled, self.result_frame(0, "both_committed")).iloc[0].n_pairs, 0)
        pooled.loc[pooled.persona.eq("b"), "score"] = np.nan
        self.assertEqual(support_audit(pooled, self.result_frame(0)).iloc[0].n_pairs, 0)

    def test_overall_explorer_has_no_subset_control(self):
        raw = previous_tests.PresentationTests().fixture()
        raw["subset"] = "ethics_all"
        data = normalize(raw, ROOT / "result/example.csv", "direct")
        with patch("pathlib.Path.write_text") as write:
            explorer(data, ROOT / "result/not-written", "Test", overall=True)
            page = write.call_args.args[0]
        self.assertNotIn("filters.subset", page)
        self.assertNotIn("heterogeneity.csv", page)
        self.assertIn("ethics_all", page)


if __name__ == "__main__":
    unittest.main()
