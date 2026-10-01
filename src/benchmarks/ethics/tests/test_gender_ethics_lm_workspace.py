"""Protect frozen-measurement and paired-availability contracts."""
import unittest

import numpy as np
import pandas as pd

from src.benchmarks.ethics.run_gender_ethics_lm_workspace import (
    availability_checks, cached_pair_reader, enable_marker_imports, validate_measurements,
)

enable_marker_imports()


class WorkspaceTests(unittest.TestCase):
    def test_cached_pairs_equal_original_across_groups_and_missing_values(self):
        from src.core.linguistic_markers.statistics import _paired_values
        fast = cached_pair_reader(_paired_values)
        rng = np.random.default_rng(18222008)
        for iteration in range(8):
            data = pd.DataFrame([
                dict(persona=persona, question_id=f"q{i}", value=rng.normal(),
                     other=rng.integers(0, 4), is_correct=(i % 2 == 0))
                for persona in ("neutral", "man", "woman") for i in range(20)
                if rng.random() > 0.15]).sample(frac=1, random_state=iteration)
            data.loc[data.index[::7], "value"] = np.nan
            for left in ("neutral", "man", "woman", "absent"):
                for right in ("neutral", "man", "woman", "absent"):
                    for column in ("value", "other"):
                        expected = _paired_values(data, left, right, column)
                        actual = fast(data, left, right, column)
                        np.testing.assert_array_equal(actual[0], expected[0])
                        np.testing.assert_array_equal(actual[1], expected[1])
            expected = _paired_values(data, "man", "woman", "value", require_both_correct=True)
            actual = fast(data, "man", "woman", "value", require_both_correct=True)
            np.testing.assert_array_equal(actual[0], expected[0])
            np.testing.assert_array_equal(actual[1], expected[1])

    def fixture(self):
        return pd.DataFrame({
            "model_id": ["m", "m"], "subset": ["justice", "justice"],
            "question_id": ["q", "q"], "persona": ["man", "woman"],
            "text_available": [True, False], "word_count": [20.0, np.nan],
            "test__count": [2.0, np.nan], "test__density": [10.0, np.nan],
            "test__present": [1.0, np.nan]})

    def test_absent_text_is_not_zero(self):
        data = self.fixture()
        self.assertTrue(validate_measurements(data, ["test"])["valid"])
        data.loc[1, "test__count"] = 0
        with self.assertRaisesRegex(ValueError, "Unavailable"):
            validate_measurements(data, ["test"])

    def test_counts_and_density_must_agree(self):
        data = self.fixture()
        data.loc[0, "test__density"] = 20
        with self.assertRaisesRegex(ValueError, "Density"):
            validate_measurements(data, ["test"])

    def test_gt_is_not_required_for_measurement_validation(self):
        data = self.fixture()
        expected = validate_measurements(data, ["test"])
        data["ground_truth"] = [1, 0]
        self.assertEqual(expected, validate_measurements(data, ["test"]))
        data["ground_truth"] = [0, 1]
        self.assertEqual(expected, validate_measurements(data, ["test"]))

    def test_availability_pairs_questions_and_rejects_incomplete_design(self):
        rows = [dict(model_id="m", subset="justice", question_id=f"q{i}",
                     persona=f"p{j}", text_available=(j != 0),
                     answer_status="committed" if j else "missing")
                for i in range(12) for j in range(9)]
        data = pd.DataFrame(rows)
        result = availability_checks(data)
        selected = result.set_index("outcome_id").loc["text_unavailable"]
        self.assertEqual(selected.n_complete_blocks, 12)
        self.assertEqual(selected.rate_range_pp, 100)
        self.assertLess(selected.p_holm, 0.05)
        with self.assertRaisesRegex(ValueError, "nine complete"):
            availability_checks(data.iloc[1:])


if __name__ == "__main__":
    unittest.main()
