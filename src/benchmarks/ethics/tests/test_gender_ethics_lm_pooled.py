"""Test pooled keys, response weighting and preservation of missingness."""
import unittest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from src.benchmarks.ethics.build_gender_ethics_lm_pooled import (
    ROOT, pool_responses, describe_markers, pair_composition, append_template,
)


class PooledTests(unittest.TestCase):
    def test_append_preserves_existing_cells_and_outputs(self):
        import nbformat
        with tempfile.TemporaryDirectory(prefix="pooled_test_", dir=Path(__file__).parent) as folder:
            directory = Path(folder)
            before = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(
                "print('old')", execution_count=4,
                outputs=[nbformat.v4.new_output("stream", name="stdout", text="old\n")])])
            nbformat.write(before, directory / "notebook_before.ipynb")
            target = directory / "analysis.ipynb"
            nbformat.write(before, target)
            append_template(directory, target)
            after = nbformat.read(target, as_version=4)
            self.assertEqual(after.cells[:1], before.cells)
            self.assertEqual(after.cells[1].id, "ethics-gabungan")
            self.assertEqual(len(after.cells), 16)
            self.assertIn(directory.relative_to(ROOT).as_posix(), after.cells[2].source)
            with self.assertRaisesRegex(ValueError, "changed since backup"):
                append_template(directory, target)

    def fixture(self):
        return pd.DataFrame([
            dict(model_id="m", persona=persona, subset=subset, question_id=q,
                 answer_status="abstain", text_available=True, word_count=100,
                 marker__count=count, marker__density=float(count), marker__present=int(count > 0))
            for subset, q, count in [("commonsense", "q", 0), ("commonsense", "q2", 0),
                                     ("virtue", "q", 9)] for persona in ("man", "woman")])

    def test_repeated_ids_in_different_subsets_remain_distinct(self):
        frame = pool_responses(self.fixture())
        self.assertEqual(frame.question_id.nunique(), 3)
        self.assertFalse(frame.duplicated(["model_id", "persona", "question_id"]).any())
        self.assertEqual(set(frame.subset), {"ethics_all"})

    def test_no_cross_subset_pairing(self):
        data = self.fixture()
        data = data.loc[~((data.subset == "commonsense") & (data.persona == "woman") & (data.question_id == "q"))]
        pooled = pool_responses(data)
        contrasts = pd.DataFrame([dict(model_id="m", left_persona="man", right_persona="woman")])
        counts = pair_composition(pooled, contrasts)
        self.assertEqual(counts.n_pairs.sum(), 2)
        self.assertEqual(counts.set_index("source_subset").loc["virtue", "n_pairs"], 1)

    def test_equal_response_weights_not_equal_subset_means(self):
        summary = describe_markers(pool_responses(self.fixture()), ["marker"])
        self.assertTrue(summary.mean_density.eq(3).all())
        self.assertTrue(summary.raw_count.eq(9).all())
        self.assertTrue(summary.n_text.eq(3).all())
        self.assertTrue(summary.median_density.eq(0).all())

    def test_surface_measurement_has_its_own_pair_support(self):
        frame = self.fixture()
        frame["ttr_300"] = 0.5
        frame.loc[0, "ttr_300"] = np.nan
        pooled = pool_responses(frame)
        contrasts = pd.DataFrame([dict(model_id="m", left_persona="man", right_persona="woman")])
        available = pair_composition(pooled, contrasts)
        ttr = pair_composition(pooled, contrasts, outcome_id="ttr_300")
        self.assertEqual(available.n_pairs.sum(), 3)
        self.assertEqual(ttr.n_pairs.sum(), 2)

    def test_unavailable_is_excluded_but_abstain_is_included(self):
        data = self.fixture()
        data.loc[data.subset.eq("virtue"), "text_available"] = False
        for column in ("marker__count", "marker__density", "marker__present"):
            data.loc[data.subset.eq("virtue"), column] = np.nan
        summary = describe_markers(pool_responses(data), ["marker"])
        self.assertTrue(summary.n_text.eq(2).all())
        self.assertTrue(summary.mean_density.eq(0).all())

    def test_duplicate_response_is_rejected(self):
        data = self.fixture()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            pool_responses(pd.concat([data, data.iloc[:1]]))


if __name__ == "__main__":
    unittest.main()
