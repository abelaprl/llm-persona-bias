"""Tests for presentation-only ETHICS direct comparisons."""
import copy
import tempfile
import unittest
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd

from src.benchmarks.ethics.present_gender_ethics_man_woman import (
    ROOT, MODEL_LABELS, summarize, consistency, rewrite_notebook, save_notebook, sha256,
)


class PresentationTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame([
            dict(subset="justice", model_id=next(iter(MODEL_LABELS)), condition="explicit",
                 outcome_id=marker, left_identity="man", right_identity="woman",
                 left_persona="man_explicit_1", right_persona="woman_explicit_1",
                 rank_biserial=effect, p_holm=p, n_pairs=100, n_nonzero=changed)
            for marker, effect, p, changed in [
                ("word_count", -.4, .01, 60), ("avg_sentence_length", .8, .2, 80),
                ("ttr_300", .2, .04, 40), ("organization_bullet", 0, 1, 0),
                ("stance_hedge", np.nan, np.nan, 0)]
        ])

    def test_summary_direction_and_missing_tests(self):
        frame = self.frame()
        before = frame.copy(deep=True)
        details, summary, highlights = summarize(frame, "numeric")
        row = summary.iloc[0]
        self.assertEqual((row.comparisons, row.tested, row.significant), (5, 4, 2))
        self.assertEqual((row.man_higher, row.woman_higher), (1, 1))
        self.assertEqual(highlights.iloc[0].outcome_id, "word_count")
        self.assertEqual(details.iloc[-1].direction, "Tidak tersedia")
        pd.testing.assert_frame_equal(frame, before)

    def test_no_significance_is_retained(self):
        frame = self.frame().assign(p_holm=1.)
        _, summary, highlights = summarize(frame, "numeric")
        self.assertEqual(summary.iloc[0].significant, 0)
        self.assertEqual(len(highlights), 0)
        self.assertIn("effect_value", highlights)

    def test_presence_uses_prevalence_not_numeric_effect(self):
        frame = self.frame().iloc[:1].assign(prevalence_delta=.03, discordant=12)
        details, summary, _ = summarize(frame, "gender_presence")
        self.assertEqual(details.iloc[0].effect_value, .03)
        self.assertEqual(details.iloc[0].n_changed, 12)
        self.assertEqual(summary.iloc[0].man_higher, 1)

    def test_duplicates_and_wrong_pairing_rejected(self):
        frame = self.frame()
        with self.assertRaises(ValueError):
            summarize(pd.concat([frame, frame]), "numeric")
        with self.assertRaises(ValueError):
            summarize(frame.assign(right_persona="woman_implicit_1"), "numeric")
        with self.assertRaises(ValueError):
            summarize(frame.assign(left_identity="woman"), "numeric")

    def test_consistency_does_not_treat_nonsignificance_as_zero(self):
        a = self.frame().iloc[:1].assign(rank_biserial=.4, p_holm=.2)
        b = a.assign(model_id=list(MODEL_LABELS)[1], rank_biserial=-.3, p_holm=.01)
        details, _, _ = summarize(pd.concat([a, b]), "numeric")
        row = consistency(details).iloc[0]
        self.assertEqual((row.man_higher, row.woman_higher, row.zero), (1, 1, 0))
        self.assertEqual((row.significant_man_higher, row.significant_woman_higher), (0, 1))

    def test_reorder_retains_all_cells_and_dependencies(self):
        original = nbformat.v4.new_notebook(cells=[
            nbformat.v4.new_code_cell(f"# source {i}", id=f"original-{i}") if i % 2 else
            nbformat.v4.new_markdown_cell(f"heading {i}", id=f"original-{i}") for i in range(38)])
        before = copy.deepcopy(original)
        output = ROOT / "result/test_presentation"
        revised = rewrite_notebook(original, output, ROOT / "notebooks/figures/test")
        ids = [c.id for c in revised.cells]
        self.assertEqual(len(ids), 44)
        self.assertEqual(len(set(ids)), 44)
        self.assertTrue(set(c.id for c in original.cells).issubset(ids))
        self.assertLess(ids.index("original-12"), ids.index("original-8"))
        self.assertLess(ids.index("original-14"), ids.index("original-8"))
        self.assertLess(ids.index("original-24"), ids.index("original-29"))
        self.assertLess(ids.index("original-29"), ids.index("original-25"))
        self.assertLess(ids.index("original-31"), ids.index("original-25"))
        self.assertEqual(original, before)
        with self.assertRaises(ValueError):
            rewrite_notebook(revised, output, ROOT / "notebooks/figures/test")

    def test_atomic_save_rejects_changed_target(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.ipynb"
            original = nbformat.v4.new_notebook()
            nbformat.write(original, path)
            original_hash = sha256(path)
            replacement = nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell("new")])
            with self.assertRaises(ValueError):
                save_notebook(replacement, path, "incorrect hash")
            self.assertEqual(sha256(path), original_hash)
            save_notebook(replacement, path, original_hash)
            self.assertEqual(nbformat.read(path, as_version=4), replacement)
            self.assertEqual(len(list(Path(directory).iterdir())), 1)


if __name__ == "__main__":
    unittest.main()
