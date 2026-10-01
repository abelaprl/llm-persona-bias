"""Lossless GSM8K rendering and installed-page checks (run with system Python)."""
import json
import unittest
from unittest.mock import patch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.benchmarks.ethics import build_gender_lm_consistent as viz


class ConsistentPresentationTest(unittest.TestCase):
    def test_direct_effect_positions_and_significance(self):
        table = pd.read_csv(viz.SOURCE / "direct_gender_paired_numeric.csv")
        snapshot = table.copy(deep=True)
        with patch.object(viz.v3, "_save", side_effect=lambda fig, *args: fig):
            fig = viz.v3._plot_direct_effect_consistency(table, viz.ROOT)
        before = [c.get_offsets().copy() for ax in fig.axes for c in ax.collections]
        viz.labels_and_contrast(fig, "15")
        after = [c.get_offsets() for ax in fig.axes for c in ax.collections]
        for a,b in zip(before,after):
            np.testing.assert_array_equal(a,b)
        triangles = sum(len(c.get_paths()[0].vertices)==4 for ax in fig.axes for c in ax.collections)
        expected = table.loc[table.analysis_role.eq("primary") & table.outcome_id.isin(viz.DIRECT_OUTCOMES), "p_holm"].lt(.05).sum()
        self.assertEqual(triangles, expected)
        pd.testing.assert_frame_equal(table, snapshot)
        plt.close(fig)

    def test_saved_sources_unchanged(self):
        manifest = json.loads((viz.OUTPUT/"manifest.json").read_text(encoding="utf-8"))
        self.assertFalse(manifest["new_inferential_tests"])
        for path, expected in manifest["source_table_hashes"].items():
            self.assertEqual(viz.digest(path), expected)

    def test_all_pages_and_print_layout(self):
        index = pd.read_csv(viz.OUTPUT/"figure_index.csv")
        self.assertEqual(len(index), 20)
        self.assertEqual(index.parent_figure.nunique(), 17)
        self.assertTrue(index.loc[index.native_14cm, "width_cm"].eq(14).all())
        self.assertTrue(index.loc[index.native_14cm, "height_cm"].le(22).all())
        for row in index.itertuples():
            self.assertEqual(viz.digest(viz.OUTPUT/"images"/row.figure), row.png_sha256)
        audits = json.loads((viz.OUTPUT/"layout_checks.json").read_text())
        for row in audits:
            if row["outside_text"] != "legacy_layout":
                self.assertEqual(row["outside_text"], [])
                self.assertGreaterEqual(row["min_font_pt"], 8)


if __name__ == "__main__":
    unittest.main()
