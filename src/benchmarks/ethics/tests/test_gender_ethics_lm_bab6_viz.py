"""Regression tests for the non-destructive Gender ETHICS Bab VI visuals."""
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.benchmarks.ethics import build_gender_ethics_lm_bab6_viz as visual


class GenderEthicsBab6VisualTest(unittest.TestCase):
    def rows(self, measurement="density"):
        records = []
        for scenario, suffix in zip(visual.SCENARIOS, ["explicit_1", "implicit_1", "implicit_2", "implicit_3"]):
            for model_index, model in enumerate(visual.MODELS):
                records.append({
                    "scope": "overall",
                    "model_id": model,
                    "marker_id": "m",
                    "measurement": measurement,
                    "left": "woman_" + suffix,
                    "right": "man_" + suffix,
                    "effect_direction": "left_minus_right",
                    "effect": -2.0 - model_index,
                    "ci_low": -3.0 - model_index,
                    "ci_high": -1.0 - model_index,
                    "matched_rank_biserial": -0.2 - model_index * .01,
                    "adjusted_p": .01 if model_index == 0 else .2,
                    "n_changed": 20 + model_index,
                })
        return pd.DataFrame(records)

    def test_density_reverses_rank_only_and_retains_source(self):
        source = self.rows("density")
        output = visual.orient_direct(source, "density")
        self.assertEqual(output.iloc[0].display_effect, .2)
        self.assertTrue(pd.isna(output.iloc[0].display_ci_low))
        self.assertEqual(output.iloc[0].effect, -2.0)
        self.assertEqual(output.iloc[0].matched_rank_biserial, -.2)
        self.assertEqual(output.iloc[0].display_direction, "Man minus Woman")
        pd.testing.assert_frame_equal(source, output[source.columns])

    def test_presence_reverses_effect_and_interval(self):
        output = visual.orient_direct(self.rows("presence"), "presence")
        row = output.iloc[0]
        self.assertEqual(row.display_effect, 2.0)
        self.assertEqual(row.display_ci_low, 1.0)
        self.assertEqual(row.display_ci_high, 3.0)
        self.assertLessEqual(row.display_ci_low, row.display_ci_high)

    def test_plot_matches_gsm8k_panel_and_model_contract(self):
        rows = visual.orient_direct(self.rows("density"), "density")
        fig = visual.plot_direct_effects(rows, ["m"], "Test", "density")
        self.assertEqual(len(fig.axes), 4)
        self.assertEqual([ax.get_title() for ax in fig.axes], visual.SCENARIOS)
        colors = set()
        for collection in fig.axes[0].collections:
            if len(collection.get_facecolors()):
                colors.add(tuple(collection.get_facecolors()[0]))
        self.assertGreaterEqual(len(colors), 7)  # six models plus the median diamond
        self.assertIn("Man − Woman", " ".join(text.get_text() for text in fig.texts))
        plt.close(fig)

    def test_na_rank_is_not_drawn_at_zero(self):
        rows = self.rows("density")
        rows.loc[0, "matched_rank_biserial"] = np.nan
        shown = visual.orient_direct(rows, "density")
        fig = visual.plot_direct_effects(shown, ["m"], "Test", "density")
        offsets = np.vstack([
            collection.get_offsets()
            for collection in fig.axes[0].collections
            if len(collection.get_offsets())
        ])
        self.assertTrue((offsets[:, 0] > 1).any())
        # No six-model median may be shown when one model is unavailable.
        self.assertEqual(len(fig.axes[0].collections), 6)
        self.assertEqual(len(fig.axes[1].collections), 7)
        plt.close(fig)

    def test_points_are_opaque_and_significance_uses_shape(self):
        rows = visual.orient_direct(self.rows("density"), "density")
        fig = visual.plot_direct_effects(rows, ["m"], "Test", "density")
        points = fig.axes[0].collections[:6]
        for point in points:
            self.assertEqual(point.get_alpha(), 1.0)
            self.assertTrue((point.get_facecolors()[:, 3] == 1).all())
            self.assertGreaterEqual(point.get_sizes()[0], 30)
        self.assertNotEqual(len(points[0].get_paths()[0].vertices), len(points[1].get_paths()[0].vertices))
        self.assertIn("▲ p Holm", " ".join(label.get_text() for label in fig.texts))
        self.assertNotIn("Warna penuh", " ".join(label.get_text() for label in fig.texts))
        self.assertNotIn("NA bukan nol", " ".join(label.get_text() for label in fig.texts))
        plt.close(fig)

    def test_builder_refuses_existing_output(self):
        with self.assertRaises(FileExistsError):
            visual.build(visual.ROOT)


if __name__ == "__main__":
    unittest.main()
