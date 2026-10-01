"""Guard scientific and visual contracts of the reference-based presentation."""
import importlib
import importlib.util
import unittest
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PREFIX = "src.benchmarks.ethics."


class ReferencePresentationTest(unittest.TestCase):
    def module(self, name):
        self.assertIsNotNone(importlib.util.find_spec(PREFIX + name), "Presentation module not implemented")
        return importlib.import_module(PREFIX + name)

    def comparisons(self):
        return pd.DataFrame([
            dict(scope="overall", model_id="model", marker_id="m", measurement="density",
                 left="woman_explicit_1", right="man_explicit_1", effect_direction="left_minus_right",
                 effect=12.5, matched_rank_biserial=np.nan, ci_low=10., ci_high=14., adjusted_p=.1),
            dict(scope="overall", model_id="model", marker_id="m", measurement="presence",
                 left="woman_explicit_1", right="man_explicit_1", effect_direction="left_minus_right",
                 effect=12.5, matched_rank_biserial=.3, ci_low=10., ci_high=14., adjusted_p=.1),
            dict(scope="justice", model_id="model", marker_id="m", measurement="presence",
                 left="woman_explicit_1", right="man_explicit_1", effect_direction="left_minus_right",
                 effect=99., matched_rank_biserial=.3, ci_low=90., ci_high=100., adjusted_p=.1),
            dict(scope="overall", model_id="model", marker_id="m", measurement="presence",
                 left="woman_implicit_1", right="woman_explicit_1", effect_direction="left_minus_right",
                 effect=-4., matched_rank_biserial=-.2, ci_low=-6., ci_high=-2., adjusted_p=.01),
        ])

    def test_direct_presence_preserves_direction_units_ci_and_scope(self):
        d = self.module("gender_lm_reference_data")
        out = d.select_contrasts(self.comparisons(), ["m"], "direct", "presence")
        self.assertEqual(len(out), 1)
        self.assertEqual(out.iloc[0].plot_value, 12.5)
        self.assertEqual(out.iloc[0].plot_ci_low, 10.)
        self.assertEqual(out.iloc[0].panel_scenario, "E")
        self.assertEqual(out.iloc[0].panel_gender, "Woman − Man")

    def test_rank_na_is_not_zero_and_mean_ci_is_not_rank_ci(self):
        d = self.module("gender_lm_reference_data")
        row = d.select_contrasts(self.comparisons(), ["m"], "direct", "density").iloc[0]
        self.assertTrue(pd.isna(row.plot_value))
        self.assertTrue(pd.isna(row.plot_ci_low))
        self.assertTrue(pd.isna(row.plot_ci_high))
        self.assertEqual(row.ci_low, 10.)

    def test_implicit_keeps_identity_and_scenario(self):
        d = self.module("gender_lm_reference_data")
        row = d.select_contrasts(self.comparisons(), ["m"], "implicit", "presence").iloc[0]
        self.assertEqual(row.panel_gender, "Woman")
        self.assertEqual(row.panel_scenario, "I1")
        self.assertEqual(row.plot_value, -4.)

    def test_rejects_duplicate_contrasts(self):
        d = self.module("gender_lm_reference_data")
        data = self.comparisons()
        with self.assertRaises(ValueError):
            d.select_contrasts(pd.concat([data, data]), ["m"], "direct", "presence")

    def test_diversity_counts_types_and_preserves_unavailable(self):
        d = self.module("gender_lm_reference_data")
        data = pd.DataFrame({"response_id": ["a", "b", "c"], "text_available": [True, True, False],
                             "m__count": [5., 0., np.nan], "n__count": [2., 0., np.nan]})
        out = d.category_diversity(data, ["m", "n"])
        self.assertEqual(out.category_count.iloc[0], 2.)
        self.assertEqual(out.category_count.iloc[1], 0.)
        self.assertTrue(pd.isna(out.category_count.iloc[2]))
        self.assertNotIn("category_count", data)

    def test_profile_combines_models_and_personas(self):
        d = self.module("gender_lm_reference_data")
        p = self.module("gender_lm_reference_plots")
        rows = [dict(model_id=model, persona=gender + "_explicit_1", marker_id="m", value=10 + k)
                for gender in ["man", "woman"] for k, model in enumerate(d.MODELS)]
        fig = p.profile(pd.DataFrame(rows), ["m"], "Profil", ["E"])
        self.assertEqual(len(fig.axes), 2)
        for ax in fig.axes:
            dots = [c for c in ax.collections if str(c.get_gid()).startswith("model:")]
            self.assertEqual(len(dots), 6)
            self.assertEqual(len({tuple(c.get_facecolor()[0]) for c in dots}), 6)
            self.assertEqual(sorted(float(c.get_offsets()[0, 0]) for c in dots), [10,11,12,13,14,15])
        self.assertGreater(fig.get_figwidth(), fig.get_figheight())
        plt.close(fig)

    def test_effect_na_is_drawn_outside_numeric_scale(self):
        d = self.module("gender_lm_reference_data")
        p = self.module("gender_lm_reference_plots")
        data = d.select_contrasts(self.comparisons(), ["m"], "direct", "density")
        data["model_id"] = d.MODELS[0]
        data["n_changed"] = 0
        fig = p.effects(data, ["m"], "Efek", "direct", "density")
        marks = [c for c in fig.axes[0].collections if c.get_gid() == "unavailable"]
        self.assertEqual(len(marks), 1)
        self.assertGreater(float(marks[0].get_offsets()[0, 0]), 1)
        plt.close(fig)

    def test_builder_refuses_overwriting_existing_result(self):
        b = self.module("build_gender_lm_reference_presentation")
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            with self.assertRaises(FileExistsError):
                b.build(target)
            self.assertEqual(list(target.iterdir()), [])

    def test_equal_profile_values_leave_six_visible_model_dots(self):
        d = self.module("gender_lm_reference_data")
        p = self.module("gender_lm_reference_plots")
        markers = ["m" + str(i) for i in range(15)]
        rows = [dict(model_id=model, persona=gender + "_explicit_1", marker_id=marker, value=0.)
                for gender in ["man","woman"] for model in d.MODELS for marker in markers]
        fig = p.profile(pd.DataFrame(rows), markers, "Profil nol", ["E"])
        fig.canvas.draw()
        ax = fig.axes[0]
        dots = [c for c in ax.collections if str(c.get_gid()).startswith("model:")]
        centers = [ax.transData.transform(c.get_offsets()[0])[1] for c in dots]
        gap = min(np.abs(np.diff(centers)))
        diameter = max(np.sqrt(c.get_sizes()[0])*fig.dpi/72 for c in dots)
        self.assertGreaterEqual(gap, diameter, "Equal values hide one model behind another")
        medians = [c for c in ax.collections if not str(c.get_gid()).startswith("model:")]
        self.assertEqual([float(c.get_offsets()[0,1]) for c in medians], list(range(15)),
                         "A median between marker rows has ambiguous category membership")
        plt.close(fig)

    def test_ecdf_vertical_labels_do_not_overlap(self):
        d = self.module("gender_lm_reference_data")
        p = self.module("gender_lm_reference_plots")
        data = pd.DataFrame([dict(model_id=d.MODELS[0], persona="neutral", category_count=0.)])
        fig = p.diversity(data)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        labels = [ax.yaxis.label for ax in fig.axes if ax.get_ylabel()]
        for i, label in enumerate(labels):
            for other in labels[i+1:]:
                self.assertFalse(label.get_window_extent(renderer).overlaps(other.get_window_extent(renderer)))
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
