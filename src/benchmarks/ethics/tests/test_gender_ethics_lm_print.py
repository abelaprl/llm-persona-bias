"""Regression tests for final-size typography and lossless presentation changes."""
import unittest
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.benchmarks.ethics import build_gender_ethics_lm_print as print_viz


class PrintPresentationTest(unittest.TestCase):
    def test_round_bound_covers_confidence_intervals(self):
        for value in [.1, 1, 2.6, 26.1, 49.5, 99.8, 123]:
            self.assertGreaterEqual(print_viz.round_bound(value), value)
        self.assertEqual(print_viz.round_bound(26.1), 30)
        self.assertEqual(print_viz.round_bound(49.5), 50)

    def test_native_width_and_legend_separation(self):
        fig, axes = plt.subplots(2, 2)
        for ax in axes.ravel():
            ax.scatter([0], [0], s=20)
            ax.set_title("E")
        print_viz.decorate(fig, list(axes.ravel()), "Test", "Rank-biserial\nMan − Woman", 18,
                           median=True, significance=True)
        self.assertAlmostEqual(fig.get_figwidth() * 2.54, 14)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for text in fig.texts:
            self.assertGreaterEqual(text.get_fontsize(), 8)
            self.assertFalse(text.get_window_extent(renderer).overlaps(fig.legends[0].get_window_extent(renderer)))
        plt.close(fig)

    def test_reflow_preserves_effect_positions(self):
        table = pd.read_csv(print_viz.ACTIVE / "figures/02_bahasa_etika_man_minus_woman.csv", float_precision="round_trip")
        snapshot = table.copy(deep=True)
        ids = list(dict.fromkeys(table.marker_id))
        fig = print_viz.direct.plot_direct_effects(table, ids, "Test", "density")
        offsets = [[c.get_offsets().copy() for c in ax.collections] for ax in fig.axes]
        print_viz.configure_effect_axes(fig, table, ids, True)
        print_viz.decorate(fig, fig.axes[:], "Test", "Rank-biserial", 21,
                           marker_ids=ids, categorical=True, significance=True, median=True)
        for ax, before in zip(fig.axes, offsets):
            for collection, positions in zip(ax.collections, before):
                np.testing.assert_array_equal(collection.get_offsets(), positions)
        pd.testing.assert_frame_equal(table, snapshot)
        plt.close(fig)

    def test_print_bundle_preserves_every_source_table(self):
        bundle = print_viz.DEFAULT_OUTPUT
        index = pd.read_csv(bundle / "figure_index.csv")
        self.assertEqual(len(index), 26)
        self.assertEqual(index.parent_figure_id.nunique(), 16)
        self.assertTrue(index.width_cm.eq(14).all())
        self.assertTrue(index.min_font_pt.ge(8).all())
        self.assertLessEqual(index.height_cm.max(), 22)
        for row in index.itertuples():
            original = bundle / "inputs" / f"{row.parent_figure_id}.csv"
            exported = bundle / "figures" / f"{row.figure_id}.csv"
            self.assertEqual(print_viz.direct.sha256(original), print_viz.direct.sha256(exported))
            self.assertEqual(print_viz.direct.sha256(exported), row.csv_sha256)
        self.assertEqual(list((bundle / "figures").glob("*.svg")), [])

    def test_all_paginated_scenarios_are_retained(self):
        index = pd.read_csv(print_viz.DEFAULT_OUTPUT / "figure_index.csv")
        for parent in ("05_profil_gender_semua_skenario", "12_konteks_rujukan"):
            rows = index.loc[index.parent_figure_id.eq(parent)].sort_values("page")
            self.assertEqual(rows.page_scope.tolist(), ["N", "E", "I1", "I2", "I3"])
        for parent in ("09_gender_terhadap_netral", "10_gender_implicit_vs_explicit"):
            self.assertEqual(index.loc[index.parent_figure_id.eq(parent), "page_scope"].tolist(), ["Man", "Woman"])


if __name__ == "__main__":
    unittest.main()
