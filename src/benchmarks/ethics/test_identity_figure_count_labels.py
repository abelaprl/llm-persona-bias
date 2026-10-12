"""Presentation-only tests; no response extraction or inference."""
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.benchmarks.ethics.identity_figure_count_labels import clean_count_text, strip_count_labels


class CountLabelTests(unittest.TestCase):
    def test_observed_coverage(self):
        self.assertEqual(clean_count_text("98.5%\n394/400"), "98.5%")
        self.assertEqual(clean_count_text("100.0%\n1,600/1,600"), "100.0%")

    def test_observed_pair_counts(self):
        for value in ["k=75 / n=388", "k=0 / n=0", "k=75.0 / n=388.0"]:
            self.assertEqual(clean_count_text(value), "")

    def test_keep_measurements_and_keys(self):
        for value in ["+0.35*†", "−0.21", "ND", "TP", "F", "k < 20", "95% CI", "100.0%",
                      "Densitas per 100 kata", "Jumlah kata/kalimat"]:
            self.assertEqual(clean_count_text(value), value)

    def test_remove_legend_prefix_only(self):
        self.assertEqual(clean_count_text("k = pasangan berubah; n = pasangan penjelasan · ND: tanpa pasangan · TP: semua selisih nol"),
                         "ND: tanpa pasangan · TP: semua selisih nol")
        self.assertEqual(clean_count_text("Pembilang/penyebut = penjelasan tersedia/JSON ditemukan; kegagalan API dan final kosong tetap dicatat."),
                         "kegagalan API dan final kosong tetap dicatat.")

    def test_plot_evidence_unchanged(self):
        fig, ax = plt.subplots()
        values = np.array([[.1, -.2], [.3, 0]])
        graphic = ax.imshow(values, vmin=-1, vmax=1)
        ax.set_xlabel("Skenario")
        ax.text(0, 0, "k=75 / n=388")
        ax.text(1, 0, "+0.35*†")
        self.assertEqual(len(strip_count_labels(fig)), 1)
        np.testing.assert_array_equal(graphic.get_array(), values)
        self.assertEqual(graphic.get_clim(), (-1, 1))
        self.assertEqual(ax.get_xlabel(), "Skenario")
        self.assertEqual(ax.texts[1].get_text(), "+0.35*†")
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
