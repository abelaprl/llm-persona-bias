"""Contract tests for the print-size, display-only Gender ETHICS prototype."""
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text

from src.benchmarks.ethics.gender_lm_reference_data import (
    SOURCE, MODELS, COLORS, digest, profiles, select_contrasts,
)


def effect_fixture(**changes):
    row = dict(effect=12.5, ci_low=10.0, ci_high=14.0,
               estimand="prevalence_difference_pp", ci_estimand="prevalence_difference_pp",
               measurement="presence", measurement_unit="percentage_points",
               adjusted_p=.08, matched_rank_biserial=np.nan)
    row.update(changes)
    return pd.DataFrame([row], index=[31])


class DisplayAdapterTest(unittest.TestCase):
    def setUp(self):
        name = "src.benchmarks.ethics.gender_lm_presentation_v2_data"
        self.assertIsNotNone(importlib.util.find_spec(name), "Display adapter is not implemented")
        self.module = importlib.import_module(name)

    def test_groups_cover_registry_without_duplicate_or_mixed_units(self):
        registry = json.loads((SOURCE / "registry.json").read_text(encoding="utf-8"))
        groups = self.module.MARKER_GROUPS
        ids = [marker for markers in groups.values() for marker in markers]
        self.assertEqual(len(ids), 41)
        self.assertEqual(len(set(ids)), 41)
        self.assertEqual(set(ids), {row["marker_id"] for row in registry})
        metadata = {row["marker_id"]: row for row in registry}
        for group_id, markers in groups.items():
            self.assertLessEqual(len(markers), 7)
            for marker in markers:
                wanted = "presence" if metadata[marker]["measurement"] == "binary" else "density"
                self.assertEqual(self.module.GROUP_MEASUREMENTS[group_id], wanted)
        self.assertEqual(len(self.module.PROTOTYPE_MARKERS), 6)

    def test_direct_orientation_keeps_source_and_does_not_multiply_pp(self):
        rows = effect_fixture()
        before = rows.copy(deep=True)
        out = self.module.effect_view(rows, "direct")
        self.assertEqual(out.display_effect.iloc[0], -12.5)
        self.assertEqual(out.display_ci_low.iloc[0], -14.0)
        self.assertEqual(out.display_ci_high.iloc[0], -10.0)
        self.assertEqual(out.display_sign.iloc[0], -1)
        self.assertEqual(out.display_direction.iloc[0], "Man - Woman")
        self.assertEqual(out.display_unit.iloc[0], "percentage_points")
        pd.testing.assert_frame_equal(rows, before)
        pd.testing.assert_frame_equal(out[before.columns], before)

    def test_density_zero_with_rank_na_remains_zero(self):
        rows = effect_fixture(effect=0., ci_low=0., ci_high=0., measurement="density",
                              estimand="mean_paired_difference", ci_estimand="mean_paired_difference",
                              measurement_unit="occurrences_per_100_words")
        out = self.module.effect_view(rows, "direct")
        self.assertEqual(out.display_effect.iloc[0], 0.)
        self.assertEqual(out.display_ci_low.iloc[0], 0.)
        self.assertTrue(pd.isna(out.matched_rank_biserial.iloc[0]))

    def test_na_is_not_converted_to_zero(self):
        out = self.module.effect_view(effect_fixture(effect=np.nan, ci_low=np.nan, ci_high=np.nan), "direct")
        self.assertTrue(out[["display_effect", "display_ci_low", "display_ci_high"]].isna().all().all())

    def test_neutral_and_implicit_keep_sign(self):
        for comparison, direction in [("neutral", "persona - N"), ("implicit", "I - E")]:
            with self.subTest(comparison=comparison):
                out = self.module.effect_view(effect_fixture(), comparison)
                self.assertEqual(out.display_effect.iloc[0], 12.5)
                self.assertEqual(out.display_ci_low.iloc[0], 10.)
                self.assertEqual(out.display_ci_high.iloc[0], 14.)
                self.assertEqual(out.display_sign.iloc[0], 1)
                self.assertEqual(out.display_direction.iloc[0], direction)

    def test_invalid_contracts_are_rejected(self):
        for changes in [dict(estimand="matched_rank_biserial"), dict(ci_estimand="other"),
                        dict(measurement_unit="percent"), dict(measurement="count"),
                        dict(ci_low=15.), dict(measurement=None)]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.module.effect_view(effect_fixture(**changes), "direct")
        for rows in [effect_fixture().iloc[:0], pd.concat([effect_fixture(), effect_fixture(measurement="density")])]:
            with self.assertRaises(ValueError):
                self.module.effect_view(rows, "direct")
        with self.assertRaises(ValueError):
            self.module.effect_view(effect_fixture(), "unknown")


def plot_fixture():
    from src.benchmarks.ethics.gender_lm_presentation_v2_data import PROTOTYPE_MARKERS, effect_view
    rows = []
    for scenario in ["E", "I1", "I2", "I3"]:
        for marker in PROTOTYPE_MARKERS:
            for model in MODELS:
                rows.append(dict(model_id=model, marker_id=marker, panel_scenario=scenario,
                                 effect=.2, ci_low=-.1, ci_high=.5, measurement="density",
                                 estimand="mean_paired_difference", ci_estimand="mean_paired_difference",
                                 measurement_unit="occurrences_per_100_words"))
    return effect_view(pd.DataFrame(rows), "direct")


class PrototypePlotTest(unittest.TestCase):
    def setUp(self):
        name = "src.benchmarks.ethics.gender_lm_presentation_v2_plots"
        self.assertIsNotNone(importlib.util.find_spec(name), "Prototype renderer is not implemented")
        self.plot = importlib.import_module(name).plot_direct_prototype
        self.data = plot_fixture()
        self.addCleanup(plt.close, "all")

    def test_models_effects_and_asymmetric_intervals_on_four_panels(self):
        fig = self.plot(self.data)
        self.assertEqual([ax.get_title() for ax in fig.axes], ["E", "I1", "I2", "I3"])
        for ax in fig.axes:
            dots = [artist for artist in ax.collections if (artist.get_gid() or "").startswith("model:")]
            intervals = [artist for artist in ax.collections if (artist.get_gid() or "").startswith("ci:")]
            self.assertEqual(len(dots), 6)
            self.assertEqual(len(intervals), 6)
            for artist, model in zip(dots, MODELS):
                self.assertEqual(artist.get_gid(), "model:" + model)
                np.testing.assert_array_equal(artist.get_offsets()[:, 0], [-.2] * 6)
                np.testing.assert_allclose(artist.get_facecolors()[0], matplotlib.colors.to_rgba(COLORS[model]))
            for interval in intervals:
                for segment in interval.get_segments():
                    np.testing.assert_array_equal(segment[:, 0], [-.5, .1])
            self.assertLessEqual(ax.get_xlim()[0], -.5)
            self.assertGreaterEqual(ax.get_xlim()[1], .5)

    def test_zero_models_have_separate_y_positions_and_all_markers_remain(self):
        self.data[["display_effect", "display_ci_low", "display_ci_high"]] = 0.
        fig = self.plot(self.data)
        for ax in fig.axes:
            dots = [artist for artist in ax.collections if (artist.get_gid() or "").startswith("model:")]
            self.assertEqual(sum(len(artist.get_offsets()) for artist in dots), 36)
            self.assertEqual(len({float(artist.get_offsets()[0, 1]) for artist in dots}), 6)
            centers = [ax.transData.transform(artist.get_offsets()[0])[1] for artist in dots]
            diameter = max(np.sqrt(artist.get_sizes()[0]) * fig.dpi / 72 for artist in dots)
            self.assertGreaterEqual(min(np.abs(np.diff(centers))), diameter)
            for artist in dots:
                np.testing.assert_array_equal(artist.get_offsets()[:, 0], [0.] * 6)
            np.testing.assert_allclose(ax.get_xlim(), [-.1, .1])

    def test_na_is_annotated_and_missing_ci_does_not_remove_dot(self):
        self.data.loc[0, ["display_effect", "display_ci_low", "display_ci_high"]] = np.nan
        self.data.loc[1, ["display_ci_low", "display_ci_high"]] = np.nan
        fig = self.plot(self.data)
        ax = fig.axes[0]
        dots = [artist for artist in ax.collections if (artist.get_gid() or "").startswith("model:")]
        self.assertEqual(sum(len(artist.get_offsets()) for artist in dots), 35)
        annotations = [t for t in ax.texts if t.get_text() == "NA"]
        self.assertEqual(len(annotations), 1)
        self.assertAlmostEqual(annotations[0].get_position()[1], -.30)
        caption = "\n".join(t.get_text() for t in fig.texts)
        self.assertIn("Estimasi NA: 1", caption)
        self.assertIn("CI NA: 1", caption)

    def test_physical_size_fonts_and_text_bounds(self):
        fig = self.plot(self.data)
        np.testing.assert_allclose(fig.get_size_inches() * 2.54, [16, 22])
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for artist in fig.findobj(Text):
            if not artist.get_visible() or not artist.get_text():
                continue
            with self.subTest(text=artist.get_text()):
                self.assertGreaterEqual(artist.get_fontsize(), 9.)
                bounds = artist.get_window_extent(renderer)
                self.assertGreaterEqual(bounds.x0, -.5)
                self.assertGreaterEqual(bounds.y0, -.5)
                self.assertLessEqual(bounds.x1, fig.bbox.width + .5)
                self.assertLessEqual(bounds.y1, fig.bbox.height + .5)
        self.assertEqual(len(fig.legends[0].get_texts()), 6)

    def test_missing_duplicate_extra_and_wrong_unit_rows_are_rejected(self):
        invalid = [self.data.iloc[1:], pd.concat([self.data, self.data.iloc[:1]]),
                   self.data.assign(display_unit="percentage_points"),
                   self.data.assign(panel_scenario="X")]
        for data in invalid:
            with self.subTest(rows=len(data)), self.assertRaises(ValueError):
                self.plot(data)


class PreviewBuilderTest(unittest.TestCase):
    def setUp(self):
        name = "src.benchmarks.ethics.build_gender_lm_presentation_v2_preview"
        self.assertIsNotNone(importlib.util.find_spec(name), "Preview builder is not implemented")
        self.module = importlib.import_module(name)

    def test_existing_empty_output_refused_without_changing_source(self):
        before = digest(SOURCE / "registry.json")
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            with self.assertRaises(FileExistsError):
                self.module.build_preview(target)
            self.assertEqual(list(target.iterdir()), [])
        self.assertEqual(digest(SOURCE / "registry.json"), before)

    def test_build_matches_all_source_columns_context_and_artifact_contract(self):
        from src.benchmarks.ethics.gender_lm_presentation_v2_data import PROTOTYPE_MARKERS
        comparisons = pd.read_csv(SOURCE / "comparisons.csv", float_precision="round_trip")
        expected = select_contrasts(comparisons, PROTOTYPE_MARKERS, "direct", "density")
        expected_context = profiles(pd.read_csv(SOURCE / "marker_summary.csv", float_precision="round_trip"))
        expected_context = expected_context.loc[expected_context.marker_id.isin(PROTOTYPE_MARKERS)]
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "preview"
            report = self.module.build_preview(target)
            self.assertEqual({p.name for p in target.iterdir()}, {
                "B02_direct_overall.png", "B02_direct_overall.svg", "B02_direct_overall.csv",
                "marker_context.csv", "source_hashes.json", "verification.json", "README.md"})
            actual = pd.read_csv(target / "B02_direct_overall.csv", float_precision="round_trip")
            context = pd.read_csv(target / "marker_context.csv", float_precision="round_trip")
            self.assertEqual(len(actual), 144)
            self.assertEqual(len(context), 324)
            pd.testing.assert_frame_equal(actual[expected.columns], expected.reset_index(drop=True),
                                          check_dtype=False, check_exact=True)
            pd.testing.assert_frame_equal(context[expected_context.columns], expected_context.reset_index(drop=True),
                                          check_dtype=False, check_exact=True)
            np.testing.assert_array_equal(actual.display_effect, -expected.effect)
            np.testing.assert_array_equal(actual.display_ci_low, -expected.ci_high)
            np.testing.assert_array_equal(actual.display_ci_high, -expected.ci_low)
            self.assertTrue(report["source_unchanged"])
            self.assertTrue(report["active_unchanged"])
            self.assertTrue(report["source_columns_equal"])
            self.assertEqual(report["panels"], 4)
            hashes = json.loads((target / "source_hashes.json").read_text(encoding="utf-8"))
            self.assertEqual(hashes["before"], hashes["after"])
            self.assertGreaterEqual(len(hashes["before"]), 41)
            from PIL import Image
            with Image.open(target / "B02_direct_overall.png") as img:
                self.assertEqual(img.size, (1889, 2598))
                self.assertAlmostEqual(img.info["dpi"][0], 300, delta=.05)
            import xml.etree.ElementTree as ET
            svg = ET.parse(target / "B02_direct_overall.svg").getroot()
            self.assertAlmostEqual(float(svg.attrib["width"].removesuffix("pt")) / 72 * 2.54, 16, places=5)
            self.assertIn("pairwise-complete", (target / "README.md").read_text(encoding="utf-8"))

    def test_duplicate_source_fails_before_output_creation(self):
        from src.benchmarks.ethics.gender_lm_presentation_v2_data import PROTOTYPE_MARKERS
        original = pd.read_csv(SOURCE / "comparisons.csv", float_precision="round_trip")
        selected = select_contrasts(original, PROTOTYPE_MARKERS, "direct", "density")
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.mkdir()
            (source / "registry.json").write_bytes((SOURCE / "registry.json").read_bytes())
            (source / "response_markers.csv.gz").write_bytes(b"hash-only fixture; never parsed")
            pd.concat([selected[original.columns], selected[original.columns].iloc[:1]]).to_csv(source / "comparisons.csv", index=False)
            pd.read_csv(SOURCE / "marker_summary.csv").to_csv(source / "marker_summary.csv", index=False)
            before = {p.name: digest(p) for p in source.iterdir()}
            target = Path(directory) / "invalid"
            with patch.object(self.module, "SOURCE", source), self.assertRaises(ValueError):
                self.module.build_preview(target)
            self.assertFalse(target.exists())
            self.assertEqual(before, {p.name: digest(p) for p in source.iterdir()})


if __name__ == "__main__":
    unittest.main()
