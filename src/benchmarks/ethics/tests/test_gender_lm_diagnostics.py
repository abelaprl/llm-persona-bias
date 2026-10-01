"""Protect response-level denominators, paired contrasts, and plotted estimates."""
import importlib
import importlib.util
import unittest
from types import SimpleNamespace
import tempfile
from pathlib import Path
import json
import hashlib

import numpy as np
import pandas as pd


MODULE = "src.benchmarks.ethics.gender_lm_diagnostic_data"


class DiagnosticDataTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec(MODULE), "Diagnostic data implementation is missing")
        self.data = importlib.import_module(MODULE)

    def test_reference_counts_responses_not_spans_and_keeps_overlapping_roles(self):
        responses = pd.DataFrame([
            dict(response_id="a", model_id="m", persona="neutral", text_available=True),
            dict(response_id="b", model_id="m", persona="neutral", text_available=True),
            dict(response_id="c", model_id="m", persona="neutral", text_available=False),
            dict(response_id="d", model_id="m", persona="woman_explicit_1", text_available=False),
        ])
        spans = pd.DataFrame([
            dict(response_id="a", marker_id="gender_x", reference_type="unknown", flags="[]"),
            dict(response_id="a", marker_id="gender_x", reference_type="unknown", flags="[]"),
            dict(response_id="a", marker_id="gender_x", reference_type="explicit_user",
                 flags='["same_gender_term_in_scenario_not_coreference"]'),
            dict(response_id="b", marker_id="other", reference_type="unknown", flags="[]"),
        ])
        result = self.data.reference_summary(responses, spans, {"gender_x"})
        neutral = result.loc[result.persona.eq("neutral")].set_index("indicator")
        self.assertEqual(neutral.loc["any_gender", "n_responses"], 1)
        self.assertEqual(neutral.loc["unknown", "n_responses"], 1)
        self.assertEqual(neutral.loc["explicit_user", "percent"], 50)
        self.assertEqual(neutral.loc["scenario_overlap", "percent"], 50)
        self.assertEqual(neutral.loc["quoted_or_hypothetical", "percent"], 0)
        self.assertTrue(neutral.n_available.eq(2).all())
        self.assertTrue(neutral.total.eq(3).all())
        self.assertTrue(result.loc[result.persona.eq("woman_explicit_1"), "percent"].isna().all())

    def test_reference_rejects_spans_on_unavailable_responses(self):
        responses = pd.DataFrame([dict(response_id="a", model_id="m", persona="neutral", text_available=False)])
        spans = pd.DataFrame([dict(response_id="a", marker_id="gender_x", reference_type="unknown", flags="[]")])
        with self.assertRaisesRegex(ValueError, "unavailable"):
            self.data.reference_summary(responses, spans, {"gender_x"})

    def test_direct_selection_preserves_sign_ci_and_registered_measurement(self):
        base = dict(model_id="m", scope="overall", marker_id="x", left="woman_implicit_2",
                    right="man_implicit_2", effect=-2.0, ci_low=-3.0, ci_high=-1.0)
        rows = pd.DataFrame([{**base, "measurement": "count"}, {**base, "measurement": "density"},
                             {**base, "measurement": "density", "right": "neutral"}])
        registry = pd.DataFrame([dict(marker_id="x", measurement="density")])
        result = self.data.direct_rows(rows, registry)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0].condition, "I2")
        self.assertEqual(result.iloc[0].effect, -2)
        self.assertEqual(result.iloc[0].ci_low, -3)
        self.assertEqual(result.iloc[0].ci_high, -1)
        self.assertEqual(len(self.data.direct_rows(rows, registry, primary_only=False)), 2)

    def test_binary_uses_presence_and_cross_condition_gender_is_rejected(self):
        base = dict(model_id="m", scope="overall", marker_id="x", left="woman_explicit_1", right="man_explicit_1")
        registry = pd.DataFrame([dict(marker_id="x", measurement="binary")])
        rows = pd.DataFrame([{**base, "measurement": "count"}, {**base, "measurement": "presence"}])
        self.assertEqual(self.data.direct_rows(rows, registry).measurement.tolist(), ["presence"])
        rows.loc[1, "right"] = "man_implicit_1"
        with self.assertRaisesRegex(ValueError, "condition"):
            self.data.direct_rows(rows, registry)

    def test_support_counts_markers_without_multiplying_pair_denominators(self):
        base = dict(model_id="m", scope="overall", condition="E", n_pairs=20,
                    n_left_total=30, n_right_total=30)
        rows = pd.DataFrame([{**base, "marker_id": "x", "n_changed": 0},
                             {**base, "marker_id": "y", "n_changed": 10}])
        result = self.data.support_summary(rows).iloc[0]
        self.assertEqual(result.n_pairs, 20)
        self.assertEqual(result.total_pairs, 30)
        self.assertEqual(result.marker_count, 2)
        self.assertEqual(result.low_support_markers, 1)
        self.assertEqual(result.low_support_percent, 50)
        rows.loc[1, "n_pairs"] = 19
        with self.assertRaisesRegex(ValueError, "denominator"):
            self.data.support_summary(rows)


class DiagnosticPlotTests(unittest.TestCase):
    def setUp(self):
        name = "src.benchmarks.ethics.gender_lm_diagnostic_plots"
        self.assertIsNotNone(importlib.util.find_spec(name), "Diagnostic plotting implementation is missing")
        self.plots = importlib.import_module(name)

    def tearDown(self):
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_point_preserves_negative_ci_and_draws_zero_but_not_na(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        negative = SimpleNamespace(effect=-2., ci_low=-3., ci_high=-1., adjusted_p=.9, n_changed=2)
        result = self.plots.draw_estimate(ax, negative, 0, "#123456")
        self.assertEqual(result, "numeric")
        self.assertEqual(list(ax.lines[0].get_xdata()), [-3., -1.])
        self.assertEqual(list(ax.lines[1].get_xdata()), [-2.])
        self.assertEqual(ax.lines[1].get_markerfacecolor(), "none")
        self.assertEqual(self.plots.draw_estimate(ax, SimpleNamespace(effect=np.nan), 1, "#123456"), "na")
        self.assertEqual(len(ax.lines), 2)
        zero = SimpleNamespace(effect=0., ci_low=0., ci_high=0., adjusted_p=1., n_changed=0)
        self.assertEqual(self.plots.draw_estimate(ax, zero, 2, "#123456"), "numeric")
        self.assertEqual(list(ax.lines[-1].get_xdata()), [0.])

    def test_subset_figure_includes_all_scopes_conditions_and_na(self):
        rows = [dict(model_id="m", marker_id="stance_hedge", condition=c, scope=s,
                     measurement="density", effect=-2., ci_low=-3., ci_high=-1., adjusted_p=.9,
                     n_pairs=20, n_changed=2)
                for c in ["E", "I1", "I2", "I3"]
                for s in ["overall", "commonsense", "deontology", "justice", "virtue"]]
        rows[0]["effect"] = np.nan
        fig, audit = self.plots.subset_figure(pd.DataFrame(rows), {"m": "Model"})
        self.assertEqual(audit, {"numeric_points": 19, "na_points": 1, "source_rows": 20})
        fig.canvas.draw()


    def test_normalization_figure_keeps_different_units_on_separate_axes(self):
        rows = [dict(model_id="m", marker_id="stance_hedge", condition=c, scope="overall",
                     measurement=m, effect=e, ci_low=e-1., ci_high=e+1., adjusted_p=.9,
                     n_pairs=20, n_changed=2)
                for c in ["E", "I1", "I2", "I3"] for m, e in [("count", 2.), ("density", -2.)]]
        fig, audit = self.plots.normalization_figure(pd.DataFrame(rows), {"m": "Model"})
        self.assertEqual(audit, {"numeric_points": 8, "na_points": 0, "source_rows": 8})
        self.assertEqual(len(fig.axes), 2)
        fig.canvas.draw()


class DiagnosticExportTests(unittest.TestCase):
    def test_export_rejects_existing_output_and_preserves_numeric_csv(self):
        name = "src.benchmarks.ethics.build_gender_lm_diagnostics"
        self.assertIsNotNone(importlib.util.find_spec(name), "Diagnostic exporter is missing")
        module = importlib.import_module(name)
        import matplotlib.pyplot as plt
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "new"
            writer = module.DiagnosticWriter(root)
            with self.assertRaises(FileExistsError):
                module.DiagnosticWriter(root)
            frame = pd.DataFrame([dict(marker_id="stance_hedge", effect=-.0123456789012345, ci_low=-.1, ci_high=.2)])
            fig, ax = plt.subplots()
            ax.plot([0, 1], [0, 1])
            writer.save(fig, frame, "test", "subset_sensitivity", "Test", ["stance_hedge"], {"source_rows": 1})
            row = writer.rows[0]
            exported = pd.read_csv(root / row["source_csv"], float_precision="round_trip")
            pd.testing.assert_frame_equal(frame, exported, check_exact=True)
            self.assertTrue((root / row["png_path"]).is_file())
            self.assertEqual(module.digest(root / row["source_csv"]), row["source_sha256"])

class DiagnosticIntegrationTests(unittest.TestCase):
    def setUp(self):
        name = "src.benchmarks.ethics.integrate_gender_lm_diagnostics"
        self.assertIsNotNone(importlib.util.find_spec(name), "Diagnostic integration is missing")
        self.integration = importlib.import_module(name)

    def test_install_copies_all_types_and_rejects_overwriting_modified_image(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle = root / "bundle"
            target = root / "notebook"
            bundle.mkdir()
            target.mkdir()
            row = dict(figure_id="diagnostic_test", png_path="figures/diagnostic_test.png",
                       svg_path="figures/diagnostic_test.svg", source_csv="diagnostic_tables/test.csv")
            for field, hash_field in [("png_path", "png_sha256"), ("svg_path", "svg_sha256"), ("source_csv", "source_sha256")]:
                path = bundle / row[field]
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(field.encode())
                row[hash_field] = hashlib.sha256(path.read_bytes()).hexdigest()
            pd.DataFrame([row]).to_csv(bundle / "diagnostic_figure_index.csv", index=False)
            for name in ["diagnostic_manifest.json", "diagnostic_visual_checks.json", "diagnostic_gallery.html"]:
                (bundle / name).write_text("{}")
            self.integration.install_assets(bundle, target)
            self.assertEqual((target / row["source_csv"]).read_bytes(), b"source_csv")
            (target / row["png_path"]).write_bytes(b"user edit")
            with self.assertRaises(FileExistsError):
                self.integration.install_assets(bundle, target)
            self.assertEqual((target / row["png_path"]).read_bytes(), b"user edit")

    def test_bundle_resolution_rejects_paths_outside_ethics_reports(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            note = root / "note"
            note.mkdir()
            (note / "diagnostic_source.json").write_text(json.dumps({"diagnostic_bundle": "outside"}))
            with self.assertRaises(ValueError):
                self.integration.selected_bundle(root, note)


if __name__ == "__main__":
    unittest.main()
