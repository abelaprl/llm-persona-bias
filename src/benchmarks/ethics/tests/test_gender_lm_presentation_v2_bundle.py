"""Full-bundle contracts: all dictionary markers, scenarios and display modes."""
import importlib
import base64
import json
import io
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stderr

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.text import Text

from src.benchmarks.ethics import gender_lm_presentation_v2_data as data
from src.benchmarks.ethics.gender_lm_reference_data import SOURCE, MODELS, PERSONAS


def exercise_explorer(test, script, views):
    """Execute the actual navigation JavaScript in a deterministic DOM harness."""
    node = shutil.which("node")
    if node is None:
        test.skipTest("Node is needed for the offline navigation contract")
    harness = r"""
const fs = require('fs'), vm = require('vm'), assert = require('assert');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const elements = {};
const element = id => elements[id] ||= {value: '', textContent: '', innerHTML: '',
  addEventListener() {}, removeAttribute(key) { delete this[key]; }};
element('view-data').textContent = JSON.stringify(input.views);
element('group').value = input.views[0].group_id;
element('mode').value = 'direct'; element('persona').value = 'Man'; element('scenario').value = 'E';
const context = {document: {getElementById: element}, location: {hash: ''},
  window: {addEventListener() {}}, assert};
vm.createContext(context); vm.runInContext(input.script, context);
vm.runInContext(`
for (const view of views) {
  location.hash = '#' + view.view_id; openHash();
  assert.equal(el('chart').src, view.svg_data_uri);
  assert.equal(el('caption').textContent, view.caption);
  assert.equal(el('table').innerHTML, view.table_html);
  assert.equal(el('download').download, view.view_id + '.svg');
  assert.equal(el('persona-label').hidden, !(view.mode === 'neutral' || view.mode === 'implicit'));
  assert.equal(el('scenario-label').hidden, view.mode !== 'profile');
  el('group').value = view.group_id; el('mode').value = view.mode;
  if (view.persona) el('persona').value = view.persona;
  if (view.scenario) el('scenario').value = view.scenario;
  render(); assert.equal(el('chart').alt, view.view_id);
}
`, context);
process.stdout.write(String(input.views.length));
"""
    result = subprocess.run([node, "-e", harness], input=json.dumps(dict(script=script, views=views)),
                            text=True, encoding="utf-8", capture_output=True, timeout=60)
    test.assertEqual(result.returncode, 0, result.stderr)
    test.assertEqual(result.stdout, str(len(views)))


class BundleDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.comparisons = pd.read_csv(SOURCE / "comparisons.csv", float_precision="round_trip")
        cls.summary = pd.read_csv(SOURCE / "marker_summary.csv", float_precision="round_trip")

    def setUp(self):
        self.assertTrue(hasattr(data, "view_data"), "Full-mode adapter is not implemented")

    def view(self, group_id, mode, **kwargs):
        return data.view_data(group_id, mode, comparisons=self.comparisons, summary=self.summary, **kwargs)

    def test_exact_81_specs_and_all_marker_mode_scenario_combinations(self):
        specs = data.view_specs()
        self.assertEqual(len(specs), 81)
        self.assertEqual(len({s["view_id"] for s in specs}), 81)
        wanted = {m for markers in data.MARKER_GROUPS.values() for m in markers}
        totals = dict(direct=0, neutral=0, implicit=0)
        coverage = {mode: set() for mode in ["profile", "direct", "neutral", "implicit"]}
        for spec in specs:
            rows = self.view(**{k: v for k, v in spec.items() if k != "view_id"})
            coverage[spec["mode"]].update(rows.marker_id)
            if spec["mode"] != "profile":
                totals[spec["mode"]] += len(rows)
                pd.testing.assert_frame_equal(rows[self.comparisons.columns], self.comparisons.loc[rows.index], check_exact=True)
                sign = -1 if spec["mode"] == "direct" else 1
                np.testing.assert_array_equal(rows.display_effect, sign * rows.effect)
            else:
                self.assertEqual(set(rows.panel_id), {"N", "Man", "Woman"})
                np.testing.assert_array_equal(rows.display_value, rows.value)
        self.assertEqual(totals, dict(direct=984, neutral=1968, implicit=1476))
        self.assertTrue(all(markers == wanted for markers in coverage.values()))
        for group in data.MARKER_GROUPS:
            subset = [s for s in specs if s["group_id"] == group]
            self.assertEqual(len([s for s in subset if s["mode"] == "profile"]), 4)
            for mode in ["neutral", "implicit"]:
                self.assertEqual({s["persona"] for s in subset if s["mode"] == mode}, {"Man", "Woman"})

    def test_scope_and_invalid_filter_arguments_are_not_silently_ignored(self):
        rows = self.view("B08", "direct", scope="justice")
        self.assertTrue(rows.scope.eq("justice").all())
        self.assertEqual(len(rows), 168)
        for group, mode, kwargs in [("bad", "direct", {}), ("B01", "bad", {}),
                                   ("B01", "profile", {}), ("B01", "profile", {"scenario": "N"}),
                                   ("B01", "direct", {"persona": "Man"}),
                                   ("B01", "direct", {"scenario": "E"}),
                                   ("B01", "neutral", {}), ("B01", "implicit", {"persona": "neutral"}),
                                   ("B01", "direct", {"scope": "bad"})]:
            with self.subTest(group=group, mode=mode, kwargs=kwargs), self.assertRaises(ValueError):
                self.view(group, mode, **kwargs)

    def test_profile_covers_2214_unique_rows_and_keeps_denominators(self):
        all_rows = pd.concat([self.view(g, "profile", scenario=s) for g in data.MARKER_GROUPS
                              for s in ["E", "I1", "I2", "I3"]])
        unique = all_rows.drop_duplicates(["scope", "model_id", "persona", "marker_id"])
        self.assertEqual(len(unique), 2214)
        self.assertEqual(set(unique.persona), set(PERSONAS))
        self.assertTrue({"raw_occurrences", "total", "n_available", "prevalence_percent",
                         "density_per_100_words"}.issubset(unique.columns))
        pd.testing.assert_frame_equal(unique[self.summary.columns].sort_index(),
                                      self.summary.loc[unique.index].sort_index(), check_exact=True)

    def test_actual_raw_counts_are_joined_from_count_rows_not_empty_density_fields(self):
        rows = self.view("B01", "profile", scenario="E")
        self.assertIn("raw_count_total", rows.columns)
        keys = ["scope", "model_id", "persona", "marker_id"]
        counts = self.summary.loc[self.summary.measurement.eq("count")].set_index(keys)
        actual = rows.set_index(keys)
        np.testing.assert_array_equal(actual.raw_count_total, counts.loc[actual.index, "raw_occurrences"])
        self.assertTrue(actual.raw_count_total.notna().all())

    def test_duplicate_input_fails_before_render(self):
        with self.assertRaises(ValueError):
            data.view_data("B01", "direct", comparisons=pd.concat([self.comparisons, self.comparisons]), summary=self.summary)

    def test_coverage_uses_all_81756_slots_and_source_pairs(self):
        coverage = data.coverage_table(self.summary, self.comparisons)
        explanation = coverage.loc[coverage.kind.eq("explanation")]
        pairs = coverage.loc[coverage.kind.eq("paired")]
        self.assertEqual(len(explanation), 54)
        self.assertEqual(len(pairs), 24)
        self.assertEqual(explanation.total.sum(), 81756)
        self.assertEqual(explanation.available.sum(), 65678)
        self.assertTrue(pairs.total.eq(1514).all())
        self.assertEqual(sorted(pairs.loc[pairs.model_id.eq(MODELS[4]), "available"]), [11, 23, 37, 47])


class FullPlotTest(unittest.TestCase):
    def setUp(self):
        self.module = importlib.import_module("src.benchmarks.ethics.gender_lm_presentation_v2_plots")
        self.assertTrue(hasattr(self.module, "render_view"), "Full renderer is not implemented")
        self.addCleanup(plt.close, "all")

    def test_all_four_modes_binary_and_density_keep_models_and_units(self):
        for group in ["B02", "B08", "B09"]:
            for mode, filters in [("direct", {}), ("profile", {"scenario": "I2"}),
                                  ("neutral", {"persona": "Man"}), ("implicit", {"persona": "Woman"})]:
                with self.subTest(group=group, mode=mode):
                    fig, rows = self.module.render_view(group, mode, **filters)
                    panels = [ax for ax in fig.axes if ax.get_gid() == "data-panel"]
                    self.assertEqual(len(panels), 4 if mode in {"direct", "neutral"} else 3)
                    self.assertAlmostEqual(fig.get_figwidth() * 2.54, 16)
                    self.assertLessEqual(fig.get_figheight() * 2.54, 24.0001)
                    for ax in panels:
                        dots = [c for c in ax.collections if (c.get_gid() or "").startswith("model:")]
                        self.assertEqual(len(dots), 6)
                    self.module.audit_geometry(fig)
                    if group == "B08":
                        self.assertTrue(rows.display_unit.eq("percent" if mode == "profile" else "percentage_points").all())
                    plt.close(fig)

    def test_six_na_models_and_seven_marker_rows_have_nonoverlapping_margin_symbols(self):
        rows = data.view_data("B08", "direct")
        rows[["display_effect", "display_ci_low", "display_ci_high"]] = np.nan
        fig = self.module.plot_view(rows, "B08", "direct")
        fig.canvas.draw()
        for ax in fig.axes:
            marks = [c for c in ax.collections if (c.get_gid() or "").startswith("unavailable:")]
            self.assertEqual(sum(len(c.get_offsets()) for c in marks), 42)
            self.assertTrue(all(np.all(c.get_offsets()[:, 0] > 1) for c in marks))
            self.assertEqual(len([t for t in ax.texts if t.get_text() == "NA"]), 1)
            centers = [c.get_offset_transform().transform(c.get_offsets()[0])[1] for c in marks]
            diameter = max(np.sqrt(c.get_sizes()[0]) * fig.dpi / 72 for c in marks)
            self.assertGreaterEqual(min(np.abs(np.diff(centers))), diameter)
        self.module.audit_geometry(fig)

    def test_zero_and_asymmetric_ci_keep_exact_coordinates(self):
        rows = data.view_data("B08", "direct")
        rows[["display_effect", "display_ci_low", "display_ci_high"]] = 0.
        fig = self.module.plot_view(rows, "B08", "direct")
        for ax in fig.axes:
            dots = [c for c in ax.collections if (c.get_gid() or "").startswith("model:")]
            self.assertEqual(sum(len(c.get_offsets()) for c in dots), 42)
            self.assertTrue(all(np.all(c.get_offsets()[:, 0] == 0) for c in dots))
        plt.close(fig)
        rows["display_effect"], rows["display_ci_low"], rows["display_ci_high"] = 2., -1., 5.
        fig = self.module.plot_view(rows, "B08", "direct")
        for ax in fig.axes:
            for lines in [c for c in ax.collections if (c.get_gid() or "").startswith("ci:")]:
                for segment in lines.get_segments():
                    np.testing.assert_array_equal(segment[:, 0], [-1., 5.])

    def test_coverage_has_two_heatmaps_and_preserves_counts_in_table(self):
        table = data.coverage_table(data.source_table("marker_summary.csv"), data.source_table("comparisons.csv"))
        fig = self.module.plot_coverage(table)
        heatmaps = [ax for ax in fig.axes if ax.images]
        self.assertEqual([ax.images[0].get_array().shape for ax in heatmaps], [(6, 9), (6, 4)])
        self.module.audit_geometry(fig)


class ExplorerTest(unittest.TestCase):
    def setUp(self):
        name = "src.benchmarks.ethics.gender_lm_presentation_v2_explorer"
        self.assertIsNotNone(importlib.util.find_spec(name), "Explorer is not implemented")
        self.module = importlib.import_module(name)

    def test_offline_document_escapes_embedded_payload_and_keeps_view_links(self):
        svg = "data:image/svg+xml;base64," + base64.b64encode(b'<svg xmlns="http://www.w3.org/2000/svg"/>').decode()
        view = dict(view_id="B08_direct_overall", group_id="B08", mode="direct", persona=None, scenario=None,
                    marker_ids=data.MARKER_GROUPS["B08"], svg_data_uri=svg,
                    caption='</script><script>alert(1)</script>', table_html="<table><tr><td>&lt;safe&gt;</td></tr></table>")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "explorer.html"
            self.module.write_explorer([view], output)
            document = output.read_text(encoding="utf-8")
            self.assertNotIn('<script>alert(1)</script>', document)
            self.assertIn('data:image/svg+xml;base64,', document)
            self.assertNotIn('src="https://', document)
            self.assertNotIn('fetch(', document)
            self.assertIn('textContent', document)
            self.assertIn('B08_direct_overall', document)
            with self.assertRaises(FileExistsError):
                self.module.write_explorer([view], output)

    def test_all_81_navigation_combinations_and_deep_links_execute(self):
        views = [{**spec, "svg_data_uri": "data:image/svg+xml;base64," + spec["view_id"],
                  "caption": spec["view_id"], "table_html": "<table>" + spec["view_id"] + "</table>"}
                 for spec in data.view_specs()]
        exercise_explorer(self, self.module.EXPLORER_SCRIPT, views)


class BundleBuildTest(unittest.TestCase):
    def setUp(self):
        name = "src.benchmarks.ethics.build_gender_lm_presentation_v2_bundle"
        self.assertIsNotNone(importlib.util.find_spec(name), "Full bundle builder is not implemented")
        self.module = importlib.import_module(name)

    def test_existing_target_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                self.module.build_bundle(Path(directory))
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_cli_rejects_ignored_filters_and_conflicting_actions_before_build(self):
        for options in [["--scope", "justice"], ["--mode", "profile"], ["--persona", "Woman"],
                        ["--scenario", "I2"], ["--group", "B02", "--mode", "direct", "--install"],
                        ["--install", "--install-existing"]]:
            with self.subTest(options=options), patch("sys.argv", ["builder", "--output", "unused"] + options), \
                    patch.object(self.module, "build_bundle") as build, \
                    patch.object(self.module, "export_view") as export, \
                    patch.object(self.module, "install_additive") as install, redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    self.module.main()
                self.assertEqual(error.exception.code, 2)
                build.assert_not_called(); export.assert_not_called(); install.assert_not_called()

    def test_links_support_windows_temporary_directory_on_another_drive(self):
        with tempfile.TemporaryDirectory() as directory:
            link = self.module._local_link(SOURCE / "evidence_explorer.html", Path(directory))
            if SOURCE.drive.lower() != Path(directory).drive.lower():
                self.assertEqual(link, (SOURCE / "evidence_explorer.html").resolve().as_uri())
            else:
                self.assertIn("evidence_explorer.html", link)

    def test_full_build_has_actual_svg_views_and_exactly_ten_standalone_images(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "bundle"
            report = self.module.build_bundle(output)
            self.assertEqual(len(list(output.glob("*.png"))), 10)
            self.assertEqual(len(list(output.glob("*.svg"))), 10)
            self.assertEqual(report["rendered_views"], 81)
            self.assertEqual(report["marker_count"], 41)
            self.assertTrue(report["protected_files_unchanged"])
            coverage = pd.read_csv(output / "visual_coverage.csv")
            self.assertEqual(len(coverage), 164)
            self.assertTrue(coverage.rendered.all())
            for mode in ["direct", "profile", "neutral", "implicit"]:
                self.assertEqual(coverage.loc[coverage['mode'].eq(mode), "marker_id"].nunique(), 41)
            html_text = (output / "explorer.html").read_text(encoding="utf-8")
            import re
            payload = re.search(r'id="view-data">(.*?)</script>', html_text, re.S).group(1)
            views = json.loads(payload)
            self.assertEqual(len(views), 81)
            for view in views:
                self.assertIn(b"<svg", base64.b64decode(view["svg_data_uri"].split(",", 1)[1]))
                self.assertIn("<table", view["table_html"])
            from src.benchmarks.ethics.gender_lm_presentation_v2_explorer import EXPLORER_SCRIPT
            exercise_explorer(self, EXPLORER_SCRIPT, views)
            import nbformat
            notebook = nbformat.read(output / "gender_lm_presentation_v2.ipynb", as_version=4)
            self.assertEqual(sum("image/png" in item.get("data", {}) for cell in notebook.cells
                                 for item in cell.get("outputs", [])), 10)
            self.assertFalse(any(item.output_type == "error" for cell in notebook.cells for item in cell.get("outputs", [])))
            inline = [cell for cell in notebook.cells if cell.cell_type == "markdown" and 'class="inline-example"' in cell.source]
            self.assertEqual(len(inline), 2)
            self.assertTrue(all("<mark>" in cell.source and "contoh_" in cell.source for cell in inline))
            self.assertTrue((output / "examples.html").is_file())
            self.assertEqual(pd.read_csv(output / "example_responses.csv").example_id.nunique(), 2)

    def test_additive_install_keeps_existing_different_file_and_reuses_identical_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, active = root / "bundle", root / "active"
            bundle.mkdir(); active.mkdir(); (active / "figures").mkdir()
            for extension in ["png", "svg", "csv"]:
                (bundle / f"B02_direct_overall.{extension}").write_bytes(b"new")
            (active / "figures/B02_direct_overall.png").write_bytes(b"old")
            (active / "figures/B02_direct_overall.svg").write_bytes(b"new")
            pd.DataFrame([dict(figure_id="B02_direct_overall", title="B02", caption="CI marginal < 100%")]).to_csv(bundle / "figure_index.csv", index=False)
            result = self.module.install_additive(bundle, active)
            self.assertEqual((active / "figures/B02_direct_overall.png").read_bytes(), b"old")
            self.assertEqual((active / "figures/B02_direct_overall_v2.png").read_bytes(), b"new")
            self.assertEqual(len(list((active / "figures").glob("*.svg"))), 1)
            self.assertTrue(result["old_files_unchanged"])
            portal = Path(result["portal"]).read_text(encoding="utf-8")
            self.assertIn("<figcaption>CI marginal &lt; 100%</figcaption>", portal)
            self.assertIn("figures/B02_direct_overall.csv", portal)


class SingleExportTest(unittest.TestCase):
    def test_scoped_export_includes_raw_counts_denominators_and_source_provenance(self):
        from src.benchmarks.ethics.build_gender_lm_presentation_v2_bundle import export_view
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "export"
            export_view(output, "B08", "neutral", persona="Woman", scope="justice")
            self.assertTrue((output / "README.md").is_file())
            context = pd.read_csv(output / "marker_context.csv")
            self.assertTrue(context.scope.eq("justice").all())
            self.assertEqual(context.marker_id.nunique(), 7)
            self.assertTrue(context.raw_count_total.notna().all())
            self.assertTrue({"n_available", "total"}.issubset(context.columns))
            provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
            self.assertEqual(provenance["scope"], "justice")
            self.assertEqual(len(provenance["source_hashes"]), 3)
            self.assertTrue(provenance["source_unchanged"])
            self.assertTrue(provenance["code_hashes"])
            self.assertEqual(len(list(output.glob("*.png"))), 1)
            with self.assertRaises(FileExistsError):
                export_view(output, "B08", "neutral", persona="Woman", scope="justice")


class InlineEvidenceTest(unittest.TestCase):
    def test_notebook_cards_include_exact_context_span_and_provenance_with_html_escaping(self):
        from src.benchmarks.ethics import gender_lm_presentation_v2_evidence as evidence
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            responses = [dict(example_id="contoh_1", response_id=persona, model_id="model", subset="justice",
                              question_id="q1", persona=persona, mode="direct", explanation_text="Before <woman> after.")
                         for persona in ["neutral", "man_explicit_1", "woman_explicit_1"]]
            spans = [dict(example_id="contoh_1", response_id="woman_explicit_1", marker_id="gender_identity_discourse",
                          rule_id="fixture", surface_name="explanation_text", start=7, end=14, text="<woman>")]
            pd.DataFrame(responses).to_csv(output / "example_responses.csv", index=False)
            pd.DataFrame(spans).to_csv(output / "example_spans.csv", index=False)
            card = evidence.inline_example(output, "contoh_1")
            self.assertIn('class="inline-example"', card)
            self.assertIn("Before <mark>&lt;woman&gt;</mark> after.", card)
            self.assertIn("gender_identity_discourse", card)
            self.assertIn("fixture", card)
            self.assertIn("7:14", card)
            self.assertIn("examples.html#contoh_1", card)
            for row in responses:
                self.assertIn(row["persona"], card)
            self.assertNotIn("<woman>", card)


if __name__ == "__main__":
    unittest.main()
