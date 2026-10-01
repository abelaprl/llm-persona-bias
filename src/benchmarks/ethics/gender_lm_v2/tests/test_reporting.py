"""Reporting contracts: truthful denominators, units and inert full evidence."""
import hashlib
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.benchmarks.ethics.gender_lm_v2.reporting import (
    build_report, summarize_markers, render_highlighted_text, write_evidence_explorer,
    _marker_pages,
)


def fixture_records():
    rows = []
    for persona, value, status in [
        ("neutral", 0, "substantive_explanation"),
        ("woman_explicit_1", 1, "substantive_explanation"),
        ("man_explicit_1", None, "missing"),
    ]:
        text = '<script>alert("x")</script> ' + "full content " * 400 + "END_OF_RESPONSE"
        rows.append(dict(model_id="model", persona=persona, subset="justice", question_id="q1",
                         raw_text=text, visible_text=text, explanation_text=text,
                         text_status=status, text_available=value is not None, word_count=100,
                         hedge__count=value, hedge__density=value, hedge__present=value,
                         ground_truth="PRIVATE_LABEL", correctness="PRIVATE_CORRECT"))
    return pd.DataFrame(rows)


class ReportingTests(unittest.TestCase):
    def test_span_lookup_preserves_legacy_grouping_order_nulls_and_provenance(self):
        from src.benchmarks.ethics.gender_lm_v2.reporting import _build_span_lookup, KEY
        spans = pd.DataFrame([
            dict(model_id="b", subset="justice", question_id="q2", persona="neutral", marker_id="second", start=1, flags=["quoted"]),
            dict(model_id="a", subset="virtue", question_id="q1", persona="woman_explicit_1", marker_id="first", start=0, flags=None),
            dict(model_id="b", subset="justice", question_id="q2", persona="neutral", marker_id="third", start=3, flags=["negated"]),
            dict(model_id="a", subset="virtue", question_id=None, persona="neutral", marker_id="dropped_missing_key", start=0, flags=None),
        ], index=[8, 2, 20, 7])
        spans["model_id"] = pd.Categorical(spans.model_id, categories=["unused", "a", "b"])
        spans["start"] = spans.start.astype("Int64")
        spans["optional_provenance"] = pd.Series(["available", pd.NA, "last", "unused"], index=spans.index, dtype="string")
        legacy = {key: group.to_dict("records") for key, group in spans.groupby(KEY + ["persona"], sort=False, observed=True)}
        actual = _build_span_lookup(spans)
        self.assertEqual(list(actual), list(legacy))
        self.assertEqual(actual, legacy)
        self.assertEqual(_build_span_lookup(pd.DataFrame()), {})
        self.assertEqual(_build_span_lookup(spans.drop(columns="question_id")), {})

    def test_low_support_retains_condition_shapes_in_rendered_forests(self):
        from src.benchmarks.ethics.gender_lm_v2.reporting import _forest, _condition_profiles, _triplet_maps
        import matplotlib.pyplot as plt

        class FigureCapture:
            def __init__(self):
                self.figures = []

            def save(self, figure, *args, **kwargs):
                self.figures.append(figure)

        rows = []
        for condition in ("explicit_1", "implicit_1", "implicit_2", "implicit_3"):
            for marker, n_changed in (("low", 2), ("supported", 20)):
                rows.append(dict(model_id="model", scope="overall", marker_id=marker, measurement="density",
                                 left="woman_" + condition, right="neutral", role="primary",
                                 contrast_id="woman_" + condition + "__minus__neutral", effect=1., ci_low=.5, ci_high=1.5,
                                 n_pairs=30, n_changed=n_changed, estimand="mean_paired_difference"))
        records = pd.DataFrame(rows)
        registry = [dict(marker_id=marker, display_family="stance") for marker in ("low", "supported")]
        book = FigureCapture()
        try:
            _forest(book, records, registry, ["model"])
            points = [line for line in book.figures[0].axes[0].lines if len(line.get_xdata()) == 1]
            self.assertEqual([line.get_marker() for line in points], ["o", "o", "s", "s", "^", "^", "v", "v"])
            self.assertEqual([line.get_markerfacecolor() for line in points[::2]], ["none"] * 4)
            self.assertTrue(all(line.get_markerfacecolor() != "none" for line in points[1::2]))
            _condition_profiles(book, records, registry, ["model"])
            profile_points = [line for ax in book.figures[-1].axes for line in ax.lines if len(line.get_xdata()) == 1]
            self.assertEqual({line.get_marker() for line in profile_points}, {"o"})
            self.assertEqual(sum(line.get_markerfacecolor() == "none" for line in profile_points), 4)
            triplets = pd.DataFrame([dict(model_id="model", scope="overall", marker_id="low", measurement="density",
                                         condition=condition, sample_scope="common_triplet", d_man_neutral=.5,
                                         d_woman_neutral=1., n_triplets=30, low_support=True)
                                     for condition in ("explicit_1", "implicit_1", "implicit_2", "implicit_3")])
            _triplet_maps(book, triplets, registry, ["model"])
            triplet_points = [line for line in book.figures[-1].axes[0].lines if len(line.get_xdata()) == 1]
            self.assertEqual([line.get_marker() for line in triplet_points], ["o", "s", "^", "v"])
            self.assertEqual([line.get_markerfacecolor() for line in triplet_points], ["none"] * 4)
            self.assertEqual(len(book.figures[-1].axes[0].texts), 0, "Dense triplet maps must use legends instead of point labels")
            legend_labels = [label.get_text() for legend in book.figures[-1].legends for label in legend.get_texts()]
            self.assertIn("low", legend_labels)
            self.assertTrue(set(("E", "I1", "I2", "I3")).issubset(legend_labels))
        finally:
            for figure in book.figures:
                plt.close(figure)

    def test_length_distribution_keeps_verdict_zero_and_excludes_unavailable(self):
        from src.benchmarks.ethics.gender_lm_v2.reporting import summarize_response_lengths
        records = pd.DataFrame([
            dict(model_id="model", persona="neutral", subset="justice", text_status=status, word_count=count)
            for status, count in [("substantive_explanation", 10), ("substantive_explanation", 30),
                                  ("verdict_only", 0), ("missing", 999), ("unknown_surface", 888),
                                  ("technical_only", 777)]
        ])
        result = summarize_response_lengths(records)
        overall = result.loc[result.scope.eq("overall")].iloc[0]
        self.assertEqual(overall.total, 6)
        self.assertEqual(overall.n_available, 3)
        self.assertEqual(overall.n_unavailable, 3)
        self.assertEqual(overall.n_zero, 1)
        self.assertEqual(overall["median"], 10)
        self.assertEqual(overall["max"], 30)
        self.assertAlmostEqual(overall["mean"], 40 / 3)
        unavailable = records.loc[records.text_status.eq("missing")]
        empty_distribution = summarize_response_lengths(unavailable).iloc[0]
        self.assertTrue(pd.isna(empty_distribution["median"]))
        self.assertEqual(empty_distribution.n_zero, 0)

    def test_small_families_share_pages_without_dropping_markers(self):
        registry = [dict(marker_id=f"m{i}", display_family=f"family{i // 2}") for i in range(18)]
        pages = list(_marker_pages(registry))
        self.assertEqual(len(pages), 3)
        self.assertTrue(all(len(ids) <= 8 for _, _, ids in pages))
        self.assertEqual([marker for _, _, ids in pages for marker in ids], [f"m{i}" for i in range(18)])

    def test_missing_is_na_and_overall_keeps_all_source_rows(self):
        result = summarize_markers(fixture_records(), [{"marker_id": "hedge", "display_family": "stance"}])
        overall = result.query("scope == 'overall' and measurement == 'density'")
        self.assertEqual(int(overall.total.sum()), 3)
        missing = overall.loc[overall.persona == "man_explicit_1"].iloc[0]
        self.assertTrue(pd.isna(missing.value))
        self.assertEqual(missing.n_available, 0)
        zero = overall.loc[overall.persona == "neutral"].iloc[0]
        self.assertEqual(zero.value, 0)

    def test_highlight_escapes_scripts_and_preserves_overlapping_spans(self):
        text = '<script>hedge</script>'
        rendered = render_highlighted_text(text, [dict(start=8, end=13, marker_id="m", rule_id="r"),
                                                   dict(start=10, end=13, marker_id="n", rule_id="s")])
        self.assertNotIn("<script>", rendered)
        self.assertIn("&lt;script&gt;", rendered)
        self.assertIn("<mark", rendered)
        self.assertIn("m", rendered)
        self.assertIn("n", rendered)

    def test_explorer_has_full_text_safe_html_and_no_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_evidence_explorer(fixture_records(), pd.DataFrame(), Path(tmp), page_size=1)
            pages = list((Path(tmp) / "evidence").glob("*.html"))
            self.assertEqual(len(pages), 1)
            page = pages[0].read_text(encoding="utf-8")
            self.assertIn("END_OF_RESPONSE", page)
            self.assertIn("&lt;script&gt;", page)
            self.assertNotIn("PRIVATE_LABEL", page)
            self.assertNotIn("PRIVATE_CORRECT", page)
            self.assertIn("unavailable", page)
            self.assertIn("woman_explicit_1", page)
            self.assertIn("neutral", page)

    def test_explorer_highlights_structure_surface_at_its_own_offsets(self):
        records = fixture_records()
        structure = "1. **Heading**\nComplete organization surface STRUCTURE_END"
        records["structure_text"] = structure
        spans = pd.DataFrame([dict(model_id="model", subset="justice", question_id="q1", persona="neutral",
                                   marker_id="organization_explicit_step", rule_id="structure_rule", start=0, end=14,
                                   surface_name="structure_text", text=structure[:14], rule_status="candidate")])
        with tempfile.TemporaryDirectory() as tmp:
            write_evidence_explorer(records, spans, Path(tmp))
            page = next((Path(tmp) / "evidence").glob("*.html")).read_text(encoding="utf-8")
            self.assertTrue("STRUCTURE_END" in page, "Full structure surface must be exported")
            self.assertTrue('data-marker="organization_explicit_step">1. **Heading**</mark>' in page,
                            "Structure span must use structure offsets")
            self.assertTrue("Surface organisasi" in page, "Surface label must distinguish organization text")

    def test_report_sources_hashes_units_and_candidate_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            comparison = pd.DataFrame([dict(model_id="model", scope="overall", marker_id="hedge",
                measurement="density", left="woman_explicit_1", right="neutral", role="primary",
                contrast_id="woman_explicit_1__minus__neutral", effect=1.5, ci_low=.5, ci_high=2.5,
                n_pairs=2, n_changed=1, estimand="mean_paired_difference", sample_scope="pairwise_complete")])
            triplet = pd.DataFrame([dict(model_id="model", scope="overall", marker_id="hedge", measurement="density",
                condition="explicit_1", d_woman_neutral=1.5, d_man_neutral=.5, d_woman_man=1.,
                n_triplets=2, sample_scope="common_triplet")])
            index = build_report(fixture_records(), pd.DataFrame(), comparison, pd.DataFrame(), root,
                                 registry=[{"marker_id": "hedge", "display_family": "stance"}], triplets=triplet)
            self.assertIn("condition_profile", set(index.figure_kind))
            self.assertIn("forest_neutral", set(index.figure_kind))
            self.assertIn("neutral_triplet_map", set(index.figure_kind))
            self.assertIn("response_length", set(index.figure_kind))
            self.assertEqual(set(index.scope), {"overall"})
            self.assertIn("justice", set(pd.read_csv(root / "marker_summary.csv").scope))
            self.assertTrue((index.scope == "overall").any())
            self.assertTrue((index.audit_status == "candidate").all())
            for row in index.itertuples():
                self.assertTrue((root / row.png_path).exists())
                self.assertTrue((root / row.svg_path).exists())
                source = root / row.source_csv
                self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), row.source_sha256)
                data = pd.read_csv(source)
                if "measurement" in data and row.figure_kind != "audit":
                    self.assertLessEqual(data.measurement.nunique(), 1)
                if row.figure_kind == "forest_neutral":
                    self.assertEqual(data.effect.iloc[0], 1.5)
                    self.assertTrue(data.low_support.iloc[0])
            report = (root / "report.md").read_text(encoding="utf-8")
            self.assertIn("3", report)
            self.assertIn("99%", report)
            self.assertIn("≥99%", report)
            self.assertIn("Woman−Man", report)
            self.assertNotIn("\ufffd", report)
            with self.assertRaises(FileExistsError):
                build_report(fixture_records(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), root)


if __name__ == "__main__":
    unittest.main()
