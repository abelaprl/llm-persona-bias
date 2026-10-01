"""Regression checks on full observed responses and measurement invariants."""
import json
from pathlib import Path
import io
import unittest


class SurfaceTests(unittest.TestCase):
    def test_verdict_is_not_explanation(self):
        from src.benchmarks.ethics.gender_lm_v2.surfaces import build_surfaces
        for text in ("1", "**0**", "Answer: 1", "1."):
            with self.subTest(text=text):
                surface = build_surfaces(text)
                self.assertEqual(surface.status, "verdict_only")
                self.assertEqual(surface.word_count, 0)
                self.assertEqual(surface.raw_text, text)

    def test_no_hidden_reasoning_fallback(self):
        from src.benchmarks.ethics.gender_lm_v2.surfaces import build_surfaces
        self.assertEqual(build_surfaces("<think>Perhaps women need help</think>1").status, "verdict_only")
        self.assertEqual(build_surfaces("<think>Perhaps women need help").status, "unknown_surface")
        self.assertEqual(build_surfaces("").status, "missing")

    def test_refusal_has_measurable_explanation(self):
        from src.benchmarks.ethics.gender_lm_v2.surfaces import build_surfaces
        surface = build_surfaces("I cannot provide a verdict without more context.")
        self.assertEqual(surface.status, "substantive_explanation")
        self.assertGreater(surface.word_count, 0)

    def test_incomplete_harmony_channel_never_becomes_explanation(self):
        from src.benchmarks.ethics.gender_lm_v2.surfaces import build_surfaces
        self.assertEqual(build_surfaces("<|meta_sep|>analysis Perhaps women need help.").status, "unknown_surface")

    def test_harmony_final_body_stops_before_later_analysis(self):
        from src.benchmarks.ethics.gender_lm_v2.surfaces import build_surfaces
        from src.benchmarks.ethics.gender_lm_v2.detectors import detect_markers
        raw = "<|channel|>final<|message|>The action has consequences.<|end|><|channel|>analysis<|message|>Perhaps women need help."
        surface = build_surfaces(raw)
        self.assertNotIn("Perhaps", surface.visible_text)
        self.assertFalse(any(x.marker_id == "stance_hedge" for x in detect_markers(surface)))


class DetectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).parent / "fixtures" / "real_responses.json"
        cls.samples = json.loads(path.read_text(encoding="utf-8"))

    def get_hits(self, sample_id, marker):
        from src.benchmarks.ethics.gender_lm_v2.surfaces import build_surfaces
        from src.benchmarks.ethics.gender_lm_v2.detectors import detect_markers, load_registry
        surface = build_surfaces(self.samples[sample_id]["response_content_full"])
        hits = detect_markers(surface, load_registry())
        for hit in hits:
            text = getattr(surface, hit.surface_name)
            self.assertEqual(hit.text, text[hit.start:hit.end])
            self.assertEqual(hit.rule_status, "candidate")
        return [hit for hit in hits if hit.marker_id == marker]

    def test_observed_empathy_contractions(self):
        for sample_id in (8, 9):
            self.assertEqual(len(self.get_hits(sample_id, "affect_empathy_validation")), 1)

    def test_observed_negated_manner_is_not_booster(self):
        for sample_id in (4, 5):
            self.assertEqual(self.get_hits(sample_id, "stance_booster"), [])

    def test_interrogative_if_excluded_other_conditionals_preserved(self):
        self.assertEqual(len(self.get_hits(1, "connective_conditional")), 3)

    def test_inline_numbered_sections_are_observed(self):
        self.assertEqual(len(self.get_hits(1, "organization_explicit_step")), 5)

    def test_rhetorical_wait_not_self_repair(self):
        self.assertEqual(self.get_hits(10, "stance_self_correction"), [])


class SourceTests(unittest.TestCase):
    def test_no_result_overwrite(self):
        from src.benchmarks.ethics.gender_lm_v2.sources import reserve_output
        with self.assertRaises(FileExistsError):
            reserve_output(Path(__file__).parent)

    def test_labels_never_enter_extraction_frame(self):
        import pandas as pd
        from src.benchmarks.ethics.gender_lm_v2.sources import load_source
        row = {"model": "m", "persona_folder": "neutral", "subset": "justice", "question_id": "q", "file": "q.json", "response_content_full": "Perhaps this is fair.", "ground_truth": 1}
        first = load_source(io.StringIO(pd.DataFrame([row]).to_csv(index=False)))
        row["ground_truth"] = 0
        second = load_source(io.StringIO(pd.DataFrame([row]).to_csv(index=False)))
        pd.testing.assert_frame_equal(first, second)
        self.assertNotIn("ground_truth", first.columns)


class ReferenceTests(unittest.TestCase):
    def test_man_is_not_substring_overlap_with_woman(self):
        from src.benchmarks.ethics.gender_lm_v2.contracts import Surface, Span
        from src.benchmarks.ethics.gender_lm_v2.references import annotate_references
        surface = Surface("I am a man", "I am a man", "I am a man", "substantive_explanation")
        span = Span("gender_first_person_identity_reference", "test", 0, 10, "I am a man", "explanation_text")
        annotate_references(surface, [span], "A woman is in the garden.")
        self.assertNotIn("same_gender_term_in_scenario_not_coreference", span.flags)


if __name__ == "__main__":
    unittest.main()
