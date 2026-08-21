from __future__ import annotations

import math
import unittest

import pandas as pd

from src.MAIN_PARSING.LINGUISTIC_MARKERS.audit_sampler import (
    build_audit_sample,
    build_recall_audit_sample,
)
from src.MAIN_PARSING.LINGUISTIC_MARKERS.extractor import (
    extract_markers,
    find_marker_matches,
)
from src.MAIN_PARSING.LINGUISTIC_MARKERS.registry import (
    MARKERS,
    MarkerSpec,
    PatternSpec,
    marker_by_id,
)
from src.MAIN_PARSING.LINGUISTIC_MARKERS.surface import prepare_text_surface


def make_record(text: str | None, benchmark: str = "gsm8k") -> dict[str, object]:
    return {
        "model_id": "test-model",
        "benchmark": benchmark,
        "subset": "all",
        "identity": "neutral",
        "condition": "neutral",
        "persona": "neutral",
        "question_id": "q1",
        "source_path": "fixture.json",
        "response_content": text,
    }


class SurfaceTests(unittest.TestCase):
    def test_typographic_quotes_are_normalized_on_measured_surface(self):
        surface = prepare_text_surface("You\u2019re a woman and \u201cyour gender\u201d is stated.", "ethics")
        self.assertIn("You're a woman", surface.lexical_text)
        self.assertIn('"your gender"', surface.lexical_text)

    def test_closed_think_is_removed_from_all_marker_counts(self):
        row = extract_markers(
            make_record("<think>Perhaps this is uncertain.</think>\nClearly the result is 4.")
        )
        self.assertTrue(row["think_block_stripped"])
        self.assertNotIn("Perhaps", row["visible_text"])
        self.assertEqual(row["stance_hedge__count"], 0)
        self.assertEqual(row["stance_booster__count"], 1)

    def test_unclosed_think_becomes_missing_not_zero(self):
        row = extract_markers(make_record("<think>Perhaps the answer is 4"))
        self.assertEqual(row["text_status"], "unseparable_think")
        self.assertFalse(row["text_available"])
        self.assertIsNone(row["stance_hedge__count"])
        self.assertIsNone(row["stance_hedge__density"])

    def test_empty_content_becomes_na(self):
        row = extract_markers(make_record(None))
        self.assertEqual(row["text_status"], "missing")
        self.assertIsNone(row["word_count"])
        self.assertIsNone(row["connective_causal__count"])

    def test_technical_answer_line_is_removed_from_explanation(self):
        surface = prepare_text_surface("Because 2 + 2 = 4.\nFinal answer: 4")
        self.assertTrue(surface.text_available)
        self.assertEqual(surface.technical_lines_stripped, 1)
        self.assertNotIn("Final answer", surface.lexical_text)
        self.assertIn("Because 2 + 2 = 4", surface.structure_text)

    def test_technical_only_response_has_no_linguistic_zero(self):
        surface = prepare_text_surface("Answer: 0")
        self.assertEqual(surface.text_status, "technical_only")
        self.assertFalse(surface.text_available)
        self.assertIsNone(surface.word_count)

    def test_ethics_binary_label_is_visible_but_excluded_from_marker_surface(self):
        row = extract_markers(make_record("1. I think this is right.", benchmark="ethics"))
        self.assertEqual(row["visible_text"], "1. I think this is right.")
        self.assertEqual(row["explanation_text"], "I think this is right.")
        self.assertEqual(row["opening_category"], "answer_first")
        self.assertEqual(row["organization_explicit_step__count"], 0)
        self.assertEqual(row["technical_lines_stripped"], 1)


class MarkerTests(unittest.TestCase):
    def test_broad_old_hedges_are_excluded(self):
        row = extract_markers(
            make_record("There are about 12 items around the box. We could calculate the total.")
        )
        self.assertEqual(row["stance_hedge__count"], 0)

    def test_context_bounded_hedges_are_detected(self):
        row = extract_markers(make_record("Perhaps it may be easier, but it could be wrong."))
        self.assertEqual(row["stance_hedge__count"], 3)

    def test_audited_epistemic_frames_are_detected(self):
        row = extract_markers(
            make_record(
                "I think this is probably intended. That seems plausible. "
                "Indeed, the result follows."
            )
        )
        self.assertEqual(row["stance_hedge__count"], 3)
        self.assertEqual(row["stance_booster__count"], 1)

    def test_deontic_must_be_is_not_an_epistemic_booster(self):
        row = extract_markers(
            make_record("Polly must be within range. The initial cost must be covered.")
        )
        self.assertEqual(row["stance_booster__count"], 0)

    def test_overlapping_patterns_count_longest_span_once(self):
        marker = MarkerSpec(
            marker_id="overlap_fixture",
            family="test",
            operational_definition="test",
            patterns=(
                PatternSpec(r"\bof course\b", "long"),
                PatternSpec(r"\bcourse\b", "short"),
            ),
        )
        matches = find_marker_matches("Of course this course continues.", marker)
        self.assertEqual([match.text.casefold() for match in matches], ["of course", "course"])

    def test_engagement_does_not_count_bare_you(self):
        row = extract_markers(make_record("You have 3 apples. Note that two are red."))
        self.assertEqual(row["engagement_reader_directive__count"], 1)

    def test_engagement_consider_requires_reader_or_clause_initial_position(self):
        row = extract_markers(
            make_record("We should consider the cost. Consider the safer alternative.")
        )
        self.assertEqual(row["engagement_reader_directive__count"], 1)

    def test_engagement_remember_excludes_first_person_memory_report(self):
        row = extract_markers(
            make_record("I remember that rule. But remember that the denominator is fixed.")
        )
        self.assertEqual(row["engagement_reader_directive__count"], 1)

    def test_code_gloss_excludes_relative_that_is(self):
        row = extract_markers(
            make_record("Choose a journey that is guided by values. That is, choose carefully.")
        )
        self.assertEqual(row["elaboration_code_gloss__count"], 1)

    def test_first_person_marker_excludes_let_me_know_signoff(self):
        row = extract_markers(make_record("Let me know if you need help. Let me calculate this."))
        self.assertEqual(row["self_reference_first_person__count"], 1)

    def test_assumption_participle_requires_clause_initial_position(self):
        row = extract_markers(
            make_record("This can cause paranoia, assuming the worst. Assuming the price is fixed, solve it.")
        )
        self.assertEqual(row["stance_assumption__count"], 1)

    def test_imperative_assumption_and_inclusive_revision_are_detected(self):
        row = extract_markers(
            make_record("Assume that x is positive. Let's re-evaluate the calculation.")
        )
        self.assertEqual(row["stance_assumption__count"], 1)
        self.assertEqual(row["stance_self_correction__count"], 1)

    def test_initial_wait_is_deliberative_not_visible_self_correction(self):
        initial = extract_markers(make_record("Wait, calculate it again."))
        revision = extract_markers(make_record("That result fails. Wait, calculate it again."))
        self.assertEqual(initial["stance_self_correction__count"], 0)
        self.assertEqual(revision["stance_self_correction__count"], 1)

    def test_self_correction_excludes_advice_and_wait_until(self):
        row = extract_markers(
            make_record(
                'Say, "I was wrong and I apologize." Wait until tomorrow. '
                "The first result fails. Actually, let me restart."
            )
        )
        self.assertEqual(row["stance_self_correction__count"], 2)

    def test_new_causal_variants_are_detected(self):
        row = extract_markers(
            make_record("For this reason, revise the total. Accordingly, use the safer option.")
        )
        self.assertEqual(row["connective_causal__count"], 2)

    def test_inclusive_we_future_is_detected(self):
        row = extract_markers(make_record("We will calculate the total next."))
        self.assertEqual(row["self_reference_inclusive_we__count"], 1)

    def test_step_rule_does_not_count_inline_arithmetic(self):
        row = extract_markers(make_record("We calculate 16 - 3 = 13, then add one."))
        self.assertEqual(row["organization_explicit_step__count"], 0)

    def test_bold_sentence_is_not_a_section_header(self):
        row = extract_markers(
            make_record("**Josh made a profit of $70,000.**\n**Steps:**\n1. Add costs.")
        )
        self.assertEqual(row["organization_section_header__count"], 1)

    def test_boxed_answer_uses_visible_answer_line_and_nested_braces(self):
        row = extract_markers(make_record(r"Final answer: \boxed{\$70{,}000}"))
        self.assertEqual(row["text_status"], "technical_only")
        self.assertEqual(row["math_boxed_answer__count"], 1)
        self.assertEqual(row["math_boxed_answer__present"], 1)

    def test_plain_arithmetic_captures_complete_multi_operand_expression(self):
        marker = marker_by_id("math_inline_arithmetic")
        text = "Total: 3 + 5 + 2 + 4 = 14. Fraction: 3/5 x 300 = 180."
        matches = find_marker_matches(text, marker)
        self.assertEqual(
            [match.text for match in matches],
            ["3 + 5 + 2 + 4 = 14", "3/5 x 300 = 180"],
        )

    def test_inline_arithmetic_does_not_cross_line_break(self):
        marker = marker_by_id("math_inline_arithmetic")
        matches = find_marker_matches("325,000 - 80,000\n= 245,000", marker)
        self.assertEqual(matches, [])

    def test_marker_not_applicable_is_na(self):
        row = extract_markers(make_record("The answer uses $x=2$.", benchmark="ethics"))
        self.assertIsNone(row["math_latex_display__count"])
        self.assertIsNone(row["math_inline_arithmetic__present"])

    def test_opening_categories_have_deterministic_precedence(self):
        cases = {
            "Hmm, let me think about this carefully.": "deliberative",
            "I cannot provide a personal judgment, but here is context.": "refusal",
            "Final answer: 18\nThe calculation follows.": "answer_first",
            "**Solution**\nWe calculate it.": "document_heading",
            "Step 1: Count the eggs.": "step_first",
            "- Count the eggs first.": "bullet_first",
            "Total eggs = 16 before breakfast.": "algebraic",
            "To find the result, subtract three.": "instructional",
            "The problem states that Janet has eggs.": "problem_restatement",
            "In this scenario, the action is wrong.": "contextual",
            "Janet ate three eggs before baking.": "narrative",
        }
        for text, expected in cases.items():
            with self.subTest(expected=expected):
                self.assertEqual(extract_markers(make_record(text))["opening_category"], expected)

    def test_ground_truth_does_not_change_extraction(self):
        base = make_record("Clearly, perhaps this result is correct.")
        first = extract_markers({**base, "ground_truth": 0})
        second = extract_markers({**base, "ground_truth": 1})
        marker_columns = [key for key in first if "__" in key]
        self.assertEqual(
            {key: first[key] for key in marker_columns},
            {key: second[key] for key in marker_columns},
        )

    def test_registry_ids_are_unique_and_candidate_versioned(self):
        ids = [marker.marker_id for marker in MARKERS]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(marker.version for marker in MARKERS))
        self.assertTrue(all("candidate" in marker.validation_status for marker in MARKERS))
        self.assertEqual(marker_by_id("stance_hedge").family, "epistemic_stance")

    def test_density_uses_available_word_count(self):
        row = extract_markers(make_record("Clearly this works."))
        self.assertEqual(row["word_count"], 3)
        self.assertTrue(math.isclose(row["stance_booster__density"], 100 / 3))

    def test_audit_sampler_caps_occurrences_not_positive_responses(self):
        extracted = [
            extract_markers(make_record("Therefore A. Thus B. Hence C.")),
            extract_markers({**make_record("Because D."), "question_id": "q2"}),
        ]
        audit = build_audit_sample(
            pd.DataFrame(extracted),
            sample_per_marker=2,
            markers=(marker_by_id("connective_causal"),),
        )
        self.assertEqual(len(audit), 2)
        self.assertTrue(
            set(audit["matched_text"].str.casefold())
            <= {"therefore", "thus", "hence", "because"}
        )

    def test_recall_audit_is_blind_and_stratified_by_condition(self):
        extracted = []
        for model in ("m1", "m2"):
            for persona in ("neutral", "man_explicit_1"):
                for qid in range(3):
                    extracted.append(
                        extract_markers(
                            {
                                **make_record("Clearly this works."),
                                "model_id": model,
                                "persona": persona,
                                "question_id": f"q{qid}",
                                "is_correct": qid == 0,
                            }
                        )
                    )
        audit = build_recall_audit_sample(
            pd.DataFrame(extracted), sample_per_condition=1, seed=7
        )
        self.assertEqual(len(audit), 4)
        self.assertNotIn("is_correct", audit.columns)
        self.assertFalse(any("__" in column for column in audit.columns))
        self.assertEqual(
            audit.groupby(["model_id", "persona"]).size().tolist(), [1, 1, 1, 1]
        )


if __name__ == "__main__":
    unittest.main()
