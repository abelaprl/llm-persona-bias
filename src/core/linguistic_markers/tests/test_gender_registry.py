from __future__ import annotations

import unittest

from src.core.linguistic_markers.extractor import (
    extract_markers,
    find_marker_matches,
)
from src.core.linguistic_markers.gender_registry import (
    ETHICS_GENDER_MARKERS,
    GENDER_MARKERS,
    compile_gender_term_pattern,
    gender_marker_by_id,
)


def make_record(text: str) -> dict[str, object]:
    return {
        "model_id": "test-model",
        "benchmark": "ethics",
        "subset": "deontology",
        "identity": "man",
        "condition": "explicit",
        "persona": "man_explicit_1",
        "question_id": "q1",
        "response_content": text,
    }


class GenderRegistryTests(unittest.TestCase):
    def test_registry_includes_ethics_specific_and_excludes_non_ethics_markers(self):
        ids = {marker.marker_id for marker in ETHICS_GENDER_MARKERS}
        self.assertNotIn("math_latex_display", ids)
        self.assertIn("framing_deontological", ids)
        self.assertIn("framing_religious", ids)
        self.assertIn("affect_empathy_validation", ids)
        self.assertEqual(len(ids), len(ETHICS_GENDER_MARKERS))

    def test_direct_address_uses_corpus_observed_frames(self):
        marker = gender_marker_by_id("gender_persona_direct_address")
        examples = (
            "As a man, you are capable of doing this task.",
            "As a woman in this profession, you may face additional barriers.",
            "You, as a woman, are being asked to decide.",
            "This could be uncomfortable for you as a female.",
            "This may feel wrong to you (as a woman).",
            "Your presence as a female teacher is appropriate.",
            "Would you like career advice as a woman in corporate law?",
        )
        for text in examples:
            with self.subTest(text=text):
                self.assertTrue(find_marker_matches(text, marker))

    def test_acknowledgment_captures_observed_ethics_phrases(self):
        row = extract_markers(
            make_record(
                "Since you're a man, the assumption is visible. "
                "Regarding your identity as a man, it should not change the verdict."
            ),
            markers=GENDER_MARKERS,
        )
        self.assertEqual(row["gender_persona_acknowledgment__present"], 1)
        self.assertGreaterEqual(row["gender_persona_acknowledgment__count"], 2)

    def test_second_person_identity_reference_excludes_immediate_hypothetical(self):
        marker = gender_marker_by_id("gender_second_person_identity_reference")
        self.assertTrue(find_marker_matches("Your gender should not affect the outcome.", marker))
        self.assertTrue(
            find_marker_matches("An AI should not infer the user's gender.", marker)
        )
        self.assertTrue(find_marker_matches("In this scenario, you are a woman working in IT.", marker))
        self.assertFalse(find_marker_matches("If you are a woman, consider the context.", marker))
        self.assertFalse(find_marker_matches("Whether you are a man is not specified.", marker))

    def test_hypothetical_user_identity_is_a_separate_marker(self):
        marker = gender_marker_by_id("gender_hypothetical_identity_reference")
        self.assertTrue(find_marker_matches("If you're a woman in this role, biases may arise.", marker))
        self.assertTrue(find_marker_matches("Assuming that you are a man, the context changes.", marker))
        self.assertFalse(find_marker_matches("If the woman is present, ask her.", marker))
        self.assertFalse(find_marker_matches("You are a woman in this scenario.", marker))

    def test_first_person_prompt_echo_is_a_separate_uptake_marker(self):
        row = extract_markers(
            make_record('The statement "I am a man" is a personal declaration of identity.'),
            markers=GENDER_MARKERS,
        )
        self.assertEqual(row["gender_first_person_identity_reference__present"], 1)

    def test_prompt_echo_relevance_denial_is_detected(self):
        row = extract_markers(
            make_record('The phrasing "I am a man" is irrelevant to the moral judgment.'),
            markers=GENDER_MARKERS,
        )
        self.assertEqual(row["gender_first_person_identity_reference__present"], 1)
        self.assertEqual(row["gender_identity_relevance_denial__present"], 1)

    def test_third_person_scenario_reference_is_not_persona_uptake(self):
        row = extract_markers(
            make_record("The woman asked the man to help her husband."),
            markers=GENDER_MARKERS,
        )
        uptake = (
            "gender_persona_direct_address",
            "gender_persona_acknowledgment",
            "gender_second_person_identity_reference",
            "gender_first_person_identity_reference",
            "gender_perspective_adoption",
        )
        self.assertTrue(all(row[f"{marker_id}__count"] == 0 for marker_id in uptake))

    def test_third_person_perspective_heading_is_excluded(self):
        marker = gender_marker_by_id("gender_perspective_adoption")
        self.assertFalse(find_marker_matches("The woman's perspective is understandable.", marker))
        self.assertFalse(find_marker_matches("Her role as a woman is discussed.", marker))
        self.assertTrue(find_marker_matches("From your female perspective, this may differ.", marker))
        self.assertTrue(
            find_marker_matches(
                "The facts are mixed. In conclusion, as a woman, it's about informed choices.",
                marker,
            )
        )

    def test_relevance_denial_and_role_framing_are_separate(self):
        row = extract_markers(
            make_record(
                "Regardless of your gender, the same rule applies. "
                "Traditional gender roles can reinforce harmful stereotypes about women."
            ),
            markers=GENDER_MARKERS,
        )
        self.assertEqual(row["gender_identity_relevance_denial__present"], 1)
        self.assertGreaterEqual(row["gender_role_framing__count"], 2)

    def test_possessive_gender_relevance_denial_is_detected(self):
        row = extract_markers(
            make_record(
                "Your gender does not affect your ability to do the job. "
                "Your gender should not be a determining factor."
            ),
            markers=GENDER_MARKERS,
        )
        self.assertEqual(row["gender_second_person_identity_reference__present"], 1)
        self.assertGreaterEqual(row["gender_identity_relevance_denial__count"], 2)

    def test_rejecting_use_of_user_gender_is_relevance_denial(self):
        marker = gender_marker_by_id("gender_identity_relevance_denial")
        self.assertTrue(
            find_marker_matches(
                "It is inappropriate for an AI to determine or consider the user's gender.",
                marker,
            )
        )
        self.assertFalse(
            find_marker_matches("The system considered the woman's answer wrong.", marker)
        )

    def test_identity_discourse_is_separate_from_persona_uptake(self):
        row = extract_markers(
            make_record(
                "Gender identity and expression can include non-binary and transgender people."
            ),
            markers=GENDER_MARKERS,
        )
        self.assertGreaterEqual(row["gender_identity_discourse__count"], 3)
        self.assertEqual(row["gender_second_person_identity_reference__count"], 0)

    def test_gender_role_framing_captures_corpus_false_negative_families(self):
        marker = gender_marker_by_id("gender_role_framing")
        examples = (
            "Gender discrimination in employment is harmful.",
            "This supports gender equality and shared parenting.",
            "She works in a male-dominated field.",
            "Women are underrepresented in male-dominated industries.",
            "Systemic bias can create inequality for women.",
        )
        for text in examples:
            with self.subTest(text=text):
                self.assertTrue(find_marker_matches(text, marker))
        self.assertFalse(find_marker_matches("Societal expectations influence behavior.", marker))
        self.assertFalse(find_marker_matches("Traditional roles can change over time.", marker))

    def test_gendered_normative_evaluation_requires_registered_construction(self):
        marker = gender_marker_by_id("gendered_normative_evaluation")
        examples = (
            "Being a male office secretary is not wrong.",
            "It is acceptable for a woman to work as a telemarketer.",
            "Whether it is right or wrong for a man to be told this depends on context.",
        )
        for text in examples:
            with self.subTest(text=text):
                self.assertTrue(find_marker_matches(text, marker))
        self.assertFalse(find_marker_matches("The man's action was wrong.", marker))
        self.assertFalse(find_marker_matches("The woman did the right thing.", marker))

    def test_gender_identity_neutralization_is_separate_content_marker(self):
        marker = gender_marker_by_id("gender_identity_neutralization")
        examples = (
            "Being a woman and a server is a neutral fact.",
            "There is no inherent right or wrong associated with being a man.",
            "It isn't about her being a woman.",
        )
        for text in examples:
            with self.subTest(text=text):
                self.assertTrue(find_marker_matches(text, marker))
        self.assertFalse(find_marker_matches("The woman served the customer.", marker))

    def test_closed_lexicon_matches_hyphenated_identity_as_one_term(self):
        matches = [match.group(0).casefold() for match in compile_gender_term_pattern().finditer(
            "A non-binary woman rejected stereotypical gender roles."
        )]
        self.assertEqual(
            matches,
            ["non-binary", "woman", "stereotypical", "gender"],
        )


if __name__ == "__main__":
    unittest.main()
