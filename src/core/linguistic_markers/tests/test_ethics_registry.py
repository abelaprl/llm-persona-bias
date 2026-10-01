from __future__ import annotations

import unittest

from src.core.linguistic_markers.ethics_registry import (
    ETHICS_MARKERS,
    ethics_marker_by_id,
    moral_polarity_sequence,
)
from src.core.linguistic_markers.extractor import (
    extract_markers,
    find_marker_matches,
)


def make_record(text: str) -> dict[str, object]:
    return {
        "model_id": "test-model",
        "benchmark": "ethics",
        "subset": "commonsense",
        "identity": "neutral",
        "condition": "neutral",
        "persona": "neutral",
        "question_id": "q1",
        "response_content": text,
    }


class EthicsRegistryTests(unittest.TestCase):
    def test_contextual_hedge_captures_strong_frames_not_bare_modals(self):
        row = extract_markers(
            make_record(
                "It depends on the context. Without more information, it is hard to judge."
            ),
            markers=ETHICS_MARKERS,
        )
        self.assertGreaterEqual(row["stance_contextual_hedge__count"], 3)
        marker = ethics_marker_by_id("stance_contextual_hedge")
        self.assertFalse(find_marker_matches("They could calculate the result.", marker))

    def test_distancing_requires_an_attributed_viewpoint(self):
        marker = ethics_marker_by_id("stance_distancing")
        self.assertTrue(find_marker_matches("Some might argue that the choice is justified.", marker))
        self.assertTrue(find_marker_matches("From a legal perspective, this differs.", marker))
        self.assertFalse(find_marker_matches("My perspective is different.", marker))

    def test_additive_excludes_bare_also(self):
        marker = ethics_marker_by_id("connective_additive")
        self.assertTrue(find_marker_matches("Furthermore, another important factor is safety.", marker))
        self.assertFalse(find_marker_matches("It is also safe.", marker))

    def test_response_stance_is_linguistic_only(self):
        base = make_record("The action is clearly wrong. In some contexts it could differ.")
        first = extract_markers({**base, "ground_truth": 0}, markers=ETHICS_MARKERS)
        second = extract_markers({**base, "ground_truth": 1}, markers=ETHICS_MARKERS)
        self.assertEqual(first["response_stance_committed__present"], 1)
        self.assertEqual(
            {key: value for key, value in first.items() if "__" in key},
            {key: value for key, value in second.items() if "__" in key},
        )

    def test_refusal_and_empathy_are_narrow(self):
        row = extract_markers(
            make_record(
                "I cannot provide a binary judgment without more context. "
                "It is understandable that you feel upset; your concern is valid."
            ),
            markers=ETHICS_MARKERS,
        )
        self.assertEqual(row["response_stance_refusal__present"], 1)
        self.assertGreaterEqual(row["affect_empathy_validation__count"], 2)

    def test_content_framing_families_remain_separate(self):
        row = extract_markers(
            make_record(
                "A professional duty to avoid harm reflects integrity and compassion. "
                "Cultural norms and religious teachings can differ."
            ),
            markers=ETHICS_MARKERS,
        )
        expected = (
            "framing_deontological",
            "framing_consequentialist",
            "framing_virtue_ethics",
            "framing_cultural",
            "framing_religious",
        )
        self.assertTrue(all(row[f"{marker_id}__present"] == 1 for marker_id in expected))

    def test_moral_polarity_sequence_uses_positioned_evaluations(self):
        result = moral_polarity_sequence(
            "At first, the action seems acceptable. After considering the harm, it is wrong."
        )
        self.assertEqual(result["moral_polarity_initial"], "positive")
        self.assertEqual(result["moral_polarity_final"], "negative")
        self.assertEqual(result["moral_polarity_shift"], 1)
        self.assertEqual(result["moral_polarity_direction"], "positive_to_negative")

    def test_bare_right_and_wrong_are_not_polarity_markers(self):
        positive = ethics_marker_by_id("moral_evaluation_positive")
        negative = ethics_marker_by_id("moral_evaluation_negative")
        self.assertFalse(find_marker_matches("Turn right and correct the wrong number.", positive))
        self.assertFalse(find_marker_matches("Turn right and correct the wrong number.", negative))

    def test_moral_polarity_rejects_questions_negation_and_technical_uses(self):
        positive = ethics_marker_by_id("moral_evaluation_positive")
        negative = ethics_marker_by_id("moral_evaluation_negative")
        self.assertFalse(
            find_marker_matches("The question is whether this is right or wrong.", positive)
        )
        self.assertFalse(
            find_marker_matches("There is no scenario in which this is acceptable.", positive)
        )
        self.assertFalse(
            find_marker_matches("The statement is wrong because the grammar is incorrect.", negative)
        )
        self.assertFalse(
            find_marker_matches("Binary classification is inappropriate here.", negative)
        )
        self.assertFalse(
            find_marker_matches(
                'Is the Label Correct? No, it is wrong. The label does not fit.', negative
            )
        )
        self.assertFalse(find_marker_matches("It doesn't seem wrong.", negative))
        self.assertTrue(find_marker_matches("The request is not justified.", negative))
        self.assertTrue(
            find_marker_matches(
                "If the decision is made with caution, it would be appropriate. "
                "Therefore, the binary answer is 1.",
                positive,
            )
        )

    def test_committed_stance_rejects_option_pairs_and_catches_numeric_commitment(self):
        marker = ethics_marker_by_id("response_stance_committed")
        self.assertFalse(
            find_marker_matches("It is difficult to say whether the action is right or wrong.", marker)
        )
        self.assertFalse(
            find_marker_matches(
                'A 1 would imply the behavior is acceptable, but the answer is 0.', marker
            )
        )
        self.assertFalse(
            find_marker_matches(
                "If the request is justified by need, then it is ethically sound.", marker
            )
        )
        self.assertTrue(find_marker_matches("After reviewing the case, I would answer: 0.", marker))
        self.assertTrue(
            find_marker_matches(
                "If I *must* answer with 0 or 1 as requested: **0** (not wrong).", marker
            )
        )

    def test_empathy_requires_user_or_feeling_anchor(self):
        marker = ethics_marker_by_id("affect_empathy_validation")
        self.assertTrue(find_marker_matches("Your feelings are valid.", marker))
        self.assertTrue(
            find_marker_matches("It is reasonable to feel upset about the exclusion.", marker)
        )
        self.assertFalse(
            find_marker_matches("Her fear reflects insecurity rather than a legitimate concern.", marker)
        )
        self.assertFalse(
            find_marker_matches("Your roommate downstairs has a legitimate concern about sleep.", marker)
        )
        self.assertTrue(
            find_marker_matches("Your concern about his behavior is a legitimate concern.", marker)
        )
        self.assertFalse(find_marker_matches("Identity statements like this are valid.", marker))

    def test_recall_audit_variants_are_registered(self):
        conditional = ethics_marker_by_id("response_stance_conditional")
        refusal = ethics_marker_by_id("response_stance_refusal")
        self.assertTrue(find_marker_matches("The verdict is context-dependent.", conditional))
        self.assertTrue(
            find_marker_matches(
                "A classic example of a moral dilemma: it is not necessarily right or wrong.",
                conditional,
            )
        )
        self.assertTrue(
            find_marker_matches("It can be appropriate under certain conditions.", conditional)
        )
        self.assertTrue(
            find_marker_matches(
                'It is not inherently "right" or "wrong." It depends on individual circumstances.',
                conditional,
            )
        )
        self.assertFalse(
            find_marker_matches("Spanish is spoken in certain contexts.", conditional)
        )
        self.assertTrue(
            find_marker_matches("I can't assign a moral judgment because context is missing.", refusal)
        )


if __name__ == "__main__":
    unittest.main()
