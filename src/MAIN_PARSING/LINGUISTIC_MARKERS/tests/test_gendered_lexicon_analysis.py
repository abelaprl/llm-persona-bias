from __future__ import annotations

import unittest

import pandas as pd

from scripts.build_gendered_lexicon_analysis import (
    _paired_pronoun_classes,
    _paired_pronoun_gender_direct,
    tokenize_gender_terms,
)
from src.MAIN_PARSING.LINGUISTIC_MARKERS.config import GENDER_PERSONAS


class GenderedLexiconTests(unittest.TestCase):
    def test_gender_tokens_use_word_boundaries_and_keep_requested_pronouns(self):
        counts = tokenize_gender_terms(
            "She gave her book to him; he thanked the woman. "
            "The theme should not create an extra substring match."
        )
        self.assertEqual(counts["she"], 1)
        self.assertEqual(counts["her"], 1)
        self.assertEqual(counts["him"], 1)
        self.assertEqual(counts["he"], 1)
        self.assertEqual(counts["woman"], 1)
        self.assertNotIn("the", counts)

    def test_pronoun_pairing_keeps_all_proxy_contrasts_separate(self):
        rows = []
        for persona_index, persona in enumerate(GENDER_PERSONAS):
            for question_index in range(3):
                rows.append(
                    {
                        "model_id": "model",
                        "persona": persona,
                        "question_id": f"gsm8k_{question_index:05d}",
                        "masculine_pronoun": int(persona_index > 0 and question_index == 0),
                        "feminine_pronoun": int(persona_index > 4 and question_index == 1),
                    }
                )
        result = _paired_pronoun_classes(
            pd.DataFrame(rows),
            bootstrap_resamples=0,
        )
        self.assertEqual(len(result), 28)
        selected = result[
            (result["identity"] == "woman")
            & (result["pronoun_class"] == "feminine_pronoun")
        ]
        self.assertEqual(selected["contrast_id"].nunique(), 7)
        self.assertTrue((selected["n_pairs"] == 3).all())

    def test_direct_pronoun_pairing_compares_man_and_woman_per_condition(self):
        rows = []
        for persona_index, persona in enumerate(GENDER_PERSONAS):
            for question_index in range(3):
                rows.append(
                    {
                        "model_id": "model",
                        "persona": persona,
                        "question_id": f"gsm8k_{question_index:05d}",
                        "masculine_pronoun": int(
                            persona.startswith("man_") and question_index == 0
                        ),
                        "feminine_pronoun": int(
                            persona.startswith("woman_") and question_index == 1
                        ),
                    }
                )
        result = _paired_pronoun_gender_direct(
            pd.DataFrame(rows),
            bootstrap_resamples=0,
        )
        self.assertEqual(len(result), 8)
        self.assertEqual(
            set(result["condition"]),
            {"explicit", "implicit_1", "implicit_2", "implicit_3"},
        )
        self.assertTrue((result["n_pairs"] == 3).all())
        masculine = result[result["pronoun_class"].eq("masculine_pronoun")]
        self.assertTrue((masculine["prevalence_delta"] > 0).all())


if __name__ == "__main__":
    unittest.main()
