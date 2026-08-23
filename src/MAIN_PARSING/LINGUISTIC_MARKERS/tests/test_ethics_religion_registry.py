from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from src.MAIN_PARSING.LINGUISTIC_MARKERS.ethics_religion_registry import (
    ETHICS_RELIGION_MARKERS,
    RELIGION_MARKERS,
)
from src.MAIN_PARSING.LINGUISTIC_MARKERS.extractor import extract_markers
from src.MAIN_PARSING.LINGUISTIC_MARKERS.religion_ethics_dataset_audit import (
    _require_openpyxl,
    _restore_incremental_audit_metadata,
    build_candidate_anchor_audit,
)


class EthicsReligionRegistryTests(unittest.TestCase):
    def extract(self, text: str) -> dict[str, object]:
        return extract_markers(
            {"benchmark": "ethics", "response_content": text},
            markers=ETHICS_RELIGION_MARKERS,
        )

    def test_all_patterns_compile(self):
        for marker in RELIGION_MARKERS:
            self.assertTrue(marker.compiled_patterns)

    def test_four_identity_families_are_separate(self):
        row = self.extract(
            "Muslims discussed Islam with Christians, a Jewish neighbor, and atheists."
        )
        self.assertEqual(row["religion_identity_islam__count"], 2)
        self.assertEqual(row["religion_identity_christian__count"], 1)
        self.assertEqual(row["religion_identity_jewish__count"], 1)
        self.assertEqual(row["religion_identity_atheist__count"], 1)

    def test_deity_figure_and_scripture_are_distinct(self):
        row = self.extract(
            "The Prophet Muhammad cited the Quran; Jesus Christ and the Bible were also named."
        )
        self.assertGreaterEqual(row["religion_figure_reference__count"], 2)
        self.assertGreaterEqual(row["religion_deity_invocation__count"], 1)
        self.assertEqual(row["religion_scripture_reference__count"], 2)

    def test_generic_faith_excludes_nonreligious_polysemy(self):
        positive = self.extract("Her religious faith remains important to her.")
        negative = self.extract(
            "The report was submitted in good faith, and he remained faithful to the contract."
        )
        self.assertEqual(positive["religion_generic_reference__count"], 2)
        self.assertEqual(negative["religion_generic_reference__count"], 0)

    def test_creator_requires_divine_title_or_religious_context(self):
        nonreligious = self.extract(
            "The creator of the app explained the creator's artistic intent."
        )
        divine_title = self.extract("The Creator guides creation.")
        divine_context = self.extract("Allah is the creator of all beings.")
        self.assertEqual(nonreligious["religion_deity_invocation__count"], 0)
        self.assertEqual(divine_title["religion_deity_invocation__count"], 1)
        self.assertGreaterEqual(divine_context["religion_deity_invocation__count"], 2)

    def test_scripture_citations_require_formal_reference(self):
        positive = self.extract(
            "Matthew 22:39 and Quran 4:19 are cited; the Bible teaches compassion."
        )
        negative = self.extract("James and Mark discussed an ordinary book.")
        self.assertGreaterEqual(positive["religion_scripture_reference__count"], 3)
        self.assertEqual(negative["religion_scripture_reference__count"], 0)

    def test_practice_and_prompt_context_terms_are_visible(self):
        row = self.extract(
            "During Ramadan they prayed at a mosque; on Shabbat they visited a synagogue. "
            "Christmas, Easter, Lent, Yom Kippur, Kaddish, and communion were also named."
        )
        self.assertGreaterEqual(row["religion_practice_community__count"], 11)

    def test_practice_polysemy_and_audited_recall_forms(self):
        positive = self.extract(
            "During Lent, she prayerfully considered two mitzvahs and remained worshipful."
        )
        negative = self.extract("She lent a charger to her neighbor.")
        self.assertEqual(positive["religion_practice_community__count"], 4)
        self.assertEqual(negative["religion_practice_community__count"], 0)

    def test_afterlife_marker_excludes_organization_and_idioms(self):
        positive = self.extract(
            "Christian doctrine teaches salvation through Christ, redemption from sin, "
            "and an immortal soul in heaven rather than hell."
        )
        negative = self.extract(
            "I donated to the Salvation Army. What the hell? The soul-crushing redemption "
            "arc was a match made in heaven."
        )
        self.assertGreaterEqual(positive["religion_afterlife_soteriology__count"], 6)
        self.assertEqual(negative["religion_afterlife_soteriology__count"], 0)

    def test_named_concepts_are_not_generic_values(self):
        positive = self.extract(
            "Tawheed, shirk, adab, ihsan, pikuach nefesh, tikkun olam, halacha, and "
            "tzniut were explicitly invoked."
        )
        negative = self.extract("Compassion, justice, humility, and stewardship matter.")
        self.assertEqual(positive["religion_named_concept_reference__count"], 8)
        self.assertEqual(negative["religion_named_concept_reference__count"], 0)

    def test_audited_named_concept_and_scripture_variants_are_registered(self):
        row = self.extract(
            "Quranic teachings mention the Ten Commandments; halakhah was discussed halachically, "
            "alongside kavod habrit. Insha'Allah."
        )
        self.assertGreaterEqual(row["religion_scripture_reference__count"], 2)
        self.assertEqual(row["religion_named_concept_reference__count"], 3)
        self.assertEqual(row["religion_formulaic_expression__count"], 1)

    def test_formulae_and_blessing_are_separate(self):
        row = self.extract(
            "Assalamu alaikum. Inshallah, God willing. May Allah guide you. Amen."
        )
        self.assertEqual(row["religion_formulaic_expression__count"], 4)
        self.assertEqual(row["religion_blessing_invocation__count"], 1)

    def test_direct_address_requires_user_anchor(self):
        positive = self.extract(
            "Given your Muslim faith, I will respond respectfully. Since you are an atheist, "
            "I will not assume a religious framework."
        )
        negative = self.extract("Many Muslims and atheists discuss ethical frameworks.")
        self.assertEqual(positive["religion_persona_direct_address__present"], 1)
        self.assertEqual(negative["religion_persona_direct_address__present"], 0)

    def test_perspective_adoption_is_not_identity_irrelevance(self):
        adopted = self.extract(
            "From a Christian perspective, the action conflicts with biblical principles."
        )
        rejected = self.extract(
            "Your religious identity as a Muslim does not change this practical judgment."
        )
        self.assertEqual(adopted["religion_perspective_adoption__present"], 1)
        self.assertEqual(adopted["religion_moral_authority_framing__present"], 1)
        self.assertEqual(rejected["religion_perspective_adoption__present"], 0)
        self.assertEqual(rejected["religion_identity_irrelevance__present"], 1)

    def test_perspective_adoption_excludes_third_party_disagreement(self):
        adopted = self.extract(
            "From a religious perspective, the action violates a moral duty."
        )
        mentioned = self.extract(
            "Some people may disagree from a religious perspective, but I use a legal test."
        )
        self.assertEqual(adopted["religion_perspective_adoption__present"], 1)
        self.assertEqual(mentioned["religion_perspective_adoption__present"], 0)

    def test_authority_requires_relation_not_bare_reference(self):
        positive = self.extract(
            "The Bible teaches mercy, and Islamic principles prohibit exploitation."
        )
        negative = self.extract("The Bible and Quran are books on the shelf.")
        self.assertEqual(positive["religion_moral_authority_framing__count"], 2)
        self.assertEqual(negative["religion_moral_authority_framing__count"], 0)

    def test_secular_humanist_framing_is_bounded(self):
        positive = self.extract(
            "Secular ethics and humanism provide a non-religious moral framework."
        )
        negative = self.extract("The secular school teaches research methods.")
        self.assertGreaterEqual(positive["religion_secular_humanist_framing__count"], 2)
        self.assertEqual(negative["religion_secular_humanist_framing__count"], 0)

    def test_pluralism_requires_religious_axis(self):
        positive = self.extract(
            "Different religions may disagree; ethical judgments vary across religious traditions."
        )
        negative = self.extract("Different people and cultures may disagree.")
        self.assertGreaterEqual(positive["religion_pluralist_framing__count"], 2)
        self.assertEqual(negative["religion_pluralist_framing__count"], 0)

    def test_identity_irrelevance_is_bidirectional(self):
        first = self.extract("Your Jewish identity is irrelevant to the safety judgment.")
        second = self.extract("This is unrelated to your religious beliefs.")
        self.assertEqual(first["religion_identity_irrelevance__present"], 1)
        self.assertEqual(second["religion_identity_irrelevance__present"], 1)

    def test_respect_acknowledgement_requires_religious_stance(self):
        positive = self.extract(
            "Regarding your atheism, I respect your perspective and will remain neutral."
        )
        negative = self.extract(
            "I respect your perspective about the vacation schedule and room assignment."
        )
        self.assertEqual(positive["religion_respect_acknowledgement__present"], 1)
        self.assertEqual(negative["religion_respect_acknowledgement__present"], 0)

    def test_missing_text_is_na_not_zero(self):
        row = extract_markers(
            {"benchmark": "ethics", "response_content": ""},
            markers=RELIGION_MARKERS,
        )
        self.assertFalse(row["text_available"])
        self.assertIsNone(row["religion_generic_reference__count"])

    def test_excel_dependency_is_checked_before_full_run(self):
        with patch(
            "src.MAIN_PARSING.LINGUISTIC_MARKERS.religion_ethics_dataset_audit."
            "importlib.util.find_spec",
            return_value=None,
        ):
            with self.assertRaisesRegex(ModuleNotFoundError, "openpyxl is required"):
                _require_openpyxl()

    def test_open_anchor_screen_uses_stripped_marker_surface(self):
        record = {
            "text_available": True,
            "visible_text": "Answer: 1 (sin)",
            "model_id": "model-a",
            "persona": "neutral",
            "condition": "neutral",
            "subset": "commonsense",
            "question_id": "ethics_commonsense_0001",
            "answer_status": "committed",
        }
        record.update(
            {f"{marker.marker_id}__present": 0 for marker in RELIGION_MARKERS}
        )
        frame = pd.DataFrame([record])
        scenarios = pd.DataFrame(
            {
                "question_id": ["ethics_commonsense_0001"],
                "scenario": ["A neutral scenario."],
            }
        )
        summary, forms, unmatched = build_candidate_anchor_audit(
            frame,
            scenarios,
            per_model_category=5,
            random_seed=123,
        )
        self.assertTrue(summary.empty)
        self.assertTrue(forms.empty)
        self.assertTrue(unmatched.empty)

    def test_incremental_audit_restores_blind_recall_source_metadata(self):
        response = pd.DataFrame(
            {
                "model_id": ["model-a", "model-b"],
                "persona": ["neutral", "islam_explicit"],
                "question_id": ["q1", "q2"],
            }
        )
        canonical = pd.DataFrame(
            {
                "model": ["model-a", "model-b"],
                "persona_folder": ["neutral", "islam_explicit"],
                "question_id": ["q1", "q2"],
                "file": ["q1.json", "q2.json"],
            }
        )
        with patch(
            "src.MAIN_PARSING.LINGUISTIC_MARKERS.religion_ethics_dataset_audit."
            "pd.read_csv",
            return_value=canonical,
        ):
            _restore_incremental_audit_metadata(response, "canonical.csv")
        self.assertEqual(response["benchmark"].tolist(), ["ethics", "ethics"])
        self.assertEqual(response["source_path"].tolist(), ["q1.json", "q2.json"])
        self.assertEqual(response["source_row"].tolist(), [2, 3])


if __name__ == "__main__":
    unittest.main()
