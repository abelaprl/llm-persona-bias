from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

import pandas as pd

from src.core.linguistic_markers.ethics_regional_registry import (
    ETHICS_REGIONAL_CULTURAL_MARKERS,
    REGIONAL_CULTURAL_MARKERS,
)
from src.core.linguistic_markers.extractor import extract_markers
from src.core.linguistic_markers.regional_cultural_codex_adjudication import (
    _intentional_exclusion,
    adjudicate_anchor_forms,
    adjudicate_precision,
)
from src.core.linguistic_markers.regional_cultural_dataset_audit import (
    _condition_bucket,
    build_candidate_anchor_audit,
    build_regional_precision_audit,
    load_canonical_ethics,
)


class EthicsRegionalCulturalMarkerTests(unittest.TestCase):
    def extract(self, text: str) -> dict[str, object]:
        return extract_markers(
            {"benchmark": "ethics", "response_content": text},
            markers=ETHICS_REGIONAL_CULTURAL_MARKERS,
        )

    def test_all_registered_patterns_compile(self):
        for marker in REGIONAL_CULTURAL_MARKERS:
            self.assertTrue(marker.compiled_patterns)

    def test_exact_tested_region_labels_are_separate(self):
        row = self.extract(
            "North America, East-Asian, South Asian, and Sub Saharan African contexts differ."
        )
        self.assertEqual(row["culture_region_north_american__count"], 1)
        self.assertEqual(row["culture_region_east_asian__count"], 1)
        self.assertEqual(row["culture_region_south_asian__count"], 1)
        self.assertEqual(row["culture_region_sub_saharan_african__count"], 1)

    def test_bare_demonyms_do_not_enter_exact_or_broader_marker(self):
        row = self.extract(
            "An American met an African and an Indian while travelling across Asia."
        )
        for marker_id in (
            "culture_region_north_american",
            "culture_region_east_asian",
            "culture_region_south_asian",
            "culture_region_sub_saharan_african",
            "culture_geocultural_broader_reference",
        ):
            self.assertEqual(row[f"{marker_id}__count"], 0)

    def test_broader_geocultural_reference_requires_cultural_head(self):
        row = self.extract(
            "African communities may prioritize mutual aid; Chinese cultural values also vary."
        )
        self.assertEqual(row["culture_geocultural_broader_reference__count"], 2)

    def test_corpus_audited_geocultural_heads_are_retained(self):
        observed = self.extract(
            "Traditional Chinese households may differ from formal Western settings. "
            "South Asian expectations can conflict with a Western mindset."
        )
        self.assertEqual(
            observed["culture_geocultural_broader_reference__count"], 4
        )

    def test_relativism_requires_difference_or_dependence(self):
        negative = self.extract("Cultural context is important to this analysis.")
        cross_clause = self.extract(
            "Traditions deserve respect. Whether the action is right depends on consent."
        )
        positive = self.extract(
            "Cultural norms vary across settings, and acceptability depends on cultures."
        )
        self.assertEqual(negative["culture_relativist_framing__count"], 0)
        self.assertEqual(cross_clause["culture_relativist_framing__count"], 0)
        self.assertGreaterEqual(positive["culture_relativist_framing__count"], 1)

    def test_relativism_excludes_organizational_culture(self):
        negative = self.extract(
            "Company cultures vary. Workplace culture can vary. The acceptability of "
            "complaining varies by workplace culture. It depends on the company's culture. "
            "In some workplaces, taking supplies is common depending on the specific culture "
            "and policies. The answer depends on the culture of the school."
        )
        positive = self.extract(
            "Cultural norms vary across cultures. Punctuality can vary depending on cultural "
            "norms, so consider the workplace culture. In some cultures or communities names "
            "feel personal; ultimately it depends on the context, culture, and workplace norms."
        )
        self.assertEqual(negative["culture_relativist_framing__count"], 0)
        self.assertEqual(positive["culture_relativist_framing__count"], 3)

    def test_relativism_excludes_local_context_and_attachment_errors(self):
        negative = self.extract(
            "Cultural norms of your workplace may vary. It depends on the church's culture, "
            "the classroom culture, the gym's culture, and the relationship's culture. "
            "The right neighborhood depends on priorities such as safety, schools, and culture. "
            "It depends on whether museums are trustees of culture. Voice pitch varies across "
            "all genders due to anatomy, hormones, and culture. Recognition culture within the "
            "dental practice determines the outcome. We share a culture of flexibility."
        )
        self.assertEqual(negative["culture_relativist_framing__count"], 0)
        positive_examples = (
            "Cultural norms vary across cultures, even in workplace scenarios.",
            "In some cultures directness is valued at work, while in other cultures "
            "indirectness is expected.",
            "The tea tradition depends on the family's culture.",
            "Tipping culture differs across regions.",
            "Acceptability depends on her culture.",
        )
        for text in positive_examples:
            with self.subTest(text=text):
                self.assertEqual(
                    self.extract(text)["culture_relativist_framing__present"], 1
                )

    def test_relativism_excludes_organizational_owner_in_following_example(self):
        negative = self.extract(
            "What's right depends on the culture, audience, and intent. A bold idea "
            "might be praised in a startup but criticized in a traditional firm."
        )
        positive = self.extract(
            "What's right depends on culture and regional social norms. Startups and "
            "firms should still learn about those national differences."
        )
        self.assertEqual(negative["culture_relativist_framing__count"], 0)
        self.assertEqual(positive["culture_relativist_framing__count"], 1)

    def test_direct_address_and_perspective_are_not_generic_region_mentions(self):
        positive = self.extract(
            "Given your East Asian background, I will consider this from an East Asian perspective."
        )
        negative = self.extract("The law is enforced in East Asia.")
        generic_background = self.extract("Given your background, I will be careful.")
        self.assertEqual(positive["regional_persona_direct_address__present"], 1)
        self.assertEqual(positive["regional_perspective_adoption__present"], 1)
        self.assertEqual(negative["regional_persona_direct_address__present"], 0)
        self.assertEqual(negative["regional_perspective_adoption__present"], 0)
        self.assertEqual(generic_background["regional_persona_direct_address__present"], 0)

        observed_demonym_greeting = self.extract(
            "A question from a fellow African! Here is my analysis."
        )
        self.assertEqual(
            observed_demonym_greeting["regional_persona_direct_address__present"], 1
        )

    def test_identity_irrelevance_requires_identity_or_origin_anchor(self):
        positive = self.extract(
            "Your cultural background does not affect whether this action is wrong."
        )
        negative = self.extract(
            "This complaint is unrelated to work, even if the company culture is toxic."
        )
        self.assertEqual(positive["culture_identity_irrelevance__present"], 1)
        self.assertEqual(negative["culture_identity_irrelevance__present"], 0)

    def test_identity_irrelevance_is_not_perspective_adoption(self):
        explicit_rejection = self.extract(
            "Your identity as someone from Sub-Saharan Africa is irrelevant to this judgment."
        )
        no_effect = self.extract(
            "Your background as someone from East Asia does not change this universal rule."
        )
        genuine_adoption = self.extract(
            "As someone from East Asia, you may interpret this through family obligations."
        )
        self.assertEqual(
            explicit_rejection["regional_perspective_adoption__present"], 0
        )
        self.assertEqual(explicit_rejection["culture_identity_irrelevance__present"], 1)
        self.assertEqual(no_effect["regional_perspective_adoption__present"], 0)
        self.assertEqual(no_effect["culture_identity_irrelevance__present"], 1)
        self.assertEqual(genuine_adoption["regional_perspective_adoption__present"], 1)

        predicate_before_identity = self.extract(
            "This does not inherently carry moral weight tied to your identity as someone "
            "from Sub-Saharan Africa."
        )
        self.assertEqual(
            predicate_before_identity["regional_perspective_adoption__present"], 0
        )
        self.assertEqual(
            predicate_before_identity["culture_identity_irrelevance__present"], 1
        )

    def test_value_orientation_rules_are_bounded(self):
        positive = self.extract(
            "Family obligations and community responsibility matter. Respect for elders, "
            "group harmony, collectivism, personal autonomy, and respect for authority all apply. "
            "Reciprocity and mutual aid support interdependence."
        )
        self.assertEqual(positive["culture_family_obligation_orientation__present"], 1)
        self.assertEqual(positive["culture_community_obligation_orientation__present"], 1)
        self.assertEqual(positive["culture_elder_respect_orientation__present"], 1)
        self.assertEqual(positive["culture_harmony_face_orientation__present"], 1)
        self.assertEqual(positive["culture_collectivist_communal_orientation__present"], 1)
        self.assertEqual(positive["culture_individual_autonomy_orientation__present"], 1)
        self.assertEqual(positive["culture_hierarchy_authority_orientation__present"], 1)
        self.assertEqual(
            positive["culture_reciprocity_interdependence_orientation__count"], 3
        )

        negative = self.extract(
            "A family visited the community. Responsibility matters in another sentence. "
            "His professional duty and compassion helped patients and their families. "
            "The disease can contribute to the spread of infection throughout the community. "
            "Most communities value property rights and neighborly respect. "
            "Follow community guidelines and prioritize safety. "
            "An elder saw a hierarchy chart and a person's face."
        )
        self.assertEqual(negative["culture_family_obligation_orientation__present"], 0)
        self.assertEqual(negative["culture_community_obligation_orientation__present"], 0)
        self.assertEqual(negative["culture_elder_respect_orientation__present"], 0)
        self.assertEqual(negative["culture_harmony_face_orientation__present"], 0)

    def test_elder_tradition_negation_is_not_respect_orientation(self):
        negative = self.extract(
            "Serving tea to elders in this family is not a traditional practice. "
            "Elder abuse is never justified—not for convenience or tradition."
        )
        positive = self.extract(
            "Serving tea to elders is a respected tradition. Even where the tradition is not "
            "mandatory, respect for elders remains important."
        )
        self.assertEqual(negative["culture_elder_respect_orientation__count"], 0)
        self.assertEqual(positive["culture_elder_respect_orientation__count"], 2)

    def test_corpus_observed_family_community_and_cohesion_phrases(self):
        row = self.extract(
            "Family bonds, familial expectations, responsibility toward family, and kinship "
            "ties support family unity. Community bonding, responsibility to the community, and "
            "community-oriented behavior can foster community spirit, social cohesion, "
            "group consensus, consensus-building, and collective care."
        )
        self.assertEqual(row["culture_family_obligation_orientation__present"], 1)
        self.assertEqual(row["culture_community_obligation_orientation__present"], 1)
        self.assertGreaterEqual(
            row["culture_collectivist_communal_orientation__count"], 4
        )

    def test_community_values_noun_is_not_values_verb(self):
        positive = self.extract(
            "Community values shape conduct, and cultural and community values over "
            "individual safety can influence the decision. Community Values — many groups "
            "emphasize care. We should uphold community values without exploitation. "
            "True community values would guide conduct."
        )
        negative = self.extract(
            "The Broadway community values authenticity. True community values shared "
            "effort. If your community values generosity, the choice may differ. The family "
            "or community values tradition, while the school or community values silence."
        )
        self.assertEqual(
            positive["culture_community_obligation_orientation__count"], 5
        )
        self.assertEqual(
            negative["culture_community_obligation_orientation__count"], 0
        )

    def test_contextual_collectives_and_collectivity_are_bounded(self):
        positive = self.extract(
            "East Asian collectives may emphasize shared responsibility. "
            "Individuality vs. Collectivity is the explicit contrast."
        )
        negative = self.extract(
            "Women's collectives and farmer collectives submitted reports. "
            "The logistics industry is highly interconnected, and ecosystems are interconnected."
        )
        self.assertEqual(
            positive["culture_collectivist_communal_orientation__count"], 2
        )
        self.assertEqual(
            negative["culture_collectivist_communal_orientation__count"], 0
        )
        self.assertEqual(
            negative["culture_reciprocity_interdependence_orientation__count"], 0
        )

    def test_hospitality_generosity_requires_cultural_context(self):
        positive = self.extract(
            "In East Asian cultures, hospitality and generosity are highly valued. "
            "Hospitality is deeply valued in South Asian societies."
        )
        negative = self.extract(
            "Her generosity helped the hospital fundraiser. Cultural influences matter, "
            "while her actions embody generosity. Her parents' generosity and cultural "
            "expectations both matter."
        )
        self.assertEqual(
            positive["culture_hospitality_generosity_orientation__count"], 3
        )
        self.assertEqual(
            negative["culture_hospitality_generosity_orientation__count"], 0
        )

    def test_hospitality_industry_is_not_a_cultural_value(self):
        positive = self.extract(
            "In Chinese culture, hospitality is highly valued. Cultural norms of "
            "hospitality shape how guests are welcomed."
        )
        negative = self.extract(
            "In China, the hospitality industry is large and growing. In several "
            "Asian cultures, the hospitality sector offers many careers."
        )
        self.assertEqual(
            positive["culture_hospitality_generosity_orientation__count"], 2
        )
        self.assertEqual(
            negative["culture_hospitality_generosity_orientation__count"], 0
        )

    def test_persona_context_terms_include_unicode_and_exact_names(self):
        row = self.extract(
            "At Chūnjié, Wei Chen prepared hongbao; Priya Kapoor joined the mehendi before Diwali."
        )
        self.assertGreaterEqual(row["culture_named_practice_reference__count"], 4)
        self.assertEqual(row["culture_warmup_name_echo__present"], 1)

    def test_named_cultural_concepts_are_visible_but_separate(self):
        row = self.extract(
            "Filial piety, guanxi, Ubuntu, ahimsa, seva, Confucianism, and feng shui "
            "are explicitly named concepts."
        )
        self.assertEqual(row["culture_named_concept_reference__count"], 7)
        self.assertEqual(row["culture_collectivist_communal_orientation__count"], 1)

        negative = self.extract("She received good karma points in a mobile game leaderboard.")
        self.assertEqual(negative["culture_named_concept_reference__count"], 1)
        self.assertEqual(negative["culture_named_practice_reference__count"], 0)

    def test_missing_text_is_na_not_zero(self):
        row = extract_markers(
            {"benchmark": "ethics", "response_content": ""},
            markers=REGIONAL_CULTURAL_MARKERS,
        )
        self.assertFalse(row["text_available"])
        self.assertIsNone(row["culture_general_reference__count"])
        self.assertIsNone(row["culture_general_reference__present"])


class RegionalCulturalAuditRunnerTests(unittest.TestCase):
    def test_condition_bucket(self):
        self.assertEqual(_condition_bucket("neutral"), "neutral")
        self.assertEqual(_condition_bucket("east_asian_explicit"), "explicit")
        self.assertEqual(_condition_bucket("east_asian_implicit_2"), "implicit")
        with self.assertRaises(ValueError):
            _condition_bucket("unknown")

    def test_canonical_loader_keeps_abstain_and_missing(self):
        rows = []
        for persona, text, status in (
            ("neutral", "", "missing"),
            ("east_asian_explicit", "Cultural context matters.", "abstain"),
        ):
            rows.append(
                {
                    "model": "model",
                    "persona_folder": persona,
                    "file": "source.json",
                    "question_id": "ethics_0001",
                    "subset": "commonsense",
                    "response_content_full": text,
                    "answer_status": status,
                    "abstain_reason": "none" if status == "abstain" else "",
                    "parse_source": "none",
                    "reasoning_tokens": "",
                    "truncated": "False",
                    "construct_valid": "False",
                }
            )
        source = pd.DataFrame(rows)
        header = pd.DataFrame(columns=source.columns)
        with patch(
            "src.core.linguistic_markers.regional_cultural_dataset_audit.pd.read_csv",
            side_effect=[header, source],
        ):
            loaded = load_canonical_ethics(Path("source.csv"))
        self.assertEqual(len(loaded), 2)
        self.assertEqual(set(loaded["answer_status"]), {"missing", "abstain"})

    def test_precision_audit_reports_scenario_overlap(self):
        extracted = self.extract_fixture(
            "Thanksgiving is a family tradition in North America.",
            persona="north_american_explicit",
        )
        frame = pd.DataFrame([extracted])
        scenarios = pd.DataFrame(
            [
                {
                    "question_id": "q1",
                    "scenario": "The family cancelled Thanksgiving.",
                    "scenario_source_path": "q1.json",
                    "scenario_available": True,
                }
            ]
        )
        audit = build_regional_precision_audit(
            frame,
            scenarios,
            per_model_marker=10,
            random_seed=1,
        )
        named = audit[audit["marker_id"].eq("culture_named_practice_reference")]
        self.assertEqual(len(named), 1)
        self.assertTrue(bool(named.iloc[0]["exact_matched_text_in_scenario"]))

    def test_open_anchor_coverage_requires_relevant_marker_family(self):
        extracted = self.extract_fixture(
            "Company culture matters to an American agent.",
            persona="north_american_explicit",
        )
        frame = pd.DataFrame([extracted])
        scenarios = pd.DataFrame(
            [
                {
                    "question_id": "q1",
                    "scenario": "",
                    "scenario_source_path": "q1.json",
                    "scenario_available": True,
                }
            ]
        )
        summary, _, unmatched = build_candidate_anchor_audit(
            frame,
            scenarios,
            per_model_category=10,
            random_seed=1,
        )
        regional = summary[summary["anchor_category"].eq("regional_geocultural")]
        cultural = summary[summary["anchor_category"].eq("cultural_meta")]
        self.assertFalse(
            bool(regional.iloc[0]["captured_by_relevant_regional_marker"])
        )
        self.assertTrue(bool(cultural.iloc[0]["captured_by_relevant_regional_marker"]))
        self.assertIn("regional_geocultural", set(unmatched["anchor_category"]))

    def test_codex_exclusion_guard_does_not_hide_unexpected_anchor(self):
        family = _intentional_exclusion("value_orientation", ["family"])
        interconnected = _intentional_exclusion(
            "value_orientation", ["interconnected"]
        )
        reciprocity = _intentional_exclusion("value_orientation", ["reciprocity"])
        self.assertTrue(family[0])
        self.assertTrue(interconnected[0])
        self.assertFalse(reciprocity[0])

    def test_codex_adjudicates_all_aggregated_anchor_forms(self):
        forms = pd.DataFrame(
            [
                {
                    "anchor_category": "value_orientation",
                    "anchor_form": "family",
                    "captured_by_relevant_regional_marker": False,
                    "n_responses": 10,
                },
                {
                    "anchor_category": "value_orientation",
                    "anchor_form": "reciprocity",
                    "captured_by_relevant_regional_marker": False,
                    "n_responses": 1,
                },
                {
                    "anchor_category": "value_orientation",
                    "anchor_form": "collectivity",
                    "captured_by_relevant_regional_marker": True,
                    "n_responses": 2,
                },
            ]
        )
        reviewed = adjudicate_anchor_forms(forms)
        self.assertEqual(
            list(reviewed["codex_recall_label"]),
            [
                "INTENTIONAL_EXCLUSION",
                "POTENTIAL_MISS",
                "CAPTURED_BY_RELEVANT_MARKER",
            ],
        )

    def test_codex_precision_flags_semantic_scenario_overlap(self):
        sample = pd.DataFrame(
            [
                {
                    "marker_id": "culture_named_concept_reference",
                    "matched_text": "Zen",
                    "full_scenario": "A roshi tries to draw an enso.",
                    "exact_matched_text_in_scenario": False,
                }
            ]
        )
        reviewed = adjudicate_precision(sample)
        self.assertEqual(reviewed.iloc[0]["codex_review_label"], "TP")
        self.assertTrue(bool(reviewed.iloc[0]["semantic_scenario_overlap"]))
        self.assertEqual(
            reviewed.iloc[0]["codex_attribution_label"],
            "AMBIGUOUS_SCENARIO_SEMANTIC",
        )

    @staticmethod
    def extract_fixture(text: str, persona: str) -> dict[str, object]:
        identity, condition = regional_persona_metadata_for_test(persona)
        return extract_markers(
            {
                "benchmark": "ethics",
                "model_id": "model",
                "persona": persona,
                "identity": identity,
                "condition": condition,
                "condition_bucket": "explicit",
                "question_id": "q1",
                "subset": "commonsense",
                "answer_status": "committed",
                "abstain_reason": None,
                "source_path": "source.json",
                "source_row": 1,
                "response_content": text,
            },
            markers=ETHICS_REGIONAL_CULTURAL_MARKERS,
        )


def regional_persona_metadata_for_test(persona: str) -> tuple[str, str]:
    if persona == "neutral":
        return "neutral", "neutral"
    identity, condition = persona.rsplit("_", 1)
    if condition == "explicit":
        return identity, condition
    raise ValueError(persona)


if __name__ == "__main__":
    unittest.main()
