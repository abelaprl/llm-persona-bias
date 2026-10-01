"""Build the ETHICS Regional-Cultural visual supplement.

This configures the shared paired-effect builder used by the Religion
notebook.  Linguistic-marker results remain descriptive while the
Regional-Cultural registry is a candidate.  Ground truth is used only for
the explicitly labelled correctness-sensitivity filter.
"""

from __future__ import annotations

from itertools import combinations

import build_religion_ethics_visual_supplement as base


base.BUNDLE_VERSION = "regional-cultural-ethics-bab4-visual-supplement-v1.0.0"
base.EXPECTED_PIPELINE_VERSION = "ethics-regional-cultural-lm-v1.0.16-candidate"
base.EXPECTED_INPUT_SHA256 = "09f96bec32feafe3426c39526c1d9a80453d09ede2edb06a25e90fbddacc89a5"
base.DOMAIN_KEY = "regional_cultural"
base.DOMAIN_LABEL = "regional-budaya"
base.DOMAIN_TITLE = "Regional-Cultural"
base.LEXICON_FILENAME = "regional_lexicon_coverage.csv"
base.LEXICON_CAPTURE_COLUMN = "captured_by_relevant_regional_marker"
base.TERM_TABLE_FILENAME = "regional_cultural_terms_explicit_response_prevalence.csv"
base.WITHIN_TABLE_FILENAME = "regional_cultural_marker_vs_neutral_descriptive.csv"
base.DIRECT_TABLE_FILENAME = "regional_cultural_marker_direct_identity_descriptive.csv"
base.TERM_FIGURE_FILENAME = "05_regional_cultural_terms_ranked_explicit_response_prevalence.png"
base.WITHIN_FIGURE_TOKEN = "regional_cultural_marker"
base.DIRECT_FIGURE_TOKEN = "regional_cultural"

base.IDENTITIES = (
    "north_american",
    "east_asian",
    "south_asian",
    "sub_saharan_african",
)
base.DIRECT_IDENTITY_PAIRS = tuple(combinations(base.IDENTITIES, 2))
base.IDENTITY_LABELS = {
    "north_american": "Amerika Utara",
    "east_asian": "Asia Timur",
    "south_asian": "Asia Selatan",
    "sub_saharan_african": "Afrika Sub-Sahara",
}
base.IDENTITY_COLORS = {
    "north_american": "#2563EB",
    "east_asian": "#D97706",
    "south_asian": "#7C3AED",
    "sub_saharan_african": "#059669",
}
base.PAIR_LABELS = {
    f"{left}_vs_{right}": f"{base.IDENTITY_LABELS[left]} − {base.IDENTITY_LABELS[right]}"
    for left, right in base.DIRECT_IDENTITY_PAIRS
}

base.RELIGION_MARKERS = (
    "culture_region_north_american",
    "culture_region_east_asian",
    "culture_region_south_asian",
    "culture_region_sub_saharan_african",
    "culture_geocultural_broader_reference",
    "culture_general_reference",
    "culture_meta_framing",
    "culture_relativist_framing",
    "regional_persona_direct_address",
    "regional_perspective_adoption",
    "culture_collectivist_communal_orientation",
    "culture_individual_autonomy_orientation",
    "culture_family_obligation_orientation",
    "culture_community_obligation_orientation",
    "culture_reciprocity_interdependence_orientation",
    "culture_hospitality_generosity_orientation",
    "culture_elder_respect_orientation",
    "culture_harmony_face_orientation",
    "culture_hierarchy_authority_orientation",
    "culture_named_concept_reference",
    "culture_named_practice_reference",
    "culture_warmup_name_echo",
    "culture_identity_irrelevance",
)
base.RELIGION_MARKER_LABELS = {
    "culture_region_north_american": "Label Amerika Utara",
    "culture_region_east_asian": "Label Asia Timur",
    "culture_region_south_asian": "Label Asia Selatan",
    "culture_region_sub_saharan_african": "Label Afrika Sub-Sahara",
    "culture_geocultural_broader_reference": "Rujukan geokultural luas",
    "culture_general_reference": "Rujukan budaya umum",
    "culture_meta_framing": "Framing budaya eksplisit",
    "culture_relativist_framing": "Framing relativis",
    "regional_persona_direct_address": "Sapaan persona regional",
    "regional_perspective_adoption": "Adopsi perspektif regional",
    "culture_collectivist_communal_orientation": "Orientasi kolektif/komunal",
    "culture_individual_autonomy_orientation": "Otonomi individual",
    "culture_family_obligation_orientation": "Kewajiban keluarga",
    "culture_community_obligation_orientation": "Kewajiban komunitas",
    "culture_reciprocity_interdependence_orientation": "Resiprositas/interdependensi",
    "culture_hospitality_generosity_orientation": "Hospitalitas/kedermawanan",
    "culture_elder_respect_orientation": "Hormat kepada tetua",
    "culture_harmony_face_orientation": "Harmoni/menjaga muka",
    "culture_hierarchy_authority_orientation": "Hierarki/otoritas",
    "culture_named_concept_reference": "Konsep budaya bernama",
    "culture_named_practice_reference": "Praktik budaya bernama",
    "culture_warmup_name_echo": "Nama warmup terbawa",
    "culture_identity_irrelevance": "Identitas tidak relevan",
}


if __name__ == "__main__":
    raise SystemExit(base.main())
