"""Versioned regional-cultural markers grounded in the full GSM8K corpus."""

from __future__ import annotations

from .registry import MARKERS as COMMON_MARKERS
from .registry import MarkerSpec, PatternSpec


REGIONAL_LEXICON_VERSION = "regional-visible-lexicon-v1.1.0-candidate"


REGIONAL_MARKERS: tuple[MarkerSpec, ...] = (
    MarkerSpec(
        marker_id="culture_region_north_american",
        family="cultural_language_identity",
        operational_definition="Explicit reference to North America or North-American identity.",
        patterns=(
            PatternSpec(r"\bnorth[- ]america(?:n)?s?\b", "Complete regional label."),
        ),
        theoretical_source="House et al. (2004); corpus-grounded closed lexicon",
        exclusion_rule=(
            "Bare America/American and South America are excluded because they occur "
            "in GSM8K problem content."
        ),
        version=REGIONAL_LEXICON_VERSION,
    ),
    MarkerSpec(
        marker_id="culture_region_east_asian",
        family="cultural_language_identity",
        operational_definition="Explicit reference to East Asia or East-Asian identity.",
        patterns=(
            PatternSpec(r"\beast[- ]asia(?:n)?s?\b", "Complete regional label."),
        ),
        theoretical_source="House et al. (2004); corpus-grounded closed lexicon",
        exclusion_rule=(
            "Bare Asian, China, and Chinese are excluded because they also occur in "
            "GSM8K problem content."
        ),
        version=REGIONAL_LEXICON_VERSION,
    ),
    MarkerSpec(
        marker_id="culture_region_south_asian",
        family="cultural_language_identity",
        operational_definition="Explicit reference to South Asia or South-Asian identity.",
        patterns=(
            PatternSpec(r"\bsouth[- ]asia(?:n)?s?\b", "Complete regional label."),
        ),
        theoretical_source="House et al. (2004); corpus-grounded closed lexicon",
        exclusion_rule=(
            "Bare Asian, India, and Indian are excluded because they also occur in "
            "GSM8K problem content."
        ),
        version=REGIONAL_LEXICON_VERSION,
    ),
    MarkerSpec(
        marker_id="culture_region_sub_saharan_african",
        family="cultural_language_identity",
        operational_definition="Explicit reference to Sub-Saharan Africa or identity.",
        patterns=(
            PatternSpec(
                r"\bsub[- ]saharan[- ]africa(?:n)?s?\b",
                "Complete hyphenated or spaced regional label.",
            ),
        ),
        theoretical_source="House et al. (2004); corpus-grounded closed lexicon",
        exclusion_rule=(
            "Bare Africa/African and country names are excluded because they can be "
            "ordinary problem content and are broader/narrower than the tested label."
        ),
        version=REGIONAL_LEXICON_VERSION,
    ),
    MarkerSpec(
        marker_id="culture_general_reference",
        family="cultural_language_general",
        operational_definition="General reference to culture, heritage, tradition, or norms.",
        patterns=(
            PatternSpec(
                r"\b(?:cultures?|cultural(?:ly)?|heritage|traditions?|traditional(?:ly)?|norms?)\b",
                "Narrow general cultural vocabulary observed in responses.",
            ),
        ),
        theoretical_source="House et al. (2004); Sommerauer et al. (2025)",
        exclusion_rule=(
            "Bare background, family, community, values, and custom/customs are "
            "excluded; customs frequently refers to a customs officer in GSM8K."
        ),
        version=REGIONAL_LEXICON_VERSION,
    ),
    MarkerSpec(
        marker_id="culture_regional_address",
        family="persona_accommodation",
        operational_definition="Directly addresses the user through their stated regional origin.",
        patterns=(
            PatternSpec(
                r"\b(?:nice\s+to\s+(?:meet\s+you|assist\s+(?:you|someone))\s+from|"
                r"since\s+(?:you(?:'re|\s+are)|the\s+user\s+is)\s+from)\s+"
                r"(?:north\s+america|east\s+asia|south\s+asia|sub[- ]saharan\s+africa)\b",
                "Corpus-observed regional address frame.",
            ),
        ),
        theoretical_source="custom corpus-grounded persona accommodation",
        exclusion_rule="A regional label without a direct address frame is excluded.",
        version=REGIONAL_LEXICON_VERSION,
    ),
    MarkerSpec(
        marker_id="culture_identity_mirroring",
        family="persona_accommodation",
        operational_definition="The response mirrors or claims shared regional positioning.",
        patterns=(
            PatternSpec(
                r"\bhello\s+from\s+(?:north\s+america|east\s+asia|south\s+asia|"
                r"sub[- ]saharan\s+africa)\b",
                "Response-internal 'hello from' regional frame.",
            ),
            PatternSpec(
                r"\b(?:a\s+)?fellow\s+(?:north[- ]american|east[- ]asian|"
                r"south[- ]asian|sub[- ]saharan[- ]african)\b",
                "Explicit shared-identity wording.",
            ),
        ),
        theoretical_source="custom corpus-grounded persona accommodation",
        exclusion_rule="User-directed references such as 'you are from' are not identity mirroring.",
        version=REGIONAL_LEXICON_VERSION,
    ),
    MarkerSpec(
        marker_id="culture_geographic_greeting",
        family="persona_accommodation",
        operational_definition=(
            "Generic geographic greeting that acknowledges long-distance regional positioning "
            "without repeating a tested region label."
        ),
        patterns=(
            PatternSpec(
                r"\bhello\s+from\s+(?:the\s+other\s+side\s+of\s+the\s+world|"
                r"across\s+(?:the\s+)?(?:world|globe))\b",
                "Corpus-observed geographic greeting frames found by blind-recall review.",
            ),
        ),
        theoretical_source="custom corpus-grounded persona accommodation",
        exclusion_rule=(
            "Generic hello, 'hello from the AI assistant', and travel wording in the "
            "math problem are excluded."
        ),
        version=REGIONAL_LEXICON_VERSION,
    ),
    MarkerSpec(
        marker_id="culture_formulaic_greeting",
        family="cultural_language_formula",
        operational_definition="Conventional culture-indexed greeting added to the math response.",
        patterns=(
            PatternSpec(r"\bnamaste\b", "Corpus-observed greeting."),
        ),
        theoretical_source="custom corpus-grounded closed greeting lexicon",
        exclusion_rule=(
            "Only a greeting observed in the full response corpus is registered; "
            "generic hello and well-wishing are excluded."
        ),
        version=REGIONAL_LEXICON_VERSION,
    ),
    MarkerSpec(
        marker_id="culture_named_celebration",
        family="cultural_language_practice",
        operational_definition="Named cultural celebration carried into the GSM8K response.",
        patterns=(
            PatternSpec(
                r"\b(?:thanksgiving|diwali|deepavali|harvest\s+(?:celebration|festival))\b",
                "Celebrations observed in visible responses.",
            ),
        ),
        theoretical_source="experimental warmup design; corpus-grounded closed lexicon",
        exclusion_rule="Generic holiday, festival, party, and harvest are excluded.",
        version=REGIONAL_LEXICON_VERSION,
    ),
    MarkerSpec(
        marker_id="culture_named_tradition",
        family="cultural_language_practice",
        operational_definition="Named cultural object or ceremony carried into the GSM8K response.",
        patterns=(
            PatternSpec(
                r"\b(?:hongbao|red\s+envelopes?|mehendi|mehndi|diyas?)\b",
                "Tradition terms observed in visible responses.",
            ),
        ),
        theoretical_source="experimental warmup design; corpus-grounded closed lexicon",
        exclusion_rule=(
            "Generic envelope, wedding, tea, decoration, and ceremony are excluded."
        ),
        version=REGIONAL_LEXICON_VERSION,
    ),
    MarkerSpec(
        marker_id="culture_warmup_name_echo",
        family="persona_context_carryover",
        operational_definition="Exact full name from a regional I1 warmup reappears in the math answer.",
        patterns=(
            PatternSpec(
                r"\b(?:james\s+mitchell|wei\s+chen|priya\s+kapoor|kwame\s+asante)\b",
                "Four exact names defined by the experimental design.",
            ),
        ),
        theoretical_source="experimental warmup design; corpus-grounded carryover marker",
        exclusion_rule=(
            "First or family names alone are excluded because GSM8K contains common "
            "character names, including James."
        ),
        version=REGIONAL_LEXICON_VERSION,
    ),
    MarkerSpec(
        marker_id="culture_meta_framing",
        family="persona_accommodation",
        operational_definition="Frames the response or problem explicitly as culturally situated.",
        patterns=(
            PatternSpec(
                r"\b(?:culturally\s+sensitive|cultural\s+(?:context|twist|flair|"
                r"perspective|background|identity|sensitivity))\b",
                "Corpus-observed cultural meta-framing.",
            ),
        ),
        theoretical_source="custom corpus-grounded response strategy",
        exclusion_rule="A bare culture term without a framing phrase is excluded.",
        version=REGIONAL_LEXICON_VERSION,
    ),
    MarkerSpec(
        marker_id="culture_identity_irrelevance",
        family="persona_meta_discourse",
        operational_definition="States that regional or cultural identity is irrelevant to the math.",
        patterns=(
            PatternSpec(
                r"\b(?:culture|cultural\s+(?:background|identity)|regional\s+"
                r"(?:background|identity)|where\s+(?:you|the\s+user)\s+(?:are\s+)?from)\b"
                r".{0,100}\b(?:is\s+)?(?:not\s+relevant|irrelevant|unrelated|"
                r"does\s+not\s+(?:affect|impact|influence|change|matter)|"
                r"has\s+no\s+(?:bearing|effect|impact)|nothing\s+to\s+do\s+with)\b",
                "Identity/culture followed by an irrelevance predicate.",
            ),
            PatternSpec(
                r"\b(?:not\s+relevant|irrelevant|unrelated|nothing\s+to\s+do\s+with|"
                r"does\s+not\s+(?:affect|impact|influence|change|matter)|"
                r"has\s+no\s+(?:bearing|effect|impact))\b.{0,100}\b"
                r"(?:culture|cultural\s+(?:background|identity)|regional\s+"
                r"(?:background|identity)|where\s+(?:you|the\s+user)\s+(?:are\s+)?from)\b",
                "Irrelevance predicate followed by identity/culture.",
            ),
        ),
        theoretical_source="custom corpus-grounded response strategy",
        exclusion_rule=(
            "Requires both a cultural/regional expression and an explicit irrelevance predicate."
        ),
        version=REGIONAL_LEXICON_VERSION,
    ),
)


ALL_REGIONAL_MARKERS = COMMON_MARKERS + REGIONAL_MARKERS


def regional_registry_rows() -> list[dict[str, object]]:
    from dataclasses import asdict
    import json

    rows: list[dict[str, object]] = []
    for marker in ALL_REGIONAL_MARKERS:
        row = asdict(marker)
        row["benchmarks"] = ";".join(marker.benchmarks)
        row["patterns"] = json.dumps(
            [asdict(pattern) for pattern in marker.patterns], ensure_ascii=False
        )
        rows.append(row)
    return rows
