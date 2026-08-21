"""Regional Cultural rule pack for Ministral 8B."""

from ..Religion_Persona.rules_ministral_8b import *  # noqa: F401,F403

RULE_PACK = 'regional_cultural_ministral_8b'

# Regional outputs sometimes place the answer only in the final bold group
# after a long single-line derivation whose step prefix is filtered by guards.
# Full-data audit: 7 newly committed rows, all 7 match the reference answer;
# no previously committed extraction changed because this is a late fallback.
AUDITED_TERMINAL_BOLD_VALUE = True
AUDITED_PRIMARY_RESULT_DECLARATION = True
AUDITED_RAW_STRONG_MARKER = True
