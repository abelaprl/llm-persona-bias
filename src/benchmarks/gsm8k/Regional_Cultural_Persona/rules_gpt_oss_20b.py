"""Regional Cultural rule pack for GPT-OSS 20B."""

from ..Religion_Persona.rules_gpt_oss_20b import *  # noqa: F401,F403

RULE_PACK = 'regional_cultural_gpt_oss_20b'

# GPT-OSS frequently ends with a rounded/bold value after equations containing
# a more precise intermediate. Reading only the terminal bold group preserves
# the model's declared final answer.
# Full-data audit: 34 newly committed rows; 31 match the reference and the
# other 3 are explicit model answers (106.12 versus reference 106), not parser
# errors. No previously committed extraction changed.
AUDITED_TERMINAL_BOLD_VALUE = True
AUDITED_PRIMARY_RESULT_DECLARATION = True
AUDITED_RAW_STRONG_MARKER = True
AUDITED_TERMINAL_EQUATION_RESULT = True
