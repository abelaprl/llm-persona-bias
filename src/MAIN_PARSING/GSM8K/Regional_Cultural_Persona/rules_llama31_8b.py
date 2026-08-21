"""Regional Cultural rule pack for Llama 3.1 8B Instruct."""

from ..Religion_Persona.rules_llama31_8b import *  # noqa: F401,F403

RULE_PACK = 'regional_cultural_llama31_8b'

# Recover only explicit terminal declarations from audited raw responses.
AUDITED_RAW_STRONG_MARKER = True
AUDITED_RAW_TERMINAL_CONNECTOR = True
