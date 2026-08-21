"""Rule pack: openai/gpt-oss-20b

Aturan boleh berbeda per model (K1). Yang TIDAK boleh berbeda ada di _core/:
normalize_number, definisi benar/salah, penyebut akurasi, larangan gold-peek.

Rule pack ini tidak boleh membaca nama persona (aturan A2, diuji V7) dan tidak
boleh memuat literal ID soal (aturan A3, diuji V6).
"""
from .._core import verbs
from .._core.priorities import (
    p_duration_target, p_leading_number, p_total_cues, p_equation_last,
    p_parallel_sum, p_copula_tail, p_or_conversion, p_predicate_verb,
    p_predicate_verb_skip_infinitive, p_total_of, p_catch_all_number,
    p_last_number,
)

RULE_PACK = "gpt_oss_20b"
MODEL_SLUG = "openai_gpt-oss-20b"

PREPROCESS = {'strip_think_blocks': True, 'tail_chars': 3000, 'last_n_lines': 20}

ANSWER_VERBS = verbs.BASE + r'|' + verbs.MODAL + r'|' + verbs.NARRATIVE

# GPT-OSS often emits a bare numeric final response, or serializes a Markdown
# table and its concluding prose on one physical line. Both recoveries were
# audited on all 21,104 Religion Persona outputs and remain model-specific.
AUDITED_WHOLE_RESPONSE_SINGLE_VALUE = True
SALVAGE_TRAILING_PROSE_AFTER_TABLE = True
AUDITED_TERMINAL_FINAL_LABEL = True

PRIORITIES = [
    p_duration_target,
    p_total_cues,
    p_equation_last,        # gpt-oss menutup ringkas dengan persamaan: "9 x $2 = $18"
    p_copula_tail,
    p_or_conversion,
    p_predicate_verb,
    p_leading_number,
    p_total_of,
]
