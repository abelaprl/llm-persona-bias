"""Llama 3.3 rule pack as loaded by the 2026-08-13 10:32 final model run."""
from .._core import verbs
from .._core.frozen_v2_priorities import (
    p_leading_number, p_total_cues, p_equation_last, p_parallel_sum,
    p_copula_tail, p_or_conversion, p_predicate_verb_skip_infinitive,
    p_total_of, p_catch_all_number,
)

RULE_PACK = 'llama33_70b'
MODEL_SLUG = 'meta-llama_llama-3.3-70b-instruct'
PREPROCESS = {'strip_think_blocks': False, 'tail_chars': 3000, 'last_n_lines': 20}
ENABLED_SOURCES = {
    'hashes', 'boxed', 'marker_line', 'currency_line', 'bold_4e',
    'conclusion', 'final_sentence', 'last_lines',
}
TERMINAL_SINGLE_VALUE_FIRST = True
ANSWER_VERBS = verbs.BASE + r'|' + verbs.MODAL + r'|' + verbs.NARRATIVE
PRIORITIES = [
    p_leading_number, p_total_cues, p_equation_last, p_copula_tail,
    p_or_conversion, p_predicate_verb_skip_infinitive, p_parallel_sum,
    p_total_of, p_catch_all_number,
]
