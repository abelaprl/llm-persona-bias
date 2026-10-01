"""Rule pack: meta-llama/llama-3.1-8b-instruct

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

RULE_PACK = "llama31_8b"
MODEL_SLUG = "meta-llama_llama-3.1-8b-instruct"

PREPROCESS = {'strip_think_blocks': False, 'tail_chars': 3000, 'last_n_lines': 20}

ANSWER_VERBS = verbs.BASE + r'|' + verbs.MODAL + r'|' + verbs.NARRATIVE

# Diaudit terhadap seluruh keluaran model: deklarasi/persamaan terakhir yang
# diikuti salam boleh di-commit hanya melalui fallback konservatif di core.
AUDITED_TERMINAL_DECLARATION = True
AUDITED_TERMINAL_EQUATION = True

PRIORITIES = [
    p_leading_number,
    p_equation_last,
    p_copula_tail,
    p_or_conversion,
    p_predicate_verb_skip_infinitive,   # llama sering pakai infinitif: "needs to make 5 vlogs"
    p_parallel_sum,                     # hanya setelah predikat gagal
    p_total_of,
    p_catch_all_number,                 # llama nyaris tak pernah pakai penanda -> perlu jaring
]
