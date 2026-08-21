"""Rule pack: CORE — pembanding antar-model (plan Sec. 3.4)

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

RULE_PACK = "core"
MODEL_SLUG = "*"

PREPROCESS = {'strip_think_blocks': True, 'tail_chars': 3000, 'last_n_lines': 20}

ANSWER_VERBS = verbs.BASE

PRIORITIES = [
    p_equation_last,
    p_copula_tail,
    p_predicate_verb,
]

# Hanya RULE 1-3. Identik untuk 6 model, dipakai sebagai analisis sensitivitas
# supaya klaim perbandingan ANTAR MODEL tidak bergantung kemurahan rule pack.
ENABLED_SOURCES = {'hashes', 'boxed', 'marker_line'}
MARKER_MODE = 'strict'   # hanya penanda formal, tanpa konektor "So,"
