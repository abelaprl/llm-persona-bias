"""Rule pack: meta-llama/llama-3.3-70b-instruct

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

RULE_PACK = "llama33_70b"
MODEL_SLUG = "meta-llama_llama-3.3-70b-instruct"

PREPROCESS = {'strip_think_blocks': False, 'tail_chars': 3000, 'last_n_lines': 20}

# Llama 3.3 menebalkan label item/step (mis. ``**Eggs**: 6``) jauh lebih
# sering daripada jawaban akhirnya.  Detektor bold umum karena itu mengambil
# angka langkah pertama sebelum sempat membaca kalimat kesimpulan.  Sumber
# lain tetap aktif; khusus rule pack ini bold 4a/4d tidak dijadikan bukti
# jawaban akhir.
ENABLED_SOURCES = {
    'hashes', 'boxed', 'marker_line', 'currency_line', 'bold_4e',
    'conclusion', 'final_sentence', 'last_lines',
}
TERMINAL_SINGLE_VALUE_FIRST = True
TERMINAL_SINGLE_BEFORE_WEAK_MARKER = True

# Pada sebagian respons satu baris, caveat "However," dan kesimpulan eksplisit
# berada dalam segmen yang sama. Guard boleh membuang caveatnya, tetapi suffix
# mulai dari penanda jawaban terakhir harus tetap tersedia untuk RULE 3.
SALVAGE_MARKER_FROM_FILTERED_SEGMENT = True
SALVAGE_WEAK_TERMINAL_MARKER = True
MARKER_TRANSACTION_PRICE = True
MARKER_LEADING_SUBJECT_RESULT = True
AUDITED_TERMINAL_DECLARATION = True

ANSWER_VERBS = verbs.BASE + r'|' + verbs.MODAL + r'|' + verbs.NARRATIVE

PRIORITIES = [
    p_leading_number,
    p_total_cues,
    p_equation_last,
    p_copula_tail,
    p_or_conversion,
    p_predicate_verb_skip_infinitive,
    p_parallel_sum,
    p_total_of,
    p_catch_all_number,
]
