"""Rule pack: mistralai/ministral-8b-2512

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

RULE_PACK = "ministral_8b"
MODEL_SLUG = "mistralai_ministral-8b-2512"

PREPROCESS = {'strip_think_blocks': False, 'tail_chars': 3000, 'last_n_lines': 20}

# Model ini lazim memisahkan heading ``### Final Answer:`` dan isi jawabannya
# menjadi dua segmen Markdown. Isi segmen berikutnya tetap merupakan bagian
# marker eksplisit, bukan fallback last-lines.
MARKER_HEADING_NEXT_SEGMENT = True

ANSWER_VERBS = verbs.BASE + r'|' + verbs.MODAL + r'|' + verbs.NARRATIVE

PRIORITIES = [
    p_duration_target,          # "will take 2 weeks" -> 2
    p_leading_number,           # "10 guests. At 10 guests..." -> 10
    p_total_cues,               # "in total", "altogether"
    p_equation_last,            # ada "=" -> angka terakhir
    p_parallel_sum,             # "27 selfies and 18 portraits" -> 45
    p_copula_tail,              # "... was 95"
    p_or_conversion,            # "2040 minutes or 34 hours" -> 34
    p_predicate_verb,           # kata kerja predikat -> angka sesudahnya
    p_total_of,                 # "total of N"
]
