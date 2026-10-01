"""Rule pack: deepseek/deepseek-r1-distill-qwen-32b

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

RULE_PACK = "deepseek_r1_distill_32b"
MODEL_SLUG = "deepseek_deepseek-r1-distill-qwen-32b"

# Late fallback audited across all 21,104 Religion-Persona rows.  It only
# commits when the complete response contains exactly one numeric value.
# Differential audit: +182 commits, 0 previously committed values changed;
# 171 match the reference answer and the remaining 11 are explicit model
# answers (model errors rather than extraction errors).
AUDITED_WHOLE_RESPONSE_SINGLE_VALUE = True

# DeepSeek sometimes appends a greeting after an otherwise explicit terminal
# declaration. Differential audit on all 887 previously uncommitted rows:
# +14 commits, 0 earlier fallback values changed; the 3 reference mismatches
# are explicit (but mathematically wrong) answers from the model.
AUDITED_TERMINAL_DECLARATION = True

# DeepSeek commonly emits ``**Answer**:`` and ``**Answer:**``. Without this
# syntax-only normalisation the strong marker is hidden by Markdown and later
# bold fallbacks may select a numbered step label instead of the answer.
NORMALIZE_BOLDED_STRONG_MARKERS = True
SALVAGE_MARKER_FROM_FILTERED_SEGMENT = True
AUDITED_LEADING_DECLARATION = True
AUDITED_TERMINAL_CONNECTOR_SINGLE_VALUE = True
AUDITED_MARKER_WORD_NUMBER = True
AUDITED_CONCISE_DECLARATION = True

# Selective post-rerun audit (21,104 rows): these late fallbacks recover four
# explicit, non-truncated DeepSeek answers without changing any prior commit.
# Two remaining responses explicitly declare the question unsolvable; they
# are classified as refusal before a hypothetical last-lines candidate can be
# surfaced. Finish-reason ``length`` is still handled by the strict guard in
# the shared skeleton before any of these fallbacks run.
AUDITED_TERMINAL_TARGET_AMOUNT = True
AUDITED_MARKER_LABELED_LIST = True
AUDITED_TERMINAL_MIXED_FRACTION = True
AUDITED_UNSOLVABLE_AS_REFUSAL = True

# Final differential audit against the canonical pre-change CSV (21,104 rows):
# 39 additional commits (35 reference matches, 4 explicit model answers that
# are mathematically wrong), 14 old wrong parses corrected, no correct parse
# changed to wrong, and no existing commit was lost. Ground truth is used only
# by this offline audit; none of the enabled extraction rules can access it.

PREPROCESS = {'strip_think_blocks': True, 'tail_chars': 3000, 'last_n_lines': 20}

ANSWER_VERBS = verbs.BASE + r'|' + verbs.MODAL + r'|' + verbs.NARRATIVE

PRIORITIES = [
    p_duration_target,      # R1 sering menutup "The car took 16 hours to travel 480 miles"
    p_total_cues,
    p_equation_last,
    p_parallel_sum,
    p_copula_tail,
    p_or_conversion,
    p_predicate_verb,
    p_leading_number,
    p_total_of,
]
