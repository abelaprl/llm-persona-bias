"""Rule pack: qwen/qwen3-next-80b-a3b-thinking

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

RULE_PACK = "qwen3_next_80b"
MODEL_SLUG = "qwen_qwen3-next-80b-a3b-thinking"

PREPROCESS = {'strip_think_blocks': True, 'tail_chars': 3000, 'last_n_lines': 20}

ANSWER_VERBS = verbs.BASE + r'|' + verbs.MODAL + r'|' + verbs.NARRATIVE

# Qwen occasionally serializes the remaining reasoning and its explicit
# Answer suffix as one physical segment beginning with Given/Assuming/Step.
# Full 21,104-row differential: +6 commits (all reference-correct), 20 prior
# values changed, 17 wrong values corrected, and 0 correct-to-wrong changes.
SALVAGE_MARKER_FROM_FILTERED_SEGMENT = True

# Late fallbacks for Markdown answer headings. Across all previously
# uncommitted rows the strong-heading rule adds 5/5 reference-correct commits;
# the weak terminal form adds one ``Thus ... is:`` + single-bold-value commit.
# Neither path can override an ordinary RULE 1-5 commit.
MARKER_HEADING_NEXT_SEGMENT = True
AUDITED_WEAK_MARKER_HEADING_NEXT_SEGMENT = True

# A duration may be restated after the Answer heading as hours + minutes.
# Commit the minute total only when that conversion agrees with an earlier
# explicit minute total in the same response (two text-only signals).
AUDITED_MARKER_HEADING_DURATION_COMPOSITION = True

PRIORITIES = [
    p_duration_target,
    p_total_cues,
    p_equation_last,
    p_copula_tail,
    p_or_conversion,
    p_predicate_verb,
    p_leading_number,
    p_total_of,
]
