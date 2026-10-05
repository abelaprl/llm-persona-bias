"""Versioned, auditable registry of linguistic-marker rules.

The registry deliberately contains no composite "reasoning quality" score.  A
marker remains an independently interpretable outcome, and overlapping
constructs are never silently summed into one number.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import cached_property
import json
import re
from typing import Literal


Measurement = Literal["density", "binary"]
Surface = Literal["lexical", "structure", "visible"]
Role = Literal["primary", "supplementary", "format", "quality_control"]


@dataclass(frozen=True)
class PatternSpec:
    pattern: str
    note: str
    flags: int = re.IGNORECASE

    def compile(self) -> re.Pattern[str]:
        return re.compile(self.pattern, self.flags)


@dataclass(frozen=True)
class MarkerSpec:
    marker_id: str
    family: str
    operational_definition: str
    patterns: tuple[PatternSpec, ...]
    measurement: Measurement = "density"
    surface: Surface = "lexical"
    benchmarks: tuple[str, ...] = ("gsm8k", "ethics")
    analysis_role: Role = "primary"
    theoretical_source: str = "custom operationalization"
    exclusion_rule: str = "Whole phrase/word matching only."
    version: str = "1.2.1-candidate"
    validation_status: str = "candidate_pending_manual_audit"

    @cached_property
    def compiled_patterns(self) -> tuple[re.Pattern[str], ...]:
        return tuple(item.compile() for item in self.patterns)


MARKERS: tuple[MarkerSpec, ...] = (
    MarkerSpec(
        marker_id="organization_explicit_step",
        family="reasoning_organization",
        operational_definition="Line-initial numbered or explicitly labelled reasoning step.",
        patterns=(
            PatternSpec(
                r"^\s*(?:#{1,4}\s*)?(?:\*{0,2}step\s*\d{1,2}\*{0,2}|\d{1,2}[.)])\s*[:.\-]?\s+(?=[A-Za-z])",
                "Restricted to line starts so arithmetic numbers are not list items.",
                re.IGNORECASE | re.MULTILINE,
            ),
        ),
        surface="structure",
        theoretical_source="Wei et al. (2022), operationalized for model responses",
        exclusion_rule="Does not match equation-internal numbers or answer-only lines.",
    ),
    MarkerSpec(
        marker_id="organization_bullet",
        family="reasoning_organization",
        operational_definition="Line-initial bullet used to organize an explanation.",
        patterns=(
            PatternSpec(
                r"^\s*[-*\u2022]\s+(?=[A-Za-z])",
                "Markdown or Unicode bullets followed by lexical content.",
                re.IGNORECASE | re.MULTILINE,
            ),
        ),
        surface="structure",
        theoretical_source="Biber (1988), custom response-format operationalization",
        exclusion_rule="Multiplication symbols and bullets without following words are excluded.",
    ),
    MarkerSpec(
        marker_id="organization_section_header",
        family="reasoning_organization",
        operational_definition="Markdown heading or short bold line that labels a reasoning section.",
        patterns=(
            PatternSpec(
                r"^\s*#{1,4}\s+[A-Za-z][^\n]{1,60}$",
                "Markdown heading.",
                re.IGNORECASE | re.MULTILINE,
            ),
            PatternSpec(
                r"^\s*\*\*(?:[A-Za-z][^*.!?\n]{0,48}:|"
                r"(?:solution|given|steps?|steps?\s+to\s+solve|"
                r"step[- ]by[- ]step\s+explanation|analysis|approach|method|"
                r"calculations?|explanation|answer|final\s+answer|result|conclusion))"
                r"\*\*\s*$",
                "Bold label ending in a colon, or a restricted conventional section label.",
                re.IGNORECASE | re.MULTILINE,
            ),
        ),
        surface="structure",
        theoretical_source="Biber (1988), custom response-format operationalization",
        exclusion_rule="Technical answer lines are removed before this rule is applied.",
    ),
    MarkerSpec(
        marker_id="stance_hedge",
        family="epistemic_stance",
        operational_definition="Explicit uncertainty or tentativeness about a proposition.",
        patterns=(
            PatternSpec(
                r"\b(?:perhaps|maybe|possibly|probably|likely|arguably|apparently|seemingly)\b",
                "Lexical hedges with comparatively narrow epistemic use.",
            ),
            PatternSpec(
                r"\b(?:it\s+(?:seems?|appears?)\s+(?:that|to)|(?:may|might|could)\s+be\b|(?:is|are)\s+likely\s+to\b)\b",
                "Context-bounded modal and evidential phrases.",
            ),
            PatternSpec(
                r"\b(?:i\s+(?:think|believe|suspect)|(?:it|this|that)\s+(?:seems?|appears?))\b",
                "First-person stance and subject-bounded evidential frames.",
            ),
        ),
        theoretical_source="Hyland (2005); Biber and Finegan (1989), adapted list",
        exclusion_rule=(
            "Bare about/around, bare could/may/might, and ability expressions such as "
            "'could calculate' are excluded."
        ),
    ),
    MarkerSpec(
        marker_id="stance_booster",
        family="epistemic_stance",
        operational_definition="Explicit emphasis on certainty or obviousness.",
        patterns=(
            PatternSpec(
                r"\b(?:clearly|obviously|definitely|certainly|undoubtedly|unquestionably|surely|indeed|in\s+fact)\b",
                "Single-word certainty boosters.",
            ),
            PatternSpec(
                r"\b(?:without\s+(?:a\s+)?doubt|of\s+course|it\s+is\s+clear\s+that)\b",
                "Multi-word certainty boosters.",
            ),
        ),
        theoretical_source="Hyland (2005); Biber and Finegan (1989), adapted list",
        exclusion_rule=(
            "Generic always, unmodified right/wrong, and must be are excluded; "
            "must be was predominantly deontic or constraint-like in the corpus audit."
        ),
    ),
    MarkerSpec(
        marker_id="stance_assumption",
        family="epistemic_stance",
        operational_definition="An assumption is made explicit before a calculation or judgment.",
        patterns=(
            PatternSpec(
                r"(?:^|(?<=[.!?;:])\s+)(?:assuming\s+(?:that|the|this)|presuming\s+that)\b",
                "Clause-initial participial assumption frame.",
                re.IGNORECASE | re.MULTILINE,
            ),
            PatternSpec(
                r"\b(?:(?:i(?:'ll|\s+will)|we|let(?:'s|\s+us))\s+assume|under\s+the\s+assumption\s+that)\b",
                "Explicit finite or prepositional assumption frame.",
            ),
            PatternSpec(
                r"(?:^|(?<=[.!?;:])\s+)assume\s+(?:that|the|a|an|this)\b",
                "Clause-initial imperative assumption frame.",
                re.IGNORECASE | re.MULTILINE,
            ),
        ),
        theoretical_source="Hyland (2005), custom task-specific operationalization",
        exclusion_rule="Given that and note that are not automatically treated as assumptions.",
    ),
    MarkerSpec(
        marker_id="stance_self_correction",
        family="epistemic_stance",
        operational_definition="Overt revision of the model's own preceding reasoning.",
        patterns=(
            PatternSpec(
                r"(?<=[.!?])\s+(?:wait[,!:]|actually[,!:]?)\s+(?=[A-Za-z])",
                "Post-sentence correction cue; Wait requires correction punctuation.",
            ),
            PatternSpec(
                r"\b(?:let\s+me\s+(?:re-?check|recalculate|correct\s+that|restart|"
                r"revise\s+that)|i\s+need\s+to\s+correct\s+(?:myself|that)|"
                r"on\s+second\s+thought|scratch\s+that|correction\s*:|recalculating)\b",
                "Explicit model-oriented self-repair phrase.",
            ),
            PatternSpec(
                r"\blet(?:'s|\s+us)\s+(?:re-?check|re-?evaluate|recalculate|reconsider|correct\s+that)\b",
                "Inclusive explicit re-check or revision phrase; generic advice scripts excluded.",
            ),
        ),
        theoretical_source="custom, motivated by reasoning-monitorability literature",
        exclusion_rule=(
            "Actually within an unpunctuated clause, Wait without correction punctuation, "
            "generic advice scripts, and quoted I-was-wrong apologies are excluded."
        ),
    ),
    MarkerSpec(
        marker_id="connective_causal",
        family="logical_connective",
        operational_definition="Explicit causal or inferential relation between propositions.",
        patterns=(
            PatternSpec(
                r"\b(?:therefore|thus|hence|consequently|accordingly|because|"
                r"as\s+a\s+result|for\s+(?:this|that)\s+reason|thereby|"
                r"it\s+follows\s+that|which\s+means|this\s+means)\b",
                "Causal and result connectives.",
            ),
        ),
        theoretical_source="Halliday and Hasan (1976); Hyland (2005)",
        exclusion_rule="Bare so is excluded because its discourse function is highly ambiguous.",
    ),
    MarkerSpec(
        marker_id="connective_contrastive",
        family="logical_connective",
        operational_definition="Explicit contrast or concession between propositions.",
        patterns=(
            PatternSpec(
                r"\b(?:however|nevertheless|nonetheless|on\s+the\s+other\s+hand|although|even\s+though|despite\s+(?:this|that|the)|in\s+contrast)\b",
                "Contrastive and concessive connectives.",
            ),
        ),
        theoretical_source="Halliday and Hasan (1976); Hyland (2005)",
        exclusion_rule="Bare but/while/yet are excluded because they are strongly polyfunctional.",
    ),
    MarkerSpec(
        marker_id="connective_conditional",
        family="logical_connective",
        operational_definition="Explicit conditional frame.",
        patterns=(
            PatternSpec(
                r"\b(?:if|unless|provided\s+that|in\s+the\s+event\s+that|on\s+condition\s+that)\b",
                "Conditional connectives.",
            ),
        ),
        theoretical_source="Halliday and Hasan (1976)",
        exclusion_rule="Only whole lexical items or phrases are counted.",
    ),
    MarkerSpec(
        marker_id="self_reference_first_person",
        family="self_reference",
        operational_definition="First-person singular reference tied to reasoning or stance.",
        patterns=(
            PatternSpec(
                r"\b(?:i\s+(?:think|believe|assume|need|will|would|find|calculate|get|conclude|interpret|understand)|let\s+me\s+(?:think|check|re-?check|calculate|recalculate|explain|clarify|break|work|walk|analy[sz]e|reason|solve|determine|find|consider|evaluate|summarize))\b",
                "Restricted first-person reasoning frames.",
            ),
        ),
        theoretical_source="Hyland (2005), adapted self-mention operationalization",
        exclusion_rule="Bare I/me/my and AI-identity statements are excluded.",
    ),
    MarkerSpec(
        marker_id="self_reference_inclusive_we",
        family="self_reference",
        operational_definition="Inclusive first-person plural used to guide joint reasoning.",
        patterns=(
            PatternSpec(
                r"\b(?:we\s+(?:need|can|have|get|find|calculate|determine|see|know|conclude|will|should|want)|let(?:'s|\s+us))\b",
                "Reasoning-functional inclusive we.",
            ),
        ),
        theoretical_source="Hyland (2005), adapted self-mention/engagement operationalization",
        exclusion_rule="Bare we/our/us are excluded.",
    ),
    MarkerSpec(
        marker_id="engagement_reader_directive",
        family="engagement",
        operational_definition="A narrow directive that explicitly guides the reader's attention.",
        patterns=(
            PatternSpec(
                r"\b(?:note\s+that|notice\s+that|recall\s+that|keep\s+in\s+mind|you\s+can\s+see\s+that|you\s+should\s+consider\s+(?:that|the|this))\b",
                "Reader-directing phrase.",
            ),
            PatternSpec(
                r"(?:^|(?<=[.!?;:])\s+)consider\s+(?:that|the|this)\b",
                "Clause-initial imperative consider.",
                re.IGNORECASE | re.MULTILINE,
            ),
            PatternSpec(
                r"(?:^|(?<=[.!?;:])\s+)(?:please\s+)?remember\s+that\b|\b(?:but|and)\s+remember\s+that\b",
                "Reader-directed imperative remember; first-person memory reports are excluded.",
                re.IGNORECASE | re.MULTILINE,
            ),
        ),
        theoretical_source="Hyland (2005), adapted engagement-marker list",
        exclusion_rule="Bare you/your and generic imperatives are excluded.",
    ),
    MarkerSpec(
        marker_id="elaboration_code_gloss",
        family="textual_elaboration",
        operational_definition="Reformulation or example that elaborates a preceding proposition.",
        patterns=(
            PatternSpec(
                r"\b(?:in\s+other\s+words|that\s+is\s+to\s+say|namely|for\s+example|such\s+as)\b|\bthat\s+is(?=\s*[,;:])|(?<!\w)(?:e\.g\.|i\.e\.)(?!\w)",
                "Code glosses and exemplifiers.",
            ),
        ),
        analysis_role="supplementary",
        theoretical_source="Hyland (2005), adapted code-gloss list",
        exclusion_rule="Phrase-level matching; punctuation variants are allowed.",
    ),
    MarkerSpec(
        marker_id="math_latex_display",
        family="mathematical_presentation",
        operational_definition="Displayed LaTeX mathematics block.",
        patterns=(
            PatternSpec(
                r"\\\[[\s\S]{1,1000}?\\\]|\$\$[\s\S]{1,1000}?\$\$",
                "Paired display delimiters.",
            ),
        ),
        surface="structure",
        benchmarks=("gsm8k",),
        analysis_role="format",
        theoretical_source="custom GSM8K presentation feature",
        exclusion_rule="Unpaired delimiters and inline LaTeX are excluded.",
    ),
    MarkerSpec(
        marker_id="math_inline_arithmetic",
        family="mathematical_presentation",
        operational_definition="Plain inline arithmetic equality.",
        patterns=(
            PatternSpec(
                r"(?<![\w.])[-+]?(?:[$\u20ac\u00a3][ \t]*)?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?"
                r"(?:[ \t]*[+\-\u00d7x*\u00f7/][ \t]*[-+]?(?:[$\u20ac\u00a3][ \t]*)?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?)+"
                r"[ \t]*=[ \t]*[-+]?(?:[$\u20ac\u00a3][ \t]*)?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[ \t]*/[ \t]*(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)?",
                "Complete plain arithmetic expression followed by an equality.",
            ),
        ),
        surface="structure",
        benchmarks=("gsm8k",),
        analysis_role="format",
        theoretical_source="custom GSM8K presentation feature",
        exclusion_rule=(
            "Operations without an equals-result structure are excluded; a multi-operand "
            "expression is counted once rather than from its final two operands."
        ),
    ),
    MarkerSpec(
        marker_id="math_boxed_answer",
        family="answer_format",
        operational_definition="A boxed final mathematical answer.",
        patterns=(
            PatternSpec(
                r"\\(?:boxed|bbox)\{(?:[^{}\n]|\{[^{}\n]{0,40}\}){1,120}\}",
                "Boxed LaTeX answer, allowing one level of nested braces.",
            ),
        ),
        measurement="binary",
        surface="visible",
        benchmarks=("gsm8k",),
        analysis_role="quality_control",
        theoretical_source="custom GSM8K answer-format feature",
        exclusion_rule="Recorded as quality control, not mathematical formality.",
    ),
    MarkerSpec(
        marker_id="ai_identity_statement",
        family="model_identity",
        operational_definition="Direct statement that the speaker is an AI/system without human experience.",
        patterns=(
            PatternSpec(
                r"\b(?:as\s+an\s+(?:ai|artificial\s+intelligence|language\s+model)|i\s+am\s+an?\s+(?:ai|artificial\s+intelligence|language\s+model)|i(?:'m|\s+am)\s+not\s+a\s+human)\b",
                "Direct AI identity statement.",
            ),
        ),
        measurement="binary",
        analysis_role="supplementary",
        theoretical_source="Diep et al. (2025), custom phrase operationalization",
        exclusion_rule="Generic refusal without an AI identity statement is excluded.",
    ),
)


def marker_by_id(marker_id: str) -> MarkerSpec:
    for marker in MARKERS:
        if marker.marker_id == marker_id:
            return marker
    raise KeyError(marker_id)


def registry_rows() -> list[dict[str, object]]:
    """Return a CSV-friendly registry with explicit rule provenance."""
    rows: list[dict[str, object]] = []
    for marker in MARKERS:
        row = asdict(marker)
        row["benchmarks"] = ";".join(marker.benchmarks)
        row["patterns"] = json.dumps(
            [asdict(pattern) for pattern in marker.patterns], ensure_ascii=False
        )
        rows.append(row)
    return rows
