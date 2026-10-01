"""ETHICS-specific discourse and content-framing marker registry.

The general registry intentionally stays benchmark-agnostic.  This module
contains constructs observed in the ETHICS response corpus and in the legacy
ETHICS notebook.  Broad legacy probes were narrowed where a bare word was
strongly polyfunctional (for example ``must``, ``right``, ``result``, ``but``,
and ``while``).

Content-framing markers are kept separate from linguistic-style markers.  They
must never be summed into a composite reasoning or morality score.
"""

from __future__ import annotations

from dataclasses import asdict
import json
import re

from .registry import MarkerSpec, PatternSpec


ETHICS_REGISTRY_VERSION = "ethics-discourse-v1.1.0-candidate"


ETHICS_MARKERS: tuple[MarkerSpec, ...] = (
    MarkerSpec(
        marker_id="stance_contextual_hedge",
        family="epistemic_stance",
        operational_definition=(
            "Strong contextual qualification that withholds an unconditional judgment."
        ),
        patterns=(
            PatternSpec(
                r"\bit\s+depends(?:\s+on|\s+whether|\s+how|\s+largely|\s+entirely)?\b",
                "Corpus-observed context-dependence frame.",
            ),
            PatternSpec(
                r"\bwithout\s+(?:more|enough|additional|further)\s+"
                r"(?:context|information|details?|clarification)\b",
                "Insufficient-context qualification.",
            ),
            PatternSpec(
                r"\b(?:can(?:not|'t)|unable\s+to)\s+"
                r"(?:determine|decide|judge|assess|conclude)\b",
                "Explicit inability to settle the judgment.",
            ),
            PatternSpec(
                r"\b(?:insufficient|not\s+enough|limited)\s+"
                r"(?:context|information|details?|evidence)\b",
                "Explicit insufficiency statement.",
            ),
            PatternSpec(
                r"\b(?:challenging|difficult|hard)\s+to\s+"
                r"(?:say|determine|judge|assess|conclude)\b",
                "Difficulty committing to a judgment.",
            ),
            PatternSpec(
                r"\b(?:no\s+clear\s+(?:answer|verdict|judgment)|"
                r"context[- ]dependent|var(?:y|ies)\s+(?:by|across|depending\s+on)\s+"
                r"(?:context|culture|circumstances?|situation))\b",
                "No-clear-verdict and context-variation family.",
            ),
            PatternSpec(
                r"\b(?:not|neither)\s+inherently\s+(?:right|wrong|ethical|unethical|"
                r"acceptable|unacceptable)(?:\s+nor\s+(?:right|wrong|ethical|unethical|"
                r"acceptable|unacceptable))?\b",
                "Explicit denial of an inherent moral polarity.",
            ),
        ),
        measurement="density",
        benchmarks=("ethics",),
        analysis_role="primary",
        theoretical_source=(
            "Hyland (2005), adapted; ETHICS full-corpus audit, 2026-08-21"
        ),
        exclusion_rule=(
            "Bare may/might/could and generic uncertainty words are excluded. "
            "This marker may overlap response_stance_conditional and must be reported separately."
        ),
        version=ETHICS_REGISTRY_VERSION,
    ),
    MarkerSpec(
        marker_id="stance_distancing",
        family="epistemic_stance",
        operational_definition=(
            "Attributes a possible position to an unspecified or external viewpoint."
        ),
        patterns=(
            PatternSpec(
                r"\b(?:one|some|others?|many\s+people|critics?|proponents?)\s+"
                r"(?:may|might|could|would)\s+(?:argue|say|view|see|consider|believe)\b",
                "External-viewpoint attribution.",
            ),
            PatternSpec(
                r"\bit\s+(?:may|might|could)\s+be\s+(?:argued|said|viewed|seen|considered)\b",
                "Impersonal passive distancing.",
            ),
            PatternSpec(
                r"\bfrom\s+(?:a|an|the|some|another|certain)\s+"
                r"(?:moral|ethical|legal|social|cultural|individual|personal|"
                r"utilitarian|deontological|religious)?\s*perspective\b",
                "Explicitly bounded perspective frame.",
            ),
        ),
        measurement="density",
        benchmarks=("ethics",),
        analysis_role="primary",
        theoretical_source="Hyland (2005), adapted; ETHICS corpus audit, 2026-08-21",
        exclusion_rule=(
            "Bare perspective and first-person statements are excluded; persona-specific "
            "perspective adoption remains a separate gender marker."
        ),
        version=ETHICS_REGISTRY_VERSION,
    ),
    MarkerSpec(
        marker_id="connective_additive",
        family="logical_connective",
        operational_definition="Explicitly adds a further proposition or consideration.",
        patterns=(
            PatternSpec(
                r"\b(?:furthermore|moreover|additionally|in\s+addition|similarly|likewise)\b",
                "Unambiguous additive connectives.",
            ),
            PatternSpec(
                r"\b(?:another|a\s+further)\s+(?:important\s+)?"
                r"(?:consideration|factor|point)\s+is\b|"
                r"\bit\s+is\s+also\s+(?:important|worthwhile|useful)\s+to\b",
                "Bounded additive frames; bare also/and are excluded.",
            ),
        ),
        benchmarks=("ethics",),
        analysis_role="primary",
        theoretical_source="Halliday and Hasan (1976); Hyland (2005)",
        exclusion_rule="Bare and/also are excluded because they are highly polyfunctional.",
        version=ETHICS_REGISTRY_VERSION,
    ),
    MarkerSpec(
        marker_id="connective_concessive",
        family="logical_connective",
        operational_definition="Marks an explicit concession before or against a judgment.",
        patterns=(
            PatternSpec(
                r"\b(?:even\s+though|granted\s+that|admittedly|notwithstanding\s+(?:this|that)|"
                r"while\s+it\s+is\s+true\s+that|although\s+it\s+is\s+true\s+that)\b",
                "Narrow concessive connectives.",
            ),
            PatternSpec(
                r"\bdespite\s+(?:this|that|these|those|the\s+(?:fact|circumstances?|concerns?))\b",
                "Bounded despite-concession frame.",
            ),
        ),
        benchmarks=("ethics",),
        analysis_role="primary",
        theoretical_source="Halliday and Hasan (1976); Hyland (2005)",
        exclusion_rule=(
            "Bare while/although are excluded here; some spans also satisfy the general "
            "contrastive marker and must not be summed."
        ),
        version=ETHICS_REGISTRY_VERSION,
    ),
    MarkerSpec(
        marker_id="response_stance_committed",
        family="response_stance",
        operational_definition="Overtly commits to a moral or task verdict in the explanation.",
        patterns=(
            PatternSpec(
                r"\b(?:this|the)\s+(?:action|behavior|behaviour|scenario|decision|choice|"
                r"statement|request|reason)\s+(?:is|would\s+be|appears?\s+to\s+be)\s+"
                r"(?:clearly\s+|definitely\s+)?(?:right|wrong|ethical|unethical|"
                r"acceptable|unacceptable|justified|unjustified)\b",
                "Entity-bound explicit verdict.",
            ),
            PatternSpec(
                r"\b(?:i\s+conclude\s+that|the\s+(?:verdict|judgment|conclusion)\s+is)\b",
                "Explicit conclusion frame.",
            ),
            PatternSpec(
                r"\btherefore\s*,?\s+(?:this|it|the\s+(?:action|behavior|decision))\s+is\s+"
                r"(?:right|wrong|ethical|unethical|acceptable|unacceptable|justified|unjustified)\b",
                "Inferentially introduced verdict.",
            ),
            PatternSpec(
                r"\b(?:i\s+would|i'd)\s+(?:answer|say|choose)\s*[:,-]?\s*[01]\b|"
                r"\bif\s+i\s+\*{0,2}must\*{0,2}\s+answer"
                r"(?:\s+with\s+[01](?:\s*(?:or|/)\s*[01])?)?"
                r"(?:\s+as\s+requested)?\s*(?::|,)?\s*\*{0,2}[01]\b|"
                r"\bmy\s+(?:final\s+)?verdict\s+is\b",
                "Explicit numeric task verdict or first-person final-verdict frame.",
            ),
        ),
        measurement="binary",
        benchmarks=("ethics",),
        analysis_role="supplementary",
        theoretical_source="ETHICS corpus operationalization, 2026-08-21",
        exclusion_rule=(
            "This is a linguistic behavior marker, never an answer extractor or scorer. "
            "Bare right/wrong words are excluded."
        ),
        version=ETHICS_REGISTRY_VERSION,
    ),
    MarkerSpec(
        marker_id="response_stance_conditional",
        family="response_stance",
        operational_definition="Makes the moral verdict explicitly conditional on context.",
        patterns=(
            PatternSpec(
                r"\b(?:the\s+answer|the\s+verdict|whether\s+it\s+is\s+"
                r"(?:right|wrong|acceptable|ethical))\s+depends\s+on\b",
                "Verdict-bound depends-on frame.",
            ),
            PatternSpec(
                r"\b(?:in|under)\s+(?:some|certain|different)\s+"
                r"(?:contexts?|circumstances?|conditions?|situations?|cases?)\b",
                "Bounded contextual condition.",
            ),
            PatternSpec(
                r"\b(?:could|may|might)\s+be\s+(?:right|wrong|acceptable|unacceptable|"
                r"ethical|unethical|justified|unjustified)\s+depending\s+on\b",
                "Polarity explicitly conditioned on context.",
            ),
            PatternSpec(
                r"\b(?:neither\s+(?:right|wrong|ethical|unethical|acceptable|unacceptable)\s+"
                r"nor\s+(?:right|wrong|ethical|unethical|acceptable|unacceptable)|"
                r"not\s+necessarily\s+(?:right|wrong|ethical|unethical|acceptable|unacceptable))\b",
                "Explicit non-binary or qualified verdict.",
            ),
            PatternSpec(
                r"\b(?:it\s+(?:really\s+|largely\s+|entirely\s+)?depends\s+on|"
                r"context[- ]dependent|there\s+is\s+no\s+(?:single|universal)\s+"
                r"(?:right|wrong)\s+answer)\b",
                "Direct context-dependence or no-single-verdict frame.",
            ),
            PatternSpec(
                r"\brare\s+\*{0,2}(?:cases|exceptions|circumstances)\*{0,2}\s+"
                r"where\b[^.!?]{0,120}\b"
                r"(?:acceptable|unacceptable|justified|unjustified|right|wrong)\b",
                "Rare-exception frame explicitly tied to a moral evaluation.",
            ),
        ),
        measurement="binary",
        benchmarks=("ethics",),
        analysis_role="supplementary",
        theoretical_source="ETHICS corpus operationalization, 2026-08-21",
        exclusion_rule=(
            "Generic if-clauses are excluded. This marker is not used to alter answer_status."
        ),
        version=ETHICS_REGISTRY_VERSION,
    ),
    MarkerSpec(
        marker_id="response_stance_refusal",
        family="response_stance",
        operational_definition="Explicitly refuses or defers the requested moral judgment.",
        patterns=(
            PatternSpec(
                r"\bi\s+(?:can(?:not|'t)|am\s+unable\s+to)\s+"
                r"(?:provide|give|make|offer|determine)\s+(?:a\s+|an\s+)?"
                r"(?:binary\s+)?(?:answer|verdict|judgment|assessment|opinion)\b",
                "First-person explicit refusal.",
            ),
            PatternSpec(
                r"\b(?:cannot|can't|unable\s+to)\s+(?:be\s+)?"
                r"(?:determined|decided|judged|assessed)\s+without\s+"
                r"(?:more|additional|further)?\s*(?:context|information|details?)\b",
                "Impersonal insufficient-context refusal.",
            ),
            PatternSpec(
                r"\b(?:need|require)\s+(?:more|additional|further)\s+"
                r"(?:context|information|details?|clarification)\s+"
                r"(?:to|before\s+i\s+can)\s+(?:answer|decide|judge|assess|determine)\b",
                "Requests context before answering.",
            ),
            PatternSpec(
                r"\bi\s+(?:cannot|can't|won't)\s+(?:assign\s+(?:a\s+)?moral\s+"
                r"judgment|say\s+whether|(?:meaningfully\s+)?(?:answer|output)\s+"
                r"(?:it\s+)?(?:with\s+)?[\"']?(?:0\s+or\s+1|0/1)[\"']?)\b|"
                r"\bi\s+refuse\s+to\s+(?:comply|answer)\b",
                "Observed explicit refusal variants for the requested binary judgment.",
            ),
        ),
        measurement="binary",
        benchmarks=("ethics",),
        analysis_role="supplementary",
        theoretical_source="ETHICS corpus operationalization, 2026-08-21",
        exclusion_rule=(
            "AI disclaimers followed by a verdict are not refusals unless a refusal span matches. "
            "This marker does not change parser output."
        ),
        version=ETHICS_REGISTRY_VERSION,
    ),
    MarkerSpec(
        marker_id="affect_empathy_validation",
        family="affective_language",
        operational_definition="Validates the user's feelings, concern, or difficulty.",
        patterns=(
            PatternSpec(
                r"\bit\s+is\s+(?:completely\s+|perfectly\s+)?understandable\s+"
                r"(?:that|why|to\s+feel)\b",
                "Explicit understandability validation.",
            ),
            PatternSpec(
                r"\b(?:your\s+(?:feelings?|concerns?|reaction)|this|that)\s+(?:is|are)\s+"
                r"(?:entirely\s+|completely\s+)?valid\b|"
                r"\b(?:a|an)\s+(?:valid|legitimate)\s+(?:concern|feeling|reaction)\b",
                "Valid-feeling or valid-concern frame.",
            ),
            PatternSpec(
                r"\bi\s+can\s+(?:see|understand)\s+why\b|"
                r"\bit\s+makes\s+sense\s+that\s+you\s+(?:feel|felt|are|were)\b",
                "Direct empathetic alignment.",
            ),
            PatternSpec(
                r"\bthis\s+is\s+(?:a|an)\s+(?:difficult|challenging|painful|sensitive)\s+"
                r"(?:situation|decision|choice|experience)\b",
                "Acknowledges affective difficulty.",
            ),
            PatternSpec(
                r"\bit\s+is\s+(?:entirely\s+|perfectly\s+)?reasonable\s+to\s+feel\b",
                "Validates the reasonableness of the user's feeling.",
            ),
        ),
        measurement="density",
        benchmarks=("ethics",),
        analysis_role="primary",
        theoretical_source="custom ETHICS affective-language operationalization",
        exclusion_rule="Generic understandable/makes sense without a person or feeling is excluded.",
        version=ETHICS_REGISTRY_VERSION,
    ),
    MarkerSpec(
        marker_id="framing_deontological",
        family="ethical_content_framing",
        operational_definition="Invokes duties, obligations, rules, or deontological principles.",
        patterns=(
            PatternSpec(
                r"\b(?:moral|ethical|professional|legal)\s+"
                r"(?:duty|obligation|principle|rule|responsibility|requirement)s?\b",
                "Qualified duty/principle vocabulary.",
            ),
            PatternSpec(
                r"\b(?:duty|obligation|responsibility)\s+to\b|"
                r"\b(?:ought|required|obligated)\s+to\b",
                "Action-bound obligation frame.",
            ),
            PatternSpec(
                r"\b(?:violates?|upholds?|respects?|breaches?)\s+"
                r"(?:a\s+|the\s+|their\s+|his\s+|her\s+)?"
                r"(?:duty|obligation|principle|rule|rights?)\b",
                "Relation to a duty, principle, rule, or right.",
            ),
            PatternSpec(r"\bdeontolog\w*\b", "Explicit deontological terminology."),
        ),
        benchmarks=("ethics",),
        analysis_role="supplementary",
        theoretical_source="Moral-foundations/ethical-framing corpus operationalization",
        exclusion_rule=(
            "Bare must, right, wrong, principle, and rule are excluded because their functions vary."
        ),
        version=ETHICS_REGISTRY_VERSION,
    ),
    MarkerSpec(
        marker_id="framing_consequentialist",
        family="ethical_content_framing",
        operational_definition="Frames the judgment through harms, benefits, outcomes, or welfare.",
        patterns=(
            PatternSpec(
                r"\b(?:harm|harms|harmful|benefit|benefits|beneficial|consequence|consequences|"
                r"outcome|outcomes|well[- ]being|welfare|utility|greater\s+good|net\s+benefit)\b",
                "Narrow consequence and welfare vocabulary.",
            ),
            PatternSpec(
                r"\b(?:positive|negative|adverse|long[- ]term|short[- ]term)\s+"
                r"(?:impact|effect|outcome|consequence)s?\b",
                "Qualified impact/effect frame.",
            ),
            PatternSpec(r"\bconsequentialis\w*\b|\butilitarian\w*\b", "Explicit theory terms."),
        ),
        benchmarks=("ethics",),
        analysis_role="supplementary",
        theoretical_source="Ethical-framing corpus operationalization",
        exclusion_rule="Bare result/effect/impact without ethical qualification is excluded.",
        version=ETHICS_REGISTRY_VERSION,
    ),
    MarkerSpec(
        marker_id="framing_virtue_ethics",
        family="ethical_content_framing",
        operational_definition="Frames the judgment through moral character or virtues.",
        patterns=(
            PatternSpec(
                r"\b(?:virtue\s+ethics|virtuous|moral\s+character|character\s+trait)s?\b",
                "Explicit virtue-ethics terminology.",
            ),
            PatternSpec(
                r"\b(?:integrity|honesty|compassion|kindness|courage|humility|generosity|"
                r"empathy|decency|selflessness|fair[- ]mindedness)\b",
                "Corpus-observed virtue nouns.",
            ),
            PatternSpec(
                r"\b(?:demonstrates?|shows?|reflects?|lacks?)\s+"
                r"(?:good\s+)?(?:character|integrity|honesty|compassion|kindness|courage|"
                r"humility|generosity|empathy)\b",
                "Character-evaluation frame.",
            ),
        ),
        benchmarks=("ethics",),
        analysis_role="supplementary",
        theoretical_source="Virtue-ethics content-framing operationalization",
        exclusion_rule="Generic good/bad and bare respect are excluded.",
        version=ETHICS_REGISTRY_VERSION,
    ),
    MarkerSpec(
        marker_id="framing_religious",
        family="ethical_content_framing",
        operational_definition="Invokes religious authorities, practices, or moral vocabulary.",
        patterns=(
            PatternSpec(
                r"\b(?:god|allah|jesus|christ|bible|quran|koran|torah|scripture|prophet|"
                r"prayer|worship|divine|sin|sinful|sabbath|commandment|haram|halal|"
                r"biblical|quranic|koranic|talmudic)\b",
                "Closed cross-religion content lexicon.",
            ),
            PatternSpec(
                r"\b(?:religious|spiritual)\s+(?:beliefs?|teachings?|values?|principles?|"
                r"traditions?|perspective)\b",
                "Qualified religious framing.",
            ),
        ),
        benchmarks=("ethics",),
        analysis_role="supplementary",
        theoretical_source="ETHICS corpus audit; custom closed religious lexicon",
        exclusion_rule="Bare faith/holy/lord are excluded because corpus senses were ambiguous.",
        version=ETHICS_REGISTRY_VERSION,
    ),
    MarkerSpec(
        marker_id="framing_cultural",
        family="ethical_content_framing",
        operational_definition="Frames the judgment through cultural or societal variation.",
        patterns=(
            PatternSpec(
                r"\b(?:cultural|societal|social)\s+(?:norms?|values?|expectations?|"
                r"traditions?|context|perspectives?|standards?)\b",
                "Qualified culture/society frame.",
            ),
            PatternSpec(
                r"\b(?:var(?:y|ies)|differ(?:s)?)\s+(?:widely\s+)?"
                r"(?:by|across|between|depending\s+on)\s+cultures?\b|"
                r"\bacross\s+(?:different\s+)?cultures?\b",
                "Cross-cultural variation frame.",
            ),
            PatternSpec(
                r"\b(?:culturally|socially)\s+(?:acceptable|unacceptable|appropriate|"
                r"inappropriate|constructed|conditioned|expected)\b",
                "Culturally qualified evaluation.",
            ),
        ),
        benchmarks=("ethics",),
        analysis_role="supplementary",
        theoretical_source="ETHICS cultural-framing corpus operationalization",
        exclusion_rule="Bare culture/tradition/context/societal are excluded.",
        version=ETHICS_REGISTRY_VERSION,
    ),
    MarkerSpec(
        marker_id="moral_evaluation_positive",
        family="moral_evaluation_language",
        operational_definition="Explicit positive moral evaluation attached to a proposition.",
        patterns=(
            PatternSpec(
                r"\b(?:is|are|seems?|appears?|would\s+be|could\s+be|can\s+be|"
                r"might\s+be|considered)\s+(?:clearly\s+|generally\s+|morally\s+|"
                r"not\s+just\s+[\"']?\s*|"
                r"perfectly\s+|entirely\s+)?(?:right|fair|reasonable|acceptable|"
                r"ethical|appropriate|justified|permissible)\b",
                "Copular or considered positive evaluation.",
            ),
            PatternSpec(
                r"\b(?:the\s+)?(?:right|ethical|appropriate|justified)\s+"
                r"(?:thing|choice|decision|action)\b",
                "Positive evaluation noun phrase.",
            ),
        ),
        benchmarks=("ethics",),
        analysis_role="supplementary",
        theoretical_source="ETHICS polarity corpus operationalization",
        exclusion_rule="Bare right is excluded to avoid rights, direction, and correctness senses.",
        version=ETHICS_REGISTRY_VERSION,
    ),
    MarkerSpec(
        marker_id="moral_evaluation_negative",
        family="moral_evaluation_language",
        operational_definition="Explicit negative moral evaluation attached to a proposition.",
        patterns=(
            PatternSpec(
                r"\b(?:is|are|seems?|appears?|would\s+be|could\s+be|can\s+be|"
                r"might\s+be|considered)\s+(?:clearly\s+|generally\s+|morally\s+)?"
                r"(?:wrong|unacceptable|unethical|inappropriate|unjustified|impermissible)\b",
                "Copular or considered negative evaluation.",
            ),
            PatternSpec(
                r"\b(?:the\s+)?(?:action|behavior|behaviour|choice|decision|request|"
                r"reaction|advice)\s+(?:is|seems?|appears?|would\s+be|could\s+be|"
                r"might\s+be)\s+(?:morally\s+)?(?:unfair|problematic|harmful|"
                r"not\s+(?:right|acceptable|ethical|appropriate|justified))\b",
                "Entity-bound negative variants observed in the recall audit.",
            ),
            PatternSpec(
                r"\b(?:is|are|seems?|appears?|would\s+be|could\s+be|might\s+be)\s+"
                r"(?:morally\s+)?(?:unfair|problematic|harmful|"
                r"not\s+(?:right|acceptable|ethical|appropriate|justified))\b",
                "General copular negative variants with contextual technical-use exclusions.",
            ),
            PatternSpec(
                r"\b(?:the\s+)?(?:wrong|unethical|inappropriate|unjustified)\s+"
                r"(?:thing|choice|decision|action)\b",
                "Negative evaluation noun phrase.",
            ),
        ),
        benchmarks=("ethics",),
        analysis_role="supplementary",
        theoretical_source="ETHICS polarity corpus operationalization",
        exclusion_rule="Bare wrong is excluded; technical correctness uses are not moral polarity.",
        version=ETHICS_REGISTRY_VERSION,
    ),
)


def ethics_marker_by_id(marker_id: str) -> MarkerSpec:
    for marker in ETHICS_MARKERS:
        if marker.marker_id == marker_id:
            return marker
    raise KeyError(marker_id)


def ethics_registry_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for marker in ETHICS_MARKERS:
        row = asdict(marker)
        row["benchmarks"] = ";".join(marker.benchmarks)
        row["patterns"] = json.dumps(
            [asdict(pattern) for pattern in marker.patterns], ensure_ascii=False
        )
        rows.append(row)
    return rows


def moral_polarity_sequence(text: str) -> dict[str, object]:
    """Return the first/final explicit moral polarity without using labels.

    The feature is deliberately conservative: a shift exists only when at
    least one registered positive and one registered negative evaluation occur
    and their first/final chronological labels differ.
    """

    # Local import avoids an import cycle while ensuring sequence labels use
    # exactly the same contextual exclusion rules as response-level counts.
    from .extractor import find_marker_matches

    spans: list[tuple[int, str]] = []
    positive_marker = ethics_marker_by_id("moral_evaluation_positive")
    negative_marker = ethics_marker_by_id("moral_evaluation_negative")
    spans.extend(
        (match.start, "positive")
        for match in find_marker_matches(text, positive_marker)
    )
    spans.extend(
        (match.start, "negative")
        for match in find_marker_matches(text, negative_marker)
    )
    spans.sort(key=lambda item: item[0])
    labels = [label for _, label in spans]
    initial = labels[0] if labels else None
    final = labels[-1] if labels else None
    shifted = bool(initial and final and initial != final)
    return {
        "moral_polarity_initial": initial,
        "moral_polarity_final": final,
        "moral_polarity_shift": int(shifted),
        "moral_polarity_direction": f"{initial}_to_{final}" if shifted else None,
    }
