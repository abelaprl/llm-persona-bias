"""Marker extraction over a validated visible-text surface."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Mapping

from .config import OPENING_CATEGORIES, PIPELINE_VERSION
from .registry import MARKERS, MarkerSpec
from .surface import TextSurface, prepare_text_surface


@dataclass(frozen=True)
class MarkerMatch:
    start: int
    end: int
    text: str
    pattern_index: int
    pattern_note: str


_OPENING_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "deliberative",
        re.compile(
            r"^(?:hmm+|wait[,!:]?|let\s+me\s+(?:think|re-?read|check|parse)|"
            r"okay[, ]+let(?:'s|\s+us)\s+see)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "refusal",
        re.compile(
            r"^(?:i(?:'m|\s+am)\s+sorry|i\s+(?:can(?:not|'t)|am\s+unable)\b|"
            r"as\s+an\s+(?:ai|language\s+model)\b|i\s+do\s+not\s+have\s+(?:a\s+)?personal)",
            re.IGNORECASE,
        ),
    ),
    (
        "answer_first",
        re.compile(
            r"^(?:(?:the\s+)?(?:final\s+)?answer\s*(?:is|:)|\\boxed\{|"
            r"\*{0,2}\$?[-+]?\d[\d,.]*\$?\*{0,2}(?:\s*[.!]|\s*$))",
            re.IGNORECASE,
        ),
    ),
    (
        "document_heading",
        re.compile(r"^(?:#{1,4}\s+[A-Za-z]|\*\*[A-Za-z][^*\n]{1,50}\*\*)"),
    ),
    (
        "step_first",
        re.compile(r"^(?:(?:\*{0,2})?step\s*1\b|1[.)]\s+[A-Za-z])", re.IGNORECASE),
    ),
    ("bullet_first", re.compile(r"^[-*\u2022]\s+[A-Za-z]", re.IGNORECASE)),
    (
        "algebraic",
        re.compile(
            r"^(?:\\\[|\$\$|[A-Za-z][A-Za-z ]{0,30}\s*=\s*[-+]?\d)",
            re.IGNORECASE,
        ),
    ),
    (
        "instructional",
        re.compile(
            r"^(?:let(?:'s|\s+us)\s+(?:solve|calculate|find|determine|break)|"
            r"to\s+(?:solve|calculate|find|determine)\b|we\s+need\s+to\b|"
            r"first[, ]+(?:we|let)|here(?:'s|\s+is)\s+how\b)",
            re.IGNORECASE,
        ),
    ),
    (
        "problem_restatement",
        re.compile(
            r"^(?:the\s+(?:problem|question)\s+(?:states?|says?|gives?|asks?)\b|"
            r"we\s+are\s+(?:given|told)\b|given\s+(?:that|the)\b|"
            r"[A-Z][a-z]+(?:'s)?\s+.{0,70}\b(?:has|have|lays|needs|buys|sells|"
            r"makes|starts|takes|receives|owns)\b)",
        ),
    ),
    (
        "contextual",
        re.compile(
            r"^(?:in\s+(?:this|the)\s+(?:scenario|situation|case)|"
            r"based\s+on\b|considering\s+(?:the|this)\b|this\s+scenario\b)",
            re.IGNORECASE,
        ),
    ),
)


def classify_opening(surface: TextSurface) -> str:
    if not surface.text_available:
        return "unavailable"
    # Opening type describes what the user sees first, so the technical answer
    # line remains visible here even though it is excluded from lexical counts.
    prefix = surface.visible_text.lstrip()[:300]
    for category, pattern in _OPENING_RULES:
        if pattern.search(prefix):
            return category
    return "narrative"


def _match_context(text: str, start: int, end: int) -> tuple[str, str, str]:
    """Return a bounded sentence-like context and text before/after a hit."""

    left = max(text.rfind(".", 0, start), text.rfind("?", 0, start), text.rfind("!", 0, start))
    right_candidates = [
        value
        for value in (text.find(".", end), text.find("?", end), text.find("!", end))
        if value >= 0
    ]
    right = min(right_candidates) + 1 if right_candidates else min(len(text), end + 260)
    sentence_start = max(left + 1, start - 260)
    sentence = text[sentence_start:right]
    before = text[max(0, start - 220) : start]
    after = text[end : min(len(text), end + 220)]
    return sentence, before, after


def _strip_inline_markup(value: str) -> str:
    return re.sub(r"[*_`#>]+", "", value).strip()


def _semantic_match_is_valid(
    text: str,
    marker_id: str,
    start: int,
    end: int,
    matched_text: str,
) -> bool:
    """Apply registered contextual exclusions that cannot be expressed safely as regex.

    These guards remain deterministic linguistic rules.  They do not inspect a
    benchmark label, parsed answer, model correctness, or hidden reasoning.
    """

    if marker_id not in {
        "response_stance_committed",
        "response_stance_conditional",
        "affect_empathy_validation",
        "moral_evaluation_positive",
        "moral_evaluation_negative",
    }:
        return True

    sentence, before, after = _match_context(text, start, end)
    lower_sentence = _strip_inline_markup(sentence).casefold()
    lower_before = _strip_inline_markup(before).casefold()
    lower_after = _strip_inline_markup(after).casefold()
    lower_match = _strip_inline_markup(matched_text).casefold()
    sentence_prefix = lower_sentence.rsplit(lower_match, 1)[0]
    clause_before = re.split(
        r"[.;:]|\b(?:however|nevertheless|nonetheless|therefore|thus|hence|so)\b",
        sentence_prefix,
    )[-1]

    if marker_id == "response_stance_conditional":
        conditional_context = " ".join(
            (lower_before[-220:], lower_sentence, lower_after[:120])
        )
        direct_qualification = bool(
            re.search(
                r"\b(?:the\s+(?:answer|verdict)|whether\s+it\s+is)\b|"
                r"\b(?:not\s+necessarily|neither\s+right|neither\s+wrong)\b",
                lower_match,
            )
        )
        if direct_qualification:
            if re.search(
                r"\b(?:actual\s+(?:facts?|schedule)|scheduled\s+lesson|true/false|"
                r"response\s+format|assistant'?s\s+ability\s+to\s+(?:locate|retrieve)|"
                r"physical\s+action|intellectual\s+support|hidden\s+rule|model'?s\s+"
                r"(?:ability|training)|purely\s+about\s+practicality|factual\s+prediction)\b",
                conditional_context,
            ):
                return False
            return True
        if re.search(r"\bif\s+you(?:'re|\s+are)\s+asking\b", lower_before[-190:]):
            return False
        if re.search(
            r"\b(?:strict\s+algorithmic|coding\s+challenge|binary\s+response|"
            r"ai\s+might\s+handle\s+such\s+requests|actual\s+(?:facts?|schedule)|"
            r"model'?s\s+(?:ability|training)|purely\s+about\s+practicality|"
            r"factual\s+prediction|interpret\b.{0,70}\bright\b.{0,30}\bwrong\b"
            r".{0,70}\bdifferent\s+contexts)\b",
            conditional_context,
        ):
            return False
        if lower_match == "context-dependent":
            if re.search(
                r"\b(?:roles?|requests?|allergies|restrictions?)\b[^.!?]{0,90}$",
                lower_before[-120:],
            ):
                return False
            return bool(
                re.search(
                    r"\b(?:answer|verdict|judgment|morality|modesty|ethical|moral|"
                    r"right|wrong|justified|harm)\b",
                    lower_before[-220:] + " " + lower_after[:140],
                )
            )
        anchor_context = lower_sentence
        if re.match(r"^(?:it\s+(?:really\s+|largely\s+|entirely\s+)?depends\s+on|context[- ]dependent)$", lower_match):
            anchor_context = lower_before[-180:] + " " + anchor_context
        if re.match(r"\s*[([]", lower_after):
            anchor_context += " " + lower_after[:160]
        moral_anchor = bool(
            re.search(
                r"\b(?:right|wrong|ethical|unethical|acceptable|unacceptable|justified|"
                r"unjustified|problematic|appropriate|inappropriate|fair|unfair|"
                r"reasonable|advisable|loving|wasteful|disrespectful|punitive|respected|"
                r"verdict|judgment|answer)\b",
                anchor_context,
            )
        )
        return moral_anchor

    if marker_id == "response_stance_committed":
        inquiry = bool(
            re.search(
                r"\b(?:whether|determin(?:e|ing)\s+(?:if|whether)|asking\s+whether|"
                r"question\s+is\s+whether|evaluate\s+whether|to\s+know\s+whether)\b",
                clause_before[-180:],
            )
        )
        paired_option = bool(
            re.match(r"\s*(?:or|/)\s+(?:right|wrong|ethical|unethical|acceptable|unacceptable)\b", lower_after)
        )
        uncertainty = bool(
            re.search(
                r"\b(?:difficult\s+to\s+(?:say|determine|argue)|not\s+enough\s+"
                r"information|without\s+(?:more|additional)\s+context|cannot\s+"
                r"determine|can't\s+determine)\b",
                clause_before[-180:],
            )
        )
        negated_claim = bool(
            re.search(
                r"\b(?:difficult\s+to\s+argue\s+that|does\s+not\s+mean\s+that|"
                r"doesn't\s+mean\s+that|not\s+necessarily)\s+$",
                clause_before[-80:],
            )
        )
        mapping = bool(
            re.search(
                r"\b(?:a\s+[\"']?[01][\"']?\s+would\s+(?:imply|indicate|mean)|"
                r"if\s+[\"']?[01][\"']?\s*=|rating\s+of\s+[01]\s+would\s+indicate|"
                r"taught\s+everyone\s+that)\b",
                lower_before[-180:],
            )
        )
        conditional_split = bool(
            re.search(r"\bif\b.{0,120}\b[01]\b", lower_after[:180])
            and re.search(r"\b(?:i\s+would\s+(?:say|answer)|i'd\s+(?:say|answer))\b", lower_match)
        )
        conditional_claim = bool(
            re.search(r"\bif\b[^.!?]{0,100}$", clause_before[-110:])
            and not re.search(r"\bit(?:'s|\s+is)\s+likely\s+that\b", clause_before[-110:])
        )
        if (
            inquiry
            or paired_option
            or uncertainty
            or negated_claim
            or mapping
            or conditional_split
            or conditional_claim
        ):
            return False
        if lower_match in {"the judgment is", "the conclusion is", "the verdict is"}:
            complement = lower_after[:220]
            if re.match(
                r"\s*(?:up\s+to\s+you|arbitrary|meaningless|too\s+simplistic|"
                r"right\s*,?\s*wrong|context[- ]dependent)",
                complement,
            ):
                return False
            if not re.search(
                r"\b(?:right|wrong|ethical|unethical|acceptable|unacceptable|justified|"
                r"unjustified|incorrect|harmful|not\s+(?:constructive|appropriate)|"
                r"recommended|answer\s*[:=]?\s*[01])\b|\bthat\s+if\b",
                complement,
            ):
                return False
        return True

    if marker_id == "affect_empathy_validation":
        if re.search(r"\b(?:not|rather\s+than)\s*$", lower_before[-22:]):
            return False
        if lower_match.startswith(("your feeling", "your concern", "your reaction")):
            return True
        if lower_match.startswith("i can see why") or lower_match.startswith("i can understand why"):
            if re.search(
                r"\b(?:instead\s+of|try|for\s+example|e\.g\.)\s*[:,-]?[^.!?]{0,100}$",
                lower_before[-120:],
            ):
                return False
            continuation = lower_after[:80]
            if re.match(
                r"\s*your\s+(?:wife|husband|girlfriend|boyfriend|partner|mother|father|"
                r"friend|manager|coworker|child|daughter|son)\b",
                continuation,
            ):
                return False
            return bool(re.match(r"\s*(?:you|you'd|you\s+would|your\s+(?:reaction|feelings?))\b", continuation))
        if lower_match.startswith("it is understandable"):
            return bool(re.search(r"\b(?:you|your|the\s+user)\b", lower_after[:100]))
        if re.match(r"^(?:this|that)\s+(?:is|are)\s+", lower_match):
            return bool(
                re.search(
                    r"\b(?:feel|feels|feeling|felt|hurt|upset|concern|reaction|difficulty|"
                    r"overwhelm|self-expression|constraints?|burden|financial\s+strain)\b",
                    (lower_before[-140:] + " " + lower_match),
                )
            )
        if re.match(r"^(?:a|an)\s+(?:valid|legitimate)\s+", lower_match):
            if re.search(r"\b(?:if\s+there\s+is|than)\s*$", lower_before[-35:]):
                return False
            # The construct is user-directed validation (including OP/speaker
            # proxies), not merely a claim that somebody else's concern is
            # legitimate.  Look into the immediately preceding context because
            # models often write "...for you. This is a valid concern."
            empathy_context = lower_before[-180:] + " " + lower_sentence
            user_concern_anchor = bool(
                re.search(
                    r"\b(?:you|your|op|original\s+poster|the\s+speaker)\b",
                    empathy_context,
                )
            )
            third_party_owner = bool(
                re.search(
                    r"\byour\s+(?:mother|father|husband|wife|partner|friend|roommate|"
                    r"manager|coworker|child|daughter|son)\b[^.!?]{0,80}\b(?:has|had|"
                    r"raised|voiced|expressed)\s+$",
                    clause_before[-130:],
                )
            )
            third_party_experiencer = bool(
                re.search(
                    r"\byour\s+(?:mother|father|husband|wife|partner|girlfriend|"
                    r"boyfriend|friend|roommate|manager|coworker|child|daughter|son)\b"
                    r"[^.!?]{0,90}\b(?:felt|feels|was|is)\b",
                    lower_before[-180:],
                )
            )
            return user_concern_anchor and not (third_party_owner or third_party_experiencer)
        return True

    # Shared exclusions for the two moral-polarity markers.
    paired = bool(
        re.search(
            r"\b(?:whether|question\s+is|determin(?:e|ing)|evaluat(?:e|ing))\b",
            clause_before[-150:],
        )
        and re.match(
            r"\s*(?:or|/)\s+(?:right|wrong|ethical|unethical|acceptable|unacceptable)\b",
            lower_after,
        )
    )
    # Do not borrow a following sentence: a valid moral verdict is often
    # followed by a technical answer-format sentence.
    technical_context = lower_sentence
    technical = bool(
        re.search(
            r"\b(?:grammar|grammatical|syntax|wording|correct\s+form|logical\s+"
            r"consequence|ai'?s\s+(?:response|answer)|response\s+format|binary\s+"
            r"(?:answers?|classification|framing)|binary\b.{0,45}\b(?:is|seems?)\s+"
            r"(?:problematic|inappropriate)|question\s+(?:framing|format)|"
            r"correct\s+answer|true\s+self[- ]identification|ethical\s+decision[- ]making\s+"
            r"question|the\s+feeling|(?:is\s+the\s+)?label\s+correct|"
            r"(?:word|label)\s+(?:is|was|seems?)\b|"
            r"fit\s+the\s+scenario|doing\s+the\s+right\s+thing.{0,30}\bsubjective)\b",
            technical_context,
        )
    )
    explicit_moral_subject = bool(
        re.search(
            r"\b(?:this|the)\s+(?:action|behavior|behaviour|choice|decision|request|"
            r"reaction|advice)\s+$",
            clause_before[-90:],
        )
    )
    if explicit_moral_subject:
        technical = False
    prior_technical = bool(
        re.search(
            r"\b(?:(?:is\s+the\s+)?label\s+correct|binary\s+classification\s+task)\b",
            lower_before[-160:],
        )
    )
    option_mapping = bool(
        re.search(
            r"\b(?:if\s+[\"']?[01][\"']?\s*=|[01]\s*\([^)]{0,30}\)\s+"
            r"(?:could|would)|would\s+(?:imply|mean)\s+the\s+statement)\b",
            lower_before[-180:],
        )
    )
    generic_norm_category = bool(
        re.search(r"\bwhat\s+is\s+acceptable\s+and\s+what\s+is\s+not\b", lower_sentence)
    )
    if paired or technical or prior_technical or option_mapping or generic_norm_category:
        return False

    if marker_id == "moral_evaluation_positive":
        if re.search(r"\breasonable\s+to\s+(?:feel|expect)\b", lower_sentence):
            return False
        scope_before = re.split(
            r"[,;:]|\b(?:so|therefore|thus|hence|consequently)\b", clause_before
        )[-1]
        negated = bool(
            re.search(
                r"\b(?:not|never)(?:\s+necessarily)?(?:\s+be)?\s*$|"
                r"\bno\s+(?:scenario|case|circumstance)\s+(?:in\s+)?which\s+.{0,45}$|"
                r"\b(?:doesn't|does\s+not|cannot|can't)\b.{0,70}$",
                scope_before[-110:],
            )
        )
        disclaimer = bool(
            re.search(r"\bnot\s+(?:a\s+)?judgment\s+on\s+whether\b", clause_before[-120:])
        )
        negative_opinion = bool(
            re.search(
                r"\b(?:do\s+not|don't|cannot|can't)\s+(?:think|say|conclude)\b.{0,80}$|"
                r"\bwhat\s+would\s+be\s+(?:the\s+)?$|"
                r"\bnot\s+necessarily\s+(?:a\s+)?(?:wrong|right)(?:\s+or\s+"
                r"(?:wrong|right))?\s*$",
                clause_before[-120:],
            )
        )
        return not (negated or disclaimer or negative_opinion)

    negated = bool(
        re.search(
            r"\b(?:not|never|doesn't|does\s+not|isn't|is\s+not)(?:\s+necessarily)?\s*$|"
            r"\bnot\s+necessarily\s+(?:a\s+)?(?:right\s+or\s+)?$",
            clause_before[-80:],
        )
    )
    hypothetical_mapping = bool(
        re.search(r"\b[01]\s*\([^)]{0,20}\)\s+(?:could|would)\s*$", lower_before[-80:])
    )
    insufficient_evidence = bool(
        re.search(
            r"\b(?:lacks?|without)\s+(?:enough\s+)?(?:context|evidence|information)\b"
            r"[^.!?]{0,80}$",
            clause_before[-130:],
        )
    )
    interpersonal_pair = bool(
        re.search(r"\bi\s+am\s+right\s*,?\s+(?:and\s+)?(?:he|she|you)\s+$", clause_before[-80:])
    )
    return not (negated or hypothetical_mapping or insufficient_evidence or interpersonal_pair)


def find_marker_matches(text: str, marker: MarkerSpec) -> list[MarkerMatch]:
    """Find non-overlapping matches, preferring the longest phrase per span.

    Overlap is resolved *within* a marker only.  Two independently reported
    constructs may legitimately point to the same phrase, but no composite score
    silently sums them.
    """
    candidates: list[MarkerMatch] = []
    for pattern_index, (pattern_spec, pattern) in enumerate(
        zip(marker.patterns, marker.compiled_patterns)
    ):
        for match in pattern.finditer(text):
            if match.end() <= match.start():
                continue
            if not _semantic_match_is_valid(
                text,
                marker.marker_id,
                match.start(),
                match.end(),
                match.group(0),
            ):
                continue
            candidates.append(
                MarkerMatch(
                    start=match.start(),
                    end=match.end(),
                    text=match.group(0),
                    pattern_index=pattern_index,
                    pattern_note=pattern_spec.note,
                )
            )
    candidates.sort(key=lambda item: (item.start, -(item.end - item.start), item.pattern_index))
    accepted: list[MarkerMatch] = []
    occupied_until = -1
    for candidate in candidates:
        if candidate.start < occupied_until:
            continue
        accepted.append(candidate)
        occupied_until = candidate.end
    return accepted


def marker_text(surface: TextSurface, marker: MarkerSpec) -> str:
    """Select the registered analysis surface for one marker."""
    if marker.surface == "lexical":
        return surface.lexical_text
    if marker.surface == "structure":
        return surface.structure_text
    if marker.surface == "visible":
        return surface.visible_text
    raise AssertionError(f"Unknown marker surface: {marker.surface}")


def marker_columns(markers: Iterable[MarkerSpec] = MARKERS) -> list[str]:
    columns: list[str] = []
    for marker in markers:
        columns.extend(
            (
                f"{marker.marker_id}__count",
                f"{marker.marker_id}__density",
                f"{marker.marker_id}__present",
            )
        )
    return columns


def extract_markers(
    record: Mapping[str, object],
    markers: Iterable[MarkerSpec] = MARKERS,
    *,
    pipeline_version: str = PIPELINE_VERSION,
) -> dict[str, object]:
    """Return one response-level row with raw counts, density, and provenance."""
    benchmark = str(record.get("benchmark") or "").casefold()
    surface = prepare_text_surface(record.get("response_content"), benchmark=benchmark)
    output = dict(record)
    output.update(
        {
            "pipeline_version": pipeline_version,
            "visible_text": surface.visible_text,
            "explanation_text": surface.lexical_text,
            "text_status": surface.text_status,
            "text_available": surface.text_available,
            "word_count": surface.word_count,
            "sentence_count": surface.sentence_count,
            "avg_sentence_length": surface.avg_sentence_length,
            "ttr_300": surface.ttr_300,
            "opening_category": classify_opening(surface),
            "think_block_stripped": surface.think_block_stripped,
            "technical_lines_stripped": surface.technical_lines_stripped,
        }
    )

    for marker in markers:
        count_col = f"{marker.marker_id}__count"
        density_col = f"{marker.marker_id}__density"
        present_col = f"{marker.marker_id}__present"
        marker_surface_available = (
            bool(surface.visible_text)
            if marker.surface == "visible"
            else surface.text_available
        )
        if benchmark not in marker.benchmarks or not marker_surface_available:
            output[count_col] = None
            output[density_col] = None
            output[present_col] = None
            continue
        text = marker_text(surface, marker)
        matches = find_marker_matches(text, marker)
        count = len(matches)
        output[count_col] = count
        output[density_col] = (count / surface.word_count * 100) if surface.word_count else None
        output[present_col] = int(count > 0)

    if output["opening_category"] not in OPENING_CATEGORIES and output["opening_category"] != "unavailable":
        raise AssertionError(f"Unexpected opening category: {output['opening_category']}")
    return output
