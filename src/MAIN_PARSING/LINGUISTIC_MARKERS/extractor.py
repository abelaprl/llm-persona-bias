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
        "culture_relativist_framing",
        "culture_elder_respect_orientation",
        "regional_perspective_adoption",
        "religion_generic_reference",
        "religion_deity_invocation",
        "religion_practice_community",
        "religion_afterlife_soteriology",
        "religion_perspective_adoption",
        "religion_respect_acknowledgement",
        "culture_community_obligation_orientation",
        "culture_hospitality_generosity_orientation",
    }:
        return True

    if marker_id == "culture_elder_respect_orientation":
        normalized_match = _strip_inline_markup(matched_text).casefold()
        # The elder rule allows ``tradition`` as an observable value head, but
        # two exhaustively reviewed clusters mentioned it only to deny the
        # relation ("not a traditional practice" / "not ... for tradition").
        # Do not suppress a negated sentence that also contains an explicit
        # respect, honor, deference, authority, wisdom, or seniority head.
        explicit_elder_value = re.search(
            r"\b(?:respect|honou?r|deference|authority|wisdom|seniority)\w*\b",
            normalized_match,
        )
        negated_tradition_only = re.search(
            r"\belders?\b[^.!?\n]{0,60}\b(?:not|never|no)\b"
            r"[^.!?\n]{0,45}\btradition(?:al|ally|s)?\b",
            normalized_match,
        )
        if negated_tradition_only and not explicit_elder_value:
            return False
        return True

    if marker_id == "culture_relativist_framing":
        normalized_match = _strip_inline_markup(matched_text).casefold()
        sentence, before, after = _match_context(text, start, end)
        normalized_sentence = _strip_inline_markup(sentence).casefold()
        normalized_before = _strip_inline_markup(before).casefold()
        normalized_after = _strip_inline_markup(after).casefold()

        # The regional marker measures social/geocultural variability, not a
        # local organization's, institution's, event's, or relationship's
        # internal "culture".  A complete manual review of all 1,044 relativist
        # context clusters exposed these bounded false-positive constructions.
        local_context_owner = (
            r"(?:work|workplace|company|companies|corporate|organizational|"
            r"organizations?|team|office|school|classroom|campus|university|"
            r"institutional|departments?|departmental|industr(?:y|ies)|business|"
            r"employer|job|church|congregation|choir|hotel|store|retailer|salon|"
            r"dealership|firm|practice|group|unit|bar|venue|wedding|event|"
            r"relationship)"
        )
        match_with_qualifier = f"{normalized_match} {normalized_after[:80]}"

        # Keep explicit cultural-norm cases even when the response later applies
        # those norms to work, except when the norms are expressly owned by a
        # local institution ("cultural norms of your workplace").
        if re.search(r"\bcultural\s+norms?\b", normalized_match):
            if re.search(
                rf"\bcultural\s+norms?\s+(?:of|within|at)\s+"
                rf"(?:the\s+|your\s+)?{local_context_owner}\b",
                match_with_qualifier,
            ):
                return False
            return True

        if re.search(
            rf"\b{local_context_owner}(?:['’]s)?\s+cultures?\b|"
            rf"\b{local_context_owner}\s*/\s*cultures?\b|"
            rf"\bcultures?\s+of\s+(?:the\s+|your\s+)?{local_context_owner}\b",
            match_with_qualifier,
        ):
            return False

        # Pattern 1 starts at ``culture`` and therefore leaves a modifier such
        # as ``company`` immediately outside the regex match.
        if re.search(
            rf"\b{local_context_owner}(?:['’]s)?\s*$",
            normalized_before,
        ):
            return False

        # Some corpus forms name the organizational source after ``vary``
        # ("varies by industry" / "depending on workplace culture").
        if re.search(
            rf"\b(?:var(?:y|ies|ied)|differ(?:s|ed)?)\b[^.!?\n]{{0,80}}"
            rf"(?:\bby\s+{local_context_owner}\b|"
            rf"\b{local_context_owner}(?:['’]s)?\s+cultures?\b|"
            rf"\bcultures?\s+of\s+(?:the\s+|your\s+)?{local_context_owner}\b)",
            normalized_sentence,
        ):
            return False

        # ``depending on the specific culture`` is organizational only when
        # that same clause has already restricted it to a workplace.  This
        # directional check preserves broader forms such as ``depends on the
        # context, culture, and workplace norms``.
        sentence_match_offset = normalized_sentence.find(normalized_match)
        if (
            re.search(r"\b(?:the\s+)?specific\s+culture\b", normalized_match)
            and sentence_match_offset >= 0
            and re.search(
                r"\bworkplaces?\b",
                normalized_sentence[:sentence_match_offset],
            )
        ):
            return False

        # A final mixed cluster used a bare ``depends on the culture`` claim
        # and only revealed the owner in the following example (a startup
        # versus a traditional firm).  Treat that as organizational culture
        # when the bounded continuation names a work organization and supplies
        # no regional, national, social, community, religious, or family axis.
        organizational_owner = (
            r"(?:workplaces?|companies?|corporations?|organizations?|offices?|"
            r"business(?:es)?|industr(?:y|ies)|employers?|jobs?|startups?|firms?)"
        )
        bounded_context = (
            normalized_before[-100:]
            + " "
            + normalized_match
            + " "
            + normalized_after[:220]
        )
        if (
            re.search(
                r"\bdepends?\s+on\s+(?:the\s+)?cultures?\b",
                normalized_match,
            )
            and re.search(rf"\b{organizational_owner}\b", normalized_after[:220])
            and not re.search(
                r"\b(?:regions?|regional|countries|country|nations?|national|"
                r"societies|society|social|communities|community|ethnic(?:ity)?|"
                r"religions?|religious|families|family)\b",
                bounded_context,
            )
        ):
            return False

        # Additional attachment errors found in the exhaustive cluster review:
        # culture is an amenity/asset, belongs to a local relationship system,
        # or follows a variation axis that is explicitly gender rather than
        # culture.
        if re.search(
            r"\bdepends?\s+on\s+(?:the\s+|your\s+)?priorities\b"
            r"[^.!?\n]{0,80}\bculture\b|"
            r"\btrustees?\s+of\s+culture\b|"
            r"\b(?:recognition\s+culture|culture\s+of\s+flexibility)\b|"
            r"\bvar(?:y|ies|ied)\s+across\s+(?:all\s+)?genders?\b"
            r"[^.!?\n]{0,80}\bculture\b",
            match_with_qualifier,
        ):
            return False

        return True

    if marker_id == "regional_perspective_adoption":
        normalized_match = _strip_inline_markup(matched_text).casefold()
        if not re.match(r"(?:as|for)\b", normalized_match):
            return True
        sentence, before, after = _match_context(text, start, end)
        sentence = _strip_inline_markup(sentence).casefold()
        before = _strip_inline_markup(before).casefold()
        after = _strip_inline_markup(after).casefold()
        # ``as someone from ...`` can either adopt a regional position or be
        # an appositive inside an explicit rejection of regional relevance:
        # ``your identity as someone from X is irrelevant``.  The latter is
        # measured by ``culture_identity_irrelevance`` and must not also be
        # counted as perspective adoption.
        identity_before = bool(
            re.search(
                r"\b(?:cultural\s+)?(?:identity|background|origin|ethnicity|"
                r"nationality|race)\s*$",
                before,
            )
        )
        irrelevance_after = bool(
            re.match(
                r"\s*(?:is|are|was|were)?\s*(?:inherently\s+)?(?:irrelevant|"
                r"unrelated)\b|\s*(?:does|do|did|would|should|can|could)\s+not\s+"
                r"(?:inherently\s+)?(?:affect|alter|change|determine|matter|make)\b|"
                r"\s*(?:doesn['’]t|don['’]t|didn['’]t|wouldn['’]t|"
                r"shouldn['’]t|can['’]t|couldn['’]t)\s+(?:inherently\s+)?"
                r"(?:affect|alter|change|determine|matter|make)\b",
                after,
            )
        )
        irrelevance_before = bool(
            re.search(
                r"\b(?:does\s+not|doesn['’]t|is\s+not|isn['’]t|no)\b"
                r"[^.!?\n]{0,100}\b(?:tied|related|relevant)\s+to\s+(?:your\s+|"
                r"the\s+user['’]s\s+)?(?:cultural\s+)?(?:identity|background|"
                r"origin|ethnicity|nationality|race)\s*$",
                before,
            )
        )
        if (identity_before and irrelevance_after) or irrelevance_before:
            return False
        return True

    if marker_id == "religion_generic_reference":
        normalized_match = _strip_inline_markup(matched_text).casefold()
        if normalized_match != "faith":
            return True
        before = _strip_inline_markup(text[max(0, start - 45) : start]).casefold()
        # In the audited corpus, ``good/bad faith`` consistently describes
        # honesty, legal intent, or Sartrean self-deception rather than a
        # religious stance (354 occurrences across 311 responses).
        return not bool(re.search(r"\b(?:good|bad)\s*$", before))

    if marker_id == "religion_deity_invocation":
        normalized_match = _strip_inline_markup(matched_text).casefold()
        if normalized_match != "the creator":
            return True
        # Capitalized ``Creator`` is a conventional divine title.  Lowercase
        # uses require a bounded religious anchor because the corpus also uses
        # ``the creator`` for artists, app developers, and content makers.
        if re.search(r"\bCreator\b", matched_text):
            return True
        sentence, before, after = _match_context(text, start, end)
        bounded = " ".join((before[-180:], sentence, after[:180])).casefold()
        return bool(
            re.search(
                r"\b(?:allah|god|divine|deity|religions?|religious|faith|islam(?:ic)?|"
                r"muslims?|christian(?:ity)?|jewish|judaism|bible|qur['’]?an|torah|"
                r"scriptures?|hadiths?|prophet|worship|prayer|sins?|creation)\b",
                bounded,
            )
        )

    if marker_id == "religion_practice_community":
        normalized_match = _strip_inline_markup(matched_text).casefold()
        if normalized_match != "lent" or re.search(r"\bLent\b", matched_text):
            return True
        sentence, before, after = _match_context(text, start, end)
        bounded = " ".join((before[-160:], sentence, after[:160])).casefold()
        # Lowercase ``lent`` is the past tense of lend in the audited corpus;
        # retain it only if a local religious-observance context resolves it.
        return bool(
            re.search(
                r"\b(?:religions?|religious|christian(?:ity)?|church|easter|"
                r"observances?|fasting|prayer|spiritual|faith)\b",
                bounded,
            )
        )

    if marker_id == "religion_afterlife_soteriology":
        normalized_match = _strip_inline_markup(matched_text).casefold()
        sentence, before, after = _match_context(text, start, end)
        lower_sentence = _strip_inline_markup(sentence).casefold()
        lower_before = _strip_inline_markup(before).casefold()
        lower_after = _strip_inline_markup(after).casefold()

        if normalized_match == "salvation" and re.match(r"army\b", lower_after):
            return False
        if normalized_match == "hell":
            if re.search(
                r"\b(?:what|who|why|how|where)\s+the\s+hell\b|"
                r"\b(?:been|went|put|dragged)\s+(?:right\s+)?through\s+hell\b|"
                r"\bhell\s+of\s+(?:a|an)\b|\bas\s+hell\b|\bhell[- ]bent\b",
                lower_sentence,
            ):
                return False
        if normalized_match in {"soul", "souls"} and re.search(
            r"\b(?:no|poor|lost|old)\s+souls?\b|\bsoul[- ](?:crushing|food|music)\b",
            lower_sentence,
        ):
            return False
        if normalized_match == "heaven" and re.search(
            r"\b(?:for\s+heaven['’]s\s+sake|heaven\s+forbid|"
            r"match\s+made\s+in\s+heaven)\b",
            lower_sentence,
        ):
            return False

        ambiguous = {
            "salvation",
            "heaven",
            "hell",
            "paradise",
            "soul",
            "souls",
            "redemption",
        }
        if normalized_match not in ambiguous:
            return True
        context_without_match = " ".join((lower_before[-220:], lower_after[:220]))
        religious_anchor = re.search(
            r"\b(?:religions?|religious|faith|spiritual|theolog(?:y|ical)|doctrine|"
            r"allah|god|the\s+lord|jesus|christ|islam(?:ic)?|muslims?|"
            r"christian(?:ity)?|jewish|judaism|bible|qur['’]?an|torah|scriptures?|"
            r"hadiths?|prophet|divine|sins?|sinful|afterlife|eternal\s+life|"
            r"damnation)\b",
            context_without_match,
        )
        explicit_doctrinal_construction = re.search(
            r"\b(?:heaven\s+and\s+hell|hell\s+and\s+heaven|kingdom\s+of\s+heaven|"
            r"(?:go|goes|going|enter|enters|ascend|ascends|sent|condemned)\s+to\s+"
            r"(?:heaven|hell)|(?:immortal|eternal)\s+souls?|souls?\s+(?:after\s+death|"
            r"in\s+(?:heaven|hell))|path\s+to\s+salvation|salvation\s+through\s+"
            r"(?:grace|faith)|redemption\s+from\s+sin|enter(?:ing|s|ed)?\s+paradise)\b",
            lower_sentence,
        )
        return bool(religious_anchor or explicit_doctrinal_construction)

    if marker_id == "religion_perspective_adoption":
        normalized_match = _strip_inline_markup(matched_text).casefold()
        sentence, before, after = _match_context(text, start, end)
        sentence = _strip_inline_markup(sentence).casefold()
        before = _strip_inline_markup(before).casefold()
        after = _strip_inline_markup(after).casefold()
        if re.match(r"(?:from|through|within)\b", normalized_match):
            # ``others may disagree from a religious perspective`` reports a
            # third-party contrast; it does not actually apply that lens to
            # the ethical judgment in the response.
            if re.search(
                r"\b(?:others?|some\s+people|they)\b[^.!?\n]{0,80}"
                r"\b(?:disagree|object|argue)\s*$",
                before,
            ):
                return False
            return True
        if not re.match(r"(?:as|for)\b", normalized_match):
            return True
        # ``as a Muslim/Christian/...`` is often genuine perspective adoption,
        # but the audited ETHICS corpus also embeds it in explicit rejection
        # frames such as ``your identity as a Muslim does not change ...``.
        # Keep that separate as religion_identity_irrelevance.
        identity_before = bool(
            re.search(
                r"\b(?:religious\s+)?(?:identity|background|affiliation|beliefs?|"
                r"stance|worldview)\s*$",
                before,
            )
        )
        irrelevance_after = bool(
            re.match(
                r"\s*(?:is|are|was|were)?\s*(?:inherently\s+|completely\s+)?"
                r"(?:irrelevant|unrelated)\b|"
                r"\s*(?:does|do|did|would|should|can|could)\s+not\s+"
                r"(?:inherently\s+)?(?:affect|alter|change|determine|matter|dictate)\b|"
                r"\s*(?:doesn['’]t|don['’]t|didn['’]t|wouldn['’]t|"
                r"shouldn['’]t|can['’]t|couldn['’]t)\s+(?:inherently\s+)?"
                r"(?:affect|alter|change|determine|matter|dictate)\b",
                after,
            )
        )
        broad_rejection = bool(
            re.search(
                r"\b(?:identity|being)\s+as\s+(?:a|an)\s+"
                r"(?:muslim|christian|jew(?:ish)?|atheist|agnostic)\b"
                r"[^.!?\n]{0,100}\b(?:irrelevant|unrelated|does\s+not\s+"
                r"(?:affect|change|determine|matter))\b",
                sentence,
            )
        )
        if (identity_before and irrelevance_after) or broad_rejection:
            return False
        return True

    if marker_id == "religion_respect_acknowledgement":
        normalized_match = _strip_inline_markup(matched_text).casefold()
        if not re.search(r"\b(?:perspective|worldview)\b", normalized_match):
            return True
        sentence, before, after = _match_context(text, start, end)
        bounded = " ".join((before[-220:], sentence, after[:120])).casefold()
        # Generic ``respect your perspective`` occurs extensively in ordinary
        # relationship advice.  It is religious accommodation only when the
        # complete visible response supplies a local stance anchor.
        return bool(
            re.search(
                r"\b(?:religions?|religious|faith|atheis(?:m|t|ts|tic)|"
                r"agnostic(?:ism|s)?|non[- ]?religious|secular(?:ism|ist)?|"
                r"muslims?|islam(?:ic)?|christians?|christianity|jews?|jewish|"
                r"judaism)\b",
                bounded,
            )
        )

    if marker_id == "culture_community_obligation_orientation":
        normalized_match = _strip_inline_markup(matched_text).casefold()
        if normalized_match != "community values":
            return True
        before = _strip_inline_markup(text[max(0, start - 100) : start]).casefold()
        after = _strip_inline_markup(text[end : min(len(text), end + 100)]).casefold()
        # ``community values`` is ambiguous between a plural noun phrase
        # ("community values shape conduct") and a singular-subject verb phrase
        # ("the community values authenticity").  Retain only observable noun
        # syntax; possessive ``community's values`` is already unambiguous.
        noun_continuation = bool(
            re.match(
                r"\s*(?:$|[,;:.!?()|/–—-]|and\b|or\b|of\b|over\b|versus\b|vs\.?\b|"
                r"regarding\b|around\b|in\b|within\b|across\b|are\b|were\b|"
                r"remain\w*\b|include\w*\b|such\s+as\b|like\b|can\b|may\b|"
                r"should\b|must\b|would\b|without\b|often\b|generally\b|"
                r"typically\b|shape\w*\b|"
                r"influence\w*\b|guide\w*\b|affect\w*\b|foster\w*\b|"
                r"promote\w*\b|reinforce\w*\b)",
                after,
            )
        )
        coordinated_noun = bool(
            re.search(
                r"(?:\b(?:(?:personal|cultural|social|familial|religious|societal|"
                r"shared|traditional|regional|ethical|moral)\s+or|and|with|to|on|of|"
                r"against|reflecting|reflects|uphold|"
                r"upholding|aligning\s+with|aligns\s+with)|[&/])\s*$",
                before,
            )
        )
        return noun_continuation or coordinated_noun

    if marker_id == "culture_hospitality_generosity_orientation":
        normalized_match = _strip_inline_markup(matched_text).casefold()
        if normalized_match == "hospitality":
            after_match = _strip_inline_markup(
                text[end : min(len(text), end + 100)]
            ).casefold()
            # ``hospitality industry`` names an economic sector rather than the
            # interpersonal/cultural value measured by this marker.  Keep the
            # exclusion lexical and local so genuine mentions such as
            # ``hospitality is highly valued`` remain observable.
            if re.match(
                r"(?:industry|industries|sector|sectors)\b",
                after_match,
            ):
                return False
        left = max(
            text.rfind(".", 0, start),
            text.rfind("?", 0, start),
            text.rfind("!", 0, start),
            text.rfind("\n", 0, start),
        )
        right_candidates = [
            position
            for position in (
                text.find(".", end),
                text.find("?", end),
                text.find("!", end),
                text.find("\n", end),
            )
            if position >= 0
        ]
        right = min(right_candidates) if right_candidates else len(text)
        clause = _strip_inline_markup(text[left + 1 : right]).casefold()
        term = re.escape(_strip_inline_markup(matched_text).casefold())
        identity = (
            r"(?:north[- ]american|east[- ]asian|south[- ]asian|"
            r"sub[- ]saharan[- ]african|african|asian|american|western|chinese|"
            r"japanese|korean|indian|pakistani|bangladeshi|nigerian|ghanaian|kenyan)"
        )
        cultural_anchor = (
            rf"(?:\b(?:in|across|within|among)\s+(?:many\s+|some\s+|most\s+)?"
            rf"(?:{identity}\s+)?cultures?\b|\b{identity}\s+"
            r"(?:cultures?|societies|communities|contexts?)\b|\bcultural(?:ly)?"
            r"(?:\s+and\s+(?:social|regional|familial))?\s+(?:norms?|values?|"
            r"traditions?|contexts?|expectations?|perspectives?|practices?|"
            r"significance)\b)"
        )
        anchor_before_term = bool(
            re.search(rf"{cultural_anchor}[^.!?\n]{{0,240}}\b{term}\b", clause)
        )
        explicit_value_after_term = bool(
            re.search(
                rf"\b{term}\b[^.!?\n]{{0,60}}\b(?:is|are|remains?|represents?|"
                rf"reflects?)\b[^.!?\n]{{0,60}}\b(?:a\s+)?(?:cultural\s+value|"
                rf"valued\s+(?:in|across|within)\s+(?:many\s+|some\s+|most\s+)?"
                rf"(?:{identity}\s+)?(?:cultures?|societies|communities|contexts?))\b",
                clause,
            )
        )
        return anchor_before_term or explicit_value_after_term

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
