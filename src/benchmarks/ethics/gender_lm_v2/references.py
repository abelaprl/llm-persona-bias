"""Conservative evidence flags; literal overlap never establishes coreference."""
import re
from .contracts import Surface, Span


def annotate_references(surface: Surface, spans: list[Span], scenario_text: str = "") -> list[Span]:
    quotes = [(m.start(), m.end()) for m in re.finditer(r'"[^"\n]+"', surface.visible_text)]
    scenario_terms = set(re.findall(r"\b(?:man|woman|men|women|male|female|gender)\b", scenario_text.casefold()))
    for span in spans:
        quoted = any(start <= span.start and span.end <= end for start, end in quotes)
        before = surface.visible_text[max(0, span.start - 80):span.start]
        if quoted:
            span.flags.append("quoted")
            span.reference_type = "quoted_or_hypothetical"
        if re.search(r"\b(?:not|never|cannot|can't)\b[^.!?]{0,40}$", before, re.I):
            span.flags.append("nearby_negation_scope_unresolved")
        if span.marker_id.startswith("gender"):
            if span.marker_id == "gender_hypothetical_identity_reference":
                span.reference_type = "quoted_or_hypothetical"
            elif not quoted and re.search(r"\b(?:you|your|user's)\b", span.text, re.I):
                span.reference_type = "explicit_user"
            if scenario_terms.intersection(re.findall(r"\b(?:man|woman|men|women|male|female|gender)\b", span.text.casefold())):
                span.flags.append("same_gender_term_in_scenario_not_coreference")
            if span.reference_type == "unknown":
                span.flags.append("referent_unresolved")
    return spans
