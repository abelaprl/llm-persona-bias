"""Frozen baseline detector bank with separately versioned candidate refinements."""
import json
from pathlib import Path
import re
from functools import lru_cache

from .contracts import Surface, Span
from ._baseline.registry import MarkerSpec, PatternSpec
from ._baseline.extractor import find_marker_matches


@lru_cache(maxsize=1)
def load_registry() -> list[dict]:
    return json.loads(Path(__file__).with_name("registry.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=2)
def _specs(serialized: str):
    registry = json.loads(serialized)
    return [(item, MarkerSpec(marker_id=item["marker_id"], family=item["family"],
                             operational_definition=item["operational_definition"],
                             patterns=tuple(PatternSpec(pattern=x["pattern"], note=x.get("note", ""), flags=x.get("flags", re.I)) for x in item["patterns"]),
                             surface=item["surface"])) for item in registry]


def detect_markers(surface: Surface, registry: list[dict] | None = None) -> list[Span]:
    if surface.status != "substantive_explanation":
        return []
    specs = _default_specs() if registry is None or registry is load_registry() else _specs(json.dumps(registry, sort_keys=True))
    spans = []
    for item, spec in specs:
        field = {"lexical": "explanation_text", "structure": "structure_text", "visible": "visible_text"}[spec.surface]
        text = getattr(surface, field)
        for hit in find_marker_matches(text, spec):
            before = text[max(0, hit.start - 100):hit.start]
            if spec.marker_id == "stance_booster" and re.search(r"\b(?:not|never)(?:\s+necessarily)?\s*$", before, re.I):
                continue
            if spec.marker_id == "connective_conditional" and hit.text.lower() == "if" and re.search(r"\b(?:determine|decide|know|judge|assess|tell|wonder|see)\s+$", before, re.I):
                # Keep pragmatic offers pending a narrower adjudicated distinction.
                if not re.search(r"\blet\s+me\s+know\s+$", before, re.I):
                    continue
            pattern = item["patterns"][hit.pattern_index]
            spans.append(Span(spec.marker_id, pattern["rule_id"], hit.start, hit.end, hit.text, field,
                              flags=["candidate_requires_independent_audit"], evidence_strength="registered_pattern_with_context_guard"))
    return spans


@lru_cache(maxsize=1)
def _default_specs():
    return _specs(json.dumps(load_registry(), sort_keys=True))
