"""Typed records shared by extraction, evidence and audit."""
from dataclasses import dataclass, field


@dataclass
class Surface:
    raw_text: str
    visible_text: str
    explanation_text: str
    status: str
    layout_fidelity: str = "unknown"
    offset_map: str = "identity_masked_normalization"
    flags: list[str] = field(default_factory=list)
    word_count: int = 0
    structure_text: str = ""


@dataclass
class Span:
    marker_id: str
    rule_id: str
    start: int
    end: int
    text: str
    surface_name: str
    flags: list[str] = field(default_factory=list)
    evidence_strength: str = "registered_pattern"
    rule_status: str = "candidate"
    reference_type: str = "unknown"


@dataclass
class AuditDecision:
    rule_id: str
    status: str
    n_reviewed: int
    false_positive: int
    reason: str
