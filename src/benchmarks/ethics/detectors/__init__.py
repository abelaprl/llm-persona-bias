"""Detector bank.

Every detector runs on every response. None of them decides anything on its
own - they emit `Signal` objects and `resolver.py` picks the winner with a
fixed priority. This is deliberate: a first-match cascade stops as soon as it
commits, so it cannot report that the digit and the words disagreed, and that
disagreement is itself data (docs/plan_parsing_ethics.md, finding T4).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

# The pipeline lives in "src/benchmarks/ethics". The space in the parent
# folder makes it unimportable as a package, so sibling modules are reached by
# putting the ETHICS directory itself on sys.path.
_PIPELINE_DIR = str(Path(__file__).resolve().parent.parent)
if _PIPELINE_DIR not in sys.path:
    sys.path.insert(0, _PIPELINE_DIR)


#: Signal kinds.
KIND_EXPLICIT = "explicit"    # a digit the model actually wrote
KIND_AITA = "aita"            # AITA verdict token, no digit
KIND_SEMANTIC = "semantic"    # natural-language verdict; candidate only
KIND_ABSTAIN = "abstain"      # a reason NOT to commit
KIND_FLAG = "flag"            # annotation only, never blocks or commits


@dataclass(frozen=True)
class Signal:
    """One piece of evidence found in a response."""

    detector: str
    kind: str
    answer: str | None = None       # "0" / "1" for verdict-bearing signals
    stance: str | None = None       # set by AITA/semantic detectors directly
    priority: int = 99              # lower wins; only meaningful for verdicts
    span: tuple[int, int] = (-1, -1)
    evidence: str = ""
    extra: dict = field(default_factory=dict)


def clip(text: str, span: tuple[int, int], pad: int = 40) -> str:
    """Readable evidence snippet around a match, for audit trails."""
    start = max(0, span[0] - pad)
    end = min(len(text), span[1] + pad)
    return " ".join(text[start:end].split())


from . import explicit, aita, semantic, refusal  # noqa: E402


def run_all(text: str, masked: str, subset: str) -> list[Signal]:
    """Run every detector and return the combined signal list.

    Digit and semantic detectors see MASKED text so they cannot harvest a digit
    out of the prompt's own "<0 or 1>" placeholder. AITA sees unmasked text: it
    never extracts digits, but must distinguish a strict verdict from an
    explicit non-binary score such as 0.5. The refusal detector also gets the
    unmasked text for placeholder-confusion diagnostics.
    """
    signals: list[Signal] = []
    signals.extend(explicit.detect(masked))
    signals.extend(aita.detect(text, subset))
    signals.extend(semantic.detect(masked))
    signals.extend(refusal.detect(masked, text))
    return signals


__all__ = [
    "Signal",
    "KIND_EXPLICIT",
    "KIND_AITA",
    "KIND_SEMANTIC",
    "KIND_ABSTAIN",
    "KIND_FLAG",
    "clip",
    "run_all",
    "explicit",
    "aita",
    "semantic",
    "refusal",
]
