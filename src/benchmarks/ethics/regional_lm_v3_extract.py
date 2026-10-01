"""Text-only Regional extraction with complete detector/resolver provenance."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import re

from src.benchmarks.ethics.run_gender_ethics_lm_workspace import enable_marker_imports

enable_marker_imports()
from src.core.linguistic_markers.ethics_regional_registry import ETHICS_REGIONAL_CULTURAL_MARKERS
from src.core.linguistic_markers.ethics_registry import moral_polarity_sequence
from src.core.linguistic_markers.extractor import (
    _semantic_match_is_valid, classify_opening, marker_text,
)
from src.core.linguistic_markers.surface import prepare_text_surface

VERSION = 'regional-lm-v3.0.0-candidate'
IDIOM_RULE = 'connective_conditional:exclude_if_you_will'
_IDIOM = re.compile(r'if you will\b', re.I)


def registry_snapshot() -> list[dict]:
    """Keep original definitions and explicitly version the one local exclusion."""
    result = []
    for marker in ETHICS_REGIONAL_CULTURAL_MARKERS:
        row = asdict(marker)
        for index, pattern in enumerate(row['patterns']):
            pattern['pattern_index'] = index
            pattern['rule_id'] = f'{marker.marker_id}:p{index}'
        row['candidate_exclusions'] = []
        if marker.marker_id == 'connective_conditional':
            row['candidate_exclusions'] = [{
                'rule_id': IDIOM_RULE, 'pattern': _IDIOM.pattern,
                'definition': 'Exclude the exact observed idiom if you will.',
                'source': 'Custom corpus-derived exclusion; two development responses.',
                'validation_status': 'ai_development_census_only_no_independent_human_validation',
            }]
        row['candidate_version'] = VERSION
        result.append(row)
    return result


def extract_response(final_text: str, *, rule_set: str) -> dict:
    """Extract without access to ground truth, model, persona, or scenario."""
    if not isinstance(final_text, str):
        raise TypeError('final_text must be a string, never a row or metadata mapping')
    if rule_set not in {'baseline', 'candidate'}:
        raise ValueError(f'Unknown rule_set: {rule_set}')
    surface = prepare_text_surface(final_text, benchmark='ethics')
    surfaces = {'visible': surface.visible_text, 'lexical': surface.lexical_text,
                'structure': surface.structure_text}
    metrics = {key: getattr(surface, key) for key in (
        'text_status', 'text_available', 'word_count', 'sentence_count',
        'avg_sentence_length', 'ttr_300', 'think_block_stripped', 'technical_lines_stripped')}
    metrics['opening_category'] = classify_opening(surface)
    polarity_keys = ('moral_polarity_initial', 'moral_polarity_final',
                     'moral_polarity_shift', 'moral_polarity_direction')
    metrics.update(moral_polarity_sequence(surface.lexical_text) if surface.text_available
                   else dict.fromkeys(polarity_keys))
    hits = []
    for marker in ETHICS_REGIONAL_CULTURAL_MARKERS:
        available = bool(surface.visible_text) if marker.surface == 'visible' else surface.text_available
        if not available:
            for suffix in ('count', 'density', 'present'):
                metrics[f'{marker.marker_id}__{suffix}'] = None
            continue
        text = marker_text(surface, marker)
        detected = []
        for index, pattern in enumerate(marker.compiled_patterns):
            for match in pattern.finditer(text):
                if match.end() <= match.start():
                    continue
                valid = _semantic_match_is_valid(text, marker.marker_id,
                                                 match.start(), match.end(), match.group())
                hit = dict(marker_id=marker.marker_id, rule_id=f'{marker.marker_id}:p{index}',
                           surface=marker.surface, start=match.start(), end=match.end(),
                           matched_text=match.group(), decision='accepted' if valid else 'rejected',
                           reason='baseline_rule' if valid else 'baseline_semantic_filter',
                           rule_version=marker.version, pattern_index=index,
                           exclusion_rule_id='')
                if (valid and rule_set == 'candidate' and marker.marker_id == 'connective_conditional'
                        and match.group().lower() == 'if' and _IDIOM.match(text, match.start())):
                    hit.update(decision='rejected', reason='idiomatic_if_you_will',
                               exclusion_rule_id=IDIOM_RULE, rule_version=VERSION)
                detected.append(hit)
        detected.sort(key=lambda h: (h['start'], -(h['end'] - h['start']), h['pattern_index']))
        occupied_until = -1
        accepted = 0
        for hit in detected:
            if hit['decision'] == 'accepted':
                if hit['start'] < occupied_until:
                    hit.update(decision='rejected', reason='within_marker_overlap')
                else:
                    occupied_until = hit['end']
                    accepted += 1
            hits.append(hit)
        metrics[f'{marker.marker_id}__count'] = accepted
        metrics[f'{marker.marker_id}__density'] = accepted / surface.word_count * 100 if surface.word_count else None
        metrics[f'{marker.marker_id}__present'] = int(accepted > 0)
    return dict(surfaces=surfaces, metrics=metrics, hits=hits,
                raw_final_sha256=hashlib.sha256(final_text.encode('utf-8')).hexdigest(),
                surface_sha256={k: hashlib.sha256(v.encode('utf-8')).hexdigest() for k, v in surfaces.items()})


def extract_batch(texts: list[str]) -> list[tuple[dict, dict]]:
    """Baseline and candidate share all rules except the exact idiom exclusion."""
    output = []
    for text in texts:
        candidate = extract_response(text, rule_set='candidate')
        baseline_metrics = dict(candidate['metrics'])
        excluded = sum(h['reason'] == 'idiomatic_if_you_will' for h in candidate['hits'])
        if excluded:
            # Independently replay affected responses instead of assuming delta arithmetic.
            baseline_metrics = extract_response(text, rule_set='baseline')['metrics']
        output.append((baseline_metrics, candidate))
    return output


if __name__ == '__main__':
    import json
    import sys
    sys.stdout.write(json.dumps(extract_batch(json.load(sys.stdin)), ensure_ascii=False))
