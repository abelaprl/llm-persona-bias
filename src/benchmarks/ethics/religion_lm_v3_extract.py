"""Final-only detector bank with accepted and excluded occurrence provenance."""
import hashlib
from src.benchmarks.ethics.religion_lm_v3_registry import get_registry, family_for, VERSION, rule_origin
from src.benchmarks.ethics.religion_lm_v3_surface import prepare_final_surface
from src.benchmarks.ethics.religion_lm_v3_scope import resolve_spans
from src.core.linguistic_markers.extractor import _semantic_match_is_valid


def detect_candidates(surface, registry):
    if not surface.explanation_available:
        return []
    surfaces = {'visible':surface.visible_text,'lexical':surface.linguistic_text,'structure':surface.structure_text}
    hits = []
    raw_hash = hashlib.sha256(surface.raw_text.encode('utf-8')).hexdigest()
    for marker in registry:
        text = surfaces[marker.surface]
        surface_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
        for pattern_index, pattern in enumerate(marker.compiled_patterns):
            for match in pattern.finditer(text):
                if match.end() <= match.start():
                    continue
                valid = _semantic_match_is_valid(text,marker.marker_id,match.start(),match.end(),match.group())
                a,b = surface.raw_span(marker.surface,match.start(),match.end())
                origin = rule_origin(marker.marker_id,pattern_index)
                hits.append(dict(marker_id=marker.marker_id,family_id=family_for(marker.marker_id),
                    rule_id=f'{marker.marker_id}:{origin}:{pattern_index}',registry_version=VERSION,
                    raw_final_sha256=raw_hash,surface_sha256=surface_hash,surface_name=marker.surface,
                    start=match.start(),end=match.end(),raw_start=a,raw_end=b,matched_text=match.group(),
                    raw_matched_text=surface.raw_text[a:b],candidate_status=('new_candidate' if origin=='candidate' else 'legacy_candidate') if valid else 'excluded',
                    exclusion_reason='' if valid else 'legacy_context_guard'))
    return hits


def summarize_response(spans, surface, registry):
    counts = {m.marker_id:0 for m in registry}
    for hit in spans:
        if hit['candidate_status'] != 'excluded':
            counts[hit['marker_id']] += 1
    row = dict(surface_status=surface.status,final_output_available=surface.final_output_available,
        explanation_available=surface.explanation_available,word_count=surface.word_count,
        sentence_count=surface.sentence_count,ttr_300=surface.ttr_300,surface_flags='|'.join(surface.flags),
        raw_final_sha256=hashlib.sha256(surface.raw_text.encode('utf-8')).hexdigest(),registry_version=VERSION)
    families = {}
    for marker in registry:
        marker_id = marker.marker_id
        count = counts[marker_id]
        row[marker_id+'__count'] = count if surface.final_output_available else None
        row[marker_id+'__density'] = count/surface.word_count*100 if surface.explanation_available else None
        row[marker_id+'__present'] = int(count>0) if surface.explanation_available else None
        row[marker_id+'__present_output'] = int(count>0) if surface.final_output_available else None
        family = family_for(marker_id)
        families[family] = families.get(family,False) or count>0
    for family, present in families.items():
        row[family+'__family_present'] = int(present) if surface.explanation_available else None
    return row


def extract_final(final_output, registry=None, ablation=False):
    registry = get_registry() if registry is None else registry
    surface = prepare_final_surface(final_output)
    candidates = detect_candidates(surface,registry)
    hits = resolve_spans(candidates,surface)
    row = summarize_response(hits,surface,registry)
    if ablation:
        from src.benchmarks.ethics.religion_lm_v3_registry import ETHICS_RELIGION_MARKERS,EXTENSIONS
        from src.core.linguistic_markers.surface import prepare_text_surface
        from src.core.linguistic_markers.extractor import find_marker_matches,marker_text
        old_surface = prepare_text_surface(final_output,benchmark='ethics')
        row['ablation_old_word_count'] = old_surface.word_count
        row['ablation_old_available'] = old_surface.text_available
        filtered = resolve_spans([h for h in candidates if ':legacy:' in h['rule_id']],surface)
        surface_only = summarize_response(filtered,surface,ETHICS_RELIGION_MARKERS)
        baseline_counts = {}
        for marker in ETHICS_RELIGION_MARKERS:
            row['ablation_new_surface_old_registry__'+marker.marker_id] = surface_only[marker.marker_id+'__count']
            text = marker_text(old_surface,marker)
            baseline_counts[marker.marker_id] = len(find_marker_matches(text,marker)) if text else None
            row['ablation_old_surface_old_registry__'+marker.marker_id] = baseline_counts[marker.marker_id]
        for marker in registry:
            text = marker_text(old_surface,marker)
            row['ablation_old_surface_new_registry__'+marker.marker_id] = baseline_counts[marker.marker_id] if marker.marker_id in baseline_counts and marker.marker_id not in EXTENSIONS else (len(find_marker_matches(text,marker)) if text else None)
    return row,hits
