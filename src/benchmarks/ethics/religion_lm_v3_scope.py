"""Within-marker overlap resolution; uncertain roles remain unresolved."""
import re


def resolve_spans(candidates, surface=None):
    resolved = []
    occupied = {}
    quote_ranges = {}
    if surface is not None:
        for name,attr in {'lexical':'linguistic_text','visible':'visible_text','structure':'structure_text'}.items():
            quote_ranges[name] = [(m.start(),m.end()) for m in re.finditer(r'"[^"\n]+"',getattr(surface,attr))]
    for hit in sorted(candidates,key=lambda r:(r['marker_id'],r['start'],-(r['end']-r['start']),r['rule_id'])):
        hit = dict(hit)
        if hit['candidate_status'] != 'excluded':
            key = (hit['marker_id'],hit['surface_name'])
            if hit['start'] < occupied.get(key,-1):
                hit['candidate_status'] = 'excluded'
                hit['exclusion_reason'] = 'within_marker_overlap'
            else:
                occupied[key] = hit['end']
        # Roles are deliberately not guessed from a small quotation window.
        hit.update(speaker_role='unresolved',referent='unresolved',discourse_scope='unresolved',polarity='unresolved')
        if surface is not None:
            quoted = any(a<hit['start'] and b>hit['end'] for a,b in quote_ranges[hit['surface_name']])
            if quoted:
                hit.update(speaker_role='quoted_or_reported',discourse_scope='double_quoted_span')
                if hit['marker_id']=='ai_identity_statement':
                    hit.update(candidate_status='excluded',exclusion_reason='quoted_identity_not_direct_self_statement')
            if 'meta_response_candidate' in surface.flags:
                hit.update(speaker_role='unresolved',discourse_scope='meta_response_or_embedded_draft')
        if hit['marker_id']=='response_binary_decline' and hit['candidate_status']!='excluded':
            hit.update(speaker_role='model_voice',referent='requested_binary_format',discourse_scope='opening_assessment_limit',polarity='declines_binary_format')
        resolved.append(hit)
    return resolved
