"""Trace regional hits to declared surfaces and literal context, not causation."""
import hashlib
import html
import json
import re
from pathlib import Path
import unicodedata
import pandas as pd
from .regional_ethics_presentation_data import ROOT, IDENTITIES, regional_marker_ids, sha256, write_json, load_provenance
from .regional_ethics_presentation_view import iter_surfaces

RAW_ROOT = ROOT / 'result/benchmarks/ethics/runs/main/v9'
ASSIGNMENTS = [('east_asian', 'implicit_1'), ('north_american', 'implicit_2'),
               ('south_asian', 'implicit_3'), ('sub_saharan_african', 'implicit_1'),
               ('east_asian', 'implicit_2'), ('sub_saharan_african', 'implicit_3')]


def validate_hit(surface, hit):
    start, end = int(hit['start']), int(hit['end'])
    if not 0 <= start < end <= len(surface) or surface[start:end] != hit['matched_text']:
        raise ValueError('Hit offsets do not match declared surface')


def context_flags(hit, persona_prompt, scenario):
    normalize = lambda text: ' '.join(text.casefold().split())
    needle = normalize(hit)
    return dict(persona_literal_overlap=None if persona_prompt is None else bool(needle) and needle in normalize(persona_prompt),
                scenario_literal_overlap=None if scenario is None else bool(needle) and needle in normalize(scenario))


def scenario_for_response(response_key, question_id, scenarios, overrides):
    return overrides.get(response_key, scenarios.get(question_id))


def excerpt_window(text, hits, max_words=100):
    words = list(re.finditer(r'\S+', text))
    if len(words) <= max_words:
        return 0, len(text)
    focus = min((int(h['start']) for h in hits), default=0)
    position = next((i for i, w in enumerate(words) if w.end() > focus), 0)
    start = min(max(0, position - 15), len(words) - max_words)
    return words[start].start(), words[start + max_words - 1].end()


def highlighted_html(text, hits):
    boundaries = {0, len(text)}
    for hit in hits:
        validate_hit(text, hit)
        boundaries.update([int(hit['start']), int(hit['end'])])
    out = []
    points = sorted(boundaries)
    for start, end in zip(points, points[1:]):
        labels = sorted({h['marker_id'] for h in hits if int(h['start']) < end and int(h['end']) > start})
        piece = html.escape(text[start:end])
        out.append(f'<mark title="{html.escape(", ".join(labels), quote=True)}">{piece}</mark>' if labels else piece)
    return ''.join(out)


def raw_path(model, persona, question):
    path = RAW_ROOT / ('Neutral Persona' if persona == 'neutral' else 'Regional Cultural Persona') / model
    if not (persona == 'neutral' and model.startswith('deepseek_')):
        path /= persona
    return path / f'{question}.json'


def prompt_snapshots(view):
    definitions = {r['persona']: r['prompt'] for r in json.loads((ROOT / 'inputs/regional_cultural_persona.json').read_text(encoding='utf-8'))}
    prompts, assistant_context, records = {}, {}, []
    for (model, persona), group in view.groupby(['model_id', 'persona'], sort=True):
        path = raw_path(model, persona, sorted(group.question_id)[0])
        prompt, warmup = None, None
        reason = 'request_unavailable'
        if path.is_file():
            raw = json.loads(path.read_text(encoding='utf-8'))
            messages = raw.get('request', {}).get('messages', [])
            if persona == 'neutral' and [m.get('role') for m in messages] == ['system', 'user']:
                prompt, reason = '', 'neutral_request_no_persona'
            elif '_implicit_' in persona and len(messages) >= 3 and messages[1].get('role') == 'user':
                prompt = messages[1]['content']
                # Assistant turns vary per request; a group snapshot is not valid corpus context.
                warmup = None
                reason = 'actual_group_request_snapshot_and_frozen_full_corpus_user_prompt_audit'
            elif persona.endswith('_explicit') and len(messages) == 2:
                expected = definitions[persona]
                if messages[1]['content'].startswith(expected + '\n\n'):
                    prompt, reason = expected, 'actual_group_request_prefix_verified'
        prompts[(model, persona)], assistant_context[(model, persona)] = prompt, warmup
        records.append(dict(model_id=model, persona=persona, prompt_available=prompt is not None,
                            prompt=prompt, assistant_warmup_available=warmup is not None,
                            matches_current_definition=prompt == definitions.get(persona, ''),
                            provenance=reason, verification_scope='frozen full-corpus user-prompt audit plus group snapshot; assistant context unavailable',
                            source_file=str(path), source_sha256=sha256(path) if path.is_file() else None))
    return prompts, assistant_context, pd.DataFrame(records)


def select_cards(view, hits, regional_ids):
    choices, candidates = [], []
    named = [m for m in ['culture_named_practice_reference', 'culture_warmup_name_echo', 'culture_named_concept_reference'] if m in regional_ids]
    for model, (identity, condition) in zip(sorted(view.model_id.unique()), ASSIGNMENTS):
        personas = ['neutral', identity + '_explicit', identity + '_' + condition]
        part = view[view.model_id.eq(model) & view.persona.isin(personas)]
        for (subset, question), group in part.groupby(['subset', 'question_id'], sort=True):
            if set(group.persona) != set(personas) or not group.text_available.all() or not group.pairing_eligible.all():
                continue
            matrix = group.set_index('persona')
            counts = matrix[[m + '__count' for m in regional_ids]]
            named_hit = any(matrix.loc[personas[2], m + '__count'] > 0 for m in named)
            region_id = 'culture_region_' + identity
            label_hit = matrix.loc[personas[1], region_id + '__count'] > 0
            candidates.append(dict(model_id=model, identity=identity, implicit_condition=condition, subset=subset,
                                   question_id=question, named_implicit=int(named_hit), label_explicit=int(label_hit),
                                   distinct_union=int(counts.gt(0).any(axis=0).sum()), count_sum=int(counts.sum().sum())))
        ranked = pd.DataFrame([c for c in candidates if c['model_id'] == model]).sort_values(
            ['named_implicit', 'label_explicit', 'distinct_union', 'count_sum', 'subset', 'question_id'],
            ascending=[False, False, False, False, True, True])
        if ranked.empty:
            raise ValueError('No matched card triplet')
        best = ranked.iloc[0].to_dict()
        for persona in personas:
            source = part[part.persona.eq(persona) & part.question_id.eq(best['question_id']) & part.subset.eq(best['subset'])].iloc[0]
            choices.append(dict(best, persona=persona, response_key=source.response_key,
                                analysis_surface_status=source.analysis_surface_status, word_count=int(source.word_count),
                                selection='purposive; named implicit, explicit label, category union, count, lexical question order; no GT/p'))
    result = pd.DataFrame(choices)
    result.attrs['candidates'] = pd.DataFrame(candidates)
    return result


def build_evidence(bundle, view, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    marker_ids = regional_marker_ids(bundle.catalog)
    chunks = []
    for chunk in pd.read_csv(bundle.source / 'marker_hits.csv.gz', chunksize=150000):
        chunks.append(chunk[chunk.marker_id.isin(marker_ids) & chunk.decision.eq('accepted') & chunk.surface.eq('lexical')])
    hits = pd.concat(chunks, ignore_index=True)
    hit_groups = {key: group.to_dict('records') for key, group in hits.groupby('response_key', sort=False)}
    cards = select_cards(view, hits, marker_ids)
    cards.attrs['candidates'].to_csv(output / 'card_candidates.csv', index=False)
    selected = set(cards.response_key)
    surface_map = {}
    for item in iter_surfaces(bundle.source / 'surface_texts.jsonl.gz'):
        key = item['response_key']
        surface = item['surfaces']['lexical']
        for hit in hit_groups.get(key, []):
            validate_hit(surface, hit)
        if key in selected:
            surface_map[key] = item
    prompts, assistant_context, provenance = prompt_snapshots(view)
    provenance.to_csv(output / 'prompt_provenance.csv', index=False)
    dataset = ROOT / 'data/ethics/ETHICS_test_4subsets_10percent_from200.json'
    scenarios = {r['id']: r['scenario'].strip() for r in json.loads(dataset.read_text(encoding='utf-8'))}
    audit = load_provenance()
    scenario_overrides = {r['response_key']: r['actual_scenario'] for r in audit['scenario_overrides']}
    metadata = view.set_index('response_key')[['model_id', 'persona', 'question_id', 'explanation_available', 'word_count']].to_dict('index')
    records = []
    for hit in hits.to_dict('records'):
        meta = metadata[hit['response_key']]
        key = (meta['model_id'], meta['persona'])
        scenario = scenario_for_response(hit['response_key'], meta['question_id'], scenarios, scenario_overrides)
        flags = context_flags(hit['matched_text'], prompts[key], scenario)
        warmup_flag = context_flags(hit['matched_text'], assistant_context[key], None)['persona_literal_overlap']
        scripts = {unicodedata.name(c, '').split(' ')[0] for c in hit['matched_text'] if c.isalpha()}
        records.append(dict(**hit, **meta, **flags, assistant_warmup_literal_overlap=warmup_flag,
                            mixed_script=len(scripts) > 1, prompt_verification_scope='frozen_full_corpus_user_prompt_audit',
                            scenario_source='actual_request_override' if hit['response_key'] in scenario_overrides else 'audited_dataset_match'))
    context = pd.DataFrame(records)
    context.to_csv(output / 'context_hits.csv.gz', index=False, compression='gzip')
    summary_rows = []
    for (model, persona), group in view.groupby(['model_id', 'persona'], sort=False):
        local = context[context.model_id.eq(model) & context.persona.eq(persona) & context.explanation_available]
        n_text = int(group.explanation_available.sum())
        for flag in ['persona_literal_overlap', 'scenario_literal_overlap', 'assistant_warmup_literal_overlap']:
            flag_available = (prompts[(model, persona)] is not None if flag == 'persona_literal_overlap' else
                              assistant_context[(model, persona)] is not None if flag == 'assistant_warmup_literal_overlap' else True)
            matches = local[local[flag].eq(True)]
            summary_rows.append(dict(model_id=model, persona=persona, flag=flag, n_text=n_text,
                                     context_available=flag_available, n_positive=matches.response_key.nunique() if flag_available else None,
                                     raw_count=len(matches) if flag_available else None,
                                     prevalence_pct=100 * matches.response_key.nunique() / n_text if flag_available and n_text else None))
    pd.DataFrame(summary_rows).to_csv(output / 'context_response_summary.csv', index=False)
    context.groupby(['model_id', 'persona', 'marker_id', 'persona_literal_overlap', 'scenario_literal_overlap'], dropna=False).agg(
        raw_count=('response_key', 'size'), n_responses=('response_key', 'nunique')).reset_index().to_csv(output / 'context_overlap.csv', index=False)
    terms = context[context.explanation_available & context.marker_id.isin(['culture_named_practice_reference', 'culture_warmup_name_echo', 'culture_named_concept_reference'])].copy()
    terms['term'] = terms.matched_text.str.casefold().str.replace(r'\s+', ' ', regex=True)
    response_terms = terms.groupby(['model_id', 'persona', 'response_key', 'term'], sort=False).agg(raw_count=('term', 'size'), word_count=('word_count', 'first')).reset_index()
    response_terms['density'] = 100 * response_terms.raw_count / response_terms.word_count
    term_summary = response_terms.groupby(['model_id', 'persona', 'term']).agg(raw_count=('raw_count', 'sum'), n_positive=('response_key', 'size'), density_sum=('density', 'sum')).reset_index()
    denominator = view.groupby(['model_id', 'persona']).explanation_available.sum().rename('n_text').reset_index()
    term_summary = term_summary.merge(denominator, on=['model_id', 'persona'], validate='many_to_one')
    term_summary['mean_response_density'] = term_summary.density_sum / term_summary.n_text
    term_summary.to_csv(output / 'named_terms.csv', index=False)
    source_manifest = json.loads((bundle.source / 'manifest.json').read_text(encoding='utf-8'))
    canonical = Path(source_manifest['source'])
    if sha256(canonical) != source_manifest['source_sha256']:
        raise ValueError('Canonical source changed')
    raw_map = {}
    for chunk in pd.read_csv(canonical, usecols=['model', 'persona_folder', 'question_id', 'response_content_full'], chunksize=10000, keep_default_na=False):
        keys = chunk.model + '|' + chunk.persona_folder + '|' + chunk.question_id
        for key, raw in zip(keys, chunk.response_content_full):
            if key in selected:
                raw_map[key] = raw
    packages, pages = [], []
    for row in cards.to_dict('records'):
        key = row['response_key']
        item = surface_map[key]
        raw = raw_map[key]
        if hashlib.sha256(raw.encode('utf-8')).hexdigest() != item['raw_final_sha256']:
            raise ValueError('Card canonical raw hash mismatch')
        surface = item['surfaces']['lexical']
        local_hits = hit_groups.get(key, [])
        start, end = excerpt_window(surface, local_hits)
        package = dict(**row, raw_final=raw, lexical=surface, hits=local_hits, excerpt=surface[start:end],
                       excerpt_start=start, excerpt_end=end, raw_final_sha256=item['raw_final_sha256'],
                       lexical_sha256=item['surface_sha256']['lexical'], scenario=scenario_for_response(key, row['question_id'], scenarios, scenario_overrides),
                       canonical_source=str(canonical))
        packages.append(package)
        pages.append('<article><h2>' + html.escape(key) + '</h2><p>' + html.escape(row['selection']) + '</p><h3>Soal</h3><pre>' + html.escape(package['scenario']) +
                     '</pre><h3>Lexical surface utuh; offset surface</h3><pre>' + highlighted_html(surface, local_hits) +
                     '</pre><details><summary>Final raw utuh (terpisah dari lexical)</summary><pre>' + html.escape(raw) + '</pre></details></article>')
    write_json(output / 'response_cards.json', packages)
    cards.to_csv(output / 'response_index.csv', index=False)
    (output / 'response_cards.html').write_text('<!doctype html><meta charset="utf-8"><title>Bukti Regional — candidate</title><style>body{font:16px sans-serif;max-width:1000px;margin:30px auto}pre{white-space:pre-wrap}article{border-top:2px solid #ddd;padding:20px}mark{background:#fef08a}</style><h1>Contoh purposif, bukan sampel audit</h1><p>Overlap literal bukan asal-usul kausal. Label-only bukan nol marker berpenjelasan. Kutipan adalah keluaran model, bukan karakterisasi penduduk.</p>' + ''.join(pages), encoding='utf-8')
    write_json(output / 'verification.json', dict(accepted_regional_lexical_hits=len(hits), all_offsets_valid=True, cards=len(cards),
               canonical_sha256=sha256(canonical), scenario_source=str(dataset), scenario_sha256=sha256(dataset),
               prompt_snapshot_groups=len(provenance), available_prompt_groups=int(provenance.prompt_available.sum()),
               full_request_scan_rows=audit['full_corpus_scan']['rows'], actual_scenario_overrides=len(scenario_overrides),
               assistant_warmup_overlap='unavailable: per-response assistant turns vary; group snapshots are not substituted', no_ground_truth_used=True))
    return {p.name: p for p in output.iterdir() if p.is_file()}
