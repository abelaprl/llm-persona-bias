"""Read-only scientific diagnostics; write only a new, non-overwriting audit bundle.

No answer extraction, correctness annotation, marker promotion, or production edits.
Run from the repository root with --output pointing to a new report directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

from .regional_ethics_presentation_data import ROOT, MODELS, load_bundle, sha256
from .regional_lm_v3_extract import extract_response

PRESENTATION = ROOT / 'result/ETHICS/reports/regional_ethics_presentation_20260919_r2'
OLD_AUDIT = ROOT / 'result/ETHICS/reports/regional_lm_v3_ai_diagnostic_20260916_r1'
SEED = 'regional-self-audit-20260919-v1'


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


def holm(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    result = np.empty(len(values))
    result[order] = np.minimum(1, np.maximum.accumulate(values[order] * np.arange(len(values), 0, -1)))
    return result


def status_diagnostics(view):
    records = []
    for model, group in view.groupby('model_id'):
        for subset, sub in group.groupby('subset'):
            # Remove incomplete scenario blocks, not missing explanations within valid blocks.
            sub = sub[sub.pairing_eligible].copy()
            sub['answer_committed'] = sub.analysis_answer_status.eq('committed')
            for outcome in ['answer_committed', 'explanation_available']:
                pivot = sub.pivot(index='question_id', columns='persona', values=outcome).dropna()
                matrix = pivot.to_numpy(dtype=float)
                n, k = matrix.shape
                totals = matrix.sum(axis=0)
                block_totals = matrix.sum(axis=1)
                denominator = k * matrix.sum() - np.square(block_totals).sum()
                q = ((k - 1) * (k * np.square(totals).sum() - matrix.sum() ** 2) / denominator
                     if denominator else np.nan)
                rates = pivot.mean()
                records.append(dict(model_id=model, subset=subset, outcome=outcome,
                                    n_question_blocks=n, n_personas=k, statistic=q,
                                    p_raw=chi2.sf(q, k - 1),
                                    effect_name='max_minus_min_persona_rate_pp',
                                    effect_size=100 * (rates.max() - rates.min()),
                                    min_rate_persona=rates.idxmin(), max_rate_persona=rates.idxmax(),
                                    min_rate=float(rates.min()), max_rate=float(rates.max()),
                                    informative_blocks=int(((block_totals > 0) & (block_totals < k)).sum()),
                                    test='cochran_q_question_blocked' if denominator else 'unavailable_no_within_block_variation',
                                    interpretation='association_not_detector_error_or_causal_persona_effect'))
    table = pd.DataFrame(records)
    table['p_holm_four_subsets'] = table.groupby(['model_id', 'outcome']).p_raw.transform(holm)
    table['p_holm_24_model_subsets'] = table.groupby('outcome').p_raw.transform(holm)
    return table


def select_responses(view, regional_ids, old_keys):
    available = view[view.explanation_available & ~view.response_key.isin(old_keys)].copy()
    available['audit_order'] = available.response_key.map(
        lambda key: hashlib.sha256((SEED + '|' + key).encode()).hexdigest())
    available = available.sort_values('audit_order')
    selected = []
    for model in MODELS:
        pool = available[available.model_id.eq(model)]
        seen = set()
        chosen = set()
        # Five hash-random responses, one from each conditioning mode.
        for condition in ['neutral', 'explicit', 'implicit_1', 'implicit_2', 'implicit_3']:
            row = pool[pool.condition.eq(condition)].iloc[0]
            selected.append((row.response_key, 'hash_random_within_model_condition'))
            chosen.add(row.response_key)
            seen.update(m for m in regional_ids if row[m + '__count'] > 0)
        # Five diversity-targeted responses; deterministic random tie breaking.
        for _ in range(5):
            remaining = pool[~pool.response_key.isin(chosen)]
            columns = [m + '__count' for m in regional_ids if m not in seen]
            scores = remaining[columns].gt(0).sum(axis=1)
            row = remaining.loc[scores.idxmax()]
            selected.append((row.response_key, 'greedy_unseen_regional_category_hash_tiebreak'))
            chosen.add(row.response_key)
            seen.update(m for m in regional_ids if row[m + '__count'] > 0)
    return selected


def run(output):
    output = Path(output).resolve()
    allowed = (ROOT / 'result/ETHICS/reports').resolve()
    if not output.is_relative_to(allowed) or output == allowed:
        raise ValueError('Audit output must be a new child directory of result/ETHICS/reports')
    if output.exists():
        raise FileExistsError('Refusing to overwrite an existing audit')
    bundle = load_bundle()
    view_path = PRESENTATION / 'tables/analysis_view.csv.gz'
    view = pd.read_csv(view_path, low_memory=False)
    regional_ids = bundle.catalog.loc[bundle.catalog.registry_scope.eq('regional_cultural_specific'), 'marker_id'].tolist()
    old_rows = json.loads((OLD_AUDIT / 'readback_18.json').read_text(encoding='utf-8'))
    old_keys = {r['model_id'] + '|' + r['persona'] + '|' + r['question_id'] for r in old_rows}
    selected = select_responses(view, regional_ids, old_keys)
    # Explicit allow-list keeps answer labels, ground truth, and correctness out of the readback.
    canonical_path = ROOT / 'result/ETHICS/reports/Regional_Cultural_Persona/_combined/all_models_results_full.csv'
    canonical = pd.read_csv(canonical_path, usecols=['model', 'persona_folder', 'question_id', 'response_content_full'],
                            keep_default_na=False)
    canonical['response_key'] = canonical.model + '|' + canonical.persona_folder + '|' + canonical.question_id
    texts = canonical.set_index('response_key').response_content_full.to_dict()
    indexed = view.set_index('response_key')
    packets, replay_failures = [], []
    for number, (key, design) in enumerate(selected, 1):
        text = texts[key]
        replay = extract_response(text, rule_set='candidate')
        row = indexed.loc[key]
        if replay['raw_final_sha256'] != row.raw_final_sha256:
            raise ValueError('Selected raw response hash mismatch: ' + key)
        for marker in bundle.catalog.marker_id:
            for suffix in ['count', 'density', 'present']:
                column = marker + '__' + suffix
                if not np.isclose(replay['metrics'][column], row[column], equal_nan=True, rtol=1e-6):
                    replay_failures.append(dict(response_key=key, column=column))
        packets.append(dict(audit_id=f'S{number:03d}', response_key=key, model_id=row.model_id,
                            persona=row.persona, subset=row.subset, sampling_design=design,
                            word_count=int(row.word_count), raw_final_sha256=row.raw_final_sha256,
                            final_text=text, surfaces=replay['surfaces'],
                            regional_hits=[h for h in replay['hits'] if h['marker_id'] in regional_ids and h['decision'] == 'accepted']))
    known = json.loads((OLD_AUDIT / 'diagnostic_labels.json').read_text(encoding='utf-8'))
    old_map = {r['model_id'] + '|' + r['persona'] + '|' + r['question_id']: r for r in old_rows}
    known_replay = []
    for label in known['labels']:
        if label['label'] != 'FP':
            continue
        key = label['response_key']
        raw = texts[key]
        if raw != old_map[key]['final_text']:
            raise ValueError('Known diagnostic text differs from canonical')
        replay = extract_response(raw, rule_set='candidate')
        matched = [h for h in replay['hits'] if all(h[f] == label[f] for f in ['marker_id', 'rule_id', 'start', 'end', 'matched_text'])]
        known_replay.append(dict(response_key=key, marker_id=label['marker_id'], rule_id=label['rule_id'],
                                 matched_text=label['matched_text'], start=label['start'], end=label['end'],
                                 old_ai_label='FP', decision_now=matched[0]['decision'] if matched else 'not_found',
                                 explanation_available=bool(indexed.loc[key, 'explanation_available']),
                                 published_marker_count=float(indexed.loc[key, label['marker_id'] + '__count'])))
    # Full-corpus source preservation and density/count checks.
    columns = [c for c in bundle.rows if c.endswith(('__count', '__density'))]
    source = bundle.rows.set_index('response_key').reindex(view.response_key)
    unchanged = all(np.allclose(source[c].to_numpy(float), view[c].to_numpy(float), equal_nan=True) for c in columns)
    density_failures = []
    eligible = view[view.explanation_available]
    for marker in bundle.catalog.marker_id:
        expected = eligible[marker + '__count'] / eligible.word_count * 100
        bad = ~np.isclose(expected, eligible[marker + '__density'], rtol=1e-6, atol=1e-8)
        if bad.any():
            density_failures.append(dict(marker_id=marker, n=int(bad.sum())))
    statuses = status_diagnostics(view)
    coverage = view.groupby(['model_id', 'persona']).agg(total=('response_key', 'size'),
            n_text=('explanation_available', 'sum')).reset_index()
    coverage['coverage_pct'] = coverage.n_text / coverage.total * 100
    summary = dict(scope='AI self-audit; no human gold, no answer scoring, no promotion',
                   source_rows=len(view), eligible_rows=len(eligible),
                   statuses=view.analysis_surface_status.value_counts().to_dict(),
                   retained_abstain_text=int(eligible.analysis_answer_status.eq('abstain').sum()),
                   pairing_excluded=int((~view.pairing_eligible).sum()),
                   frozen_count_density_columns=len(columns), frozen_count_density_unchanged=unchanged,
                   density_failures=density_failures, selected_full_responses=len(packets),
                   selected_total_words=sum(p['word_count'] for p in packets),
                   regional_markers_in_selected_hits=len({h['marker_id'] for p in packets for h in p['regional_hits']}),
                   selected_regional_hits=sum(len(p['regional_hits']) for p in packets),
                   selected_regional_rules=len({h['rule_id'] for p in packets for h in p['regional_hits']}),
                   selected_replay_failures=replay_failures,
                   known_old_fp_hits_still_accepted=sum(x['decision_now'] == 'accepted' for x in known_replay),
                   release_gate=bundle.release_gate,
                   diagnostic_notes=['Sample excludes the 18 old diagnostic responses.',
                     '30 hash-random responses within model/condition plus 30 diversity-targeted responses.',
                     'Sampling is not a representative hit sample or rule-wise precision/recall estimate.',
                     'Coverage tests are paired association diagnostics, not detector-error tests.',
                     'No human annotations are created or inferred.'])
    output.mkdir()
    dump(output / 'summary.json', summary)
    dump(output / 'readback_60.json', packets)
    dump(output / 'known_false_positive_replay.json', known_replay)
    statuses.to_csv(output / 'persona_status_diagnostics.csv', index=False)
    coverage.to_csv(output / 'coverage_by_persona.csv', index=False)
    pd.DataFrame([{k: p[k] for k in ['audit_id', 'response_key', 'sampling_design', 'word_count', 'raw_final_sha256']}
                  for p in packets]).to_csv(output / 'sample_manifest.csv', index=False)
    dump(output / 'provenance.json', dict(seed=SEED, inputs={**bundle.hashes,
            str(view_path): sha256(view_path), str(canonical_path): sha256(canonical_path),
            str(OLD_AUDIT / 'diagnostic_labels.json'): sha256(OLD_AUDIT / 'diagnostic_labels.json')},
            script_sha256=sha256(__file__), output_hashes={p.name: sha256(p) for p in output.iterdir()}))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
