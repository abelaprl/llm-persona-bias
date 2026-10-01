"""Explicit explanation eligibility over frozen surfaces, not a new detector."""
import gzip
import hashlib
import json
import re

import numpy as np
import pandas as pd

from .regional_ethics_presentation_data import POLICY, load_provenance

# Frozen from religion_lm_v3_surface.py BINARY; format-only classification.
BINARY = re.compile(r'\s*(?:[\s*`#]*(?:(?:final\s+)?(?:answer|label|verdict)\s*:?\s*)?)?[<\[({*`\s]*[01][>\])}*`\s.!:]*\s*', re.I)


def classify_surface(visible, lexical, legacy_status):
    if legacy_status == 'missing':
        return 'missing'
    if BINARY.fullmatch(visible):
        return 'label_only'
    if legacy_status != 'available' or not any(c.isalpha() for c in lexical):
        return 'technical_only'
    return 'linguistic_text'


def iter_surfaces(path):
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        for line in handle:
            item = json.loads(line)
            for name, surface in item['surfaces'].items():
                if hashlib.sha256(surface.encode('utf-8')).hexdigest() != item['surface_sha256'][name]:
                    raise ValueError(f'Surface hash mismatch: {item["response_key"]}/{name}')
            yield item


def build_view(bundle, surfaces_path):
    frame = bundle.rows.copy()
    lookup = frame.set_index('response_key')[['text_status', 'raw_final_sha256']].to_dict('index')
    statuses = {}
    for item in iter_surfaces(surfaces_path):
        key = item['response_key']
        if key in statuses or key not in lookup:
            raise ValueError('Duplicate or unknown surface key')
        if item['raw_final_sha256'] != lookup[key]['raw_final_sha256']:
            raise ValueError('Raw hash mismatch')
        statuses[key] = classify_surface(item['surfaces']['visible'], item['surfaces']['lexical'], lookup[key]['text_status'])
    if set(statuses) != set(lookup):
        raise ValueError('Missing surface rows')
    frame['analysis_surface_status'] = frame.response_key.map(statuses)
    frame['explanation_available'] = frame.analysis_surface_status.eq('linguistic_text')
    frame['analysis_policy_version'] = POLICY
    if frame.loc[frame.explanation_available, 'word_count'].le(0).any():
        raise ValueError('Eligible explanation with zero words')
    return apply_pairing_exclusions(frame, {r['response_key'] for r in load_provenance()['scenario_overrides']})


def apply_pairing_exclusions(view, excluded_keys):
    if not set(excluded_keys).issubset(set(view.response_key)):
        raise ValueError('Unknown pairing exclusion key')
    result = view.copy()
    result['pairing_eligible'] = ~result.response_key.isin(excluded_keys)
    result['pairing_exclusion_reason'] = np.where(result.pairing_eligible, '', 'source_scenario_mismatch')
    return result


def paired_engine_frame(view, catalog):
    frame = engine_frame(view, catalog)
    frame['text_available'] &= view.pairing_eligible.astype(bool)
    return frame


def engine_frame(view, catalog):
    frame = view.copy()
    frame['text_available'] = frame.explanation_available.astype(bool)
    frame['validated_committed'] = frame.analysis_answer_status.eq('committed')
    frame['analysis_is_correct'] = pd.NA
    columns = [c for c in frame if c.endswith(('__count', '__density', '__present')) or c in ['word_count', 'avg_sentence_length', 'ttr_300']]
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors='raise').astype(float)
        frame.loc[~frame.text_available, column] = np.nan
    return frame


def coverage_table(view):
    records = []
    for (model, persona), group in view.groupby(['model_id', 'persona'], sort=False):
        text = group[group.explanation_available]
        status, answers = group.analysis_surface_status.value_counts(), group.analysis_answer_status.value_counts()
        row = dict(model_id=model, persona=persona, identity=group.identity.iloc[0], condition=group.condition.iloc[0],
                   total=len(group), n_text=len(text), coverage_pct=100 * len(text) / len(group),
                   abstain_text=int(text.analysis_answer_status.eq('abstain').sum()),
                   word_total=int(text.word_count.sum()), median_words=text.word_count.median(),
                   q25_words=text.word_count.quantile(.25), q75_words=text.word_count.quantile(.75),
                   mean_words=text.word_count.mean(), analysis_policy_version=POLICY)
        for name in ['label_only', 'technical_only', 'missing']:
            row['surface_' + name] = int(status.get(name, 0))
        for name in ['committed', 'abstain', 'missing']:
            row[name] = int(answers.get(name, 0))
            row[('commit' if name == 'committed' else name) + '_rate'] = row[name] / len(group)
        if 'pairing_eligible' in group:
            row['pairing_excluded'] = int((~group.pairing_eligible).sum())
            row['pairing_text_eligible'] = int((group.pairing_eligible & group.explanation_available).sum())
        records.append(row)
    return pd.DataFrame(records)
