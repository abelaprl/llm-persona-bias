"""Frozen Regional r5 inputs; no extraction or scoring is performed here."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'result/ETHICS/reports/regional_lm_v3_20260916_r5'
LEGACY = ROOT / 'result/ETHICS/reports/overall_statistics_20260913/regional'
ACTIVE = ROOT / 'notebooks/regional/linguistic marker/ETHICS'
POLICY = 'regional-explanation-v1-pairing-v2-20260919'
PROVENANCE = ROOT / 'docs/regional_ethics_prompt_provenance_20260919.json'
PROVENANCE_HASH = '03bd024f610bab7d862a61069a6aa28caec2ed88fb708810a6ba5692304ec39a'
IDENTITIES = ['north_american', 'east_asian', 'south_asian', 'sub_saharan_african']
IDENTITY_LABELS = dict(zip(IDENTITIES, ['Amerika Utara', 'Asia Timur', 'Asia Selatan', 'Afrika Sub-Sahara']))
CONDITIONS = ['explicit', 'implicit_1', 'implicit_2', 'implicit_3']
CONDITION_LABELS = dict(zip(['neutral'] + CONDITIONS, ['N', 'E', 'I1', 'I2', 'I3']))
PERSONAS = ['neutral'] + [f'{i}_{c}' for i in IDENTITIES for c in CONDITIONS]
MODELS = ['meta-llama_llama-3.1-8b-instruct', 'meta-llama_llama-3.3-70b-instruct',
          'mistralai_ministral-8b-2512', 'deepseek_deepseek-r1-distill-qwen-32b',
          'openai_gpt-oss-20b', 'qwen_qwen3-next-80b-a3b-thinking']
MODEL_LABELS = dict(zip(MODELS, ['Llama 3.1 8B', 'Llama 3.3 70B', 'Ministral 8B', 'DeepSeek R1 32B', 'GPT-OSS 20B', 'Qwen3 Next 80B']))
MODEL_COLORS = dict(zip(MODELS, ['#2563eb', '#7c3aed', '#059669', '#dc2626', '#d97706', '#0891b2']))
FROZEN_HASHES = {
    'response_level.csv.gz': 'cb5e6b8949364391cd6c9e53c0e8a833eec3bf10af2654a832ab05506c3a5c83',
    'registry.json': '67c400dade02d2903d7e196aea4e50f55b1a37cbf1a5c7ce3f0d70de4863ca64',
    'surface_texts.jsonl.gz': 'a89ee32529c1a87f9104fb8c929bfb5ab6308e3cf5de242ff2b86ccbdf849465',
    'marker_hits.csv.gz': '528fc1cf7a3967f1f073506c13111e748428ef2aa6dd8ef3bb2cc38262730143',
    'release_gate.json': '74c5ade31c8c950722bb143ab94d0ba437851694a8dd3c4ae93e3b978170e9b4',
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


def load_provenance():
    if sha256(PROVENANCE) != PROVENANCE_HASH:
        raise ValueError('Pairing audit changed; revalidation required')
    audit = json.loads(PROVENANCE.read_text(encoding='utf-8'))
    overrides = audit['scenario_overrides']
    if len(overrides) != 20 or len({r['response_key'] for r in overrides}) != 20:
        raise ValueError('Incomplete approved pairing exclusions')
    for item in overrides:
        if sha256(ROOT / item['raw_source_path']) != item['raw_source_sha256']:
            raise ValueError('Scenario source changed since audit')
    return audit


def persona_parts(persona):
    if persona == 'neutral':
        return 'neutral', 'neutral'
    for identity in IDENTITIES:
        for condition in CONDITIONS:
            if persona == f'{identity}_{condition}':
                return identity, condition
    raise ValueError(f'Unknown Regional persona: {persona}')


def validate_rows(rows):
    keys = ['model_id', 'persona', 'subset', 'question_id']
    if rows[keys + ['response_key']].isna().any().any():
        raise ValueError('Missing response key')
    if rows.duplicated(keys).any() or rows.response_key.duplicated().any():
        raise ValueError('Duplicate response keys')
    expected = rows.model_id + '|' + rows.persona + '|' + rows.question_id
    if not expected.eq(rows.response_key).all():
        raise ValueError('Response key does not match columns')


def validate_gate(gate):
    if type(gate.get('release_eligible')) is not bool:
        raise ValueError('release_eligible must be a JSON boolean')
    result = dict(gate)
    value = gate.get('human_review_complete', False)
    if str(value).lower() not in {'true', 'false'}:
        raise ValueError('Invalid human_review_complete')
    result['human_review_complete'] = str(value).lower() == 'true'
    return result


def join_catalog(registry, dictionary):
    if len(registry) != 53 or registry.marker_id.nunique() != 53:
        raise ValueError('Expected exactly 53 distinct frozen markers')
    catalog = registry.merge(dictionary[['outcome_id', 'registry_scope', 'registered_measurement']],
                             left_on='marker_id', right_on='outcome_id', validate='one_to_one', how='left')
    if catalog.registry_scope.isna().any() or not catalog.measurement.eq(catalog.registered_measurement).all():
        raise ValueError('Incomplete or inconsistent legacy scope mapping')
    if len(regional_marker_ids(catalog)) != 23 or catalog.measurement.value_counts().to_dict() != {'density': 45, 'binary': 8}:
        raise ValueError('Unexpected frozen registry composition')
    catalog['scope_source'] = str(LEGACY / 'outcome_dictionary.csv')
    return catalog


def regional_marker_ids(catalog):
    return catalog.loc[catalog.registry_scope.eq('regional_cultural_specific'), 'marker_id'].tolist()


@dataclass
class RegionalBundle:
    source: Path
    rows: pd.DataFrame
    catalog: pd.DataFrame
    hashes: dict
    release_gate: dict


def load_bundle(source=SOURCE, legacy=LEGACY):
    source, legacy = Path(source).resolve(), Path(legacy).resolve()
    hashes = {str(source / name): sha256(source / name) for name in FROZEN_HASHES}
    for name, expected in FROZEN_HASHES.items():
        if hashes[str(source / name)] != expected:
            raise ValueError(f'Frozen source changed: {name}')
    for path in [source / 'manifest.json', legacy / 'outcome_dictionary.csv']:
        hashes[str(path)] = sha256(path)
    audit = load_provenance()
    hashes[str(PROVENANCE)] = PROVENANCE_HASH
    for relative, expected in audit['source_hashes'].items():
        path = str(ROOT / relative)
        if hashes.get(path, sha256(path)) != expected:
            raise ValueError(f'Audited input changed: {relative}')
        hashes[path] = expected
    for item in audit['scenario_overrides']:
        hashes[str(ROOT / item['raw_source_path'])] = item['raw_source_sha256']
    rows = pd.read_csv(source / 'response_level.csv.gz', low_memory=False)
    validate_rows(rows)
    if len(rows) != 154428 or set(rows.model_id) != set(MODELS) or set(rows.persona) != set(PERSONAS):
        raise ValueError('Frozen corpus shape changed')
    if not rows.groupby(['model_id', 'persona']).size().eq(1514).all():
        raise ValueError('Incomplete model/persona grid')
    catalog = join_catalog(pd.DataFrame(json.loads((source / 'registry.json').read_text(encoding='utf-8'))),
                           pd.read_csv(legacy / 'outcome_dictionary.csv'))
    gate = validate_gate(json.loads((source / 'release_gate.json').read_text(encoding='utf-8')))
    for path, expected in hashes.items():
        if sha256(path) != expected:
            raise ValueError(f'Input changed during read: {path}')
    return RegionalBundle(source, rows, catalog, hashes, gate)
