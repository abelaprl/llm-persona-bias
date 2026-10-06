"""Recompute paired Regional statistics for explanation-eligible responses."""
import json
import inspect
import importlib.metadata
from pathlib import Path
import numpy as np
import pandas as pd
from .run_gender_ethics_lm_workspace import enable_marker_imports
enable_marker_imports()
from . import regional_cultural_ethics_linguistic_markers_v2 as engine
from .build_ethics_lm_overall_statistics import pool_frame, support_audit, validate_results
from .regional_ethics_presentation_data import POLICY, LEGACY, MODELS, sha256, write_json
from .regional_ethics_presentation_view import engine_frame, paired_engine_frame


def presentation_effect(row):
    scale = 100 if row.outcome_scale == 'binary' else 1
    unit = 'pp' if scale == 100 else ('per 100 kata' if row.outcome_scale == 'density_per_100_words' else row.get('outcome_id', 'native'))
    return {**{k: float(row[k]) * scale for k in ['effect', 'ci_low', 'ci_high']}, 'unit': unit}


def planned_holm(values):
    p = np.asarray(values, dtype=float)
    available = np.isfinite(p)
    adjusted = engine.holm_adjust(np.where(available, p, 1.0))
    adjusted[~available] = np.nan
    return adjusted


def presentation_inference(table):
    result = table.copy()
    for column in ['p_raw', 'p_holm', 'p_bh_outcomes', 'test_used']:
        result['engine_' + column] = result[column]
    empty = result.n_pairs.eq(0)
    result.loc[empty, ['p_raw', 'p_bh_outcomes']] = np.nan
    result.loc[empty, 'test_used'] = 'unavailable_no_pairs'
    groups = ['population', 'model_id', 'subset', 'outcome_id', 'multiplicity_family', 'identity']
    result['p_holm'] = result.groupby(groups, dropna=False).p_raw.transform(planned_holm)
    result['significant_holm_0_05'] = result.p_holm.lt(.05)
    return result


def validate_statistics(table, frame):
    dictionary = table[['outcome_id']].drop_duplicates()
    native = table.copy()
    native['population'] = native.get('engine_population', native.population)
    for column in ['p_raw', 'p_holm', 'p_bh_outcomes', 'test_used']:
        if 'engine_' + column in native:
            native[column] = native['engine_' + column]
    validate_results(engine, native, dictionary)
    groups = ['population', 'model_id', 'subset', 'outcome_id', 'multiplicity_family', 'identity']
    for _, family in table.groupby(groups, dropna=False):
        expected_size = {'direct_identity_24': 24, 'neutral_sensitivity_4': 4, 'primary_implicit_explicit_3': 3}[family.multiplicity_family.iloc[0]]
        if len(family) != expected_size:
            raise ValueError('Incomplete planned Holm family')
        np.testing.assert_allclose(family.p_holm, planned_holm(family.p_raw), equal_nan=True, rtol=1e-12)
    if 'value_col' in native:
        support_audit(frame, native)


def subset_diagnostics(frame, table):
    records = []
    for (model, left_persona, right_persona), tests in table.groupby(['model_id', 'left_persona', 'right_persona'], sort=False):
        part = frame[frame.model_id.eq(model)]
        left = part[part.persona.eq(left_persona)].set_index('question_id')
        right = part[part.persona.eq(right_persona)].set_index('question_id').reindex(left.index)
        for row in tests.itertuples(index=False):
            keep = left.text_available & right.text_available & left[row.value_col].notna() & right[row.value_col].notna()
            if row.population == 'text_both_committed':
                keep &= left.validated_committed & right.validated_committed
            for subset in engine.SUBSETS:
                mask = keep & left.source_subset.eq(subset)
                delta = left.loc[mask, row.value_col] - right.loc[mask, row.value_col]
                records.append(dict(model_id=model, left_persona=left_persona, right_persona=right_persona,
                                    population=row.population, outcome_id=row.outcome_id, source_subset=subset,
                                    n_pairs=len(delta), mean_delta=delta.mean(), median_delta=delta.median(),
                                    effect=delta.mean() if row.outcome_scale == 'binary' else delta.median(),
                                    outcome_scale=row.outcome_scale))
    return pd.DataFrame(records)


def build_statistics(view, catalog, output, bootstrap_resamples=10000):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    dictionary, outcomes = engine.build_outcome_dictionary(catalog)
    descriptive_frame = pool_frame(engine_frame(view, catalog))
    frame = pool_frame(paired_engine_frame(view, catalog))
    dictionary.to_csv(output / 'outcome_dictionary.csv', index=False)
    engine.build_descriptives(descriptive_frame, outcomes).to_csv(output / 'overall_descriptive.csv', index=False)
    weights = descriptive_frame.groupby(['model_id', 'persona', 'source_subset']).agg(total=('question_id', 'size'), n_text=('text_available', 'sum')).reset_index()
    pair_weights = frame.groupby(['model_id', 'persona', 'source_subset']).text_available.sum().rename('n_pairing_text').reset_index()
    weights = weights.merge(pair_weights, on=['model_id', 'persona', 'source_subset'], validate='one_to_one')
    weights['text_share'] = weights.n_text / weights.groupby(['model_id', 'persona']).n_text.transform('sum')
    weights.to_csv(output / 'subset_weights.csv', index=False)
    inputs = dict(policy=POLICY, bootstrap_resamples=bootstrap_resamples,
                  engine_sha256=sha256(engine.__file__), adapter_sha256=sha256(__file__),
                  view_sha256=__import__('hashlib').sha256(pd.util.hash_pandas_object(view, index=True).values.tobytes()).hexdigest(),
                  catalog_sha256=__import__('hashlib').sha256(catalog.to_json().encode()).hexdigest())
    dependencies = {Path(inspect.getfile(fn)) for fn in [pool_frame, engine_frame, paired_engine_frame, engine._bootstrap_ci, engine._paired_numeric_test, engine.holm_adjust]}
    inputs['dependency_hashes'] = {str(p): sha256(p) for p in sorted(dependencies)}
    inputs['versions'] = {name: importlib.metadata.version(name) for name in ['pandas', 'numpy', 'scipy']}
    outputs, supports = [], []
    for model in [m for m in MODELS if m in set(frame.model_id)]:
        print(f'[statistics] {model}', flush=True)
        part = frame[frame.model_id.eq(model)]
        checkpoint = output / ('checkpoint_' + model)
        checkpoint.mkdir(exist_ok=True)
        manifest = checkpoint / 'manifest.json'
        cached = checkpoint / 'contrasts.csv.gz'
        if manifest.exists():
            saved = json.loads(manifest.read_text(encoding='utf-8'))
            if saved['inputs'] != inputs or saved['sha256'] != sha256(cached):
                raise ValueError('Checkpoint provenance mismatch')
            table = pd.read_csv(cached)
        else:
            within = engine.build_within_identity_pairs(part, outcomes, bootstrap_resamples)
            direct = engine.build_direct_identity_pairs(part, outcomes, bootstrap_resamples)
            table = pd.concat([within, direct], ignore_index=True).merge(dictionary[['outcome_id', 'value_col']], on='outcome_id', validate='many_to_one')
            table.to_csv(cached, index=False, compression='gzip')
            write_json(manifest, dict(inputs=inputs, sha256=sha256(cached)))
        validate_results(engine, table, dictionary)
        supports.append(support_audit(part, table))
        outputs.append(table)
        print(f'[statistics] saved {len(table):,} contrasts', flush=True)
    table = pd.concat(outputs, ignore_index=True)
    table = presentation_inference(table)
    table['engine_population'] = table.population
    table['population'] = table.population.map({'all_visible': 'text_all', 'both_committed': 'text_both_committed'})
    if table.population.isna().any():
        raise ValueError('Unexpected pooled population')
    table['analysis_policy_version'] = POLICY
    validate_statistics(table, frame)
    table.to_csv(output / 'paired_contrasts.csv.gz', index=False, compression='gzip')
    support = pd.concat(supports, ignore_index=True)
    support['population'] = support.population.map({'all_visible': 'text_all', 'both_committed': 'text_both_committed'})
    support.to_csv(output / 'pair_composition.csv', index=False)
    subset_diagnostics(frame, table).to_csv(output / 'subset_effect_diagnostics.csv', index=False)
    old = pd.concat([pd.read_csv(LEGACY / 'within.csv'), pd.read_csv(LEGACY / 'direct.csv')], ignore_index=True)
    old.population = old.population.map({'all_visible': 'text_all', 'both_committed': 'text_both_committed'})
    keys = ['population', 'model_id', 'subset', 'left_persona', 'right_persona', 'outcome_id', 'outcome_scale']
    values = ['n_pairs', 'effect', 'ci_low', 'ci_high', 'p_holm', 'effect_size']
    comparison = table[keys + values].merge(old[keys + values], on=keys, suffixes=('_new', '_old'), validate='one_to_one')
    comparison['direction_reversed'] = comparison.effect_new * comparison.effect_old < 0
    comparison['significance_changed'] = comparison.p_holm_new.lt(.05) != comparison.p_holm_old.lt(.05)
    comparison['n_changed'] = comparison.n_pairs_new != comparison.n_pairs_old
    comparison.to_csv(output / 'old_new_comparison.csv.gz', index=False, compression='gzip')
    write_json(output / 'provenance.json', dict(inputs=inputs, seed_scheme='native stable SHA256 seeds; engine BUNDLE_VERSION retained',
               main_rows=int(table.population.eq('text_all').sum()), sensitivity_rows=int(table.population.eq('text_both_committed').sum()),
               old_new_rows=len(comparison), direction_reversed=int(comparison.direction_reversed.sum()),
               significance_changed=int(comparison.significance_changed.sum()),
               pairing_excluded=int((~view.pairing_eligible).sum()),
               note='Pooled v2 vs pooled r5/new eligibility + approved exclusion of 20 mismatched neutral responses from pairing only; descriptives retain them. R5 per-subset inference is not relabelled pooled.'))
    return {p.stem: p for p in output.glob('*.csv*')}
