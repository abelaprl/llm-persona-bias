"""Display-only adapters from the frozen first candidate, without new inference."""
from pathlib import Path

import numpy as np
import pandas as pd

from .gender_lm_reference_data import ROOT
from .gender_lm_candidate_gsm8k import PRIMARY

PREVIOUS = ROOT / 'notebooks/gender/linguistic marker/candidate'
SHARED_MARKERS = ['organization_explicit_step', 'stance_hedge', 'connective_causal',
                  'connective_contrastive', 'self_reference_first_person',
                  'self_reference_inclusive_we', 'engagement_reader_directive']
FILES = {
    'GSM8K': dict(coverage='gsm8k_01_coverage.csv', length='gsm8k_02_verbosity.csv',
                  profile='gsm8k_03_absolute_profile.csv',
                  effects=['gsm8k_04_direct_effects_page1.csv', 'gsm8k_05_direct_effects_page2.csv']),
    'ETHICS': dict(coverage='ethics_E01_coverage.csv', length='ethics_E02_verbosity.csv',
                   profile='ethics_all_marker_profiles.csv', effects=['ethics_all_direct_effects.csv']),
}


def input_paths():
    paths = []
    for spec in FILES.values():
        for value in spec.values():
            paths.extend(PREVIOUS / 'tables' / name for name in (value if isinstance(value, list) else [value]))
    return paths


def read_table(name):
    return pd.read_csv(PREVIOUS / 'tables' / name, float_precision='round_trip')


def adapt_length(source, benchmark):
    out = source.copy(deep=True)
    out['estimate'] = out['median']
    out['lower'] = out['q1' if benchmark == 'GSM8K' else 'q25']
    out['upper'] = out['q3' if benchmark == 'GSM8K' else 'q75']
    out['interval_kind'] = 'IQR'
    return out


def adapt_effect(source, benchmark, detail=False):
    """Keep all source fields; new display columns name the estimand explicitly."""
    out = source.copy(deep=True)
    if benchmark == 'GSM8K':
        if detail and not out.outcome_id.isin(PRIMARY[3:]).all():
            raise ValueError('Only marker density outcomes can use this detail adapter')
        if not (out.left_identity.eq('man') & out.right_identity.eq('woman')).all():
            raise ValueError('GSM8K must be Man minus Woman')
        out['marker_id'] = out.outcome_id
        out['panel'] = out.panel_scenario
        out['estimate'] = out.median_paired_delta if detail else out.rank_biserial
        out['lower'] = out.ci_low if detail else np.nan
        out['upper'] = out.ci_high if detail else np.nan
        out['changed'] = out.n_nonzero
        out['corrected_p'] = out.p_holm
        out['estimand_display'] = 'median_paired_difference' if detail else 'matched_rank_biserial'
        out['display_unit'] = 'occurrences_per_100_words' if detail else 'rank_biserial'
    elif benchmark == 'ETHICS':
        if not (out.left.str.startswith('woman_') & out.right.str.startswith('man_') &
                out.effect_direction.eq('left_minus_right')).all():
            raise ValueError('ETHICS frozen orientation must be Woman minus Man')
        if not (out.estimand.eq(out.ci_estimand).all() and out.ci_level.eq(.95).all()
                and out.ci_simultaneous.eq(False).all()):
            raise ValueError('ETHICS CI must match its estimand and be 95% marginal')
        out['panel'] = out.panel_id
        out['estimate'] = -out.effect
        out['lower'], out['upper'] = -out.ci_high, -out.ci_low
        out['changed'] = out.n_changed
        out['corrected_p'] = out.adjusted_p
        out['estimand_display'] = out.estimand
        out['display_unit'] = out.measurement_unit
    else:
        raise ValueError('Unknown benchmark')
    return out


def load(benchmark):
    spec = FILES[benchmark]
    coverage = read_table(spec['coverage'])
    if benchmark == 'GSM8K':
        coverage = coverage.assign(available=coverage.n_text_available, total=coverage.target,
                                   percent=coverage.text_available_pct)
    else:
        coverage = coverage.loc[coverage.kind.eq('explanation')].copy()
        coverage['persona'] = coverage.setting
    length = adapt_length(read_table(spec['length']), benchmark)
    profiles = read_table(spec['profile'])
    if benchmark == 'GSM8K':
        profiles['marker_id'] = profiles.outcome_id
        profiles['estimate'] = profiles.prevalence_pct
        profiles['raw_count_total'] = profiles.raw_occurrences
    else:
        profiles['estimate'] = profiles.display_value
    selected = profiles.loc[profiles.marker_id.isin(SHARED_MARKERS) & profiles.persona.isin(
        ['neutral', 'man_explicit_1', 'woman_explicit_1'])].copy()
    selected['panel'] = selected.persona.map({'neutral': 'N', 'man_explicit_1': 'Man E',
                                             'woman_explicit_1': 'Woman E'})
    selected['display_unit'] = 'percent'
    raw = pd.concat([read_table(name) for name in spec['effects']], ignore_index=True)
    effects = adapt_effect(raw, benchmark)
    common = effects.loc[effects.marker_id.isin(SHARED_MARKERS)].copy()
    detail_raw = raw.loc[raw['outcome_id' if benchmark == 'GSM8K' else 'marker_id'].isin(SHARED_MARKERS)]
    detail = adapt_effect(detail_raw, benchmark, detail=True)
    return dict(coverage=coverage, length=length, profile=selected, effects=effects,
                overview=common, detail=detail, all_profiles=profiles)
