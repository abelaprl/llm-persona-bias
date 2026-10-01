"""Presentation views of frozen statistics; no new inferential tests."""
import numpy as np
import pandas as pd

KEY = ['model_id', 'condition', 'marker_id', 'measurement']
SUBSETS = ['commonsense', 'deontology', 'justice', 'virtue']


def atlas_data(summary):
    """Use prevalence for the overview, retaining density and raw counts."""
    data = summary.loc[summary.scope.eq('overall')]
    keys = ['scope', 'model_id', 'persona', 'marker_id']
    prevalence = data.loc[data.measurement.eq('presence')].copy()
    counts = data.loc[data.measurement.eq('count'), keys + ['value', 'raw_occurrences']]
    counts = counts.rename(columns={'value': 'mean_count', 'raw_occurrences': 'total_raw_occurrences'})
    density = data.loc[data.measurement.eq('density'), keys + ['value']]
    density = density.rename(columns={'value': 'mean_density'})
    return prevalence.merge(counts, on=keys, validate='one_to_one').merge(
        density, on=keys, validate='one_to_one')


def subset_view(primary):
    """Count opposing nonzero subset signs, never average tests or effects."""
    rows = []
    for keys, group in primary.groupby(KEY, sort=False):
        if group.scope.duplicated().any():
            raise ValueError('Duplicate subset estimate')
        effects = group.set_index('scope').effect
        overall = effects.get('overall', np.nan)
        subset = effects.reindex(SUBSETS)
        available = subset.dropna()
        status = 'complete' if len(available) == 4 else 'partial'
        count = np.nan
        if pd.isna(overall) or available.empty:
            status = 'unavailable'
        elif overall == 0:
            status = 'zero_overall'
        else:
            count = int((np.sign(available) == -np.sign(overall)).sum())
        rows.append(dict(zip(KEY, keys)) | dict(overall_effect=overall,
                    opposite_count=count, n_available_subsets=len(available),
                    n_zero_subsets=int(available.eq(0).sum()), status=status))
    return pd.DataFrame(rows)


def normalization_view(direct):
    """Compare sign categories; retain both source rows without re-estimation."""
    data = direct.loc[direct.scope.eq('overall')]
    keys = KEY[:-1]
    count = data.loc[data.measurement.eq('count')].set_index(keys).add_prefix('count_')
    density = data.loc[data.measurement.eq('density')].set_index(keys).add_prefix('density_')
    result = count.join(density, how='outer', validate='one_to_one').reset_index()
    a, b = result.count_effect, result.density_effect
    result['status'] = np.select(
        [a.isna() | b.isna(), a.eq(0) & b.eq(0), a.eq(0) | b.eq(0), np.sign(a).eq(np.sign(b))],
        ['unavailable', 'both_zero', 'one_zero', 'same'], default='opposite')
    known = result.count_adjusted_p.notna() & result.density_adjusted_p.notna()
    result['holm_decision_changed'] = (
        result.count_adjusted_p.lt(.05) != result.density_adjusted_p.lt(.05)).astype('boolean')
    result.loc[~known, 'holm_decision_changed'] = pd.NA
    return result


def interaction_view(source, registry):
    primary = registry.set_index('marker_id').measurement.replace({'binary': 'presence'})
    result = source.loc[source.scope.eq('overall') &
                        source.measurement.eq(source.marker_id.map(primary))].copy()
    labels = {('implicit_1', 'implicit_2'): '1−2', ('implicit_1', 'implicit_3'): '1−3',
              ('implicit_2', 'implicit_3'): '2−3'}
    result['condition'] = [labels[(left, right)] for left, right in zip(result.left, result.right)]
    return result
