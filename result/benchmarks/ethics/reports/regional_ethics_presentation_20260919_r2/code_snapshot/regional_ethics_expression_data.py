"""Per-response regional category diversity and all-marker profiles."""
import numpy as np
import pandas as pd
from .regional_ethics_presentation_data import POLICY

LENGTH_BANDS = ['1–49', '50–99', '100–199', '200–399', '400+']


def validate_counts(view, marker_ids):
    if view.response_key.isna().any() or view.response_key.duplicated().any():
        raise ValueError('Response keys must be unique')
    text = view[view.explanation_available]
    counts = text[[m + '__count' for m in marker_ids]].to_numpy(float)
    if not np.isfinite(counts).all() or (counts < 0).any() or (counts != np.floor(counts)).any():
        raise ValueError('Counts must be finite nonnegative integers')
    if text.word_count.le(0).any():
        raise ValueError('Explanation needs positive word count')
    return text, counts


def profile_table(view, marker_ids):
    validate_counts(view, marker_ids)
    rows = []
    for (model, persona), group in view.groupby(['model_id', 'persona'], sort=False):
        text = group[group.explanation_available]
        for marker in marker_ids:
            counts = text[marker + '__count']
            density = text[marker + '__density']
            np.testing.assert_allclose(density, counts * 100 / text.word_count, rtol=1e-5, atol=1e-6)
            rows.append(dict(model_id=model, persona=persona, identity=group.identity.iloc[0], condition=group.condition.iloc[0],
                             marker_id=marker, n_total=len(group), n_text=len(text), n_positive=int(counts.gt(0).sum()),
                             count_sum=int(counts.sum()), mean_density=density.mean(),
                             prevalence_pct=100 * counts.gt(0).mean() if len(text) else np.nan, analysis_policy_version=POLICY))
    return pd.DataFrame(rows)


def diversity_rows(view, marker_ids):
    text, counts = validate_counts(view, marker_ids)
    result = view[[c for c in view if '__' not in c]].copy()
    result['distinct_markers'] = np.nan
    result['raw_marker_count'] = np.nan
    result.loc[text.index, 'distinct_markers'] = (counts > 0).sum(axis=1)
    result.loc[text.index, 'raw_marker_count'] = counts.sum(axis=1)
    result['marker_density'] = result.raw_marker_count * 100 / result.word_count.where(result.explanation_available)
    result['length_band'] = pd.cut(result.word_count.where(result.explanation_available), [0, 49, 99, 199, 399, np.inf], labels=LENGTH_BANDS)
    return result


def diversity_tables(rows, n_markers=23):
    summaries, distributions, lengths = [], [], []
    for (model, persona), group in rows.groupby(['model_id', 'persona'], sort=False):
        text = group[group.explanation_available]
        key = dict(model_id=model, persona=persona, analysis_policy_version=POLICY)
        record = dict(key, n_total=len(group), n_text=len(text), zero_marker_text=int(text.distinct_markers.eq(0).sum()),
                      abstain_text=int(text.analysis_answer_status.eq('abstain').sum()),
                      mean_distinct_markers=text.distinct_markers.mean(), median_distinct_markers=text.distinct_markers.median(),
                      raw_marker_count=int(text.raw_marker_count.sum()), mean_marker_density=text.marker_density.mean())
        for status in ['committed', 'abstain', 'missing']:
            record[status] = int(group.analysis_answer_status.eq(status).sum())
        summaries.append(record)
        for value in range(n_markers + 1):
            distributions.append(dict(key, distinct_markers=value, n=int(text.distinct_markers.eq(value).sum()), n_text=len(text),
                                      ecdf=text.distinct_markers.le(value).mean() if len(text) else np.nan))
        for band in LENGTH_BANDS:
            part = text[text.length_band.eq(band)]
            lengths.append(dict(key, length_band=band, n=len(part), mean_distinct_markers=part.distinct_markers.mean(),
                                median_distinct_markers=part.distinct_markers.median(), raw_marker_count=int(part.raw_marker_count.sum()),
                                mean_marker_density=part.marker_density.mean()))
    return dict(diversity_summary=pd.DataFrame(summaries), diversity_distribution=pd.DataFrame(distributions),
                diversity_by_length=pd.DataFrame(lengths))
