"""Frozen-data adapters for religious expression heatmaps and response cards."""
import html
import re

import numpy as np
import pandas as pd

from src.benchmarks.ethics.religion_ethics_gsm8k_style_data import (
    LABELS, MODELS, PERSONAS, CONDITIONS,
)

RELIGION_MARKERS = [marker for marker in LABELS if marker.startswith('religion_')]
CARD_IDENTITIES = dict(zip(MODELS, ['islam', 'christian', 'jewish', 'islam', 'atheist', 'jewish']))
GROUPS = {
    'identity': ('Identitas / rujukan umum', '#dbeafe'),
    'source': ('Kitab / figur agama', '#ccfbf1'),
    'lexical': ('Konsep / praktik / formula', '#fef3c7'),
    'framing': ('Pola framing / respons identitas', '#fce7f3'),
    'overlap': ('Beberapa kategori bertumpuk', '#e2e8f0'),
}


def marker_group(marker):
    if marker in ['religion_scripture_reference', 'religion_figure_reference']:
        return 'source'
    if marker.startswith('religion_identity_') and marker != 'religion_identity_irrelevance':
        return 'identity'
    if marker == 'religion_generic_reference':
        return 'identity'
    if marker in ['religion_deity_invocation', 'religion_practice_community',
                  'religion_afterlife_soteriology', 'religion_named_concept_reference',
                  'religion_formulaic_expression']:
        return 'lexical'
    return 'framing'


def heatmap_table(summary, markers=RELIGION_MARKERS, models=MODELS, personas=PERSONAS):
    keys = ['model_id', 'persona', 'marker_id']
    selected = summary.loc[summary.marker_id.isin(markers) & summary.model_id.isin(models)
                           & summary.persona.isin(personas)].copy()
    if selected.duplicated(keys).any():
        raise ValueError('Duplicate heatmap cells')
    index = pd.MultiIndex.from_product([models, personas, markers], names=keys)
    if set(map(tuple, selected[keys].values)) != set(index):
        raise ValueError('Missing heatmap cells; absence cannot be interpreted as zero')
    selected = selected.set_index(keys).loc[index].reset_index()
    # Frozen v3 prevalence is already expressed in percentage points, not fractions.
    selected['prevalence_pct'] = selected.prevalence_text
    observed = selected.prevalence_pct.dropna()
    if not observed.between(0, 100).all():
        raise ValueError('Frozen prevalence must be in the 0-100 percentage range')
    selected.loc[selected.linguistic_available.eq(0), 'prevalence_pct'] = np.nan
    return selected


def select_card_rows(rows, markers=RELIGION_MARKERS, assignments=CARD_IDENTITIES):
    keys = ['model_id', 'subset', 'question_id']
    if rows.duplicated(keys + ['persona']).any():
        raise ValueError('Duplicate response keys in card candidates')
    counts = [m + '__count' for m in markers]
    all_candidates, selected = [], []
    for model, identity in assignments.items():
        print(f'  Memeringkat contoh: {model} / {identity}', flush=True)
        personas = ['neutral'] + [identity + '_' + condition for condition in CONDITIONS]
        available = rows[rows.model_id.eq(model) & rows.persona.isin(personas)]
        candidates = []
        for (subset, question), group in available.groupby(['subset', 'question_id'], sort=True):
            if set(group.persona) != set(personas):
                continue
            by_persona = group.set_index('persona')
            explicit = by_persona.loc[identity + '_explicit']
            implicit_scores = []
            for condition in CONDITIONS[1:]:
                implicit = by_persona.loc[identity + '_' + condition]
                # Missing responses contribute no observed hits to example ranking only.
                # Their original NA counts/status are retained in the selected records.
                joined = np.nan_to_num(explicit[counts].to_numpy(dtype=float), nan=0.)
                joined += np.nan_to_num(implicit[counts].to_numpy(dtype=float), nan=0.)
                implicit_scores.append((int((joined > 0).sum()), int(joined.sum()), condition))
            diversity, total, chosen = sorted(implicit_scores, key=lambda item: (-item[0], -item[1], item[2]))[0]
            candidates.append(dict(model_id=model, identity=identity, subset=subset, question_id=question,
                                   implicit_condition=chosen, distinct_markers=diversity, raw_hits=total,
                                   combined_words=int(explicit.word_count + by_persona.loc[identity + '_' + chosen].word_count)))
        if not candidates:
            raise ValueError(f'No complete N/E/I1/I2/I3 card candidates for {model}/{identity}')
        ranked = pd.DataFrame(candidates).sort_values(
            ['distinct_markers', 'raw_hits', 'combined_words', 'subset', 'question_id'],
            ascending=[False, False, True, True, True]).reset_index(drop=True)
        ranked['rank'] = np.arange(1, len(ranked) + 1)
        ranked['selected'] = ranked['rank'].eq(1)
        all_candidates.append(ranked)
        winner = ranked.iloc[0]
        chosen_personas = ['neutral', identity + '_explicit', identity + '_' + winner.implicit_condition]
        triplet = available.loc[available.subset.eq(winner.subset) & available.question_id.eq(winner.question_id)]
        triplet = triplet.set_index('persona').loc[chosen_personas].reset_index()
        triplet['card_id'] = f'card_{len(selected) + 1:02d}'
        triplet['card_identity'] = identity
        selected.append(triplet)
    return pd.concat(selected, ignore_index=True), pd.concat(all_candidates, ignore_index=True)


def highlight_segments(text, hits):
    boundaries = {0, len(text)}
    for hit in hits:
        start, end = int(hit['raw_start']), int(hit['raw_end'])
        if not 0 <= start < end <= len(text) or text[start:end] != hit['raw_matched_text']:
            raise ValueError(f'Occurrence does not match raw final text: {start}:{end}')
        boundaries.update([start, end])
    points = sorted(boundaries)
    segments = []
    for start, end in zip(points, points[1:]):
        markers = sorted({h['marker_id'] for h in hits if h['raw_start'] < end and h['raw_end'] > start})
        groups = {marker_group(marker) for marker in markers}
        group = 'overlap' if len(groups) > 1 else next(iter(groups), None)
        segments.append(dict(start=start, end=end, text=text[start:end], markers=markers, group=group))
    return segments


def highlight_html(text, hits):
    output = []
    for segment in highlight_segments(text, hits):
        escaped = html.escape(segment['text'])
        if segment['group']:
            title = html.escape('; '.join(segment['markers']), quote=True)
            color = GROUPS[segment['group']][1]
            escaped = f'<mark style="background:{color}" title="{title}">{escaped}</mark>'
        output.append(escaped)
    return ''.join(output)


def excerpt_window(text, hits, max_words=100):
    tokens = list(re.finditer(r'\S+', text))
    if len(tokens) <= max_words:
        return 0, len(text)
    best = None
    for index in range(len(tokens) - max_words + 1):
        start, end = tokens[index].start(), tokens[index + max_words - 1].end()
        included = [hit for hit in hits if hit['raw_start'] >= start and hit['raw_end'] <= end]
        score = (len({hit['marker_id'] for hit in included}), len(included), -index)
        if best is None or score > best[0]:
            best = score, start, end
    return best[1], best[2]
