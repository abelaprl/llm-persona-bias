"""Observed search probes and development-only full-response readback queues."""
import re
import pandas as pd

# Search probes from the recorded 72-response review, not production rules.
PROBES = {
    'ai_identity_variation': {'pattern':r'\bas a (?:neutral|helpful) ai(?: assistant)?\b','sample_id':'R013,R014'},
    'binary_decline': {'pattern':r'\bi cannot (?:give a binary|answer this with a binary)\b','sample_id':'R065,R069'},
    'niyyah': {'pattern':r'\bniyyah\b','sample_id':'R069'},
    'inline_heading': {'pattern':r'(?<!\n)\s###\s+\*\*','sample_id':'R037'},
}


def build_discovery_queue(rows, question_split, prior_sample_keys, sampling_design, seed):
    data = rows.merge(question_split,on=['subset','question_id'],validate='many_to_one')
    pool = data.loc[data.partition.eq('development') & ~data.response_key.isin(prior_sample_keys)]
    selected = []
    for _, group in pool.groupby('model_id',sort=True):
        selected.append(group.sample(n=min(len(group),sampling_design.get('per_model',12)),random_state=seed))
    return pd.concat(selected) if selected else pool.iloc[:0]


def scan_observed_forms(rows, form_specs=PROBES):
    records = []
    for name, spec in form_specs.items():
        counts = rows.final_output.str.count(spec['pattern'],flags=re.I)
        part = rows[['model_id','persona','subset']].copy()
        part['response_hits'] = counts.gt(0).astype(int)
        part['occurrence_hits'] = counts
        part = part.groupby(['model_id','persona','subset'],as_index=False).sum()
        part['form_id'] = name
        part['source_sample_id'] = spec['sample_id']
        part['search_pattern'] = spec['pattern']
        part['review_status'] = 'search_probe_only'
        records.append(part)
    return pd.concat(records,ignore_index=True)


def targeted_readback(rows, split, seed=18222008):
    data = rows.merge(split,on=['subset','question_id'],validate='many_to_one')
    data = data.loc[data.partition.eq('development')]
    selections = []
    for form_id, spec in PROBES.items():
        pool = data.loc[data.final_output.str.contains(spec['pattern'],flags=re.I,regex=True)]
        for model, group in pool.groupby('model_id',sort=True):
            selected = group.sample(n=min(len(group),10),random_state=seed).copy()
            selected['form_id'] = form_id
            selected['model_hits_development'] = len(group)
            selections.append(selected)
    return pd.concat(selections,ignore_index=True)
