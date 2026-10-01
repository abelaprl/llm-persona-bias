"""Whole-ETHICS paired estimates, question bootstrap and explicit denominators."""
from itertools import combinations
import hashlib
import numpy as np
import pandas as pd
from scipy import stats
from src.benchmarks.ethics.religion_lm_v3_registry import get_registry, family_for

IDENTITIES = ('islam','christian','jewish','atheist')
CONDITIONS = ('explicit','implicit_1','implicit_2','implicit_3')


def holm_adjust(values):
    values = np.asarray(values,float)
    result = np.full(len(values),np.nan)
    indices = np.flatnonzero(np.isfinite(values))
    ordered = indices[np.argsort(values[indices],kind='stable')]
    result[ordered] = np.minimum(1,np.maximum.accumulate(values[ordered]*np.arange(len(ordered),0,-1)))
    return result


def build_pairs(rows,left_persona,right_persona):
    keys = ['model_id','subset','question_id']
    left = rows.loc[rows.persona.eq(left_persona)].drop(columns='persona')
    right = rows.loc[rows.persona.eq(right_persona)].drop(columns='persona')
    return left.merge(right,on=keys,suffixes=('_left','_right'),validate='one_to_one')


def contrast_specs():
    for identity in IDENTITIES:
        for condition in CONDITIONS:
            yield 'vs_neutral',identity+'_'+condition,'neutral'
        for condition in CONDITIONS[1:]:
            yield 'implicit_vs_explicit',identity+'_'+condition,identity+'_explicit'
    for left,right in combinations(IDENTITIES,2):
        for condition in CONDITIONS:
            yield 'direct_identity',left+'_'+condition,right+'_'+condition


def bootstrap_effects(differences, subsets, seed, resamples=10000, equal_subset=False):
    d = np.asarray(differences,float)
    if d.ndim == 1:
        d = d[:,None]
    if not np.isfinite(d).all():
        raise ValueError('Bootstrap input must be a common finite paired population')
    n,k = d.shape
    if n == 0:
        return {name:np.full(k,np.nan) for name in ('effect','ci_low','ci_high','p_raw')}
    rng = np.random.default_rng(seed)
    labels = np.unique(subsets)
    groups = [np.flatnonzero(subsets==label) for label in labels] if equal_subset else [np.arange(n)]
    estimate = sum(d[g].mean(axis=0)/len(groups) for g in groups)
    draws = np.zeros((resamples,k))
    # A multinomial vector resamples whole question rows jointly across outcomes.
    for g in groups:
        values = d[g]
        active = np.ptp(values,axis=0)>0
        draws[:,~active] += values[0,~active]/len(groups)
        if active.any():
            # Chunk limits peak memory; invariant outcomes require no sampling.
            for start in range(0,resamples,250):
                b = min(250,resamples-start)
                weights = rng.multinomial(len(g),np.full(len(g),1/len(g)),size=b)
                draws[start:start+b,active] += weights @ values[:,active] / len(g)/len(groups)
    low,high = np.quantile(draws,[.025,.975],axis=0)
    # Centered bootstrap null test; Monte Carlo correction prevents zero p-values.
    p = (1+(np.abs(draws-estimate)>=np.abs(estimate)-1e-14).sum(axis=0))/(resamples+1)
    p[np.all(d==0,axis=0)] = 1
    if n < 2 or (equal_subset and any(len(g)<2 for g in groups)):
        low[:] = high[:] = p[:] = np.nan
    return dict(effect=estimate,ci_low=low,ci_high=high,p_raw=p)


def summarize_overall(rows):
    markers = [m.marker_id for m in get_registry()]
    records = []
    for (model,persona), group in rows.groupby(['model_id','persona'],sort=True):
        for marker in markers:
            eligible = group.explanation_available
            final = group.final_output_available
            records.append(dict(model_id=model,persona=persona,subset='ethics_all',marker_id=marker,family_id=family_for(marker),
                total=len(group),final_available=int(final.sum()),linguistic_available=int(eligible.sum()),
                label_only=int(group.surface_status.eq('label_only').sum()),abstain_text=int((group.answer_status.eq('abstain')&eligible).sum()),
                raw_count=group[marker+'__count'].sum(min_count=1),word_total=group.loc[eligible,'word_count'].sum(),
                prevalence_text=group[marker+'__present'].mean()*100,prevalence_output=group[marker+'__present_output'].mean()*100,
                mean_response_density=group[marker+'__density'].mean(),status='candidate_pending_human_review'))
    return pd.DataFrame(records)


def estimate_contrasts(rows,seed=18222008,resamples=10000):
    markers = [m.marker_id for m in get_registry()]
    records = []
    for model,group in rows.groupby('model_id',sort=True):
        for contrast_index,(kind,left,right) in enumerate(contrast_specs()):
            pair = build_pairs(group,left,right)
            if pair.empty:
                continue
            both_text = pair.explanation_available_left & pair.explanation_available_right
            both_output = pair.final_output_available_left & pair.final_output_available_right
            committed = pair.answer_status_left.eq('committed') & pair.answer_status_right.eq('committed')
            quality = r'high_token_repetition_candidate|mixed_latin_cjk|symbol_repetition_candidate|meta_response_candidate'
            no_flag = ~pair.surface_flags_left.fillna('').str.contains(quality) & ~pair.surface_flags_right.fillna('').str.contains(quality)
            populations = [('text_all',both_text,False),('output_all',both_output,False),
                ('both_committed_text',both_text&committed,False),('equal_subset_text',both_text,True),
                ('no_quality_flag_text',both_text&no_flag,False)]
            cache = {}
            for population,mask,equal in populations:
                suffixes = [('prevalence_output','__present_output',100)] if population=='output_all' else [('prevalence_text','__present',100),('density','__density',1)]
                selected = pair.loc[mask]
                columns = [(marker,metric,suffix,scale) for metric,suffix,scale in suffixes for marker in markers]
                d = np.column_stack([(selected[m+s+'_left'].to_numpy(float)-selected[m+s+'_right'].to_numpy(float))*scale for m,_,s,scale in columns])
                unique_seed = int(hashlib.sha256(f'{seed}|{model}|{kind}|{left}|{right}|{population}'.encode()).hexdigest()[:8],16)
                cache_key = (tuple(selected.index),equal,tuple(suffixes))
                if cache_key not in cache:
                    cache[cache_key] = bootstrap_effects(d,selected.subset.to_numpy(),unique_seed,resamples,equal)
                result = cache[cache_key]
                for i,(marker,metric,_,_) in enumerate(columns):
                    record = dict(model_id=model,subset='ethics_all',contrast_kind=kind,left_persona=left,right_persona=right,
                        population=population,marker_id=marker,family_id=family_for(marker),metric=metric,n_pairs=len(selected),
                        total_pairs=len(pair),bootstrap_resamples=resamples,status='candidate_pending_human_review')
                    record.update({k:float(v[i]) for k,v in result.items()})
                    for subset in ('commonsense','deontology','justice','virtue'):
                        record['n_'+subset] = int(selected.subset.eq(subset).sum())
                    records.append(record)
            if (contrast_index+1)%13==0:
                print(f'Contrasts {model}: {contrast_index+1}/52',flush=True)
        print(f'Statistics complete: {model}',flush=True)
    result = pd.DataFrame(records)
    group_cols = ['model_id','population','metric','contrast_kind','family_id']
    result['p_holm'] = result.groupby(group_cols,dropna=False).p_raw.transform(lambda values:holm_adjust(values.to_numpy()))
    return result
