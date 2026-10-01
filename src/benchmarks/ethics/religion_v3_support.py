"""Add auditable raw numerators and marginal estimates to paired contrasts."""
import numpy as np
import pandas as pd
import hashlib
from src.benchmarks.ethics.religion_lm_v3_statistics import build_pairs,contrast_specs


def paired_marker_support(pair,markers,equal_subset=False,seed=18222008,resamples=10000):
    result = pd.DataFrame({'marker_id':markers})
    groups = [np.flatnonzero(pair.subset.to_numpy()==s) for s in pair.subset.unique()] if equal_subset else [np.arange(len(pair))]
    by_side={}
    for side in ('left','right'):
        counts = pair[[m+'__count_'+side for m in markers]].to_numpy(float)
        words = pair['word_count_'+side].to_numpy(float)
        density = np.divide(100*counts,words[:,None],out=np.full(counts.shape,np.nan),where=words[:,None]>0)
        presence = (counts>0).astype(float)*100
        by_side[side]=dict(counts=counts,density=density,presence=presence)
        def average(values):
            if not len(pair):return np.full(len(markers),np.nan)
            return sum(values[g].mean(axis=0)/len(groups) for g in groups)
        result['raw_count_'+side] = counts.sum(axis=0)
        result['word_total_'+side] = words.sum()
        result['positive_responses_'+side] = (counts>0).sum(axis=0)
        result['density_'+side] = average(density)
        result['prevalence_'+side] = average(presence)
    left_yes=by_side['left']['counts']>0;right_yes=by_side['right']['counts']>0
    result['discordant_left_only']=(left_yes&~right_yes).sum(axis=0)
    result['discordant_right_only']=(~left_yes&right_yes).sum(axis=0)
    for metric in ('density','prevalence'):
        name='presence' if metric=='prevalence' else metric
        differences=by_side['left'][name]-by_side['right'][name]
        constant=np.zeros(len(markers),dtype=bool)
        if len(pair):
            constant[:]=True
            for group in groups:
                d=differences[group]
                constant &= np.isfinite(d).all(axis=0)&(np.ptp(np.where(np.isfinite(d),d,0.),axis=0)<=1e-12)
        result['constant_'+metric+'_difference']=constant
        from src.benchmarks.ethics.religion_v3_randomization import paired_sign_flip
        weights=np.zeros(len(pair))
        for group in groups:
            if len(group):weights[group]=1/(len(groups)*len(group))
        if metric=='density' or equal_subset:
            p,methods,informative=paired_sign_flip(differences,weights,seed,resamples)
            result[metric+'_sign_flip_p']=p
            result[metric+'_sign_flip_method']=methods
        else:informative=(np.abs(differences)>1e-12).sum(axis=0)
        result['informative_'+metric+'_pairs']=informative
    return result


def augment_contrasts(frame,contrasts,markers):
    records = []
    for model,group in frame.groupby('model_id',sort=True):
        for kind,left,right in contrast_specs():
            pair = build_pairs(group,left,right)
            cache={}
            both_text = pair.explanation_available_left & pair.explanation_available_right
            both_output = pair.final_output_available_left & pair.final_output_available_right
            committed = pair.answer_status_left.eq('committed') & pair.answer_status_right.eq('committed')
            quality = r'high_token_repetition_candidate|mixed_latin_cjk|symbol_repetition_candidate|meta_response_candidate'
            no_flag = ~pair.surface_flags_left.fillna('').str.contains(quality) & ~pair.surface_flags_right.fillna('').str.contains(quality)
            for population,mask,equal in [('text_all',both_text,False),('output_all',both_output,False),
                    ('both_committed_text',both_text&committed,False),('equal_subset_text',both_text,True),
                    ('no_quality_flag_text',both_text&no_flag,False)]:
                cache_key=(mask.to_numpy().tobytes(),equal)
                if cache_key not in cache:
                    seed=int.from_bytes(hashlib.sha256(f'18222008|{model}|{kind}|{left}|{right}|{population}|paired_sign_flip'.encode()).digest()[:8],'big')
                    cache[cache_key]=paired_marker_support(pair.loc[mask],markers,equal,seed)
                table=cache[cache_key].copy()
                table['model_id']=model;table['contrast_kind']=kind
                table['left_persona']=left;table['right_persona']=right;table['population']=population
                table['support_n_pairs']=int(mask.sum())
                records.append(table)
        print('Paired inference support finished: '+str(model),flush=True)
    keys = ['model_id','contrast_kind','left_persona','right_persona','population','marker_id']
    merged = contrasts.merge(pd.concat(records,ignore_index=True),on=keys,validate='many_to_one')
    assert merged.support_n_pairs.eq(merged.n_pairs).all()
    for side in ('left','right'):
        merged['estimate_'+side] = np.where(merged.metric.eq('density'),merged['density_'+side],merged['prevalence_'+side])
    finite = merged.effect.notna()
    assert np.allclose(merged.loc[finite,'effect'],(merged.estimate_left-merged.estimate_right).loc[finite])
    # Raw totals describe selected observations even when equal-subset inference is unsupported.
    unsupported = merged.inference_scope_status.eq('insufficient_four_subset_support')
    merged.loc[unsupported,['estimate_left','estimate_right']] = np.nan
    return merged.drop(columns=['density_left','density_right','prevalence_left','prevalence_right','support_n_pairs'])
