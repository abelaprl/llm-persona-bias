"""Paired availability diagnostics and sensitivity direction comparisons."""
import numpy as np
import pandas as pd
from scipy import stats
from src.benchmarks.ethics.religion_lm_v3_statistics import build_pairs,holm_adjust

def availability_diagnostics(rows):
    records = []
    for model,group in rows.groupby('model_id'):
        for persona in sorted(set(group.persona)-{'neutral'}):
            pair = build_pairs(group,persona,'neutral')
            for metric in ('explanation_available','final_output_available'):
                a,b = pair[metric+'_left'].to_numpy(bool),pair[metric+'_right'].to_numpy(bool)
                yes_no,no_yes = int((a&~b).sum()),int((~a&b).sum())
                discordant = yes_no+no_yes
                records.append(dict(model_id=model,persona=persona,reference='neutral',subset='ethics_all',metric=metric,n_pairs=len(pair),
                    left_available=int(a.sum()),right_available=int(b.sum()),left_only=yes_no,right_only=no_yes,
                    effect_pp=float((a.astype(float)-b).mean()*100) if len(a) else None,
                    p_raw=stats.binomtest(yes_no,discordant).pvalue if discordant else 1,
                    test='exact paired McNemar',interpretation='availability difference, not detector error or equal marker prevalence'))
    result = pd.DataFrame(records)
    result['p_holm'] = result.groupby(['model_id','metric']).p_raw.transform(lambda p:holm_adjust(p.to_numpy()))
    return result

def sensitivity_directions(contrasts):
    keys = ['model_id','contrast_kind','left_persona','right_persona','marker_id','metric']
    base = contrasts.loc[contrasts.population.eq('text_all'),keys+['effect','ci_low','ci_high','n_pairs','p_holm']]
    sensitivity = contrasts.loc[contrasts.population.isin(['both_committed_text','equal_subset_text','no_quality_flag_text'])]
    joined = sensitivity.merge(base,on=keys,suffixes=('','_primary'),validate='many_to_one')
    joined['direction_reversed'] = joined.effect*joined.effect_primary<0
    joined['paired_n_difference'] = joined.n_pairs-joined.n_pairs_primary
    joined['effect_difference'] = joined.effect-joined.effect_primary
    return joined
