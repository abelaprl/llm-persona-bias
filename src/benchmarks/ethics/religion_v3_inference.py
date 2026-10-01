"""Conservative inference for sparse or degenerate paired marker outcomes."""
from functools import lru_cache
import numpy as np
from scipy import stats
from src.benchmarks.ethics.religion_lm_v3_statistics import holm_adjust


@lru_cache(maxsize=None)
def exact_mcnemar(left_only,right_only):
    n=left_only+right_only
    if not n:return 1.,False
    value=float(stats.binomtest(min(left_only,right_only),n,p=.5).pvalue)
    floor=np.finfo(float).tiny
    return max(value,floor),value<floor


def holm_planned(values):
    values=np.asarray(values,float)
    finite=np.isfinite(values)
    if not np.all((values[finite]>=0)&(values[finite]<=1)):raise ValueError('Invalid p-value')
    adjusted=holm_adjust(np.where(finite,values,1.))
    adjusted[~finite]=np.nan
    return adjusted


def repair_inference(frame):
    result=frame.copy()
    for column in ('ci_low','ci_high','p_raw','p_holm'):
        if column+'_bootstrap_shard' not in result:result[column+'_bootstrap_shard']=result[column]
        result[column]=result[column].astype(float)
    result['p_holm']=np.nan
    eligible=result.inference_scope_status.eq('available')&result.n_pairs.ge(2)
    presence=result.metric.str.startswith('prevalence')
    constant=np.where(presence,result.constant_prevalence_difference,result.constant_density_difference)
    result['bootstrap_degenerate']=constant&eligible
    result['ci_status']=np.where(result.ci_low.notna()&result.ci_high.notna(),'available','insufficient_support')
    result.loc[result.bootstrap_degenerate,['ci_low','ci_high']]=np.nan
    result.loc[result.bootstrap_degenerate,'ci_status']='degenerate_question_bootstrap'
    informative=np.where(presence,result.informative_prevalence_pairs,result.informative_density_pairs)
    sparse=(informative<6)&eligible&~result.bootstrap_degenerate
    result.loc[sparse,['ci_low','ci_high']]=np.nan
    result.loc[sparse,'ci_status']='fewer_than_six_informative_pairs'
    result.loc[~eligible,['ci_low','ci_high','p_raw']]=np.nan
    result.loc[~eligible,'ci_status']='insufficient_support'
    result['p_method']='not_estimated_insufficient_support'
    result['p_underflow_floor']=False
    exact=presence&~result.population.eq('equal_subset_text')&eligible
    values=[exact_mcnemar(int(a),int(b)) for a,b in result.loc[exact,['discordant_left_only','discordant_right_only']].itertuples(index=False,name=None)]
    if values:
        result.loc[exact,'p_raw']=np.array([p for p,_ in values],dtype=float)
        result.loc[exact,'p_underflow_floor']=np.array([flag for _,flag in values],dtype=bool)
        result.loc[exact,'p_method']='exact_paired_McNemar'
    for name,mask in [('density',~presence&eligible),('prevalence',presence&~exact&eligible)]:
        result.loc[mask,'p_raw']=result.loc[mask,name+'_sign_flip_p']
        result.loc[mask,'p_method']=result.loc[mask,name+'_sign_flip_method']
    return result
