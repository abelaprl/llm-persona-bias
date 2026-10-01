"""Evaluate human category annotations; no fabricated gold or answer scoring."""
import numpy as np
import pandas as pd

KEYS = ['audit_id','marker_id']

def validate_annotations(frame):
    if frame.empty or frame.duplicated(KEYS).any() or not frame.present.isin([0,1]).all():
        raise ValueError('Complete, unique binary annotations required, including explicit negatives')

def reviewer_agreement(first,second):
    for frame in (first,second):
        validate_annotations(frame)
        if frame.reviewer_id.nunique()!=1:
            raise ValueError('One identified reviewer per independent file is required')
    if first.reviewer_id.iloc[0]==second.reviewer_id.iloc[0]:
        raise ValueError('Independent reviewer IDs required')
    pair = first.merge(second,on=KEYS,suffixes=('_a','_b'),validate='one_to_one',how='outer')
    if pair[['present_a','present_b']].isna().any().any():
        raise ValueError('Reviewer universes differ')
    observed = float(pair.present_a.eq(pair.present_b).mean())
    a,b = pair.present_a.mean(),pair.present_b.mean()
    expected = a*b+(1-a)*(1-b)
    return dict(n=len(pair),agreement=observed,cohen_kappa=(observed-expected)/(1-expected) if expected<1 else None,
        unit='response_marker_presence_before_adjudication',independent_reviewers=[first.reviewer_id.iloc[0],second.reviewer_id.iloc[0]])

def evaluate_against_gold(predictions,gold,sampling_design,seed=18222008,resamples=10000):
    for frame in (predictions,gold): validate_annotations(frame)
    pair = predictions.merge(gold,on=KEYS,suffixes=('_pred','_gold'),how='outer',validate='one_to_one')
    if pair[['present_pred','present_gold']].isna().any().any():
        raise ValueError('Prediction/gold universe differs; absence must be explicitly annotated')
    pair = pair.merge(sampling_design,on='audit_id',validate='many_to_one',how='left')
    if pair.inclusion_probability.isna().any() or not pair.inclusion_probability.between(0,1,inclusive='right').all():
        raise ValueError('Known positive inclusion probabilities required')
    rng = np.random.default_rng(seed)
    results = []
    for marker,group in pair.groupby('marker_id',sort=True):
        pred,truth = group.present_pred.to_numpy(bool),group.present_gold.to_numpy(bool)
        events = np.column_stack([pred&truth,pred&~truth,~pred&truth]).astype(float)
        weighted = events/group.inclusion_probability.to_numpy()[:,None]
        totals = weighted.sum(axis=0)
        precision = totals[0]/(totals[0]+totals[1]) if totals[0]+totals[1]>0 else None
        recall = totals[0]/(totals[0]+totals[2]) if totals[0]+totals[2]>0 else None
        clusters = pd.DataFrame(weighted,columns=['tp','fp','fn'])
        clusters[['subset','question_id']] = group[['subset','question_id']].reset_index(drop=True)
        values = clusters.groupby(['subset','question_id'])[['tp','fp','fn']].sum().to_numpy()
        draws = rng.multinomial(len(values),np.full(len(values),1/len(values)),size=resamples) @ values
        intervals = {}
        for name,den in [('precision',draws[:,0]+draws[:,1]),('recall',draws[:,0]+draws[:,2])]:
            valid = den>0
            sample = draws[valid,0]/den[valid]
            lo,hi = np.quantile(sample,[.025,.975]) if len(sample) and len(values)>1 else (None,None)
            intervals[name+'_ci_low'],intervals[name+'_ci_high'] = lo,hi
        results.append(dict(marker_id=marker,n=len(group),gold_positive=int(truth.sum()),predicted_positive=int(pred.sum()),
            tp=int(events[:,0].sum()),fp=int(events[:,1].sum()),fn=int(events[:,2].sum()),precision=precision,recall=recall,
            **intervals,status='insufficient_positive_evidence' if not truth.any() else 'evaluated_human_annotations'))
    return dict(markers=results,unit='response_marker_presence',weighting='inverse inclusion probability',
        confidence_interval='question-cluster percentile bootstrap; not a per-rule precision audit',resamples=resamples,seed=seed)

def exact_span_agreement(first,second):
    keys = ['audit_id','marker_id','raw_start','raw_end']
    a,b = set(first[keys].itertuples(index=False,name=None)),set(second[keys].itertuples(index=False,name=None))
    return dict(exact_shared=len(a&b),first_spans=len(a),second_spans=len(b),
        span_f1=2*len(a&b)/(len(a)+len(b)) if a or b else None,unit='exact raw-final offsets')
