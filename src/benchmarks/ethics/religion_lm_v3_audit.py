"""Question-disjoint audit design and fail-closed human validation gates."""
import hashlib
import math
import pandas as pd


def make_question_split(keys, development_questions, seed):
    rows = keys[['subset','question_id']].drop_duplicates().sort_values(['subset','question_id']).copy()
    assignments = []
    for subset, group in rows.groupby('subset', sort=True):
        ordered = sorted(group.question_id, key=lambda q: hashlib.sha256(f'{seed}|{subset}|{q}'.encode()).hexdigest())
        forced = [q for q in ordered if (subset,q) in development_questions]
        remaining = [q for q in ordered if (subset,q) not in development_questions]
        n_dev = max(len(forced), round(.6*len(ordered)))
        dev = set(forced + remaining[:n_dev-len(forced)])
        rest = [q for q in remaining if q not in dev]
        calibration = set(rest[:min(round(.2*len(ordered)),len(rest))])
        for q in ordered:
            assignments.append(dict(subset=subset,question_id=q,partition='development' if q in dev else 'calibration' if q in calibration else 'held_out'))
    return pd.DataFrame(assignments).sort_values(['subset','question_id']).reset_index(drop=True)


def promote_rules(report):
    reasons = []
    if report.get('reviewer_type') != 'human' or len(set(report.get('independent_reviewers', []))) < 2:
        reasons.append('two_independent_human_reviewers_required')
    for field, threshold in [('precision',.98),('recall',.95),('precision_ci_low',.95),('recall_ci_low',.90)]:
        value = report.get(field)
        if not isinstance(value,(int,float)) or not math.isfinite(value) or value < threshold:
            reasons.append(field+'_target_not_met')
    required = 100 if report.get('rule_type') == 'semantic' else 50
    total = report.get('total_rule_hits', required)
    required = total if total < 50 else required
    if report.get('precision_samples',0) < required:
        reasons.append('insufficient_precision_samples')
    if report.get('false_positives') != 0:
        reasons.append('precision_audit_not_clean')
    if report.get('held_out_independent') is not True or report.get('per_model_readback_complete') is not True:
        reasons.append('audit_provenance_incomplete')
    return {'approved':not reasons, 'reasons':reasons}


def export_annotation_package(rows, split, per_cell=10, seed=18222008):
    joined = rows.merge(split,on=['subset','question_id'],validate='many_to_one')
    pool = joined.loc[joined.partition.eq('held_out')]
    samples = []
    for _, group in pool.groupby(['model_id','persona'], sort=True):
        n = min(per_cell,len(group))
        selected = group.sample(n=n,random_state=seed).copy()
        selected['inclusion_probability'] = n/len(group)
        samples.append(selected)
    selected = pd.concat(samples).sort_values('response_key').reset_index(drop=True)
    selected['audit_id'] = [f'H{i+1:05d}' for i in range(len(selected))]
    return selected
