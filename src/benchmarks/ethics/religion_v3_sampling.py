"""Independent per-cell RNG streams and explicit four-subset recall strata."""
import hashlib
import pandas as pd

ALLOCATION = {'commonsense':3,'deontology':3,'justice':2,'virtue':2}

def sample_recall(rows,split,seed=18222008):
    joined = rows.merge(split,on=['subset','question_id'],validate='many_to_one')
    pool = joined.loc[joined.partition.eq('held_out')]
    selected = []
    for (model,persona),cell in pool.groupby(['model_id','persona'],sort=True):
        for subset,n in ALLOCATION.items():
            group = cell.loc[cell.subset.eq(subset)].sort_values('response_key')
            if len(group)<n: raise ValueError('Insufficient held-out stratum for prespecified allocation')
            stream = int(hashlib.sha256(f'{seed}|{model}|{persona}|{subset}'.encode()).hexdigest()[:8],16)
            sample = group.sample(n=n,random_state=stream).copy()
            sample['inclusion_probability'] = n/len(group)
            sample['sampling_stratum'] = model+'|'+persona+'|'+subset
            sample['stream_seed'] = stream
            selected.append(sample)
    result = pd.concat(selected).sort_values('response_key').reset_index(drop=True)
    result['audit_id'] = [f'V3H{i+1:05d}' for i in range(len(result))]
    return result
