"""Replay a completed contrast with one BLAS thread and compare every outcome."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
from pathlib import Path
import hashlib
import sys
import time
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from src.benchmarks.ethics.religion_lm_v3_statistics import build_pairs,bootstrap_effects
from src.benchmarks.ethics.religion_lm_v3_registry import get_registry
from src.benchmarks.ethics.religion_lm_v3_io import KEYS,create_run_directory,write_json,build_source_manifest


def main():
    base=ROOT/'result/benchmarks/ethics/reports'
    shard=base/'religion_lm_v3_candidate_20260915_r2_shard_b'
    out=create_run_directory(base/'religion_lm_v3_thread_check_20260916_r1')
    markers=[m.marker_id for m in get_registry()]
    suffixes=[('prevalence_text','__present',100),('density','__density',1)]
    columns=[(m,metric,suffix,scale) for metric,suffix,scale in suffixes for m in markers]
    usecols=KEYS+['explanation_available']+[m+s for m,_,s,_ in columns]
    frame=pd.read_csv(shard/'response_level/markers.csv.gz',usecols=usecols)
    model='qwen_qwen3-next-80b-a3b-thinking';left='islam_explicit';right='neutral'
    pair=build_pairs(frame.loc[frame.model_id.eq(model)],left,right)
    selected=pair.loc[pair.explanation_available_left&pair.explanation_available_right]
    differences=np.column_stack([(selected[m+s+'_left'].to_numpy(float)-selected[m+s+'_right'].to_numpy(float))*scale for m,_,s,scale in columns])
    seed=int(hashlib.sha256(f'18222008|{model}|vs_neutral|{left}|{right}|text_all'.encode()).hexdigest()[:8],16)
    start=time.perf_counter();result=bootstrap_effects(differences,selected.subset.to_numpy(),seed,10000);elapsed=time.perf_counter()-start
    old=pd.read_csv(shard/'statistics/paired_contrasts.csv.gz')
    old=old.loc[old.model_id.eq(model)&old.left_persona.eq(left)&old.right_persona.eq(right)&old.population.eq('text_all')].set_index(['marker_id','metric'])
    index=pd.MultiIndex.from_tuples([(m,metric) for m,metric,_,_ in columns])
    deltas={}
    for key,values in result.items():
        expected=old.loc[index,key].to_numpy(float)
        deltas[key]=float(np.max(np.abs(values-expected)))
        assert np.allclose(values,expected,rtol=0,atol=1e-12,equal_nan=True),(key,deltas[key])
    report=dict(status='single_thread_replay_matches_completed_contrast',n_pairs=len(selected),outcomes=len(columns),
        bootstrap_resamples=10000,bootstrap_seconds=elapsed,max_absolute_differences=deltas,
        environment={k:os.environ[k] for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},
        source_hashes=build_source_manifest([shard/'response_level/markers.csv.gz',shard/'statistics/paired_contrasts.csv.gz',Path(__file__)]))
    write_json(out/'manifest.json',report)
    print({k:v for k,v in report.items() if k!='source_hashes'},flush=True)


if __name__=='__main__':main()
