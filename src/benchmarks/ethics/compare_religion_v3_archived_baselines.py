"""Check whether current presentation and archived marker baseline share counts."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import pandas as pd
from collections import Counter
from src.benchmarks.ethics.religion_lm_v3_io import KEYS,create_run_directory,write_json,build_source_manifest


def main():
    base=ROOT/'result/ETHICS/reports'
    source=base/'Religion_Persona/linguistic_markers_religion_candidate_20260824_v1_1_10/response_level_markers.csv.gz'
    current=base/'Religion_Persona/linguistic_markers/analysis/religion_ethics_linguistic_markers_v2_0_2_candidate_20260827/dataset/ethics_religion_marker_analysis_response_level.csv.gz'
    out=create_run_directory(base/'religion_lm_v3_baseline_identity_20260916_r1')
    usecols=lambda c:c in KEYS+['text_available','word_count'] or c.endswith('__count')
    old=pd.read_csv(source,usecols=usecols).set_index(KEYS)
    counts=Counter();n=0
    for chunk in pd.read_csv(current,usecols=usecols,chunksize=5000):
        chunk=chunk.set_index(KEYS)
        previous=old.loc[chunk.index];n+=len(chunk)
        for column in previous:
            mismatch=~(chunk[column].eq(previous[column])|(chunk[column].isna()&previous[column].isna()))
            counts[column]+=int(mismatch.sum())
    if n!=len(old):raise ValueError('Baseline universes differ')
    pd.DataFrame([dict(column=k,mismatched_responses=v) for k,v in counts.items()]).to_csv(out/'column_comparison.csv',index=False)
    result=dict(rows=n,compared_columns=len(counts),mismatched_cells=sum(counts.values()),
        sources=build_source_manifest([source,current,Path(__file__)]),ground_truth_used=False,
        scope='raw marker counts and surface denominator; not scoring or inference equivalence')
    write_json(out/'manifest.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='sources'}),flush=True)


if __name__=='__main__':main()
