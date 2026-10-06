"""Read-only corpus inspection; writes new planning evidence, not marker rules."""
from pathlib import Path
import hashlib
import json
import re
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = Path(__file__).resolve().parent
SOURCE = ROOT / 'result/ETHICS/reports/Religion_Persona/_combined/all_models_results_full.csv'
MARKERS = ROOT / 'result/ETHICS/reports/Religion_Persona/linguistic_markers_religion_candidate_20260824_v1_1_10/response_level_markers.csv.gz'
SEED = 18222008
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
columns = ['model','persona_folder','question_id','subset','response_content_full','answer_status']
raw = pd.read_csv(SOURCE,usecols=columns,keep_default_na=False,dtype=str)
raw['source_row'] = np.arange(len(raw))
raw = raw.rename(columns={'model':'model_id','persona_folder':'persona','response_content_full':'final_output'})
header = pd.read_csv(MARKERS,nrows=0).columns
counts = [c for c in header if c.endswith('__count')]
markers = pd.read_csv(MARKERS,usecols=['model_id','persona','question_id','subset','text_status','text_available','word_count']+counts,low_memory=False)
keys = ['model_id','persona','question_id','subset']
data = raw.merge(markers,on=keys,validate='one_to_one')
text = data.final_output
data['empty_output'] = text.str.strip().eq('')
data['bare_binary_label'] = text.str.fullmatch(r'\s*[01]\s*')
data['think_tag'] = text.str.contains(r'</?think\b',case=False,regex=True)
data['channel_marker'] = text.str.contains(r'<\|(?:start|channel|im_start|im_sep|message)\|>',regex=True)
assert not data.think_tag.any() and not data.channel_marker.any(), 'Separate technical channels before any text export'
data['analysis_heading'] = text.str.contains(r'(?im)^\s*(?:analysis|reasoning)\s*:')
religion_cols = [c for c in counts if c.startswith('religion_')]
data['religion_hit_total'] = data[religion_cols].sum(axis=1,min_count=1)
data['bare_label_marked_available'] = data.bare_binary_label & data.text_available
data['available_without_letters'] = data.text_available & ~text.str.contains(r'[A-Za-z]')
inventory = data.groupby(['model_id','persona','subset','answer_status']).agg(
    total=('source_row','size'),empty_output=('empty_output','sum'),bare_binary_label=('bare_binary_label','sum'),
    bare_label_marked_available=('bare_label_marked_available','sum'),text_available=('text_available','sum'),
    available_without_letters=('available_without_letters','sum'),analysis_heading=('analysis_heading','sum')).reset_index()
inventory.to_csv(OUTPUT/'corpus_inventory.csv',index=False)
inventory.groupby('model_id').sum(numeric_only=True).reset_index().to_csv(OUTPUT/'model_inventory.csv',index=False)
identities = ['islam','christian','jewish','atheist']
subsets = ['commonsense','deontology','justice','virtue']
selected = []
rng = np.random.default_rng(SEED)
for mi, model in enumerate(sorted(data.model_id.unique())):
    pool = data.loc[data.model_id.eq(model)]
    used = set()
    def choose(frame, purpose):
        frame = frame.loc[~frame.source_row.isin(used)].sort_values('source_row')
        assert len(frame),(model,purpose)
        row = frame.iloc[int(rng.integers(len(frame)))].copy()
        used.add(row.source_row)
        row['selection_purpose'] = purpose
        selected.append(row)
    prose = pool.loc[pool.word_count.ge(20) & ~pool.analysis_heading]
    for j, identity in enumerate(identities):
        choose(prose.loc[prose.persona.eq(identity+'_explicit') & prose.subset.eq(subsets[(mi+j)%4])], 'explicit_identity_'+identity)
    for j in range(3):
        identity = identities[(mi+j)%4]
        choose(prose.loc[prose.persona.eq(identity+f'_implicit_{j+1}') & prose.subset.eq(subsets[(mi+j+1)%4])],f'implicit_{j+1}')
    choose(prose.loc[prose.persona.eq('neutral')], 'neutral')
    choose(prose.loc[prose.answer_status.eq('abstain')], 'visible_abstain')
    choose(prose.loc[prose.religion_hit_total.eq(0)], 'no_religion_hit_prose_challenge')
    short = pool.loc[pool.bare_binary_label]
    if not len(short):
        short = pool.loc[pool.word_count.le(pool.word_count.quantile(.1)) & pool.word_count.gt(0) & ~pool.analysis_heading]
    choose(short, 'bare_label_or_short_output')
    long = prose.loc[prose.word_count.ge(prose.word_count.quantile(.9))]
    choose(long, 'long_output')

sample = pd.DataFrame(selected).reset_index(drop=True)
sample.insert(0,'sample_id',[f'R{i+1:03d}' for i in range(len(sample))])
sample['final_sha256'] = sample.final_output.map(lambda s:hashlib.sha256(s.encode('utf-8')).hexdigest())
sample.drop(columns=['final_output']).to_csv(OUTPUT/'sample_index.csv',index=False)
export_cols = ['sample_id','model_id','persona','subset','question_id','source_row','answer_status','selection_purpose','word_count','text_status','text_available','final_sha256','final_output']
sample[export_cols+counts].to_json(OUTPUT/'full_final_samples.json',orient='records',force_ascii=False,indent=2)
for model, rows in sample.groupby('model_id',sort=False):
    parts = []
    for row in rows.itertuples():
        found = {c.removesuffix('__count'):int(getattr(row,c)) for c in counts if pd.notna(getattr(row,c)) and getattr(row,c)>0}
        parts.append(f'===== {row.sample_id} | {row.persona} | {row.subset} | {row.question_id} | source_row={row.source_row} | {row.selection_purpose} =====\n{row.final_output}\nCURRENT COUNTS: {json.dumps(found)}\n')
    (OUTPUT/(model+'_full.txt')).write_text('\n'.join(parts),encoding='utf-8')
manifest = dict(source=str(SOURCE),source_sha256=sha(SOURCE),marker_source=str(MARKERS),marker_sha256=sha(MARKERS),
    rows=len(data),sample_responses=len(sample),seed=SEED,read_columns=columns,ground_truth_read=False,
    thinking_content_exported=False,think_tags=int(data.think_tag.sum()),channel_markers=int(data.channel_marker.sum()),
    full_response_read_status='pending_assistant_readback',sampling='purposeful stratified discovery; not a precision/recall estimate; 12 per model',
    sample_per_model=sample.groupby('model_id').size().to_dict(),bare_binary_labels=int(data.bare_binary_label.sum()),
    bare_labels_marked_available=int(data.bare_label_marked_available.sum()),script_sha256=sha(Path(__file__)))
(OUTPUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps(manifest,indent=2))
