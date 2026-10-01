"""Locate already-recorded discovery evidence; these probes are not detectors."""
import hashlib
import json
import re
import pandas as pd
from src.benchmarks.ethics.religion_lm_v3_io import ROOT,KEYS,SOURCE_COLUMNS,write_json
from src.benchmarks.ethics.religion_lm_v3_registry import family_for
from src.benchmarks.ethics.religion_lm_v3_surface import prepare_final_surface

PROBES={
 'G01':r'\b(?:this response|the response|as a helpful ai)\b',
 'G02':r'\bi cannot (?:give a binary|answer this with a binary)\b',
 'G03':r'\b(?:cannot and will not|provide a binary|answer this as a binary)\b',
 'G04':r'\bniyyah\b','G05':r'#{2,4}\s+\*\*[^*\n]+\*\*',
 'G06':r'\b(?:digital ai|large language model)\b','G07':r'\b(?:maslahah|amanah|darurat)\b',
 'G08':r'\b(?:anavah|bitachon|emunah)\b','G09':r'\bboundaries of the rules\b',
 'G10':r'\b(?:consent|autonomy)\b','G11':r'(?:\*\s*){64,}',
 'G12':r'\bunequivocally\b','G13':r'\b(?:jewish|religio\w*)\b',
 'G14':r'\b(?:qur.?an|scripture|hadith)\b','G15':r'\b(?:I|my)\b',
 'G16':r'\b(?:human rights|rights|dignity)\b',
}


def enrich_gap_provenance(output):
    base=ROOT/'result/ETHICS/reports'
    gaps=pd.read_csv(base/'religion_lm_v3_development_review_20260915_r1/discovery_gaps.csv',keep_default_na=False)
    readback=pd.read_csv(base/'religion_lm_v3_development_review_20260915_r1/readback_index.csv')
    gap_keys=set(gaps.response_key)
    keys=gap_keys|set(readback.response_key);texts={}
    source=base/'Religion_Persona/_combined/all_models_results_full.csv'
    for chunk in pd.read_csv(source,usecols=SOURCE_COLUMNS,dtype=str,keep_default_na=False,chunksize=5000):
        chunk=chunk.rename(columns={'model':'model_id','persona_folder':'persona','response_content_full':'final_output'})
        for values,text in zip(chunk[KEYS].itertuples(index=False,name=None),chunk.final_output):
            key=json.dumps(list(values),separators=(',',':'))
            if key in keys:texts[key]=text
    if set(texts)!=keys:raise ValueError('Gap evidence keys missing')
    hashes={key:(hashlib.sha256(text.encode('utf-8')).hexdigest(),
        hashlib.sha256(prepare_final_surface(text).visible_text.encode('utf-8')).hexdigest()) for key,text in texts.items()}
    reconciled=[]
    for row in readback.itertuples(index=False):
        raw_hash,visible_hash=hashes[row.response_key]
        relation='raw_final' if row.raw_final_sha256==raw_hash else ('normalized_visible_final' if row.raw_final_sha256==visible_hash else 'unresolved_source_difference')
        if relation=='unresolved_source_difference':raise ValueError('Readback source mismatch: '+row.sample_id)
        reconciled.append(dict(sample_id=row.sample_id,response_key=row.response_key,source_file=row.source_file,
            source_raw_final_sha256=raw_hash,readback_text_sha256=row.raw_final_sha256,normalized_visible_sha256=visible_hash,
            readback_text_relation=relation,reviewer_type='AI_discovery'))
    pd.DataFrame(reconciled).to_csv(output/'readback_hash_reconciliation.csv',index=False)
    records=[]
    for row in gaps.to_dict('records'):
        text=texts[row['response_key']]
        if re.search(r'</?think\b|<\|(?:start|channel|im_start)\|>',text,re.I):
            raise ValueError('Unexpected hidden/channel block in previously reviewed final evidence')
        matches=list(re.finditer(PROBES[row['gap_id']],text,re.I))
        if row['gap_id']=='G16':row['proposed_marker_id']='theme_rights_dignity_candidate'
        row['proposed_family_id']=family_for(row['proposed_marker_id'])
        row['raw_final_sha256']=hashlib.sha256(text.encode('utf-8')).hexdigest()
        row['evidence_role']='anchor locating a recorded gap; not proof of semantic interpretation or an implemented rule'
        if not matches:
            records.append(dict(row,span_status='anchor_not_located_requires_record_review',evidence_text=''))
        for match in matches:
            records.append(dict(row,raw_span_start=match.start(),raw_span_end=match.end(),
                span_status='raw_final_anchor_located',evidence_text=match.group()))
    result=pd.DataFrame(records)
    result.to_csv(output/'discovery_gaps_with_spans.csv',index=False)
    write_json(output/'discovery_gap_full_responses.json',[dict(response_key=k,final_output=v,
        raw_final_sha256=hashlib.sha256(v.encode('utf-8')).hexdigest()) for k,v in sorted(texts.items()) if k in gap_keys])
    return result.loc[result.span_status.eq('anchor_not_located_requires_record_review'),'gap_id'].tolist()
