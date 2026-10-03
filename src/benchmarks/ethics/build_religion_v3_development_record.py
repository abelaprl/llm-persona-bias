"""Record completed AI discovery readback separately from human validation."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import pandas as pd
from src.benchmarks.ethics.religion_lm_v3_io import create_run_directory,write_json,build_source_manifest
from src.benchmarks.ethics.religion_lm_v3_registry import registry_rows,FAMILIES,THEME_MARKERS,EXTENSIONS,rule_origin
from src.benchmarks.ethics.religion_lm_v3_extract import extract_final

def main():
    base = ROOT/'result/benchmarks/ethics/reports'
    prep = base/'religion_lm_v3_preparation_20260915_r1'
    out = create_run_directory(base/'religion_lm_v3_development_review_20260915_r1')
    files = [(prep/'targeted_full_responses.json','D'),(prep/'open_discovery_round_1.json','O1-'),
        (prep/'open_discovery_round_2.json','O2-'),(base/'religion_lm_v3_partition_revision_20260915_r1/rare_full_responses.json','Q'),
        (base/'religion_lm_v3_theme_discovery_20260915_r1/full_responses.json','T')]
    samples = []
    for path,prefix in files:
        for i,row in enumerate(json.loads(path.read_text(encoding='utf-8'))):
            row['sample_id'] = row.get('readback_id',prefix+f'{i+1:02d}')
            row['source_file'] = str(path)
            samples.append(row)
    records,hit_records = [],[]
    for row in samples:
        result,hits = extract_final(row['final_output'])
        records.append(dict(sample_id=row['sample_id'],response_key=row['response_key'],model_id=row['model_id'],persona=row['persona'],
            subset=row['subset'],question_id=row['question_id'],raw_final_sha256=result['raw_final_sha256'],
            source_file=row['source_file'],reviewer_type='AI_discovery',full_final_read=True,human_audit=False,
            surface_status=result['surface_status'],surface_flags=result['surface_flags']))
        for hit in hits:
            if hit['candidate_status']!='excluded':
                hit_records.append(dict(sample_id=row['sample_id'],response_key=row['response_key'],model_id=row['model_id'],rule_id=hit['rule_id'],marker_id=hit['marker_id']))
    pd.DataFrame(records).to_csv(out/'readback_index.csv',index=False)
    hits = pd.DataFrame(hit_records).drop_duplicates(['response_key','rule_id'])
    hits.to_csv(out/'readback_rule_hits.csv',index=False)
    hits.groupby(['model_id','rule_id']).size().rename('ai_full_responses_read').reset_index().to_csv(out/'readback_by_model_rule.csv',index=False)
    gap_specs = [
        ('G01','D001','ai_identity_statement','function','Meta-response mengandung draft jawaban; kata AI bukan otomatis suara model langsung.','scope_flagged_unresolved','meta_response_candidate'),
        ('G02','D032','response_binary_decline','form','Bentuk opening give a binary/answer this with a binary.','candidate_implemented','response_binary_decline'),
        ('G03','D058','response_binary_decline','form','cannot and will not, provide a binary, answer this as a binary belum semua ditangkap.','open_needs_rule_specific_scan',''),
        ('G04','Q004','religion_named_concept_reference','form','niyyah ditambahkan sebagai rujukan leksikal.','candidate_implemented','religion_named_concept_reference'),
        ('G05','D068','organization_section_header','form','Heading Markdown inline; variasi tanpa bold dan level ## belum diperluas.','candidate_implemented_partial','organization_section_header'),
        ('G06','O1-24','ai_identity_statement','form','digital AI dan large language model di luar ekstensi neutral/helpful.','open_needs_rule_specific_scan',''),
        ('G07','O1-33','religion_named_concept_reference','category','maslahah/amanah/darurat dan konsep lain butuh inventori bentuk serta audit masing-masing.','open_needs_rule_specific_scan',''),
        ('G08','O2-37','religion_named_concept_reference','category','anavah/bitachon/emunah dan transliterasi Yahudi lain belum masuk kamus tertutup.','open_needs_rule_specific_scan',''),
        ('G09','T043','theme_boundaries','context','Boundaries of the rules bukan bukti batas pribadi; tema v3 hanya lexical mention.','intentional_semantic_exclusion','theme_boundaries'),
        ('G10','O2-39','theme_consent_autonomy','category','Consent/autonomy dan tema moral tidak otomatis religion reasoning.','candidate_implemented_lexical_only','theme_consent_autonomy'),
        ('G11','D039','organization_section_header','quality','Repetisi simbol tetap masuk utama dengan sensitivity exclusion terpisah.','quality_flagged','symbol_repetition_candidate'),
        ('G12','O2-60','stance_booster','form','unequivocally belum diperluas; jangan menyebut booster lengkap.','open_needs_rule_specific_scan',''),
        ('G13','O2-57','religion_identity_irrelevance','context','Identitas dalam kutipan/user/scenario serta negasi/adopsi membutuhkan adjudikasi.','open_function_scope',''),
        ('G14','O1-36','religion_moral_authority_framing','function','Kutipan scriptural dan klaim otoritas model berbeda; kebenaran doktrin bukan target detektor.','open_function_scope',''),
        ('G15','O2-10','self_reference_first_person','function','Narasi orang pertama dapat memainkan tokoh; bukan pengalaman pribadi model.','open_function_scope',''),
        ('G16','T026','theme_consent_autonomy','category','Human rights/dignity bukan sinonim otomatis consent/autonomy; perlu kategori terpisah berbukti.','open_needs_rule_specific_scan',''),
    ]
    by_id = {r['sample_id']:r for r in samples}
    gaps = []
    for gap_id,sample,marker,kind,note,decision,rule in gap_specs:
        r = by_id[sample]
        gaps.append(dict(gap_id=gap_id,sample_id=sample,response_key=r['response_key'],round_id=sample.split('-')[0],
            raw_span_start=None,raw_span_end=None,gap_type=kind,proposed_marker_id=marker,proposed_function='unresolved',
            proposed_referent='unresolved',decision=decision,decision_reason=note,reviewer_type='AI_discovery',related_rule_id=rule,
            span_status='response_level_issue_full_text_linked'))
    pd.DataFrame(gaps).to_csv(out/'discovery_gaps.csv',index=False)
    matrix = []
    for marker in registry_rows():
        evidence = hits.loc[hits.marker_id.eq(marker['marker_id'])]
        related = [g for g in gaps if g['proposed_marker_id']==marker['marker_id']]
        matrix.append(dict(family_id=marker['family_id'],marker_id=marker['marker_id'],definition_version=marker['version'],
            source_type='custom corpus-derived or inherited candidate; see registry source',pattern_count=len(marker['patterns']),
            discovery_response_count=evidence.response_key.nunique(),positive_example_ids='|'.join(evidence.sample_id.drop_duplicates()),
            negative_example_ids='synthetic negative fixtures in tests; not gold',ambiguous_example_ids='|'.join(g['sample_id'] for g in related),
            function_scope_status='lexical only' if marker['marker_id'].startswith('theme_') else 'partial; unresolved retained',
            unresolved_gap_count=sum(g['decision'].startswith('open') for g in related),validation_status='candidate_pending_human_review'))
    pd.DataFrame(matrix).to_csv(out/'completeness_matrix.csv',index=False)
    pd.DataFrame([dict(round_id='O1',new_responses=60,status='read_in_full',new_relevant_gaps=True,saturation=False),
        dict(round_id='O2',new_responses=60,status='read_in_full',new_relevant_gaps=True,saturation=False)]).to_csv(out/'discovery_rounds.csv',index=False)
    write_json(out/'manifest.json',dict(status='AI_discovery_completed_for_recorded_samples_not_human_validation',read_events=len(records),
        unique_responses=pd.DataFrame(records).response_key.nunique(),registry_markers=len(matrix),gap_count=len(gaps),
        saturation_claim=False,precision_or_recall_claim=False,source_hashes=build_source_manifest([p for p,_ in files]+[Path(__file__)])))
    print(out,flush=True)

if __name__=='__main__': main()
