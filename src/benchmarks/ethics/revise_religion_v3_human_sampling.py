"""Replace correlated recall draws; preserve all superseded packages."""
import json
from pathlib import Path
import shutil
import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import pandas as pd
from src.benchmarks.ethics.religion_lm_v3_io import SOURCE_COLUMNS,KEYS,create_run_directory,write_json,sha256
from src.benchmarks.ethics.religion_v3_sampling import sample_recall,ALLOCATION
from src.benchmarks.ethics.religion_lm_v3_surface import prepare_final_surface
from src.benchmarks.ethics.religion_lm_v3_registry import registry_rows

def main():
    base = ROOT/'result/benchmarks/ethics/reports'
    partition = base/'religion_lm_v3_partition_revision_20260915_r1'
    previous = base/'religion_lm_v3_human_audit_20260915_r1'
    output = create_run_directory(base/'religion_lm_v3_human_audit_20260916_r2')
    source = base/'Religion_Persona/_combined/all_models_results_full.csv'
    rows = pd.read_csv(source,usecols=SOURCE_COLUMNS,dtype=str,keep_default_na=False).rename(columns={'model':'model_id','persona_folder':'persona','response_content_full':'final_output'})
    rows['response_key'] = [json.dumps(list(k),separators=(',',':')) for k in rows[KEYS].itertuples(index=False,name=None)]
    split = pd.read_csv(partition/'question_split.csv')
    selected = sample_recall(rows,split)
    selected.drop(columns='final_output').to_csv(output/'sampling_design_key.csv',index=False)
    blind = []
    for row in selected.itertuples(index=False):
        surface = prepare_final_surface(row.final_output)
        # Source offsets are packaged explicitly; hidden blocks never enter displayed text.
        blind.append(dict(audit_id=row.audit_id,final_output=surface.visible_text,
            raw_final_sha256=__import__('hashlib').sha256(row.final_output.encode('utf-8')).hexdigest(),
            visible_to_raw_offsets=surface.maps['visible']))
    write_json(output/'full_final_blind.json',blind)
    markers = registry_rows()
    write_json(output/'registry.json',markers)
    template = pd.DataFrame([dict(audit_id=key,marker_id=m['marker_id'],present='',reviewer_id='',notes='') for key in selected.audit_id for m in markers])
    for reviewer in ('reviewer_a','reviewer_b','adjudication'):
        directory = output/reviewer; directory.mkdir()
        template.to_csv(directory/'category_annotations.csv',index=False)
        for filename in ('span_annotations.csv','completion.csv'):
            shutil.copyfile(previous/reviewer/filename,directory/filename)
    shutil.copyfile(previous/'codebook.md',output/'codebook.md')
    shutil.copyfile(partition/'question_split.csv',output/'question_split.csv')
    (output/'README.md').write_text('# Paket recall manusia efektif\n\nMenggantikan paket 20260915: seed kelompok yang sama sebelumnya menghasilkan 1.020 respons dari hanya 10 pertanyaan dan nol justice. Paket ini memakai stream RNG berbeda per model/persona/subset; total tetap 10 respons per model/persona.\n\nKuota per cell: commonsense=3, deontology=3, justice=2, virtue=2. Gunakan inverse inclusion probability pada sampling_design_key.csv. Split pertanyaan tidak diubah.\n\nDua reviewer mengisi file secara independen sebelum adjudikasi. Blank bukan label nol. Baca seluruh final, termasuk no-hit/abstain. visible_to_raw_offsets memetakan karakter tampilan ke offset final asli untuk anotasi span; raw_final_sha256 mengikat sumber. Jangan menyebut paket ini gold sebelum anotasi lengkap.\n',encoding='utf-8')
    write_json(output/'manifest.json',dict(status='blank_independent_stratified_templates_not_gold',responses=len(selected),
        distinct_question_clusters=selected[['subset','question_id']].drop_duplicates().shape[0],
        subset_counts=selected.subset.value_counts().to_dict(),allocation_per_model_persona=ALLOCATION,
        seed=18222008,markers=len(markers),human_reviewers_completed=0,
        supersedes=[str(previous),str(partition/'human_recall_blind.json')],
        question_split_unchanged=True,question_split_sha256=sha256(partition/'question_split.csv'),
        source_sha256=sha256(source),registry_sha256=sha256(output/'registry.json'),
        sampling_script_sha256=sha256(ROOT/'src/benchmarks/ethics/religion_v3_sampling.py')))
    print('Revised recall package: '+str(len(selected))+' responses; '+str(selected.question_id.nunique())+' questions',flush=True)

if __name__=='__main__': main()
