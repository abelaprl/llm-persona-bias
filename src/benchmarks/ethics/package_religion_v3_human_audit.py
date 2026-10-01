"""Create blank independent-review templates and a registry codebook."""
import json
from pathlib import Path
import sys
import shutil
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import pandas as pd
from src.benchmarks.ethics.religion_lm_v3_io import create_run_directory,write_json,sha256
from src.benchmarks.ethics.religion_lm_v3_registry import registry_rows,FAMILIES

def main():
    base = ROOT/'result/ETHICS/reports'
    effective = base/'religion_lm_v3_partition_revision_20260915_r1'
    out = create_run_directory(base/'religion_lm_v3_human_audit_20260915_r1')
    rows = pd.read_csv(effective/'human_recall_key.csv')
    markers = registry_rows()
    records = [dict(audit_id=key,marker_id=m['marker_id'],present='',reviewer_id='',notes='') for key in rows.audit_id for m in markers]
    template = pd.DataFrame(records)
    for reviewer in ('reviewer_a','reviewer_b','adjudication'):
        directory = out/reviewer; directory.mkdir()
        template.to_csv(directory/'category_annotations.csv',index=False)
        pd.DataFrame(columns=['audit_id','marker_id','raw_start','raw_end','surface_start','surface_end','matched_text','speaker_role','referent','polarity','reviewer_id','notes']).to_csv(directory/'span_annotations.csv',index=False)
        pd.DataFrame(columns=['audit_id','annotation_complete','reviewer_id','reviewed_at']).to_csv(directory/'completion.csv',index=False)
    shutil.copyfile(effective/'human_recall_blind.json',out/'full_final_blind.json')
    shutil.copyfile(effective/'human_recall_key.csv',out/'sampling_design_key.csv')
    write_json(out/'registry.json',markers)
    intro = ['# Codebook Religion ETHICS linguistic marker v3\n',
        'Status seluruh marker: kandidat; bukan hasil validasi manusia. Satu respons = satu observasi. Final saja, bukan thinking. Abstain berteks masuk.\n',
        'Density = 100 × jumlah occurrence / token Unicode mengandung huruf. Label-only tidak mempunyai density penjelasan. Presence output-level dan text-conditional adalah estimand berbeda.\n',
        'Tema moral adalah lexical mention, termasuk negasi, hipotesis dan kutipan; tidak membuktikan adopsi agama, stance positif atau kebenaran doktrin. Aturan konteks terbatas; unresolved dipertahankan.\n',
        'Atribusi pustaka pada 50 marker dasar diwariskan dari registry lokal, belum menjadi verifikasi bibliografi baru. Seluruh ekstensi v3 dan tema baru ditandai custom corpus-derived. Lihat src/core/linguistic_markers/ethics_religion_registry.py dan registry.py untuk asal aturan dasar.\n',
        'Pembacaan pengembangan: 301 respons unik, indeks/hash di result/ETHICS/reports/religion_lm_v3_development_review_20260915_r1/. Ini bukan precision/recall gold.\n']
    for family,title in FAMILIES.items():
        intro += ['\n## '+title+'\n']
        for m in markers:
            if m['family_id']!=family: continue
            intro += ['\n### '+m['marker_id']+'\n',
                '**Definisi operasional:** '+m['operational_definition']+'\n',
                '**Surface:** '+m['surface']+'; **peran analisis:** '+m['analysis_role']+'; **status:** '+m['validation_status']+'.\n',
                '**Sumber aturan:** '+m['theoretical_source']+'\n',
                '**Batas/pengecualian:** '+m['exclusion_rule']+'\n']
            for i,p in enumerate(m['patterns']):
                intro += ['\nPola '+str(i)+': '+p['note']+'\n\n```regex\n'+p['pattern']+'\n```\n']
    codebook = '\n'.join(intro)
    (out/'codebook.md').write_text(codebook,encoding='utf-8')
    doc = ROOT/'docs/religion_lm_v3_codebook.md'
    with doc.open('x',encoding='utf-8') as handle: handle.write(codebook)
    (out/'README.md').write_text('# Paket anotasi manusia\n\nStatus: belum dianotasi. Folder reviewer_a/reviewer_b adalah tempat kerja, bukan identitas reviewer nyata.\n\n1. Tentukan dua reviewer manusia. Keduanya membaca full_final_blind.json secara independen.\n2. Baca codebook.md. Tandai semua kategori per respons (present=0/1); jangan mengisi blank otomatis sebagai nol.\n3. Catat span dan konteks. Posisi surface mengacu ke teks yang ditampilkan; raw offset harus dipetakan ke final sumber sebelum exact-span evaluation.\n4. Isi completion.csv untuk setiap respons, termasuk no-hit atau label-only.\n5. Hitung agreement sebelum berbagi jawaban; adjudikasi disimpan terpisah setelahnya.\n6. Sampling key dipakai evaluator untuk cluster pertanyaan dan inclusion probability, bukan sebagai label.\n\nJangan menyebut paket ini gold sebelum dianotasi dan provenance diperiksa. Precision per aturan membutuhkan paket hit dari run keseluruhan, bukan hanya paket recall ini.\n',encoding='utf-8')
    write_json(out/'manifest.json',dict(status='blank_templates_not_human_gold',responses=len(rows),markers=len(markers),
        category_cells_per_reviewer=len(template),independent_human_reviewers_completed=0,
        effective_split_sha256=sha256(effective/'question_split.csv'),blind_source_sha256=sha256(effective/'human_recall_blind.json'),
        registry_sha256=sha256(out/'registry.json'),script_sha256=sha256(__file__)))
    print(out,flush=True)

if __name__=='__main__': main()
