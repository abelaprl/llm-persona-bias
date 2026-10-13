"""Additive Religion/Regional paired galleries using a single plotting grammar."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import nbformat
from nbclient import NotebookClient
import pandas as pd

from .gender_lm_reference_data import ROOT, digest
from .gender_lm_candidate_common import save_panel
from .identity_lm_unified_bundle import CONTRACT, CAUTION, make_notebook, write_gallery
from . import identity_lm_unified_plots as plots


def adapter(dimension):
    if dimension not in {'religion','regional'}:
        raise ValueError('Unknown persona dimension')
    return importlib.import_module('.'+dimension+'_lm_unified_data',package=__package__)


def default_output(dimension):
    return ROOT/'notebooks'/dimension/'linguistic marker/candidate/unified_v1'


def pairing_evidence(dimension):
    """Read a concrete counterexample without reading ground-truth labels."""
    model='meta-llama_llama-3.1-8b-instruct'
    identity='islam' if dimension=='religion' else 'north_american'
    folder='Religion_Persona' if dimension=='religion' else 'regional_cultural_persona'
    paths=[ROOT/'result/benchmarks/gsm8k/runs/main/Gender_Persona'/model/'neutral/gsm8k_00511.json']
    paths.extend(ROOT/'result/benchmarks/gsm8k/runs/main'/folder/model/(identity+'_'+c)/'gsm8k_00511.json'
                 for c in ['explicit','implicit_1'])
    records=[]
    for condition,path in zip(['N','E','I1'],paths):
        question=json.loads(path.read_text(encoding='utf-8'))['input']['question']
        records.append(dict(condition=condition,model_id=model,question_id='gsm8k_00511',
                            source_file=str(path),question=question,
                            question_sha256=hashlib.sha256(question.encode('utf-8')).hexdigest()))
    if records[0]['question']==records[1]['question'] or records[1]['question']!=records[2]['question']:
        raise ValueError('Pairing audit counterexample changed; review sources before building')
    return records


def withhold_effects(source):
    """Retain frozen numbers as provenance, never show unverified pairs as effects."""
    rows=source.copy(deep=True)
    for name in ['estimate','lower','upper']:
        rows['source_'+name]=rows[name]
        rows[name]=float('nan')
    if 'corrected_p' in rows:
        rows['source_corrected_p']=rows.corrected_p
        rows['corrected_p']=float('nan')
    rows['display_status']='withheld_question_pairing_not_validated'
    rows['pairing_validity']='unverified_for_paired_interpretation'
    rows['n_pairs_semantics']='key_matched_not_question_validated'
    return rows


def panel_data(frame, data, scenario, markers=None):
    selected=frame.loc[frame.scenario.eq(scenario)&frame.identity.isin(data['identities'])].copy()
    if 'row_id' not in selected:
        selected['row_id']=selected.marker_id
    if markers is not None:
        selected=selected.loc[selected.row_id.isin(markers)].copy()
    selected['panel']=selected.identity.map(data['identity_labels'])
    return selected


def axis_label(rows):
    units=rows.display_unit.unique().tolist()
    if len(units)!=1:
        raise ValueError(f'Mixed effect units: {units}')
    if units[0]=='rank_biserial':
        return 'Rank-biserial persona − N', 'rank'
    if units[0] in {'percentage_points','pp'}:
        return 'Delta prevalence (poin persentase)', 'effect'
    if units[0]!='occurrences_per_100_words':
        raise ValueError('Unsupported effect unit: '+str(units[0]))
    estimands=' '.join(str(x) for x in rows.estimand_display.unique())
    if 'median' in estimands:
        return 'Delta median (kemunculan/100 kata)', 'effect'
    return 'Delta rerata (kemunculan/100 kata)', 'effect'


def figures(dimension, benchmark, data):
    """Yield figures one at a time so large catalogs do not stay in memory."""
    context=dimension.title()+' '+benchmark
    panels=[data['identity_labels'][i] for i in data['identities']]
    markers=data['core_markers']
    labels=data['marker_labels']
    notes=data['notes']
    def record(code,title,caption,fig,table,section='main'):
        return dict(code=code,title=title,caption=caption,fig=fig,table=table,section=section)
    yield record('U01','U01 · Cakupan teks',notes['coverage'],
                 plots.coverage(data['coverage'],data['personas'],data['persona_labels'],context),data['coverage'])
    yield record('U02','U02 · Panjang respons',notes['length'],
                 plots.length(data['length'],data['personas'],data['persona_labels'],context,
                              'Titik = median; garis = IQR, bukan CI.\n'
                              'Seluruh teks tersedia sesuai populasi sumber; lihat caption.'),data['length'])
    def three(scenario,section):
        profile=panel_data(data['profile'],data,scenario,markers)
        profile['display_unit']='percent'
        suffix='' if scenario=='E' else '_'+scenario
        title=f'U03 · Profil marker — {scenario}'
        yield record('U03'+suffix,title,notes['profile'],
                     plots.matrix(profile,panels,markers,labels,context,title,'Prevalence (% teks tersedia)',
                                  'Marker bersama; profil seluruh teks tersedia, bukan pasangan.\n'
                                  'N/E/I1/I2/I3 lengkap serta raw count tersedia pada tabel.',mode='profile',support=False),
                     profile,section)
        for family,code in [('overview','U04'),('detail','U05')]:
            rows=panel_data(data[family],data,scenario,markers)
            xlabel,mode=axis_label(rows)
            intervals=family=='detail'
            title=f'{code} · Perbedaan {scenario} − N'
            note=('Titik = effect size; CI sumber ditampilkan hanya pada U05.\n' if not intervals else
                  'Garis = CI 95% marginal sesuai estimand, bukan terkoreksi Holm.\n')
            note+='Titik kosong: <20 pasangan berubah atau dukungan belum tersedia.\nDukungan, p Holm, estimand dan denominator: CSV.'
            warning=data.get('pairing_warning')
            if warning:
                title=f'{code} · Kontras ditahan — {scenario} − N'
                note=('NA = ditahan karena pairing isi soal belum tervalidasi.\n'
                      'Angka sumber tidak dihitung ulang; lihat PAIRING_AUDIT.md.')
            fig=plots.matrix(rows,panels,markers,labels,context,title,xlabel,note,
                             intervals=intervals,mode=mode)
            if warning:
                for ax in fig.axes:
                    ax.text(.5,.5,'DITAHAN\nPairing soal belum valid',ha='center',va='center',
                            transform=ax.transAxes,fontsize=9,color='#9a3412',
                            bbox=dict(facecolor='white',edgecolor='none',pad=3))
            yield record(code+suffix,title,notes[family],fig,rows,section)
    yield from three('E','main')
    terms=panel_data(data['terms'],data,'E')
    orders={p:terms.loc[terms.panel.eq(p)].sort_values('term_order').row_id.drop_duplicates().tolist() for p in panels}
    yield record('U06','U06 · Kata dan ungkapan teramati',notes['terms'],
                 plots.matrix(terms,panels,orders,{r:r for ids in orders.values() for r in ids},context,
                              'U06 · Kata dan ungkapan teramati','Kemunculan per 100 kata',
                              'Hit dibagi total kata; bukan rerata density per respons.\n'
                              'Pilihan deskriptif; istilah dapat merujuk konteks soal.\n'
                              'NA = tidak tersedia; daftar istilah dan batasannya: CSV.',mode='positive',support=False),terms)
    example=data['examples']
    identity=data['identity_labels'][example.identity.iloc[0]]
    yield record('U07','U07 · Konteks kalimat',notes['examples'],
                 plots.examples(example,context,'Identitas: '+identity+'. Ilustrasi purposif, bukan sampel representatif.\n'
                                'Teks penuh, sumber dan posisi cuplikan: CSV.'),example)
    for scenario in ['I1','I2','I3']:
        yield from three(scenario,'detail')
    catalog_ids=data['catalog'].marker_id.tolist()
    for start in range(0,len(catalog_ids),6):
        selected=catalog_ids[start:start+6]
        profile=panel_data(data['profile'],data,'E',selected)
        profile['display_unit']='percent'
        code=f'CAT{start//6+1:02}'
        title=f'{code} · Katalog marker {start+1}–{min(start+6,len(catalog_ids))}'
        yield record(code,title,
                     'Katalog seluruh marker sumber: prevalence pada E untuk tiap identitas. '
                     'Tidak diseleksi menurut besar efek atau p-value. Profil N dan I1–I3 serta kontras yang diadaptasi ada pada CSV.',
                     plots.matrix(profile,panels,selected,labels,context,title,'Prevalence (% teks tersedia)',
                                  'Semua marker sumber; panel identitas pada skenario E.\n'
                                  'Definisi, sumber kamus, raw count dan denominator pada CSV.',mode='profile',support=False),
                     profile,'detail')


def read_pair(dimension):
    module=adapter(dimension)
    pair={b:module.load(b) for b in ['GSM8K','ETHICS']}
    for key in ['identities','identity_labels','personas','persona_labels','core_markers']:
        if pair['GSM8K'][key]!=pair['ETHICS'][key]:
            raise ValueError('Cross-benchmark display contract differs: '+key)
    gsm=pair['GSM8K']
    gsm['pairing_evidence']=pairing_evidence(dimension)
    gsm['pairing_warning']='Kontras GSM8K ditahan: ditemukan ID sama dengan isi soal berbeda.'
    for name in ['overview','detail']:
        gsm[name]=withhold_effects(gsm[name])
        gsm['notes'][name]=(gsm['pairing_warning']+' Template ditampilkan tanpa nilai. '
                            'Kolom source_* dan statistik lama dipertahankan sebagai arsip, '
                            'bukan hasil yang layak ditafsirkan. Audit dan perhitungan ulang belum dilakukan.')
    return module,pair


def check_layout(dimension):
    _,pair=read_pair(dimension)
    matched={}
    for benchmark,data in pair.items():
        for item in figures(dimension,benchmark,data):
            try:
                geometry=plots.audit_geometry(item['fig'])
                sig=plots.signature(item['fig'])
                if not item['code'].startswith('CAT'):
                    if item['code'] in matched and matched[item['code']]!=sig:
                        raise ValueError('Mismatched pair '+item['code'])
                    matched[item['code']]=sig
                print(dimension,benchmark,item['code'],geometry['min_font_pt'],'pt',flush=True)
            finally:
                plt.close(item['fig'])
    return len(matched)


def build(dimension,output=None):
    output=Path(output or default_output(dimension)).resolve()
    if output.exists():
        raise FileExistsError('Use a new output folder: '+str(output))
    module=adapter(dimension)
    base=ROOT/'notebooks'/dimension
    previous_files=[p for p in base.rglob('*') if p.is_file() and
                    not any(s in {'archive','_arsip'} for s in p.parts)]
    evidence=pairing_evidence(dimension)
    protected=set(module.input_paths())|set(previous_files)|{Path(r['source_file']) for r in evidence}
    print(f'[{dimension} 1/5] Hash sumber dan berkas lama',flush=True)
    before={str(p):digest(p) for p in sorted(protected)}
    _,pair=read_pair(dimension)
    output.mkdir(parents=True,exist_ok=False)
    for name in ['figures','tables']:
        (output/name).mkdir()
    status=dict(status='building',dimension=dimension,created_at=datetime.now(timezone.utc).isoformat())
    def write_json(name,value):
        (output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    write_json('manifest.json',status)
    try:
        records,marker_coverage,checks,old_links=[],[],{},[]
        pd.DataFrame(evidence).to_csv(output/'tables/pairing_counterexample.csv',index=False)
        (output/'PAIRING_AUDIT.md').write_text(
            '# Kontras GSM8K ditahan\n\n'
            'Pada Llama 3.1, `gsm8k_00511` adalah soal Gerald pada Neutral tetapi soal Andy pada E dan I1. '
            'Bukti teks, path JSON, dan SHA-256 pertanyaan tersedia di '
            '[pairing_counterexample.csv](tables/pairing_counterexample.csv). '
            '\n\nModul statistik GSM8K lama di `src/core/linguistic_markers/` '
            '(`religion_statistics.py` dan `regional_statistics.py`, baris 95–107) memasangkan hanya '
            'berdasarkan question_id. Keunikan ID dan hash file tidak membuktikan kesamaan soal. '
            'Satu counterexample ini membatalkan klaim bahwa seluruh pairing sudah valid; '
            '**bukan** bukti bahwa setiap model atau setiap pasangan salah.\n\n'
            'Seluruh U04/U05 GSM8K (E dan I1–I3) ditahan secara konservatif hingga audit pairing '
            'berdasarkan isi soal dilakukan. Estimate/CI display = NA, bukan nol. Kolom source_* '
            'dan statistik asli hanya arsip provenance, tidak boleh dipakai untuk kesimpulan. '
            'Grafik deskriptif tetap mengikuti populasi sumber. U07 dipilih terpisah sebagai contoh yang diperiksa. '
            'Statistik ETHICS tidak diubah.\n\n'
            'Perbaikan memerlukan audit isi soal seluruh pasangan lalu penghitungan ulang statistik '
            'dan Holm pada pasangan yang sah. Langkah tersebut belum dilakukan dalam tugas penyajian ini.\n',
            encoding='utf-8')
        print(f'[{dimension} 2/5] Render bersama dan katalog marker',flush=True)
        for benchmark,data in pair.items():
            for name in ['coverage','length','profile','overview','detail','terms','examples','catalog']:
                data[name].to_csv(output/'tables'/f'{benchmark.lower()}_{name}.csv',index=False)
            for label,path in data.get('previous_links',{}).items():
                path=Path(path)
                if not path.is_absolute():
                    path=ROOT/path
                if not path.exists():
                    raise FileNotFoundError(path)
                old_links.append((benchmark+' · '+label,Path(os.path.relpath(path,output)).as_posix().replace(' ','%20')))
            for item in figures(dimension,benchmark,data):
                print('  '+benchmark+' '+item['code'],flush=True)
                sig=plots.signature(item['fig'])
                record=save_panel(output,benchmark.lower()+'_'+item['code'],item['title'],item['title'],item['caption'],
                                  item['fig'],item['table'],item['section'])
                record.update(benchmark=benchmark,pair_id=item['code'],layout=sig)
                records.append(record)
                if item['code'].startswith('CAT'):
                    marker_coverage.extend(dict(benchmark=benchmark,marker_id=m,figure=record['png'],
                                                shown_in_main=m in data['core_markers']) for m in item['table'].row_id.unique())
                else:
                    if item['code'] in checks and checks[item['code']]['signature']!=sig:
                        raise ValueError('Mismatched benchmark templates: '+item['code'])
                    checks[item['code']]=dict(signature=sig,matching=benchmark=='ETHICS')
        print(f'[{dimension} 3/5] Galeri, sumber dan metodologi',flush=True)
        pd.DataFrame(CONTRACT,columns=['pair_id','question','visual_reference','shared_template']).to_csv(output/'visual_contract.csv',index=False)
        pd.DataFrame(marker_coverage).to_csv(output/'marker_coverage.csv',index=False)
        pd.DataFrame(records).drop(columns=['geometry','layout']).to_csv(output/'figure_index.csv',index=False)
        write_json('figure_manifest.json',records)
        write_json('layout_checks.json',checks)
        write_gallery(output,dimension,records,old_links)
        (output/'README.md').write_text(
            f'# {dimension.title()} — template visual bersama\n\n[Buka galeri](index.html). '
            '[Notebook GSM8K](GSM8K_unified.ipynb) · [Notebook ETHICS](ETHICS_unified.ipynb).\n\n'
            '**Peringatan: U04/U05 GSM8K ditahan karena masalah pairing soal.** '
            'Lihat [audit sumber](PAIRING_AUDIT.md). Kandidat visual ini bukan validasi ulang statistik.\n\n'
            '## SPEC dan DESIGN\n\n'+CAUTION+'\n\n'
            'Perubahan hanya penyajian statistik yang sudah tersedia. Tidak ada API, sampling statistik, '
            'ekstraksi ulang, perubahan kamus, maupun pengujian inferensial baru. U01–U07 memakai satu fungsi '
            'plot per jenis analisis pada kedua benchmark. Semua skenario tersedia; I1–I3 dan katalog lengkap '
            'diletakkan di detail agar cerita utama tetap terbaca.\n\n'
            'Profil dan katalog mencakup seluruh marker sumber. Tabel kontras hanya memuat estimand '
            'yang diadaptasi ke template; statistik lain, termasuk kontras binary GSM8K, tetap '
            'tersedia pada paket sumber yang ditautkan dari galeri.\n\n'
            '## IMPLEMENTATION\n\n'
            'PNG 240 dpi, SVG, dan CSV tersedia. Lebar 16 cm; DejaVu Sans minimal 9 pt. '
            'Warna serta posisi model tetap. E/I1/I2/I3 bukan urutan waktu atau dosis. '
            'Kontras utama adalah persona − N, bukan perbandingan langsung dua identitas. '
            'U06 menghitung 100 × hit / total kata; bukan rerata density respons. '
            'Kutipan U07 berupa substring nyata; pemilihan contoh tidak mewakili prevalensi.\n\n'
            +'\n\n'.join(f'### {b}\n\n'+'\n\n'.join(str(x) for x in d['notes'].values()) for b,d in pair.items())+
            '\n\n## CHECKLIST dan TESTING\n\n'
            '- Seluruh marker sumber ditelusuri melalui marker_coverage.csv dan katalog.\n'
            '- Geometri pasangan diperiksa di layout_checks.json.\n'
            '- Hasil eksekusi notebook, jumlah gambar dan hash tersedia di verification.json.\n'
            '- Seluruh artefak sebelum tugas ini tetap utuh; sumber dicek sebelum/sesudah.\n\n'
            '## INTEGRATION\n\n'
            'Ini kandidat terpisah, belum mengganti notebook utama. Reproduksi dari D:\\TugasAkhir ke **folder baru**:\n\n'
            f'```powershell\npython -X utf8 -B -m src.benchmarks.ethics.build_identity_lm_unified '
            f'--dimension {dimension} --output "notebooks/{dimension}/linguistic marker/candidate/unified_rerun"\n```\n',encoding='utf-8')
        print(f'[{dimension} 4/5] Eksekusi notebook',flush=True)
        notebook_checks={}
        for benchmark,data in pair.items():
            nb=make_notebook(dimension,benchmark,records,output,data['notes'])
            NotebookClient(nb,timeout=240,kernel_name='python3',resources={'metadata':{'path':str(output)}}).execute()
            errors=[o for c in nb.cells if c.cell_type=='code' for o in c.outputs if o.output_type=='error']
            pngs=sum('image/png' in o.get('data',{}) for c in nb.cells if c.cell_type=='code' for o in c.outputs)
            if errors or pngs!=sum(r['benchmark']==benchmark for r in records):
                raise ValueError('Incomplete notebook '+benchmark)
            nbformat.write(nb,output/f'{benchmark}_unified.ipynb')
            notebook_checks[benchmark]=dict(errors=0,embedded_pngs=pngs)
        print(f'[{dimension} 5/5] Verifikasi keutuhan sumber',flush=True)
        after={path:digest(Path(path)) for path in before}
        if before!=after:
            raise ValueError('Protected source or old artifact changed')
        write_json('source_hashes.json',dict(before=before,after=after,unchanged=True))
        names=['build_identity_lm_unified.py','identity_lm_unified_plots.py','identity_lm_unified_bundle.py',
               dimension+'_lm_unified_data.py','gender_lm_candidate_common.py','gender_lm_unified_plots.py',
               'gender_lm_reference_data.py','gender_lm_presentation_v2_plots.py']
        write_json('code_hashes.json',{str(Path(__file__).with_name(n)):digest(Path(__file__).with_name(n)) for n in names})
        coverage=pd.DataFrame(marker_coverage).groupby('benchmark').marker_id.nunique().to_dict()
        if coverage!={b:d['catalog'].marker_id.nunique() for b,d in pair.items()}:
            raise ValueError('Some catalog markers were not plotted')
        verification=dict(notebooks=notebook_checks,figures=len(records),matching_pairs=len(checks),
                          main_pairs=7,protected_files=len(before),old_files_unchanged=True,marker_coverage=coverage,
                          min_font_pt=min(r['geometry']['min_font_pt'] for r in records),
                          gsm8k_inference_status='withheld_pending_question_pairing_audit',
                          withheld_figures=8)
        write_json('verification.json',verification)
        status['status']='complete_visual_candidate_with_gsm8k_inference_hold'
        write_json('manifest.json',status)
        return verification
    except Exception as error:
        status.update(status='incomplete_do_not_use',error=str(error))
        write_json('manifest.json',status)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dimension',required=True,choices=['religion','regional'])
    parser.add_argument('--output',type=Path)
    parser.add_argument('--check-layout',action='store_true')
    args=parser.parse_args()
    result=check_layout(args.dimension) if args.check_layout else build(args.dimension,args.output)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
