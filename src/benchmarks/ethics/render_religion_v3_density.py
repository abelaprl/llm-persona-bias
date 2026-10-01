"""Add a prespecified density effect view and notebook navigation."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.benchmarks.ethics.religion_lm_v3_io import ROOT,sha256,write_json
from src.benchmarks.ethics.religion_lm_v3_visuals import MODEL_LABELS
from src.benchmarks.ethics.religion_lm_v3_statistics import IDENTITIES

def render_density(output):
    data = pd.read_csv(output/'statistics/paired_contrasts.csv.gz')
    chosen = data.loc[data.population.eq('text_all')&data.metric.eq('density')&data.contrast_kind.eq('vs_neutral')&
        data.left_persona.str.endswith('_explicit')&data.marker_id.str.startswith('theme_')].copy()
    figure_id = '13_density_moral_themes'
    table = output/f'figure_tables/{figure_id}.csv';chosen.to_csv(table,index=False)
    fig,axes = plt.subplots(2,3,figsize=(23,13),layout='constrained')
    finite = np.abs(chosen[['ci_low','ci_high','effect']].to_numpy(float))
    limit = max(1,float(np.nanmax(finite))*1.05) if np.isfinite(finite).any() else 1
    colors = ['#0072b2','#d55e00','#009e73','#cc79a7']
    for ax,model in zip(axes.flat,sorted(chosen.model_id.unique())):
        group = chosen.loc[chosen.model_id.eq(model)].sort_values(['left_persona','marker_id'])
        labels = []
        for y,(_,r) in enumerate(group.iterrows()):
            identity = r.left_persona.split('_')[0]
            color = colors[list(IDENTITIES).index(identity)]
            if np.isfinite(r.ci_low) and np.isfinite(r.ci_high): ax.plot([r.ci_low,r.ci_high],[y,y],color=color,lw=1)
            if np.isfinite(r.effect):ax.scatter(r.effect,y,color=color,s=13)
            labels.append(identity+' | '+r.marker_id.removeprefix('theme_').replace('_',' ')+'  n='+str(r.n_pairs))
        ax.set_yticks(range(len(labels)),labels,fontsize=7);ax.invert_yaxis();ax.axvline(0,color='#999999',lw=.7)
        ax.set_xlim(-limit,limit);ax.set_xlabel('Selisih occurrence per 100 kata; explicit − neutral; CI 95%')
        ax.set_title(MODEL_LABELS[model])
    fig.suptitle('Tema moral leksikal | efek density berpasangan, keseluruhan ETHICS',fontsize=16)
    fig.text(.01,.008,'KANDIDAT — belum tervalidasi manusia. Tema termasuk negasi/kutipan; bukan endorsement atau indikator kualitas moral.',fontsize=9)
    paths = []
    for ext in ('png','svg'):
        path = output/f'figures/{figure_id}.{ext}';fig.savefig(path,dpi=220,bbox_inches='tight',facecolor='white');paths.append(str(path.relative_to(output)))
    plt.close(fig)
    entry = dict(figure_id=figure_id,scope='ethics_all',caption='Delapan tema ditetapkan dari discovery sebelum estimasi; density adalah rata-rata selisih respons berpasangan, bukan pooled token density.',
        outcome_ids=sorted(chosen.marker_id.unique()),table=str(table.relative_to(output)),table_sha256=sha256(table),files=paths,sha256={p:sha256(output/p) for p in paths})
    write_json(output/'figure_manifest_additional.json',[entry])
    notebook = json.loads((output/'religion_ethics_linguistic_markers_v3_candidate.ipynb').read_text(encoding='utf-8'))
    notebook['cells'][0]['source'].append('\n\nPada forest plot, baris tanpa titik berarti efek tidak tersedia; titik tanpa garis CI berarti interval tidak dapat diestimasi. Perhatikan n pasangan. Efek text-conditional hanya berlaku pada pasangan dengan teks di kedua sisi; lihat output-level dan availability untuk konteks seleksi.\n')
    notebook['cells'][0]['source'].append('\n\n[Penjelajah seluruh 52 kontras, termasuk I1/I2/I3](evidence/all_contrasts_explorer.html). Filter model, marker, populasi dan metrik; tabel menyertakan jumlah occurrence mentah kedua sisi. CI 95% bersifat marginal; koreksi Holm dilaporkan terpisah.\n')
    notebook['cells'][0]['source'].append('\n\nIstilah penjelasan pada grafik mengikuti kategori operasional linguistic_text: final dengan kata linguistik, termasuk verdict leksikal atau prosa pendek. Kategori ini tidak membuktikan bahwa model memberikan argumentasi panjang. Label-only merujuk pada format verdict biner teknis yang didefinisikan surface.\n')
    notebook['cells'][0]['source'].append('\n\n[Kamus dan definisi](registry/markers.json) · [Statistik deskriptif](statistics/overall_descriptive.csv) · [Seluruh kontras dan sensitivitas](statistics/paired_contrasts.csv.gz) · [Denominator](inventory/coverage.csv) · [Audit aturan](audit/rule_quotas.csv).\n')
    notebook['cells'][0]['source'].append('\n\n[PDF A4: 24 panel per model](religion_ethics_marker_publication_panels.pdf). Setiap panel tetap keseluruhan ETHICS; tersedia juga PNG/SVG pada figures/publication. Gambar enam model di bawah berfungsi sebagai ringkasan keseluruhan.\n')
    notebook['cells'][0]['source'].append('\n\nPrevalence tanpa pembobotan memakai McNemar eksak. Uji density dan prevalence equal-subset memakai paired sign-flip (eksak hingga 16 pasangan informatif; selebihnya 10.000 pengacakan). Asumsinya pertukaran kondisi dalam pertanyaan sah di bawah hipotesis nol. CI bootstrap tetap perkiraan; interval degenerat, kurang dari enam pasangan informatif, atau dukungan strata tidak cukup ditandai NA. Holm mempertahankan seluruh hipotesis yang direncanakan. [Audit perbaikan inferensi](statistics/inference_audit_manifest.json).\n')
    notebook['cells'].insert(-1,dict(cell_type='markdown',metadata={},source=['## Density tema moral\n\n'+entry['caption']+'\n\n![Density]('+paths[0]+')\n\n[Tabel pasangan]('+entry['table']+')\n']))
    for i,cell in enumerate(notebook['cells']):cell['id']=hashlib.sha256(str(i).encode()).hexdigest()[:12]
    name = 'religion_ethics_linguistic_markers_v3_presentation_candidate.ipynb'
    write_json(output/name,notebook)
    destination = ROOT/'notebooks/religion'/('linguistic_marker_religion_ethics_v3_candidate_'+output.name.removeprefix('religion_lm_v3_candidate_')+'.ipynb')
    prefix = os.path.relpath(output,destination.parent).replace('\\','/')+'/'
    for cell in notebook['cells']:
        converted=[]
        for line in cell['source']:
            line=line.replace('](religion_ethics_marker_publication_panels.pdf)',']('+prefix+'religion_ethics_marker_publication_panels.pdf)')
            for folder in ('figures','figure_tables','evidence','registry','statistics','inventory','audit','comparison_with_baseline'):
                line=line.replace(']('+folder+'/',']('+prefix+folder+'/')
            converted.append(line)
        cell['source']=converted
    write_json(destination,notebook)
    return entry,name,str(destination)
