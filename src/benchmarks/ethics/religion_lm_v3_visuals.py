"""Export whole-ETHICS tables, figures, evidence and a candidate notebook."""
import html
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.benchmarks.ethics.religion_lm_v3_registry import FAMILIES,get_registry,family_for
from src.benchmarks.ethics.religion_lm_v3_statistics import IDENTITIES,CONDITIONS
from src.benchmarks.ethics.religion_lm_v3_io import sha256,write_json,ROOT
from src.benchmarks.ethics.religion_lm_v3_extract import extract_final

PERSONAS = ['neutral']+[i+'_'+c for i in IDENTITIES for c in CONDITIONS]
PERSONA_LABELS = ['N']+[i[0].upper()+'-'+c for i in ('Muslim','Christian','Jewish','Atheist') for c in ('E','I1','I2','I3')]
MODEL_LABELS = {'deepseek_deepseek-r1-distill-qwen-32b':'DeepSeek R1 Distill 32B',
 'meta-llama_llama-3.1-8b-instruct':'Llama 3.1 8B','meta-llama_llama-3.3-70b-instruct':'Llama 3.3 70B',
 'mistralai_ministral-8b-2512':'Ministral 8B','openai_gpt-oss-20b':'GPT-OSS 20B',
 'qwen_qwen3-next-80b-a3b-thinking':'Qwen3 Next 80B A3B'}

def availability_table(rows):
    result = rows.groupby(['model_id','persona','surface_status']).size().rename('total').reset_index()
    result['percent'] = 100*result.total/result.groupby(['model_id','persona']).total.transform('sum')
    result['subset'] = 'ethics_all'
    return result

def heatmap_table(rows,markers,personas=PERSONAS):
    return rows.pivot(index='marker_id',columns='persona',values='prevalence_text').reindex(index=markers,columns=personas)

def highlight_final(text):
    _,hits = extract_final(text)
    accepted = [h for h in hits if h['candidate_status']!='excluded']
    positions = sorted({0,len(text)}|{h[k] for h in accepted for k in ('raw_start','raw_end')})
    pieces = []
    for a,b in zip(positions,positions[1:]):
        active = [h for h in accepted if h['raw_start']<=a and h['raw_end']>=b]
        segment = html.escape(text[a:b])
        if active:
            title = '; '.join(h['rule_id']+' | '+h['speaker_role']+' | '+h['discourse_scope'] for h in active)
            segment = '<mark title="'+html.escape(title,quote=True)+'">'+segment+'</mark>'
        pieces.append(segment)
    return ''.join(pieces)

def render_report(output):
    output = Path(output)
    for name in ('figures','figure_tables','evidence'): (output/name).mkdir(exist_ok=False)
    frame = pd.read_csv(output/'response_level/markers.csv.gz')
    summary = pd.read_csv(output/'statistics/overall_descriptive.csv')
    contrasts = pd.read_csv(output/'statistics/paired_contrasts.csv.gz')
    models = sorted(frame.model_id.unique())
    figures = []
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    def save(fig,figure_id,table,caption,markers):
        table_path = output/f'figure_tables/{figure_id}.csv'
        table.to_csv(table_path,index=False)
        fig.text(.01,.008,'KANDIDAT — audit manusia belum selesai. Seluruh ETHICS; satu respons = satu observasi.',fontsize=9,color='#5a5a5a')
        paths = []
        for ext in ('png','svg'):
            path = output/f'figures/{figure_id}.{ext}'
            fig.savefig(path,dpi=220,bbox_inches='tight',facecolor='white')
            paths.append(str(path.relative_to(output)))
        plt.close(fig)
        figures.append(dict(figure_id=figure_id,scope='ethics_all',caption=caption,outcome_ids=markers,table=str(table_path.relative_to(output)),
            files=paths,sha256={p:sha256(output/p) for p in paths},table_sha256=sha256(table_path)))
    available = availability_table(frame)
    fig,axes = plt.subplots(2,3,figsize=(19,10),layout='constrained')
    statuses = [('linguistic_text','Penjelasan','#238b8e'),('label_only','Label saja','#e6b450'),('empty_final','Final kosong','#d4d4d4'),
        ('technical_only','Nonprosa','#8f79a8'),('unseparable_channel','Channel ambigu','#c75858')]
    for ax,model in zip(axes.flat,models):
        data = available.loc[available.model_id.eq(model)].pivot(index='persona',columns='surface_status',values='percent').reindex(PERSONAS).fillna(0)
        bottom = np.zeros(len(PERSONAS))
        for status,label,color in statuses:
            values = data[status].to_numpy() if status in data else np.zeros(len(PERSONAS))
            ax.barh(PERSONA_LABELS,values,left=bottom,color=color,label=label,height=.8); bottom+=values
        ax.invert_yaxis(); ax.set_xlim(0,100);ax.set_xlabel('% dari seluruh respons');ax.set_title(MODEL_LABELS.get(model,model))
    handles,labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=5,bbox_to_anchor=(.5,1.045))
    save(fig,'01_availability',available,'Denominator seluruh respons per model/persona. N=neutral; M/C/J/A=Muslim/Christian/Jewish/Atheist; E=explicit; I1–I3=implicit.',[])
    for index,(family,title) in enumerate(FAMILIES.items(),2):
        markers = [m.marker_id for m in get_registry() if family_for(m.marker_id)==family]
        if not markers: continue
        table = summary.loc[summary.marker_id.isin(markers)]
        fig,axes = plt.subplots(2,3,figsize=(24,max(8,len(markers)*.85+3)),layout='constrained')
        cmap = plt.get_cmap('viridis').copy();cmap.set_bad('#c9c9c9')
        for ax,model in zip(axes.flat,models):
            data = heatmap_table(table.loc[table.model_id.eq(model)],markers)
            im = ax.imshow(np.ma.masked_invalid(data.to_numpy(float)),aspect='auto',vmin=0,vmax=100,cmap=cmap)
            ax.set_xticks(range(len(PERSONAS)),PERSONA_LABELS,rotation=90)
            ax.set_yticks(range(len(markers)),[m.replace('religion_','').replace('organization_','').replace('theme_','').replace('_',' ') for m in markers],fontsize=8)
            ax.set_title(MODEL_LABELS.get(model,model)); ax.tick_params(length=0)
        fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.65,label='% respons berteks dengan marker (abu-abu = NA)')
        fig.suptitle(title+' | prevalence bersyarat pada penjelasan',fontsize=17)
        save(fig,f'{index:02d}_atlas_{family}',table,'Kehadiran marker pada respons berteks; denominator, jumlah mentah, dan densitas ada pada CSV. Nol berbeda dari NA. Kemunculan tema tidak membuktikan adopsi agama.',markers)
    # Fixed before results: these outcomes explain reference, accommodation and limits.
    focus = ['religion_perspective_adoption','religion_moral_authority_framing','religion_named_concept_reference','religion_identity_irrelevance','ai_identity_statement','response_binary_decline']
    colors = ['#0072b2','#d55e00','#009e73','#cc79a7']
    for number,kind in enumerate(('vs_neutral','implicit_vs_explicit','direct_identity'),10):
        data = contrasts.loc[contrasts.contrast_kind.eq(kind)&contrasts.population.eq('text_all')&contrasts.metric.eq('prevalence_text')&contrasts.marker_id.isin(focus)]
        if kind in ('vs_neutral','direct_identity'):
            data = data.loc[data.left_persona.str.endswith('_explicit')]
        else:
            data = data.loc[data.left_persona.str.endswith('_implicit_1')]
        fig,axes = plt.subplots(2,3,figsize=(23,12),layout='constrained')
        for ax,model in zip(axes.flat,models):
            group = data.loc[data.model_id.eq(model)].sort_values(['left_persona','right_persona','marker_id'])
            labels = []
            for y,(_,r) in enumerate(group.iterrows()):
                identity = r.left_persona.split('_')[0]
                color = colors[list(IDENTITIES).index(identity)]
                if np.isfinite(r.ci_low) and np.isfinite(r.ci_high):
                    ax.plot([r.ci_low,r.ci_high],[y,y],color=color,lw=1)
                if np.isfinite(r.effect): ax.scatter(r.effect,y,s=13,color=color)
                labels.append(r.left_persona.split('_')[0]+'−'+r.right_persona.split('_')[0]+' | '+r.marker_id.replace('religion_','').replace('_',' ')+'  n='+str(r.n_pairs))
            ax.set_yticks(range(len(labels)),labels,fontsize=6.5);ax.invert_yaxis();ax.axvline(0,color='#999999',lw=.7)
            ax.set_xlim(-100,100);ax.set_xlabel('Selisih prevalence (poin persentase), kiri − kanan; CI 95%')
            ax.set_title(MODEL_LABELS.get(model,model))
        save(fig,f'{number:02d}_forest_{kind}',data,'Efek berpasangan pada pertanyaan sama; subset termasuk kunci. Grafik explicit atau implicit_1 sebagai ringkasan tetap; seluruh 52 kontras/model, density, Holm dan sensitivitas tersedia di paired_contrasts.csv.gz.',focus)
    evidence_path = ROOT/'result/ETHICS/reports/religion_lm_v3_preparation_20260915_r1/targeted_full_responses.json'
    samples = json.loads(evidence_path.read_text(encoding='utf-8'))
    cards = []
    for r in samples:
        cards.append('<details><summary>'+html.escape(r.get('readback_id','')+' | '+r['model_id']+' | '+r['persona'])+'</summary><p><code>'+html.escape(r['response_key'])+'</code></p><div class="response">'+highlight_final(r['final_output'])+'</div></details>')
    (output/'evidence/full_responses.html').write_text('<!doctype html><html lang="id"><meta charset="utf-8"><title>Bukti marker Religion ETHICS</title><style>body{font:16px/1.65 system-ui;max-width:1100px;margin:40px auto;padding:20px}details{border:1px solid #ddd;padding:14px;margin:12px 0}.response{white-space:pre-wrap;overflow-wrap:anywhere}mark{background:#ffebad}code{font-size:12px}summary{cursor:pointer}</style><h1>Bukti respons akhir utuh</h1><p>Kandidat, bukan gold manusia. Sorot marker untuk rule dan scope. Contoh pengembangan terarah; bukan sampel representatif. Substring dan klaim di dalamnya adalah keluaran model.</p>'+''.join(cards)+'</html>',encoding='utf-8')
    write_json(output/'figure_manifest.json',figures)
    intro = '# Religion ETHICS — linguistic marker kandidat v3\n\nPool seluruh empat subset; enam model dan 17 persona. Tidak ada akurasi ETHICS gabungan. Marker final-only mencakup abstain berteks. Status kandidat: audit dua reviewer dan recall independen belum selesai.\n\nBuka `manifest.json`, `registry/markers.json` serta tabel setiap gambar untuk denominator dan provenance. M/C/J/A = Muslim/Christian/Jewish/Atheist; E = explicit; I1–I3 = implicit; N = neutral.'
    cells = [dict(cell_type='markdown',metadata={},source=intro.splitlines(keepends=True))]
    for f in figures:
        cells.append(dict(cell_type='markdown',metadata={},source=[f"## {f['figure_id']}\n\n{f['caption']}\n\n![{f['figure_id']}]({f['files'][0]})\n\n[Tabel pembentuk]({f['table']})\n"]))
    cells.append(dict(cell_type='markdown',metadata={},source=['## Audit dan keterbatasan\n\n[Bukti lengkap dengan highlight](evidence/full_responses.html). Tema leksikal tidak sama dengan dukungan model, truth doktrinal atau moral judgement. Scope yang tidak pasti tetap unresolved. Dua putaran discovery masih menemukan gap; tidak ada klaim coverage hampir sempurna. Sensitivitas tidak memperbaiki cacat upstream pada prompt/scoring.']))
    write_json(output/'religion_ethics_linguistic_markers_v3_candidate.ipynb',dict(cells=cells,metadata={},nbformat=4,nbformat_minor=5))
    return figures
