"""Readable A4 model panels; every panel pools all ETHICS subsets."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from src.benchmarks.ethics.religion_lm_v3_io import sha256,write_json
from src.benchmarks.ethics.religion_lm_v3_visuals import MODEL_LABELS,PERSONAS

FOCUS={
 'religion_perspective_adoption':'Adopsi perspektif',
 'religion_moral_authority_framing':'Otoritas moral agama',
 'religion_named_concept_reference':'Konsep agama',
 'religion_identity_irrelevance':'Identitas tak relevan',
 'ai_identity_statement':'Pernyataan AI',
 'response_binary_decline':'Batas format biner',
}
THEMES={'theme_harm':'Harm/suffering','theme_compassion':'Compassion/empathy',
 'theme_fairness':'Fairness/equality','theme_consent_autonomy':'Consent/autonomy',
 'theme_boundaries':'Boundaries','theme_responsibility':'Responsibility',
 'theme_intention':'Intention','theme_honesty':'Honesty/deception'}
IDENTITY={'islam':'M','christian':'C','jewish':'J','atheist':'A'}
COLORS={'islam':'#0072b2','christian':'#d55e00','jewish':'#009e73','atheist':'#cc79a7'}


def short_persona(persona):
    if persona=='neutral':return 'N'
    identity,condition=persona.split('_',1)
    return IDENTITY[identity]+'-'+('E' if condition=='explicit' else 'I'+condition[-1])


def render_publication(output,source=None,preview=False):
    output=Path(output);source=Path(source) if source else output/'statistics/paired_contrasts.csv.gz'
    data=pd.read_csv(source)
    figures=output/'figures/publication';figures.mkdir(parents=True,exist_ok=False)
    tables=output/'figure_tables/publication';tables.mkdir(parents=True,exist_ok=False)
    primary=data.loc[data.population.eq('text_all')].copy()
    for name in ('left_persona','right_persona'):
        primary[name]=pd.Categorical(primary[name],categories=PERSONAS,ordered=True)
    density=primary.loc[primary.metric.eq('density')&primary.contrast_kind.eq('vs_neutral')&
        primary.left_persona.str.endswith('_explicit')&primary.marker_id.isin(THEMES)]
    finite=np.abs(density[['effect','ci_low','ci_high']].to_numpy(float))
    density_limit=max(1,float(np.nanmax(finite))*1.05) if np.isfinite(finite).any() else 1
    specs=[('vs_neutral','Explicit − neutral','prevalence_text',FOCUS,100),
        ('implicit_vs_explicit','Implicit 1 − explicit','prevalence_text',FOCUS,100),
        ('direct_identity','Perbandingan identitas pada explicit','prevalence_text',FOCUS,100),
        ('density_themes','Density tema moral: explicit − neutral','density',THEMES,density_limit)]
    records=[]
    pdf_path=output/'religion_ethics_marker_publication_panels.pdf'
    if pdf_path.exists():raise FileExistsError(pdf_path)
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    with PdfPages(pdf_path) as pdf:
        for model in sorted(primary.model_id.unique()):
            for kind,title,metric,labels,limit in specs:
                actual_kind='vs_neutral' if kind=='density_themes' else kind
                condition='_implicit_1' if kind=='implicit_vs_explicit' else '_explicit'
                chosen=primary.loc[primary.model_id.eq(model)&primary.contrast_kind.eq(actual_kind)&
                    primary.metric.eq(metric)&primary.marker_id.isin(labels)&primary.left_persona.str.endswith(condition)].copy()
                chosen['marker_id']=pd.Categorical(chosen.marker_id,categories=list(labels),ordered=True)
                chosen=chosen.sort_values(['left_persona','right_persona','marker_id'])
                figure_id=model+'__'+kind
                table=tables/(figure_id+'.csv');chosen.to_csv(table,index=False)
                fig,ax=plt.subplots(figsize=(8.27,11.69))
                fig.subplots_adjust(left=.40,right=.97,bottom=.21,top=.90)
                tick_labels=[]
                for y,row in enumerate(chosen.itertuples(index=False)):
                    color=COLORS[row.left_persona.split('_')[0]]
                    if np.isfinite(row.ci_low) and np.isfinite(row.ci_high):ax.plot([row.ci_low,row.ci_high],[y,y],color=color,lw=1.5)
                    if np.isfinite(row.effect):ax.scatter(row.effect,y,color=color,s=22,zorder=3)
                    else:ax.text(0,y,'NA',ha='center',va='center',fontsize=7,color='#666',bbox=dict(facecolor='white',edgecolor='none',pad=1))
                    tick_labels.append(short_persona(row.left_persona)+'−'+short_persona(row.right_persona)+' | '+labels[row.marker_id]+'  n='+str(row.n_pairs))
                ax.set_yticks(range(len(tick_labels)),tick_labels,fontsize=8)
                ax.invert_yaxis();ax.axvline(0,color='#888',lw=.8);ax.set_xlim(-limit,limit)
                ax.tick_params(axis='x',labelsize=9);ax.grid(axis='x',color='#ddd',lw=.5,zorder=0)
                unit='occurrence per 100 kata' if metric=='density' else 'poin persentase'
                ax.set_xlabel('Selisih '+unit+'\nkiri − kanan; CI 95% marginal',fontsize=10)
                fig.suptitle(MODEL_LABELS.get(model,model)+'\n'+title+' | keseluruhan ETHICS',fontsize=12,y=.965)
                status='PREVIEW LAYOUT — belum laporan integrasi' if preview else 'KANDIDAT — audit manusia belum selesai'
                status_text=fig.text(.055,.13,status,fontsize=9,weight='bold')
                legend_text=fig.text(.055,.08,'M=Muslim; C=Christian; J=Jewish; A=Atheist; N=neutral; E=explicit; I1=implicit 1.\n'
                    'Satu pasangan = pertanyaan sama, kedua final berteks. n tampil pada setiap baris.\n'
                    'CI tidak tersedia: titik tanpa garis; efek tidak tersedia: NA. Holm dan raw count pada CSV.',fontsize=8)
                caveat_text=fig.text(.055,.025,'Tema leksikal mencakup kutipan/negasi; bukan endorsement atau kualitas moral.\n'
                    'Seluruh I1/I2/I3 dan 52 kontras/model tersedia pada penjelajah, bukan dipool.',fontsize=8)
                fig.canvas.draw();renderer=fig.canvas.get_renderer()
                boxes=[t.get_window_extent(renderer) for t in (status_text,legend_text,caveat_text)]
                assert ax.get_tightbbox(renderer).y0>boxes[0].y1+2,'Axis and footer overlap'
                assert boxes[0].y0>boxes[1].y1+2 and boxes[1].y0>boxes[2].y1+2,'Footer blocks overlap'
                for text in ax.get_yticklabels()+ax.get_xticklabels()+[ax.xaxis.label,fig._suptitle,status_text,legend_text,caveat_text]:
                    box=text.get_window_extent(renderer)
                    assert box.x0>=0 and box.x1<=fig.bbox.width and box.y0>=0 and box.y1<=fig.bbox.height,'Text clipped'
                paths=[]
                for extension in ('png','svg'):
                    path=figures/(figure_id+'.'+extension);fig.savefig(path,dpi=220,facecolor='white');paths.append(str(path.relative_to(output)))
                pdf.savefig(fig,facecolor='white');plt.close(fig)
                records.append(dict(figure_id=figure_id,model_id=model,scope='ethics_all',kind=kind,metric=metric,
                    table=str(table.relative_to(output)),table_sha256=sha256(table),files=paths,
                    sha256={p:sha256(output/p) for p in paths},outcome_ids=list(labels),page_size='A4 portrait',preview=preview,layout_check_passed=True))
            print('Publication panels: '+model,flush=True)
    manifest=dict(panels=records,count=len(records),scope='ethics_all',source_sha256=sha256(source),preview=preview,
        pdf=str(pdf_path.relative_to(output)),pdf_sha256=sha256(pdf_path),script_sha256=sha256(__file__))
    write_json(output/'publication_panel_manifest.json',manifest)
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False);render_publication(args.output,args.source,preview=True)
