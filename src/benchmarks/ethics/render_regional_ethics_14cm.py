"""Render frozen regional ETHICS tables; never recalculate tests or publish assets."""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import re
import shutil
import subprocess
import sys
import importlib.metadata

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Text
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
ACTIVE = ROOT / 'notebooks/regional/ethics accuracy analysis'
THESIS = Path('D:/ProposalTA-18222008')
CHAPTER = THESIS / 'Bab VI - ETHICS Regional Akurasi.tex'
MODELS = ['qwen_qwen3-next-80b-a3b-thinking', 'meta-llama_llama-3.1-8b-instruct',
          'mistralai_ministral-8b-2512', 'deepseek_deepseek-r1-distill-qwen-32b',
          'openai_gpt-oss-20b', 'meta-llama_llama-3.3-70b-instruct']
NAMES = ['Qwen3 Next 80B-A3B', 'Llama 3.1 8B', 'Ministral 8B',
         'DeepSeek R1-Distill 32B', 'GPT-OSS 20B', 'Llama 3.3 70B']
GROUPS = [('au_at', ['north_american', 'east_asian'], 'Amerika Utara–Asia Timur'),
          ('as_ass', ['south_asian', 'sub_saharan_african'], 'Asia Selatan–Afrika Sub-Sahara')]
SHORT = dict(zip(['north_american', 'east_asian', 'south_asian', 'sub_saharan_african'], ['AU','AT','AS','ASS']))
SCENARIOS = ['explicit','implicit_1','implicit_2','implicit_3']
REASONS = {'both_sides':'Kedua sisi / tidak berpihak', 'analysis_without_verdict':'Analisis tanpa verdict',
           'no_opinion_stop':'Tidak punya opini lalu berhenti', 'explicit_binary_refusal':'Menolak jawaban biner',
           'task_substitution':'Mengganti tugas', 'ai_refusal':'Penolakan sebagai AI',
           'insufficient_context':'Konteks dianggap kurang', 'placeholder_confusion':'Bingung placeholder',
           'returns_question':'Mengulang pertanyaan', 'capability_mismatch':'Mengaku tidak mampu',
           'detector_disagreement':'Detektor tidak sepakat', 'conditional_only':'Jawaban hanya bersyarat'}
WIDTH = 14 / 2.54


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dec(value, signed=False):
    if not np.isfinite(value):
        return 'NA'
    return (f'{value:+.1f}' if signed else f'{value:.1f}').replace('.', ',').replace('-', '−')


def personas(ids):
    return [identity + '_' + scenario for identity in ids for scenario in SCENARIOS]


def labels(ids):
    return [SHORT[identity] + '\n' + scenario for identity in ids for scenario in ['E','I1','I2','I3']]


def ratio(numerator, denominator):
    return numerator / denominator.replace(0, np.nan)


class Renderer:
    def __init__(self):
        self.run = ROOT / 'result/analyses/accuracy/runs' / ('regional_ethics_bab6_14cm_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        self.run.mkdir(parents=True, exist_ok=False)
        for folder in ['bab6', 'pelengkap', 'pelengkap_committed', 'tables', 'proof']:
            (self.run / folder).mkdir()
        self.figures, self.cells, self.layout, self.mapping = [], [], [], []
        self.chapter = CHAPTER.read_text(encoding='utf-8')
        self.old = []
        for block in re.findall(r'\\begin\{figure\}.*?\\end\{figure\}', self.chapter, re.S):
            path = re.search(r'\{(image/[^}]+)\}', block).group(1)
            label = re.search(r'\\label\{([^}]+)\}', block).group(1)
            self.old.append((path, label))
        protected = [p for folder in [ACTIVE/'tables', ACTIVE/'figures', ACTIVE] for p in folder.glob('*') if p.is_file()]
        protected += [CHAPTER, Path(__file__)] + [THESIS/p for p, _ in self.old]
        self.baseline = {str(p.resolve()):sha(p) for p in protected}
        self.summary = pd.read_csv(ACTIVE/'tables/ethics_regional_parity_condition_summary.csv')
        self.effects = pd.read_csv(ACTIVE/'tables/ethics_regional_parity_paired_effects_vs_neutral.csv')
        self.reasons = pd.read_csv(ACTIVE/'tables/ethics_regional_parity_abstain_reasons.csv')
        self.category_source = pd.read_csv(ACTIVE/'tables/ethics_regional_parity_category_summary.csv')
        plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':8, 'axes.titlesize':9,
                            'axes.labelsize':8, 'xtick.labelsize':8, 'ytick.labelsize':8,
                            'legend.fontsize':8, 'pdf.fonttype':42, 'savefig.facecolor':'white'})
        print('RUN:', self.run, flush=True)

    def validate(self):
        s, e, r = self.summary, self.effects, self.reasons
        assert len(self.old) == 10
        assert len(s) == 408 and not s.duplicated(['model','persona','subset']).any()
        assert set(s.model) == set(MODELS)
        assert set(s.persona) == set(personas(GROUPS[0][1])+personas(GROUPS[1][1])+['neutral'])
        assert (s.total == s.committed + s.abstain + s.missing).all()
        assert (s.nonmissing == s.total - s.missing).all()
        assert (s.committed == s.n_answer_wrong + s.n_answer_acceptable).all()
        for metric, numerator, denominator in [('commit_rate','committed','total'), ('abstain_rate','abstain','total'),
                                               ('missing_rate','missing','total'), ('pct_answer_wrong','n_answer_wrong','committed')]:
            np.testing.assert_allclose(s[metric], ratio(s[numerator],s[denominator]), equal_nan=True)
        valid = s[s.subset.isin(['commonsense','justice'])]
        np.testing.assert_allclose(valid.accuracy_committed, ratio(valid.n_correct,valid.committed), equal_nan=True)
        np.testing.assert_allclose(valid.effective_success_rate_nonmissing, ratio(valid.n_correct,valid.nonmissing), equal_nan=True)
        assert s[~s.subset.isin(['commonsense','justice'])].accuracy_committed.isna().all()
        assert len(e) == 960 and e.family_size.eq(16).all() and e.persona_neutral.eq('neutral').all()
        assert not e.duplicated(['outcome','model','persona','subset']).any()
        assert e.groupby(['model','subset','outcome']).size().eq(16).all()
        assert (e.significant_holm_0_05 == e.holm_adjusted_p_value.lt(.05)).all()
        np.testing.assert_allclose(e.risk_difference_pp, e.rate_a_pct-e.rate_b_pct, atol=1e-10, equal_nan=True)
        assert set(r.abstain_reason) == set(REASONS)
        assert not r.duplicated(['model','persona_folder','abstain_reason']).any()
        counts = ['total','nonmissing','committed','abstain','missing','n_answer_wrong','n_answer_acceptable']
        self.all_status = s.groupby(['model','persona'])[counts].sum()
        reason_sums = r.groupby(['model','persona_folder']).reason_count.sum()
        reason_sums.index.names = ['model','persona']
        pd.testing.assert_series_equal(reason_sums.sort_index(),self.all_status.abstain.sort_index(),check_names=False)
        denominators = self.all_status.reindex(pd.MultiIndex.from_frame(r[['model','persona_folder']])).reset_index(drop=True)
        for denominator, metric in [('abstain','pct_of_abstain'),('nonmissing','pct_of_nonmissing')]:
            np.testing.assert_array_equal(r[denominator], denominators[denominator])
            np.testing.assert_allclose(r[metric],100*ratio(r.reason_count,r[denominator]),equal_nan=True)
        pool = self.all_status.reset_index()
        pool['category'] = pool.persona.str.replace(r'_implicit_[123]$', '_implicit_avg', regex=True)
        self.category = pool.groupby(['model','category'])[counts].sum()
        pd.testing.assert_frame_equal(self.category.sort_index(),self.category_source.set_index(['model','category'])[counts].sort_index(),check_dtype=False)
        for stem, frame in [('condition_summary_frozen',s),('paired_effects_frozen',e),('abstain_reasons_frozen',r),
                            ('category_counts',self.category.reset_index())]:
            frame.to_csv(self.run/'tables'/f'{stem}.csv',index=False)
        self.checks = {'source_responses':int(s.total.sum()),'status_counts':s[['committed','abstain','missing']].sum().astype(int).to_dict(),
                       'summary_rows':len(s),'effect_rows':len(e),'holm_family_size':16,
                       'significant_by_outcome':e.groupby('outcome').significant_holm_0_05.sum().astype(int).to_dict(),
                       'reason_categories_exhaust_abstain':True,'formulas_and_pooling_match':True,
                       'statistical_tests_rerun':False,'parser_changed':False,'physical_print_checked':False}

    def heat(self, ax, values, xlabels, ylabels, annotations=None, limit=None, cmap='YlOrBr'):
        a = np.asarray(values,float)
        im = ax.pcolormesh(np.arange(a.shape[1]+1)-.5,np.arange(a.shape[0]+1)-.5,np.ma.masked_invalid(a),
                          vmin=-limit if limit else 0,vmax=limit if limit else 100,cmap=cmap,shading='flat',rasterized=False)
        ax.set(xlim=(-.5,a.shape[1]-.5),ylim=(a.shape[0]-.5,-.5))
        ax.set_xticks(range(a.shape[1]),xlabels)
        ax.set_yticks(range(a.shape[0]),ylabels)
        ax.tick_params(length=0,pad=4)
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.set_xticks(np.arange(-.5,a.shape[1],1),minor=True)
        ax.set_yticks(np.arange(-.5,a.shape[0],1),minor=True)
        ax.grid(which='minor',color='white',linewidth=.5)
        ax.tick_params(which='minor',bottom=False,left=False)
        for (i,j),v in np.ndenumerate(a):
            label = annotations[i,j] if annotations is not None else dec(v)
            rgba = im.cmap(im.norm(v)) if np.isfinite(v) else (1,1,1,1)
            lum = np.dot(rgba[:3],[.2126,.7152,.0722])
            t = ax.text(j,i,label,ha='center',va='center',fontsize=8,color='white' if lum<.5 else '#202020')
            t._cell_annotation = True
        return im

    def colorbar(self, fig, im, unit, y=.09):
        ticks=np.linspace(im.norm.vmin,im.norm.vmax,5)
        cb = fig.colorbar(im,cax=fig.add_axes([.34,y,.60,.022]),orientation='horizontal',ticks=ticks)
        cb.set_ticklabels([dec(v) for v in ticks])
        cb.set_label(unit,fontsize=8)
        cb.solids.set_rasterized(False)

    def save(self, fig, stem, folder, caption, source_number, metric, denominator, group, subsets):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        min_font = 100
        for ax in fig.axes:
            for t in ax.texts:
                if getattr(t,'_cell_annotation',False):
                    b=t.get_window_extent(renderer);x,y=t.get_position()
                    p0=ax.transData.transform((x-.5,y-.5));p1=ax.transData.transform((x+.5,y+.5))
                    fits=b.width<abs(p1[0]-p0[0]) and b.height<abs(p1[1]-p0[1])
                    self.layout.append({'figure':stem,'text':t.get_text(),'fits':bool(fits)})
                    assert fits,(stem,t.get_text(),'cell overflow')
            for axis in [ax.xaxis,ax.yaxis]:
                boxes=[t.get_window_extent(renderer) for t in axis.get_ticklabels() if t.get_visible() and t.get_text()]
                assert all(not a.overlaps(b) for a,b in zip(boxes,boxes[1:])),(stem,'ticks overlap')
        for t in fig.findobj(Text):
            if t.get_visible() and t.get_text():
                b=t.get_window_extent(renderer)
                min_font=min(min_font,t.get_fontsize())
                assert b.x0>=-1 and b.y0>=-1 and b.x1<=fig.bbox.width+1 and b.y1<=fig.bbox.height+1,(stem,t.get_text(),'outside canvas')
        assert min_font>=8
        for ext in ['png','pdf']:
            p=self.run/folder/f'{stem}.{ext}'
            assert not p.exists()
            fig.savefig(p,dpi=300)
        self.figures.append(dict(stem=stem,folder=folder,caption=caption,source_number=source_number,metric=metric,
                                 denominator=denominator,group=group,subsets=','.join(subsets),width_cm=14,
                                 height_cm=fig.get_figheight()*2.54,min_font_pt=min_font,vector=True))
        plt.close(fig)
        print(f'[{len(self.figures):02d}] {folder}/{stem}',flush=True)

    def matrices(self):
        specs=[('06','accuracy_committed',['commonsense','justice'],'Akurasi committed','Akurasi (% committed)','committed','bab6','RdYlGn'),
               ('07','effective_success_rate_nonmissing',['commonsense','justice'],'Keberhasilan efektif','Benar / nonmissing (%)','nonmissing','pelengkap','RdYlGn'),
               ('08','pct_answer_wrong',['deontology','virtue'],'Keputusan Wrong','Wrong (% committed)','committed','pelengkap','RdYlBu_r')]
        for number,metric,subsets,title,unit,denominator,folder,cmap in specs:
            for tag,ids,group in GROUPS:
                selected=personas(ids)+['neutral']
                fig,axes=plt.subplots(2,1,figsize=(WIDTH,6.6))
                fig.subplots_adjust(left=.32,right=.985,bottom=.22,top=.86,hspace=.52)
                for ax,subset in zip(axes,subsets):
                    d=self.summary[self.summary.subset.eq(subset)].set_index(['model','persona'])
                    values=d[metric].unstack().reindex(index=MODELS,columns=selected)*100
                    coverage=d.commit_rate.unstack().reindex(index=MODELS,columns=selected)*100
                    ann=np.array([[dec(v)+'\nc'+dec(c) for v,c in zip(vr,cr)] for vr,cr in zip(values.to_numpy(),coverage.to_numpy())])
                    im=self.heat(ax,values,labels(ids)+['N'],NAMES,ann,cmap=cmap)
                    ax.set_title(subset.title(),pad=7)
                    for model in MODELS:
                        for persona in selected:
                            self.cells.append(dict(figure=f'{number}_{metric}_{tag}',kind='summary',metric=metric,model=model,persona=persona,subset=subset,
                                                   value_pct=values.loc[model,persona],coverage_pct=coverage.loc[model,persona]))
                fig.suptitle(f'ETHICS regional: {title}\n{group}',fontsize=10,y=.98)
                self.colorbar(fig,im,unit)
                fig.text(.5,.014,'c = committed / seluruh respons (%); N = netral.',ha='center',fontsize=8)
                caption=f'{title} pada {group}, per subset. Penyebut: {denominator}; c menunjukkan commit rate terhadap seluruh respons. '+self.abbreviations(ids)
                self.save(fig,f'{number}_{metric}_{tag}',folder,caption,number,metric,denominator,tag,subsets)
        specs=[('10','coverage','Delta commit rate','pasangan nonmissing',['commonsense','deontology','justice','virtue']),
               ('11','conditional_accuracy','Delta akurasi','pasangan sama-sama committed',['commonsense','justice']),
               ('12','effective_success','Delta keberhasilan efektif','pasangan nonmissing',['commonsense','justice']),
               ('13','stance_wrong','Delta proporsi Wrong','pasangan sama-sama committed',['deontology','virtue'])]
        for number,outcome,title,denominator,subsets in specs:
            source=self.effects[self.effects.outcome.eq(outcome)]
            limit=max(1,float(source.risk_difference_pp.abs().max()))
            for tag,ids,group in GROUPS:
                for start in range(0,len(subsets),2):
                    subset_pair=subsets[start:start+2]
                    suffix=f'_{start//2+1}' if len(subsets)>2 else ''
                    stem=f'{number}_{outcome}_{tag}{suffix}'
                    fig,axes=plt.subplots(2,1,figsize=(WIDTH,6.6))
                    fig.subplots_adjust(left=.32,right=.985,bottom=.22,top=.86,hspace=.52)
                    for ax,subset in zip(axes,subset_pair):
                        d=source[source.subset.eq(subset)].set_index(['model','persona'])
                        values=d.risk_difference_pp.unstack().reindex(index=MODELS,columns=personas(ids))
                        sig=d.significant_holm_0_05.unstack().reindex_like(values)
                        ann=np.array([[dec(v,True)+('\n*' if s else '') for v,s in zip(vr,sr)] for vr,sr in zip(values.to_numpy(),sig.to_numpy())])
                        im=self.heat(ax,values,labels(ids),NAMES,ann,limit=limit,cmap='RdBu_r')
                        ax.set_title(subset.title(),pad=7)
                        for model in MODELS:
                            for persona in personas(ids):
                                row=d.loc[(model,persona)]
                                self.cells.append(dict(figure=stem,kind='effect',metric=outcome,model=model,persona=persona,subset=subset,
                                                       value_pct=values.loc[model,persona],p_holm=row.holm_adjusted_p_value,
                                                       significant=bool(sig.loc[model,persona]),n_pairs=int(row.n_pairs),family_size=int(row.family_size)))
                    fig.suptitle(f'ETHICS regional: {title} persona − N\n{group}',fontsize=10,y=.98)
                    self.colorbar(fig,im,'Delta (poin persentase)')
                    fig.text(.5,.014,'* p Holm < 0,05; keluarga 16/model/subset/outcome.',ha='center',fontsize=8)
                    caption=f'{title} persona−N pada {group}, dalam poin persentase; {denominator}. Bintang: Exact McNemar setelah Holm <0,05, keluarga 16. '+self.abbreviations(ids)
                    self.save(fig,stem,'bab6' if number=='11' else 'pelengkap',caption,number,outcome,denominator,tag,subset_pair)

    @staticmethod
    def abbreviations(ids):
        full={'north_american':'Amerika Utara','east_asian':'Asia Timur','south_asian':'Asia Selatan','sub_saharan_african':'Afrika Sub-Sahara'}
        return '; '.join(SHORT[i]+' = '+full[i] for i in ids)+'.'

    def distributions(self):
        colors=['#B80F2E','#0086A8','#D59000','#89939F']
        records=[]
        for tag,ids,group in GROUPS:
            selected=[i+'_'+s for i in ids for s in ['explicit','implicit_avg']]+['neutral']
            ticks=[SHORT[i]+'\n'+s for i in ids for s in ['E','I']]+['N']
            for committed_only in [False,True]:
                metric='committed' if committed_only else 'total'
                stem=f'00_distribution_{metric}_{tag}'
                fig,axes=plt.subplots(3,2,figsize=(WIDTH,7.0))
                fig.subplots_adjust(left=.085,right=.99,top=.82,bottom=.14,hspace=.76,wspace=.35)
                columns=['n_answer_wrong','n_answer_acceptable']+([] if committed_only else ['abstain','missing'])
                status=['Wrong','Acceptable','Abstain','Missing'][:len(columns)]
                for ax,model,name in zip(axes.flat,MODELS,NAMES):
                    d=self.category.loc[model].loc[selected]
                    bottom=np.zeros(len(selected))
                    for col,label,color in zip(columns,status,colors):
                        values=100*ratio(d[col],d[metric])
                        ax.bar(range(5),values,bottom=bottom,width=.88,color=color,label=label)
                        for j,v in enumerate(values):
                            if np.isfinite(v) and v>=12:
                                ax.text(j,bottom[j]+v/2,dec(v),ha='center',va='center',fontsize=8,color='white' if label in ['Wrong','Acceptable'] else '#202020')
                            records.append(dict(figure=stem,model=model,category=selected[j],status=label,count=int(d[col].iloc[j]),denominator=int(d[metric].iloc[j]),denominator_type=metric,percent=v))
                        bottom+=values.to_numpy()
                    np.testing.assert_allclose(bottom,100)
                    ax.set(ylim=(0,100),yticks=[0,50,100])
                    ax.set_xticks(range(5),ticks)
                    ax.set_title(name.replace('DeepSeek R1-Distill 32B','DeepSeek R1-Distill\n32B'),fontsize=8,pad=7)
                fig.suptitle('ETHICS regional: distribusi '+('committed' if committed_only else 'seluruh respons')+'\n'+group,fontsize=10,y=.985)
                fig.legend(*axes.flat[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.91),ncol=len(columns),frameon=False)
                fig.text(.5,.028,'Sumbu y: % '+('committed' if committed_only else 'seluruh respons')+'; gabungan empat subset.\nI = gabungan hitungan I1–I3; N = netral.',ha='center',fontsize=8)
                caption='Distribusi '+('/'.join(status))+f' pada {group}, penyebut '+('committed' if committed_only else 'seluruh respons')+'. Wrong berarti keputusan tindakan salah, bukan jawaban model salah. '+self.abbreviations(ids)
                self.save(fig,stem,'pelengkap_committed' if committed_only else 'bab6',caption,'00','distribution',metric,tag,['all'])
        pd.DataFrame(records).to_csv(self.run/'tables/distribution_cells.csv',index=False)

    def reason_figures(self):
        records=[]
        for denominator,number,folder in [('abstain','04','bab6'),('nonmissing','05','pelengkap')]:
            for tag,ids,group in GROUPS:
                selected=personas(ids)+['neutral']
                for part in range(2):
                    stem=f'{number}_reasons_{denominator}_{tag}_{part+1}'
                    fig,axes=plt.subplots(3,1,figsize=(WIDTH,7.85))
                    fig.subplots_adjust(left=.405,right=.99,top=.91,bottom=.13,hspace=.43)
                    for ax,model in zip(axes,MODELS[part*3:part*3+3]):
                        d=self.reasons[self.reasons.model.eq(model)]
                        counts=d.pivot(index='abstain_reason',columns='persona_folder',values='reason_count').reindex(index=REASONS,columns=selected).fillna(0)
                        den=self.all_status.loc[model].loc[selected,denominator]
                        values=100*counts.div(den.replace(0,np.nan),axis=1)
                        ticks=[lab.replace('\n',' ')+'\n'+str(int(n)) for lab,n in zip(labels(ids)+['N'],den)]
                        im=self.heat(ax,values,ticks,list(REASONS.values()))
                        ax.tick_params(pad=3)
                        ax.set_title(NAMES[MODELS.index(model)],pad=6)
                        for reason in REASONS:
                            for persona in selected:
                                records.append(dict(figure=stem,model=model,persona=persona,reason=reason,count=int(counts.loc[reason,persona]),denominator=int(den[persona]),denominator_type=denominator,percent=values.loc[reason,persona]))
                    fig.suptitle(f'ETHICS regional: alasan / {denominator}\n{group} ({part+1}/2)',fontsize=10,y=.985)
                    self.colorbar(fig,im,f'% dari {denominator}; angka bawah kolom = n',y=.060)
                    caption=f'Alasan abstain pada {group}, kelompok model {part+1}/2. Persentase dari n respons {denominator} per skenario (empat subset); skala 0–100%. '+self.abbreviations(ids)
                    self.save(fig,stem,folder,caption,number,'reason_share',denominator,tag,['all'])
        df=pd.DataFrame(records)
        for row in self.reasons.itertuples():
            for denominator,source_pct in [('abstain',row.pct_of_abstain),('nonmissing',row.pct_of_nonmissing)]:
                actual=df[(df.model==row.model)&(df.persona==row.persona_folder)&(df.reason==row.abstain_reason)&(df.denominator_type==denominator)]
                assert len(actual)>0 and actual['count'].eq(row.reason_count).all()
                np.testing.assert_allclose(actual.percent,source_pct,equal_nan=True)
        df.to_csv(self.run/'tables/reason_cells.csv',index=False)

    def finish(self):
        # Keep the historical token asset byte-identical, with an explicit exception.
        token=ACTIVE/'figures/14_ethics_regional_completion_tokens_by_category.png'
        target=self.run/'pelengkap'/token.name
        shutil.copy2(token,target)
        with Image.open(target) as image:
            image.load(); height=14*image.height/image.width
        self.figures.append(dict(stem=target.stem,folder='pelengkap',caption='Rata-rata completion token per model dan kategori persona. Aset historis dipertahankan identik; bukan hasil render ulang.',
                                 source_number='14',metric='completion_tokens',denominator='sumber historis',group='all',subsets='all',width_cm=14,height_cm=height,min_font_pt=None,vector=False))
        cells=pd.DataFrame(self.cells)
        effect_cells=cells[cells.kind.eq('effect')]
        assert len(effect_cells)==len(self.effects)
        assert not effect_cells.duplicated(['metric','model','persona','subset']).any()
        merged=effect_cells.merge(self.effects,left_on=['metric','model','persona','subset'],right_on=['outcome','model','persona','subset'],validate='one_to_one',suffixes=('_plot','_source'))
        for a,b in [('value_pct','risk_difference_pp'),('p_holm','holm_adjusted_p_value'),('n_pairs_plot','n_pairs_source'),('family_size_plot','family_size_source')]:
            np.testing.assert_allclose(merged[a].astype(float),merged[b].astype(float),equal_nan=True)
        assert (merged.significant==merged.significant_holm_0_05).all()
        cells.to_csv(self.run/'tables/heatmap_cells.csv',index=False)
        pd.DataFrame(self.layout).to_csv(self.run/'tables/layout_audit.csv',index=False)
        for old,label in self.old:
            number=Path(old).name[:2]
            for f in self.figures:
                if f['source_number']==number:
                    self.mapping.append(dict(old_path=old,old_label=label,new_path=f['folder']+'/'+f['stem']+('.pdf' if f['vector'] else '.png'),
                                             metric=f['metric'],denominator=f['denominator'],caption=f['caption'],proposed_location='Bab VI; pelengkap bukan instruksi pemindahan' if f['folder']=='pelengkap' else 'Bab VI',
                                             status='angka cocok sumber; menunggu review visual'))
        for f in self.figures:
            if f['source_number']=='04':
                self.mapping.append(dict(old_path='',old_label='',new_path=f['folder']+'/'+f['stem']+'.pdf',metric=f['metric'],denominator=f['denominator'],caption=f['caption'],
                                         proposed_location='Tambahan Bab VI; berbeda dari gambar 05/nonmissing',status='gambar tambahan, bukan pengganti 05'))
        pd.DataFrame(self.mapping).to_csv(self.run/'mapping_integrasi.csv',index=False)
        assert set(x['old_path'] for x in self.mapping if x['old_path']) == set(p for p,_ in self.old)
        self.figures.sort(key=lambda f:(['bab6','pelengkap','pelengkap_committed'].index(f['folder']),f['stem']))
        for i,f in enumerate(self.figures,1): f['proof_page']=i
        pd.DataFrame(self.figures).to_csv(self.run/'figure_index.csv',index=False)
        self.make_proof()
        for p in self.run.glob('*/*.png'):
            with Image.open(p) as image: image.load()
        assert all(sha(p)==h for p,h in self.baseline.items())
        self.checks.update({'protected_files_unchanged':len(self.baseline),'thesis_modified':False,
                            'figure_count':len(self.figures),'new_vector_figures':sum(f['vector'] for f in self.figures),
                            'main_figures':sum(f['folder']=='bab6' for f in self.figures),'all_960_effects_preserved':True,
                            'all_annotations_fit':True,'new_min_font_pt':8,'visual_review_completed':False})
        self.write_json('checks.json',self.checks)
        self.write_json('manifest.json',{'created_at':datetime.now().isoformat(),'source_hashes':self.baseline,
                        'output_hashes':{str(p.relative_to(self.run)):sha(p) for p in self.run.rglob('*') if p.is_file()},
                        'versions':{n:importlib.metadata.version(n) for n in ['numpy','pandas','matplotlib','Pillow']},
                        'command':'.venv/Scripts/python.exe src/benchmarks/ethics/render_regional_ethics_14cm.py',
                        'statistics_recomputed':False,'thesis_modified':False})
        (self.run/'CAPTIONS.md').write_text('# Caption siap pakai\n\n'+'\n\n'.join('## '+f['stem']+'\n\n'+f['caption'] for f in self.figures)+'\n\n'+
            '## Catatan metode pendamping\n\nN diulang hanya sebagai referensi visual. I menggabungkan hitungan I1–I3; I1–I3 bukan urutan intensitas. '
            'Koreksi Holm tetap 16 per model/subset/outcome, tanpa prasyarat Q tambahan. Delta adalah poin persentase pada pasangan yang memenuhi syarat, '
            'bukan Cohen’s h atau selisih dua akurasi marginal. Alasan memiliki kategori eksklusif pada tabel sumber; NA untuk penyebut nol. '
            'n pasangan/discordant berada di tables/paired_effects_frozen.csv. Akurasi hanya Commonsense/Justice dan dibaca bersama coverage. '
            'Pembagian gambar tidak memvalidasi ulang parser maupun pairing historis.\n',encoding='utf-8')
        guide='# ETHICS regional — paket 14 cm\n\n.tex belum diubah. Angka beku dipertahankan; cetak fisik belum diperiksa.\n\n'
        for folder in ['bab6','pelengkap','pelengkap_committed']:
            guide+='## '+folder+'\n\n'
            for f in self.figures:
                if f['folder']==folder: guide+=f"- [{f['stem']}.png]({folder}/{f['stem']}.png) — proof halaman {f['proof_page']}\n"
            guide+='\n'
        guide+='PDF vektor berdampingan dengan PNG kecuali token historis. [Proof A4](proof/regional_14cm.pdf); [caption](CAPTIONS.md); [mapping integrasi](mapping_integrasi.csv). '
        guide+='Folder pelengkap tidak berarti gambar lama boleh dihapus/dipindahkan dari tesis. Sumber token tetap raster dan perlu penilaian keterbacaan tersendiri.\n'
        (self.run/'MULAI_DI_SINI.md').write_text(guide,encoding='utf-8')
        print(json.dumps(self.checks,indent=2),flush=True)

    def write_json(self,name,obj):
        (self.run/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')

    def make_proof(self):
        # A real A4 LaTeX proof preserves vector assets and complete captions.
        tex=r'''\documentclass[12pt,a4paper]{article}
\usepackage{fontspec,graphicx,caption}
\usepackage[left=4cm,right=3cm,top=3cm,bottom=3cm]{geometry}
\setmainfont{Times New Roman}
\setlength{\parindent}{0pt}
\captionsetup{font=small,labelformat=empty}
\begin{document}
'''
        def escape(s):
            return s.replace('&',r'\&').replace('%',r'\%').replace('_',r'\_')
        for f in self.figures:
            extension='pdf' if f['vector'] else 'png'
            tex+='\\noindent\\includegraphics[width=14cm]{../'+f['folder']+'/'+f['stem']+'.'+extension+'}\n'
            tex+='\\captionof{figure}{'+escape(f['caption'])+'}\n\\clearpage\n'
        tex+='\\end{document}\n'
        path=self.run/'proof/regional_14cm.tex'
        path.write_text(tex,encoding='utf-8')
        result=subprocess.run(['xelatex','-disable-installer','-interaction=nonstopmode','-halt-on-error',path.name],cwd=path.parent,capture_output=True,text=True,encoding='utf-8',errors='replace')
        if result.returncode:
            raise RuntimeError(result.stdout[-4000:]+result.stderr[-1000:])
        log=path.with_suffix('.log').read_text(encoding='utf-8',errors='replace')
        assert 'Overfull \\vbox' not in log
        result=subprocess.run(['pdfinfo',str(path.with_suffix('.pdf'))],capture_output=True,text=True)
        pages=int(re.search(r'Pages:\s+(\d+)',result.stdout).group(1))
        assert pages==len(self.figures),(pages,len(self.figures),'proof overflow')
        self.checks['proof_pages']=pages
        subprocess.run(['pdftoppm','-r','85','-png',str(path.with_suffix('.pdf')),str(path.parent/'page')],check=True,capture_output=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    job=Renderer()
    job.validate()
    job.matrices()
    job.distributions()
    job.reason_figures()
    job.finish()
