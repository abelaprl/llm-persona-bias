"""Shared print templates for Religion/Regional, with frozen display data only."""
import textwrap

import matplotlib.pyplot as plt
import numpy as np

from .gender_lm_reference_data import MODELS, MODEL_LABELS, COLORS
from .gender_lm_presentation_v2_plots import PRINT_STYLE, audit_geometry
from .gender_lm_unified_plots import _dots, _grid, _ticks, _legend, layout_signature


def signature(fig):
    return layout_signature(fig)


def heading(fig, context, title, note, legend=True):
    fig.suptitle(title, y=.978, fontsize=12, fontweight='bold')
    fig.text(.5, .936, context+' · candidate · template bersama', ha='center', fontsize=9)
    if legend:
        _legend(fig)
    fig.text(.035, .022, '\n'.join(textwrap.fill(s,85) for s in note.splitlines()),
             va='bottom', fontsize=9, linespacing=1.35)


def _validate_grid(rows, keys, expected):
    if rows.duplicated(keys).any() or set(rows[keys].itertuples(index=False,name=None)) != expected:
        raise ValueError('Incomplete, extra or duplicate display grid')


def coverage(rows, personas, labels, context):
    _validate_grid(rows, ['persona','model_id'], {(p,m) for p in personas for m in MODELS})
    if rows[['available','total','percent']].isna().any().any() or rows.total.le(0).any():
        raise ValueError('Coverage denominator unavailable')
    with plt.rc_context(PRINT_STYLE):
        fig,ax=plt.subplots(figsize=(16/2.54,26/2.54))
        values=rows.pivot(index='persona',columns='model_id',values='percent').loc[personas,MODELS]
        counts=rows.pivot(index='persona',columns='model_id',values='available').loc[personas,MODELS]
        ax.imshow(values,cmap='Blues',vmin=0,vmax=100,aspect='auto')
        for i in range(len(personas)):
            for j in range(len(MODELS)):
                ax.text(j,i,f'{int(counts.iloc[i,j])}\n{values.iloc[i,j]:.1f}%',ha='center',va='center',
                        fontsize=9,color='white' if values.iloc[i,j]>65 else '#172554')
        ax.set_yticks(range(len(personas)),[labels[p] for p in personas])
        ax.set_xticks(range(6),['L3.1','L3.3','Min','DS','GPT','Qwen'])
        for label,model in zip(ax.get_xticklabels(),MODELS):
            label.set_color(COLORS[model])
        ax.tick_params(length=0)
        fig.subplots_adjust(left=.31,right=.97,top=.88,bottom=.205)
        heading(fig,context,'U01 · Berapa teks yang dapat diukur?',
                'Angka = teks tersedia dan persentase terhadap target.\n'
                'N = netral; E = eksplisit; I1–I3 = narasi implisit. Missing bukan nol marker.')
    return fig


def length(rows, personas, labels, context, note):
    _validate_grid(rows,['persona','model_id'],{(p,m) for p in personas for m in MODELS})
    with plt.rc_context(PRINT_STYLE):
        fig,ax=plt.subplots(figsize=(16/2.54,26/2.54))
        fig.subplots_adjust(left=.31,right=.95,top=.88,bottom=.235)
        for i,model in enumerate(MODELS):
            data=rows.loc[rows.model_id.eq(model)].set_index('persona').loc[personas]
            _dots(ax,data,model,np.arange(len(personas))+(i-2.5)/6,'personas')
        ax.set_yticks(range(len(personas)),[labels[p] for p in personas])
        ax.set_ylim(len(personas)-.5,-.5)
        _ticks(ax,0,max(float(rows.upper.max())*1.08,1.))
        _grid(ax)
        ax.set_xlabel('Jumlah kata: median dan IQR')
        heading(fig,context,'U02 · Seberapa panjang responsnya?',note)
    return fig


def matrix(rows, panels, row_ids, labels, context, title, xlabel, note,
           intervals=False, mode='effect', support=True):
    """Use identical four-panel axes for profiles, effects and observed forms."""
    varying=isinstance(row_ids,dict)
    order=row_ids if varying else {p:row_ids for p in panels}
    if len(panels)>4 or len({len(order[p]) for p in panels})!=1:
        raise ValueError('Use one to four equally sized panels')
    expected={(p,r,m) for p in panels for r in order[p] for m in MODELS}
    _validate_grid(rows,['panel','row_id','model_id'],expected)
    if rows.display_unit.nunique()!=1:
        raise ValueError('Mixed units cannot share an axis')
    cols=['estimate','lower','upper'] if intervals else ['estimate']
    values=rows[cols].to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError('Infinite coordinate')
    finite=values[np.isfinite(values)]
    bound=max(float(np.abs(finite).max())*1.1,.05) if len(finite) else 1.
    with plt.rc_context(PRINT_STYLE):
        fig,axes=plt.subplots(2,2,figsize=(16/2.54,24/2.54))
        fig.subplots_adjust(left=.285,right=.94,top=.87,bottom=.25,
                            wspace=1.15 if varying else .30,hspace=.33)
        for i,ax in enumerate(axes.flat):
            if i>=len(panels):
                ax.axis('off')
                continue
            panel=panels[i]
            ids=order[panel]
            ax.set_title(panel,fontsize=10,fontweight='bold',pad=8)
            for j,model in enumerate(MODELS):
                data=rows.loc[rows.panel.eq(panel)&rows.model_id.eq(model)].set_index('row_id').loc[ids]
                _dots(ax,data,model,np.arange(len(ids))+(j-2.5)/6,panel,
                      intervals=intervals,support=support)
            if varying:
                for annotation in ax.texts:
                    if annotation.get_text()=='NA':
                        annotation.set_position((1.04,1.13))
            ax.set_yticks(range(len(ids)),[textwrap.fill(labels.get(r,r),14 if varying else 18) for r in ids])
            ax.tick_params(labelleft=varying or i%2==0)
            ax.set_ylim(len(ids)-.5,-.5)
            _grid(ax)
            for boundary in np.arange(.5,len(ids)-.5):
                ax.axhline(boundary,color='#f1f5f9',linewidth=.6,zorder=0)
            if mode=='profile':
                ax.set_xlim(0,100); ax.set_xticks([0,50,100])
            elif mode=='rank':
                ax.set_xlim(-1.03,1.03); ax.set_xticks([-1,0,1])
            elif mode=='positive':
                _ticks(ax,0,bound)
            else:
                _ticks(ax,-bound,bound)
            if mode not in {'profile','positive'}:
                ax.axvline(0,color='#94a3b8',linewidth=.7,zorder=0)
        fig.text(.61,.193,xlabel,ha='center',fontsize=9)
        heading(fig,context,title,note)
    return fig


def examples(rows, context, note):
    """Three literal same-question excerpts; no generated paraphrase."""
    if set(rows.panel)!= {'N','E','I1'} or len(rows)!=3:
        raise ValueError('Examples must contain exactly N/E/I1')
    if rows.model_id.nunique()!=1 or rows.question_id.nunique()!=1 or rows.identity.nunique()!=1:
        raise ValueError('Examples must share one model, question and identity')
    for row in rows.itertuples():
        if row.full_text[int(row.excerpt_start):int(row.excerpt_end)] != row.excerpt:
            raise ValueError('Excerpt does not match source text')
    with plt.rc_context(PRINT_STYLE):
        fig=plt.figure(figsize=(16/2.54,22/2.54))
        heading(fig,context,'U07 · Seperti apa konteks kalimatnya?',note,legend=False)
        first=rows.iloc[0]
        fig.text(.05,.882,MODEL_LABELS[first.model_id]+' · '+str(first.question_id),fontsize=9)
        for i,panel in enumerate(['N','E','I1']):
            row=rows.loc[rows.panel.eq(panel)].iloc[0]
            y=.82-i*.235
            fig.text(.05,y,panel,fontsize=10,fontweight='bold',color=COLORS[row.model_id])
            excerpt=('… ' if row.excerpt_start>0 else '')+row.excerpt+(' …' if row.excerpt_end<len(row.full_text) else '')
            fig.text(.05,y-.035,textwrap.fill(excerpt,77),fontsize=9,va='top',linespacing=1.35)
            fig.text(.05,y-.17,textwrap.fill(row.evidence_note,83),fontsize=9,va='top',color='#475569')
    return fig
