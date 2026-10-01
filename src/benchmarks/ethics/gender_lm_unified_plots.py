"""One visual grammar per question, shared by both benchmarks."""
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

from .gender_lm_reference_data import MODELS, MODEL_LABELS, COLORS, PERSONAS, SCENARIOS
from .gender_lm_reference_plots import LABELS as ETHICS_LABELS
from .gender_lm_candidate_gsm8k import OUTCOME_LABELS
from .gender_lm_presentation_v2_plots import PRINT_STYLE

LABELS = {**ETHICS_LABELS, **OUTCOME_LABELS}
PERSONA_LABELS = ['N', 'Man E', 'Woman E', 'Man I1', 'Woman I1',
                  'Man I2', 'Woman I2', 'Man I3', 'Woman I3']


def layout_signature(fig):
    """Inspect actual axes and artists, rather than trusting template metadata."""
    return dict(size_cm=[round(float(x) * 2.54, 4) for x in fig.get_size_inches()],
                panels=[dict(position=[round(float(x), 5) for x in ax.get_position().bounds],
                             title=ax.get_title(), visible=ax.axison,
                             interval_models=[c.get_gid().split(':')[-1] for c in ax.collections
                                              if (c.get_gid() or '').startswith('interval:')],
                             model_ids=[c.get_gid().split(':')[-1] for c in ax.collections
                                        if (c.get_gid() or '').startswith('model:')]) for ax in fig.axes])


def assert_matching_layouts(first, second):
    if first != second:
        raise ValueError('Corresponding benchmark figures have different visual layouts')


def _legend(fig):
    handles = [Line2D([], [], color=COLORS[m], marker='o', linestyle='', markersize=3.5,
                      label=MODEL_LABELS[m]) for m in MODELS]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .105),
               ncol=3, frameon=False, columnspacing=.8, handletextpad=.4)


def _heading(fig, benchmark, title, note, legend=True, subtitle_y=.942):
    fig.suptitle(title, y=.98, fontsize=12, fontweight='bold')
    fig.text(.5, subtitle_y, f'Gender {benchmark} · candidate · template bersama', ha='center', fontsize=9)
    if legend:
        _legend(fig)
    lines = '\n'.join(textwrap.fill(line, 85) for line in note.splitlines())
    fig.text(.035, .025, lines, fontsize=9, va='bottom', linespacing=1.35)


def _grid(ax):
    ax.grid(axis='x', color='#e2e8f0', linewidth=.6)
    ax.set_axisbelow(True)
    for side in ['left', 'right', 'top']:
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color('#94a3b8')
    ax.tick_params(axis='y', length=0, pad=7)


def _ticks(ax, low, high):
    ax.set_xlim(low, high)
    ticks = MaxNLocator(nbins=3, min_n_ticks=3).tick_values(low, high)
    ax.set_xticks(ticks[(ticks >= low) & (ticks <= high)])


def _dots(ax, data, model, y, panel, intervals=True, support=False):
    valid = data.estimate.notna().to_numpy()
    if intervals:
        ci = valid & data.lower.notna().to_numpy() & data.upper.notna().to_numpy()
        if (data.lower > data.upper).any():
            raise ValueError('Reversed interval')
        lines = ax.hlines(y[ci], data.lower[ci], data.upper[ci], color=COLORS[model], linewidth=.7)
        lines.set_gid(f'interval:{panel}:{model}')
    colors = [COLORS[model] if not support or n >= 20 else 'white'
              for n in (data.changed[valid] if support else np.full(valid.sum(), 20))]
    dots = ax.scatter(data.estimate[valid], y[valid], s=12, facecolors=colors,
                      edgecolors=COLORS[model], linewidths=.7, zorder=3, clip_on=False)
    dots.set_gid(f'model:{panel}:{model}')
    if (~valid).any():
        missing = ax.scatter(np.full((~valid).sum(), 1.04), y[~valid], marker='x', s=10,
                             color=COLORS[model], transform=ax.get_yaxis_transform(), clip_on=False)
        missing.set_gid(f'missing:{panel}:{model}')
        if not any(t.get_text() == 'NA' for t in ax.texts):
            ax.text(1.04, 1.03, 'NA', transform=ax.transAxes, ha='center', fontsize=9)


def plot_coverage(rows, benchmark, note):
    with plt.rc_context(PRINT_STYLE):
        fig, ax = plt.subplots(figsize=(16 / 2.54, 13 / 2.54))
        percent = rows.pivot(index='model_id', columns='persona', values='percent').loc[MODELS, PERSONAS]
        count = rows.pivot(index='model_id', columns='persona', values='available').loc[MODELS, PERSONAS]
        ax.imshow(percent, cmap='Blues', vmin=0, vmax=100, aspect='auto')
        for i in range(6):
            for j in range(9):
                value = percent.iloc[i, j]
                ax.text(j, i, f'{int(count.iloc[i, j])}\n{value:.1f}%', ha='center', va='center',
                        fontsize=9, color='white' if value > 65 else '#172554')
        ax.set_yticks(range(6), [MODEL_LABELS[m] for m in MODELS])
        for label, model in zip(ax.get_yticklabels(), MODELS):
            label.set_color(COLORS[model])
        ax.set_xticks(range(9), ['N', 'M-E', 'W-E', 'M-I1', 'W-I1', 'M-I2', 'W-I2', 'M-I3', 'W-I3'],
                      rotation=45, ha='right')
        ax.tick_params(length=0)
        fig.subplots_adjust(left=.24, right=.98, top=.82, bottom=.23)
        _heading(fig, benchmark, 'U01 · Berapa teks yang dapat diukur?', note, legend=False, subtitle_y=.90)
    return fig


def plot_length(rows, benchmark, note):
    with plt.rc_context(PRINT_STYLE):
        fig, ax = plt.subplots(figsize=(16 / 2.54, 22 / 2.54))
        fig.subplots_adjust(left=.205, right=.95, top=.86, bottom=.255)
        for i, model in enumerate(MODELS):
            data = rows.loc[rows.model_id.eq(model)].set_index('persona').loc[PERSONAS]
            _dots(ax, data, model, np.arange(9) + (i-2.5)*.12, 'personas')
        ax.set_yticks(range(9), PERSONA_LABELS)
        ax.set_ylim(8.5, -.5)
        high = max(float(rows.upper.max())*1.08, 1.)
        _ticks(ax, 0, high)
        _grid(ax)
        ax.set_xlabel('Jumlah kata: median dan IQR')
        _heading(fig, benchmark, 'U02 · Seberapa panjang responsnya?', note)
    return fig


def _matrix(rows, markers, panels, benchmark, title, xlabel, note, profile=False, rank=False, intervals=True):
    expected = {(p, m, model) for p in panels for m in markers for model in MODELS}
    keys = ['panel', 'marker_id', 'model_id']
    if rows.duplicated(keys).any() or set(rows[keys].itertuples(index=False, name=None)) != expected:
        raise ValueError('Incomplete or duplicated model/marker/panel grid')
    if rows.display_unit.nunique() != 1:
        raise ValueError('Mixed units cannot share one effect axis')
    columns = ['estimate'] if profile or not intervals else ['estimate', 'lower', 'upper']
    values = rows[columns].to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError('Infinite coordinate')
    finite = values[np.isfinite(values)]
    bound = max(float(np.abs(finite).max())*1.10, .05) if len(finite) else 1.
    with plt.rc_context(PRINT_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(16 / 2.54, 24 / 2.54))
        fig.subplots_adjust(left=.275, right=.94, top=.87, bottom=.24, wspace=.29, hspace=.32)
        for i, ax in enumerate(axes.flat):
            if i >= len(panels):
                ax.axis('off')
                continue
            panel = panels[i]
            ax.set_title(panel, fontsize=10, fontweight='bold', pad=8)
            for j, model in enumerate(MODELS):
                data = rows.loc[rows.model_id.eq(model) & rows.panel.eq(panel)].set_index('marker_id').loc[markers]
                _dots(ax, data, model, np.arange(len(markers))+(j-2.5)*.12, panel,
                      intervals=intervals and not profile and not rank, support=not profile)
            ax.set_yticks(range(len(markers)), [textwrap.fill(LABELS[m], 18) for m in markers])
            ax.tick_params(labelleft=i%2 == 0)
            ax.set_ylim(len(markers)-.5, -.5)
            _grid(ax)
            ax.axvline(0, color='#94a3b8', linewidth=.7, zorder=0)
            for boundary in np.arange(.5, len(markers)-.5):
                ax.axhline(boundary, color='#f1f5f9', linewidth=.6, zorder=0)
            if profile:
                ax.set_xlim(0, 100)
                ax.set_xticks([0, 50, 100])
            elif rank:
                ax.set_xlim(-1.03, 1.03)
                ax.set_xticks([-1, 0, 1])
            else:
                _ticks(ax, -bound, bound)
        fig.text(.60, .188, xlabel, fontsize=9, ha='center')
        _heading(fig, benchmark, title, note)
    return fig


def plot_effect(rows, markers, benchmark, code, xlabel, note, rank=False, panels=None, intervals=True):
    return _matrix(rows, markers, SCENARIOS if panels is None else panels, benchmark,
                   code + ' · Bagaimana perbedaan Man–Woman?', xlabel, note, rank=rank, intervals=intervals)


def plot_profile(rows, markers, benchmark, note):
    return _matrix(rows, markers, ['N', 'Man E', 'Woman E'], benchmark,
                   'U03 · Seberapa umum marker ini?', 'Prevalence (% respons tersedia)', note, profile=True)


def plot_terms(rows, benchmark, note):
    terms = rows.sort_values('term_order').term.drop_duplicates().tolist()
    with plt.rc_context(PRINT_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(16 / 2.54, 20 / 2.54), sharex=True, sharey=True)
        fig.subplots_adjust(left=.29, right=.94, top=.87, bottom=.255, wspace=.29)
        for ax, panel in zip(axes, ['Man E', 'Woman E']):
            for i, model in enumerate(MODELS):
                data = rows.loc[rows.model_id.eq(model) & rows.panel.eq(panel)].set_index('term').loc[terms]
                _dots(ax, data, model, np.arange(len(terms))+(i-2.5)*.12, panel, intervals=False)
            ax.set_title(panel, fontsize=10, fontweight='bold')
            ax.set_yticks(range(len(terms)), [textwrap.fill(t, 18) for t in terms])
            ax.set_ylim(len(terms)-.5, -.5)
            _ticks(ax, 0, max(float(rows.estimate.max())*1.12, .05))
            _grid(ax)
        fig.text(.6, .19, 'Kemunculan per 100 kata', ha='center', fontsize=9)
        _heading(fig, benchmark, 'U06 · Kata dan ungkapan apa yang muncul?', note)
    return fig


def plot_examples(rows, benchmark, note):
    with plt.rc_context(PRINT_STYLE):
        fig = plt.figure(figsize=(16 / 2.54, 22 / 2.54))
        _heading(fig, benchmark, 'U07 · Seperti apa konteks kalimatnya?', note, legend=False)
        first = rows.iloc[0]
        fig.text(.05, .885, MODEL_LABELS[first.model_id] + ' · ' + str(first.question_id), fontsize=9)
        for i, panel in enumerate(['N', 'Man E', 'Woman E']):
            row = rows.loc[rows.panel.eq(panel)].iloc[0]
            y = .82 - i*.235
            fig.text(.05, y, panel, fontsize=10, fontweight='bold', color=COLORS[row.model_id])
            fig.text(.05, y-.035, textwrap.fill(row.excerpt, 78), fontsize=9, va='top', linespacing=1.35)
            fig.text(.05, y-.17, textwrap.fill(row.evidence_note, 83), fontsize=9, va='top', color='#475569')
    return fig
