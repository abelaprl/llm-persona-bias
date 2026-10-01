"""GSM8K-style shared model dots and explicit persona comparison panels."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

from .gender_lm_compact_plots import MODELS, SHORT, style

COLORS = dict(zip(MODELS, ['#2563EB', '#7C3AED', '#059669', '#DC2626', '#D97706', '#0891B2']))
SCENARIOS = {'explicit_1': 'E', 'implicit_1': 'I1', 'implicit_2': 'I2', 'implicit_3': 'I3'}
PANELS = {'direct': ['Woman−Man'], 'neutral': ['Man−Neutral', 'Woman−Neutral'],
          'explicit': ['Man: I−E', 'Woman: I−E']}


def select_contrasts(comparisons, registry, kind):
    """Select frozen primary measurements without changing signs or statistics."""
    primary = registry.set_index('marker_id').measurement.replace({'binary': 'presence'})
    data = comparisons.loc[comparisons.scope.eq('overall') &
                           comparisons.measurement.eq(comparisons.marker_id.map(primary))].copy()
    left_suffix = data.left.str.split('_', n=1).str[1]
    right_suffix = data.right.str.split('_', n=1).str[1]
    gender = data.left.str.split('_', n=1).str[0]
    if kind == 'direct':
        keep = gender.eq('woman') & data.right.str.startswith('man_') & left_suffix.eq(right_suffix)
        panels = pd.Series('Woman−Man', index=data.index)
    elif kind == 'neutral':
        keep = gender.isin(['man', 'woman']) & data.right.eq('neutral')
        panels = gender.map({'man': 'Man−Neutral', 'woman': 'Woman−Neutral'})
    elif kind == 'explicit':
        keep = left_suffix.str.startswith('implicit_') & data.right.eq(gender + '_explicit_1')
        panels = gender.map({'man': 'Man: I−E', 'woman': 'Woman: I−E'})
    else:
        raise ValueError(f'Unknown contrast kind: {kind}')
    result = data.loc[keep].copy()
    result['scenario'] = left_suffix.loc[keep].map(SCENARIOS)
    result['panel'] = panels.loc[keep]
    if result.scenario.isna().any() or result.duplicated(['model_id', 'marker_id', 'panel', 'scenario']).any():
        raise ValueError('Unknown scenario or duplicate contrast')
    return result


def model_legend(fig, models, colors, y=.045):
    handles = [Line2D([], [], marker='o', color=colors[m], lw=0, markersize=4, label=name)
               for m, name in models.items()]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, y), ncol=3,
               frameon=False, fontsize=8.5, columnspacing=1.0, handletextpad=.35)


def contrast_figure(data, markers, kind, measurement, title, *, models=MODELS):
    """Every panel contains every model; missing estimates get a separate gutter."""
    style()
    panels = PANELS[kind]
    scenarios = ['I1', 'I2', 'I3'] if kind == 'explicit' else ['E', 'I1', 'I2', 'I3']
    height = min(26, max(14, len(markers) * .61 * len(panels) + 5.2))
    fig, axes = plt.subplots(len(panels), len(scenarios),
                              figsize=(16 / 2.54, height / 2.54), squeeze=False)
    fig.subplots_adjust(left=.255, right=.985, bottom=.23 if len(panels) == 1 else .14,
                        top=.86 if len(panels) == 1 else .90, wspace=.14, hspace=.22)
    colors = {m: COLORS.get(m, list(COLORS.values())[i % 6]) for i, m in enumerate(models)}
    rank = measurement == 'density'
    value = 'matched_rank_biserial' if rank else 'effect'
    show_na = not np.isfinite(data[value].to_numpy(float)).all()
    limits = data[['effect', 'ci_low', 'ci_high']].to_numpy(float)
    limit = 1. if rank else max(.1, float(np.nanmax(np.abs(limits))) if np.isfinite(limits).any() else 1.)
    offsets = np.linspace(-.33, .33, len(models))
    audit = dict(numeric_points=0, na_points=0, ci_segments=0, source_rows=len(data),
                 models_per_panel=len(models), panels=len(panels) * len(scenarios), plotted_estimand=value)
    for row, panel in enumerate(panels):
        for col, scenario in enumerate(scenarios):
            ax = axes[row, col]
            ax.axvline(0, color='#64748b', lw=.7, zorder=0)
            if show_na:
                ax.axvspan(limit * 1.10, limit * 1.40, color='#f1f5f9', zorder=0)
                ax.text(limit * 1.25, 1.013, 'NA', transform=ax.get_xaxis_transform(),
                        fontsize=8.5, ha='center', color='#64748b')
            for i, model in enumerate(models):
                part = data.loc[data.model_id.eq(model) & data.panel.eq(panel) & data.scenario.eq(scenario)]
                if set(part.marker_id) != set(markers) or part.marker_id.duplicated().any():
                    raise ValueError(f'Incomplete panel: {model}, {panel}, {scenario}')
                part = part.set_index('marker_id').reindex(markers)
                values = part[value].to_numpy(float)
                finite = np.isfinite(values)
                y = np.arange(len(markers)) + offsets[i]
                significance = part.adjusted_p.lt(.05).to_numpy()
                color = colors[model]
                sizes = 6 + 2 * np.log10(part.n_changed.fillna(0).to_numpy(float) + 1)
                alpha = np.where(significance, 1., .48)
                if not rank:
                    for j in range(len(markers)):
                        low, high = part.ci_low.iloc[j], part.ci_high.iloc[j]
                        if finite[j] and np.isfinite(low) and np.isfinite(high):
                            ax.plot([low, high], [y[j], y[j]], color=color, lw=.55, alpha=alpha[j], zorder=1)
                            audit['ci_segments'] += 1
                if finite.any():
                    dots = ax.scatter(values[finite], y[finite], s=sizes[finite], c=color,
                                      alpha=alpha[finite], linewidths=.2, edgecolors='white', zorder=3)
                    dots.set_gid(model)
                if (~finite).any():
                    missing = ax.scatter(np.full((~finite).sum(), limit * 1.25), y[~finite],
                                         s=8, marker='x', c=color, linewidths=.7, zorder=3)
                    missing.set_gid(model)
                audit['numeric_points'] += int(finite.sum())
                audit['na_points'] += int((~finite).sum())
            ax.set_xlim(-limit * 1.08, limit * (1.40 if show_na else 1.08))
            ax.set_ylim(len(markers) - .45, -.6)
            ax.set_xticks([-limit, 0, limit])
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.2g}'))
            ax.set_yticks(range(len(markers)), [SHORT.get(k, k) for k in markers] if col == 0 else [''] * len(markers))
            ax.tick_params(axis='y', length=0, pad=3)
            ax.grid(axis='x', color='#e2e8f0', lw=.5)
            ax.set_title(scenario if kind == 'direct' else f'{panel}\n{scenario}', fontsize=9, pad=16)
            ax.spines[['top', 'right', 'left']].set_visible(False)
    fig.suptitle(title, fontsize=12, y=.978)
    unit = 'Rank-biserial selisih densitas (−1 sampai +1)' if rank else 'Selisih prevalensi (poin persentase); CI nominal 95%'
    fig.text(.62, .16 if len(panels) == 1 else .105, unit, ha='center', fontsize=8.5)
    model_legend(fig, models, colors, y=.048 if len(panels) == 1 else .036)
    fig.text(.5, .016, 'Pekat: p Holm < 0,05 · ukuran: n_changed · × di kolom NA: tak tersedia',
             ha='center', fontsize=8.5)
    assert audit['numeric_points'] + audit['na_points'] == len(data)
    return fig, audit


def persona_profile(lengths):
    """One color per model, with Man/Woman displayed as adjacent panels."""
    style()
    fig, axes = plt.subplots(1, 2, figsize=(16 / 2.54, 12 / 2.54), sharey=True)
    fig.subplots_adjust(left=.13, right=.98, top=.78, bottom=.30, wspace=.15)
    for ax, gender in zip(axes, ['man', 'woman']):
        personas = ['neutral'] + [gender + '_' + k for k in SCENARIOS]
        for model in MODELS:
            part = lengths.loc[lengths.model_id.eq(model)].set_index('persona').reindex(personas)
            ax.plot(range(5), part['median'], color=COLORS[model], marker='o', ms=3, lw=1)
        ax.set_xticks(range(5), ['N', 'E', 'I1', 'I2', 'I3'])
        ax.set_title(gender.title())
        ax.grid(axis='y', alpha=.25)
    axes[0].set_ylabel('Median kata per respons tersedia')
    fig.suptitle('Panjang respons menurut model dan persona', fontsize=12, y=.96)
    model_legend(fig, MODELS, COLORS, y=.11)
    fig.text(.5, .035, 'N adalah baseline yang sama; verdict-only = 0 kata.\nGaris menghubungkan skenario kategorikal, bukan urutan waktu.', ha='center')
    return fig


def reference_figure(reference):
    style()
    from .gender_lm_diagnostic_data import PERSONAS, INDICATORS
    from .gender_lm_compact_plots import PERSONA_CODES
    titles = ['Semua rujukan gender', 'Pengguna eksplisit', 'Kutipan / hipotetis',
              'Belum terklasifikasi', 'Istilah sama dengan skenario']
    fig, axes = plt.subplots(3, 2, figsize=(16 / 2.54, 20 / 2.54), sharex=True, sharey=True)
    fig.subplots_adjust(left=.12, right=.98, top=.90, bottom=.18, hspace=.52, wspace=.16)
    for ax, indicator, title in zip(axes.flat, INDICATORS, titles):
        for model in MODELS:
            part = reference.loc[reference.model_id.eq(model) & reference.indicator.eq(indicator)]
            values = part.set_index('persona').reindex(PERSONAS).percent
            ax.plot(range(9), values, color=COLORS[model], marker='o', ms=2.5, lw=.8)
        ax.set_title(title, fontsize=9)
        ax.set_xticks(range(9), PERSONA_CODES, rotation=90)
        ax.tick_params(axis='x', labelbottom=True)
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=.25)
    axes.flat[-1].set_visible(False)
    fig.supylabel('% respons tersedia', fontsize=9, x=.015)
    fig.suptitle('Konteks rujukan gender: seluruh model bersama', fontsize=12, y=.97)
    model_legend(fig, MODELS, COLORS, y=.065)
    fig.text(.5, .022, 'Kategori bertumpang tindih; klasifikasi otomatis candidate.\nKesamaan istilah dengan skenario bukan bukti coreference.', ha='center')
    return fig
