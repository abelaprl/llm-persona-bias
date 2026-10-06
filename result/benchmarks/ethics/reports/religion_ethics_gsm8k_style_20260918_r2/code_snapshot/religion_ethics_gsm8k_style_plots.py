"""Render ETHICS results using identity panels and the GSM8K model palette."""
from itertools import combinations
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from src.MAIN_PARSING.ETHICS.religion_ethics_gsm8k_style_data import (
    IDENTITIES, IDENTITY_LABELS, CONDITIONS, CONDITION_LABELS, MODELS, MODEL_LABELS,
    MODEL_COLORS, PERSONAS, LABELS, RELIGION_FOCUS, GENERAL_FOCUS, STANCE_FOCUS,
    MORAL_FOCUS, THEME_FOCUS, select_contrasts,
)


class Renderer:
    def __init__(self, output):
        self.output = Path(output)
        (self.output / 'figures').mkdir()
        (self.output / 'figure_tables').mkdir()
        self.index = []
        plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                             'axes.titlesize': 16, 'axes.titleweight': 'bold',
                             'axes.spines.top': False, 'axes.spines.right': False,
                             'axes.labelsize': 12, 'svg.fonttype': 'none'})

    def finish(self, fig, figure_id, title, table, caption, section, legend=True, median=False):
        fig.suptitle(title, fontsize=22, weight='bold', y=.99)
        if legend:
            handles = [Line2D([], [], marker='o', linestyle='', color=MODEL_COLORS[m],
                              markersize=7, label=MODEL_LABELS[m]) for m in MODELS]
            if median:
                handles.append(Line2D([], [], marker='D', linestyle='', color='#17202e',
                                      markersize=7, label='Median 6 model'))
            fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .035),
                       ncol=4 if median else 3, frameon=False, fontsize=12)
        footer = 'ETHICS keseluruhan · final berteks termasuk abstain · hasil kandidat'
        fig.text(.01, .008, footer, fontsize=9, color='#64748b')
        fig.tight_layout(rect=(.015, .12 if legend else .06, .99, .94), h_pad=2.6, w_pad=2.2)
        for extension in ['png', 'svg']:
            fig.savefig(self.output / f'figures/{figure_id}.{extension}', dpi=170, facecolor='white')
        plt.close(fig)
        table.to_csv(self.output / f'figure_tables/{figure_id}.csv', index=False)
        self.index.append(dict(figure_id=figure_id, title=title, section=section,
                               png=f'figures/{figure_id}.png', svg=f'figures/{figure_id}.svg',
                               table=f'figure_tables/{figure_id}.csv', caption=caption, rows=len(table)))
        print(f'  [{len(self.index):02d}] {figure_id}', flush=True)

    def coverage(self, table):
        pivot = table.pivot(index='model_id', columns='persona', values='coverage_pct').reindex(index=MODELS, columns=PERSONAS)
        fig, ax = plt.subplots(figsize=(21, 7))
        im = ax.imshow(pivot, aspect='auto', cmap='YlGnBu', vmin=0, vmax=100)
        labels = ['N'] + [f'{IDENTITY_LABELS[i]}\n{CONDITION_LABELS[c]}' for i in IDENTITIES for c in CONDITIONS]
        ax.set_xticks(range(17), labels, fontsize=10)
        ax.set_yticks(range(6), [MODEL_LABELS[m] for m in MODELS])
        for y in range(6):
            for x in range(17):
                value = pivot.iloc[y, x]
                ax.text(x, y, f'{value:.1f}', ha='center', va='center',
                        color='white' if value > 60 else '#0f172a', fontsize=10)
        fig.colorbar(im, ax=ax, shrink=.7, label='Respons dengan teks linguistik (%)')
        self.finish(fig, '01_coverage_model_persona', 'Cakupan teks pada 17 skenario persona', table,
                    'Persentase memakai seluruh respons sebagai denominator. Label-only, final kosong dan technical-only '
                    'tetap tercatat di tabel. Nol marker berbeda dari tidak tersedianya teks.', 'coverage', legend=False)

    def condition_profile(self, table, figure_id, title, value, section, ylabel, interval=None, aligned=False):
        fig, axes = plt.subplots(2, 2, figsize=(17, 11))
        exported = []
        for ax, identity in zip(axes.flat, IDENTITIES):
            personas = ['neutral'] + [f'{identity}_{c}' for c in CONDITIONS]
            selected = table[table.persona.isin(personas)].copy()
            if aligned:
                selected = selected[selected.marker_id.eq('religion_identity_' + identity)]
            selected['panel_identity'] = identity
            exported.append(selected)
            for n, model in enumerate(MODELS):
                group = selected[selected.model_id.eq(model)].set_index('persona').reindex(personas)
                x = np.arange(5) + (n - 2.5) * .095
                ax.plot(x, group[value], marker='o', ms=5, lw=.7, alpha=.85, color=MODEL_COLORS[model])
                if interval:
                    ax.vlines(x, group[interval[0]], group[interval[1]], color=MODEL_COLORS[model], lw=1.2)
            ax.set_xticks(range(5), ['N', 'E', 'I1', 'I2', 'I3'])
            ax.set_title(IDENTITY_LABELS[identity]); ax.set_ylabel(ylabel)
            ax.grid(axis='y', color='#e2e8f0'); ax.set_axisbelow(True)
            ax.set_ylim(bottom=0)
            if aligned:
                ax.set_ylim(0, 105)
        caption = ('Teks linguistik, termasuk abstain; setiap model ditampilkan terpisah. Neutral yang sama diulang '
                   'pada panel identitas. Median dan IQR menggambarkan panjang respons, bukan CI.' if interval else
                   'Marker identitas yang sesuai dengan masing-masing panel; neutral memakai marker yang sama. '
                   'Prevalence bersyarat pada teks, bukan bukti agama atau keyakinan model. Count dan denominator di tabel.')
        self.finish(fig, figure_id, title, pd.concat(exported), caption, section)

    def dot_profiles(self, summary, markers, figure_id, title, section, metric='prevalence_text'):
        data = summary[summary.marker_id.isin(markers) & summary.condition.eq('explicit')].copy()
        fig, axes = plt.subplots(2, 2, figsize=(18, max(10, len(markers) * .67 + 4)))
        for ax, identity in zip(axes.flat, IDENTITIES):
            group = data[data.identity.eq(identity)]
            for n, model in enumerate(MODELS):
                values = group[group.model_id.eq(model)].set_index('marker_id').reindex(markers)[metric]
                ax.scatter(values, np.arange(len(markers)) + (n - 2.5) * .065,
                           color=MODEL_COLORS[model], s=28, zorder=3)
            medians = group.groupby('marker_id')[metric].agg(['median', 'count']).reindex(markers)
            median_values = medians['median'].where(medians['count'].eq(6))
            ax.scatter(median_values, np.arange(len(markers)), marker='D', s=42, color='#17202e', zorder=4)
            ax.set_yticks(range(len(markers)), [LABELS[m] for m in markers], fontsize=10)
            ax.set_title(IDENTITY_LABELS[identity]); ax.invert_yaxis()
            ax.grid(axis='x', color='#e2e8f0'); ax.set_axisbelow(True)
            ax.set_xlim(-2, 103)
            ax.set_xlabel('Prevalence pada respons berteks (%)')
        self.finish(fig, figure_id, title, data,
                    'Skenario explicit; warna membedakan model. Berlian adalah median enam model dan tidak '
                    'merupakan uji inferensial. Kemunculan marker tidak otomatis berarti endorsement. '
                    'Raw count, denominator teks dan density per 100 kata tersedia pada tabel.', section, median=True)

    def terms(self, table):
        explicit = table[table.condition.eq('explicit')]
        aggregated = explicit.groupby(['identity', 'term']).agg(
            mean_model_density=('mean_response_density', 'mean'), raw_count=('raw_count', 'sum'),
            models=('model_id', 'nunique'), n_present=('n_present', 'sum'), n_text=('n_text', 'sum')).reset_index()
        selected = aggregated.sort_values(['identity', 'mean_model_density', 'term'], ascending=[True, False, True])
        selected = selected.groupby('identity', sort=False).head(12).copy()
        fig, axes = plt.subplots(2, 2, figsize=(18, 13))
        for ax, identity in zip(axes.flat, IDENTITIES):
            group = selected[selected.identity.eq(identity)]
            ax.hlines(range(len(group)), 0, group.mean_model_density, color='#93c5fd', lw=2)
            ax.scatter(group.mean_model_density, range(len(group)), color='#2563eb', s=45)
            ax.set_yticks(range(len(group)), [textwrap.shorten(t, width=38, placeholder='…') for t in group.term], fontsize=11)
            ax.set_title(IDENTITY_LABELS[identity]); ax.invert_yaxis()
            ax.grid(axis='x', color='#e2e8f0'); ax.set_axisbelow(True)
            ax.set_xlabel('Rata-rata density respons per 100 kata\n(bobot model setara)')
            ax.set_xlim(left=0)
        self.finish(fig, '05_religious_terms_ranked_explicit', 'Istilah agama yang muncul pada skenario explicit', selected,
                    'Dua belas span ternormalisasi teratas per identitas dari occurrence religion_reference yang sudah '
                    'tersimpan; span persis lintas marker dideduplikasi. Density dihitung per respons, dirata-ratakan '
                    'per model, lalu enam model berbobot setara. Istilah dapat berupa frasa; ini bukan kamus baru.',
                    'terms', legend=False)

    def forest(self, panels, markers, figure_id, title, metric, section, columns=4):
        rows = int(np.ceil(len(panels) / columns))
        fig, axes = plt.subplots(rows, columns, figsize=(5.7 * columns, max(9, rows * (len(markers) * .42 + 2.5))), squeeze=False)
        data = pd.concat([group.assign(panel=label) for label, group in panels], ignore_index=True)
        finite = data[['effect', 'ci_low', 'ci_high']].to_numpy(float).ravel()
        finite = finite[np.isfinite(finite)]
        extent = max(float(np.max(np.abs(finite))) * 1.10, 1 if metric == 'prevalence_text' else .025) if len(finite) else 1
        for ax, (label, group) in zip(axes.flat, panels):
            for n, model in enumerate(MODELS):
                model_data = group[group.model_id.eq(model)].set_index('marker_id')
                if model_data.index.duplicated().any():
                    raise ValueError(f'Ambiguous forest rows: {figure_id}, {label}, {model}')
                for j, marker in enumerate(markers):
                    if marker not in model_data.index:
                        continue
                    record = model_data.loc[marker]
                    y = j + (n - 2.5) * .105
                    color = MODEL_COLORS[model]
                    if not np.isfinite(record.effect):
                        ax.text(1.005, y, 'NA', transform=ax.get_yaxis_transform(), color=color, fontsize=5)
                        continue
                    alpha = 1 if np.isfinite(record.p_holm) and record.p_holm < .05 else .30
                    has_ci = np.isfinite(record.ci_low) and np.isfinite(record.ci_high)
                    if has_ci:
                        ax.plot([record.ci_low, record.ci_high], [y, y], color=color, alpha=alpha, lw=1)
                    ax.scatter(record.effect, y, s=20, edgecolors=color,
                               facecolors=color if has_ci else 'white', alpha=alpha, linewidths=.9, zorder=3)
            support = group.n_pairs
            support_text = f'n pasangan: {int(support.min())}–{int(support.max())}' if len(support) else 'Tidak ada pasangan'
            ax.set_title(f'{label}\n{support_text}', fontsize=13)
            ax.set_yticks(range(len(markers)), [LABELS[m] for m in markers], fontsize=9)
            ax.set_ylim(len(markers) - .35, -.65)
            ax.axvline(0, color='#94a3b8', ls='--', lw=1)
            ax.set_xlim(-extent, extent); ax.grid(axis='x', color='#e2e8f0'); ax.set_axisbelow(True)
            ax.set_xlabel('Delta prevalence (poin persentase)' if metric == 'prevalence_text' else 'Delta density per 100 kata', fontsize=10)
        for ax in list(axes.flat)[len(panels):]:
            ax.set_visible(False)
        caption = ('Effect, CI 95% marginal, Holm dan n pasangan disalin dari inferensi v3; kiri − kanan. '
                   'Titik pekat: p Holm < 0,05; titik pudar: tidak lolos atau p tidak tersedia. Titik kosong: CI NA. '
                   'NA berarti effect tidak tersedia. Skala x sama antarpanel pada gambar ini. '
                   'Populasi text_all, mencakup abstain berteks. Daftar marker ditetapkan sebelum render; tabel lengkap tetap tersedia.')
        self.finish(fig, figure_id, title, data, caption, section)

    def sensitivity(self, table):
        populations = [('both_committed_text', 'Keduanya committed'), ('equal_subset_text', 'Bobot subset setara'),
                       ('no_quality_flag_text', 'Tanpa flag kualitas')]
        metrics = [('prevalence_text', 'Delta prevalence (pp)'), ('density', 'Delta density / 100 kata')]
        fig, axes = plt.subplots(2, 3, figsize=(19, 12))
        for row, (metric, unit) in enumerate(metrics):
            for col, (population, label) in enumerate(populations):
                ax = axes[row, col]
                group = table[table.population.eq(population) & table.metric.eq(metric)]
                valid = group[group.evaluable]
                for model in MODELS:
                    points = valid[valid.model_id.eq(model)]
                    ax.scatter(points.effect_main, points.effect_sensitivity, color=MODEL_COLORS[model], s=6, alpha=.20, rasterized=True)
                values = valid[['effect_main', 'effect_sensitivity']].to_numpy(float)
                extent = max(np.abs(values).max() * 1.05, .1) if values.size else 1
                ax.plot([-extent, extent], [-extent, extent], color='#475569', ls='--', lw=1)
                ax.set_xlim(-extent, extent); ax.set_ylim(-extent, extent)
                ax.set_title(f'{label}\n{len(valid):,} evaluable; {int(valid.direction_reversed.sum()):,} berbalik', fontsize=13)
                ax.set_xlabel('Utama: ' + unit); ax.set_ylabel('Sensitivitas: ' + unit)
                ax.grid(color='#e2e8f0'); ax.set_axisbelow(True)
        self.finish(fig, '10_sensitivity', 'Stabilitas efek pada tiga analisis sensitivitas', table,
                    'Satu titik = satu kontras model × marker × pasangan persona, bukan respons independen. '
                    'Semua kontras ditampilkan, termasuk tidak signifikan. NA dikeluarkan dari scatter dan disimpan '
                    'di tabel. Pembalikan arah tidak menyatakan pembalikan temuan signifikan.', 'sensitivity')

    def availability(self, table):
        selected = table[table.metric.eq('explanation_available')].copy()
        fig, axes = plt.subplots(2, 2, figsize=(17, 11))
        for ax, identity in zip(axes.flat, IDENTITIES):
            group = selected[selected.identity.eq(identity)]
            for n, model in enumerate(MODELS):
                data = group[group.model_id.eq(model)].set_index('condition').reindex(CONDITIONS)
                for x, (_, record) in enumerate(data.iterrows()):
                    alpha = 1 if record.p_holm < .05 else .3
                    ax.scatter(x + (n - 2.5) * .10, record.effect_pp, color=MODEL_COLORS[model], s=40, alpha=alpha)
            ax.set_title(IDENTITY_LABELS[identity]); ax.set_xticks(range(4), ['E', 'I1', 'I2', 'I3'])
            ax.axhline(0, color='#94a3b8', ls='--'); ax.set_ylim(-105, 105)
            ax.set_ylabel('Delta ketersediaan teks vs N (pp)'); ax.grid(axis='y', color='#e2e8f0'); ax.set_axisbelow(True)
        self.finish(fig, '13_differential_availability', 'Perbedaan ketersediaan teks terhadap neutral', selected,
                    'Pasangan pertanyaan sama; effect dan McNemar/Holm dari tabel availability v3. Titik pekat '
                    'menandai p Holm < 0,05. Ini mengukur availability, bukan error detektor. Sumber tidak menyediakan '
                    'CI availability, sehingga gambar tidak menambahkan interval.', 'availability')


def render_all(output, data):
    renderer = Renderer(output)
    renderer.coverage(data['coverage'])
    renderer.condition_profile(data['coverage'], '02_verbosity_by_identity_condition',
                               'Panjang respons menurut identitas dan skenario', 'median_words', 'verbosity',
                               'Jumlah kata: median dan IQR', ('q25_words', 'q75_words'))
    renderer.condition_profile(data['summary'], '03_aligned_identity_mentions',
                               'Penyebutan identitas yang sesuai dengan persona', 'prevalence_text', 'identity',
                               'Respons berteks dengan marker (%)', aligned=True)
    renderer.dot_profiles(data['summary'], RELIGION_FOCUS, '04_religious_accommodation_explicit',
                          'Akomodasi bahasa agama pada persona explicit', 'accommodation')
    renderer.terms(data['terms'])
    for identity in IDENTITIES:
        for prefix, markers, metric, section, title in [
            ('07', RELIGION_FOCUS, 'prevalence_text', 'religion_effect', 'Efek marker agama terhadap neutral'),
            ('08', GENERAL_FOCUS, 'density', 'general_effect', 'Efek marker umum terhadap neutral')]:
            table = select_contrasts(data['contrasts'], 'vs_neutral', markers, metric)
            panels = [(CONDITION_LABELS[c] + ' − N', table[table.left_persona.eq(f'{identity}_{c}')]) for c in CONDITIONS]
            renderer.forest(panels, markers, f'{prefix}_{identity}_{section}_vs_neutral',
                            f'{IDENTITY_LABELS[identity]} — {title}', metric, section)
    direct = select_contrasts(data['contrasts'], 'direct_identity', RELIGION_FOCUS)
    for condition in CONDITIONS:
        panels = []
        for left, right in combinations(IDENTITIES, 2):
            group = direct[direct.left_persona.eq(f'{left}_{condition}') & direct.right_persona.eq(f'{right}_{condition}')]
            if len(group) != len(MODELS) * len(RELIGION_FOCUS):
                raise ValueError(f'Unexpected direct-identity direction or missing source rows: {left}, {right}, {condition}')
            panels.append((f'{IDENTITY_LABELS[left]} − {IDENTITY_LABELS[right]}', group))
        renderer.forest(panels, RELIGION_FOCUS, f'09_direct_identity_{condition}',
                        f'Perbandingan langsung antaridentitas — {CONDITION_LABELS[condition]}',
                        'prevalence_text', 'direct_identity', columns=3)
    implicit = select_contrasts(data['contrasts'], 'implicit_vs_explicit', RELIGION_FOCUS)
    for identity in IDENTITIES:
        panels = [(CONDITION_LABELS[c] + ' − E', implicit[implicit.left_persona.eq(f'{identity}_{c}')]) for c in CONDITIONS[1:]]
        renderer.forest(panels, RELIGION_FOCUS, f'09b_{identity}_implicit_vs_explicit',
                        f'{IDENTITY_LABELS[identity]} — implicit dibanding explicit',
                        'prevalence_text', 'implicit_vs_explicit', columns=3)
    renderer.sensitivity(data['sensitivity'])
    renderer.dot_profiles(data['summary'], STANCE_FOCUS, '11_evaluative_stance_explicit',
                          'Bahasa sikap evaluatif pada persona explicit', 'stance')
    renderer.availability(data['availability'])
    renderer.dot_profiles(data['summary'], MORAL_FOCUS, '14_moral_basis_explicit',
                          'Dasar pembenaran moral pada persona explicit', 'moral')
    renderer.dot_profiles(data['summary'], THEME_FOCUS, '15_moral_themes_explicit',
                          'Tema moral pada persona explicit', 'moral')
    index = pd.DataFrame(renderer.index)
    index.to_csv(Path(output) / 'figure_index.csv', index=False)
    return index
