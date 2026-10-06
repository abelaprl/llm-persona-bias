"""Page-sized views of frozen results and descriptive category diversity."""
import copy
import math
import re
import textwrap
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text
import numpy as np
import pandas as pd

from src.MAIN_PARSING.ETHICS.religion_ethics_gsm8k_style_plots import Renderer
from src.MAIN_PARSING.ETHICS.religion_ethics_gsm8k_style_data import (
    MODELS, MODEL_LABELS, MODEL_COLORS, PERSONAS, IDENTITIES, IDENTITY_LABELS,
    CONDITIONS, CONDITION_LABELS,
)
from src.MAIN_PARSING.ETHICS.religion_ethics_expression import RELIGION_MARKERS, GROUPS, excerpt_window
from src.MAIN_PARSING.ETHICS.religion_ethics_expression_plots import _draw_highlighted_text
from src.MAIN_PARSING.ETHICS.religion_ethics_gsm8k_style_data import LABELS

WIDTH_CM = 16
FONT = 'DejaVu Sans'
LENGTH_BANDS = ['1–49', '50–99', '100–199', '200–399', '400+']


def diversity_rows(rows, markers):
    """Count present dictionary categories, not words or semantic diversity."""
    if rows.response_key.isna().any() or rows.response_key.duplicated().any():
        raise ValueError('Response keys must be present and unique')
    result = rows.drop(columns=[c for c in rows if '__' in c]).copy()
    available = rows.surface_status.eq('linguistic_text')
    counts = rows.loc[available, [m + '__count' for m in markers]].to_numpy(float)
    if not np.isfinite(counts).all() or (counts < 0).any() or (counts != np.floor(counts)).any():
        raise ValueError('Available text requires nonnegative integer marker counts')
    if not rows.loc[available, 'word_count'].gt(0).all():
        raise ValueError('Available text requires a positive word count')
    result['distinct_markers'] = np.nan
    result['raw_marker_count'] = np.nan
    result.loc[available, 'distinct_markers'] = (counts > 0).sum(axis=1)
    result.loc[available, 'raw_marker_count'] = counts.sum(axis=1)
    result['marker_density'] = result.raw_marker_count * 100 / result.word_count.where(available)
    result['length_band'] = pd.cut(result.word_count.where(available),
                                   [0, 49, 99, 199, 399, np.inf], labels=LENGTH_BANDS)
    return result


def diversity_tables(rows, marker_count=20):
    summaries, distribution, lengths = [], [], []
    for (model, persona), group in rows.groupby(['model_id', 'persona'], sort=False):
        text = group[group.surface_status.eq('linguistic_text')]
        n = len(text)
        key = dict(model_id=model, persona=persona)
        record = dict(key, total=len(group), linguistic_available=n,
                      text_availability_rate=n / len(group),
                      abstain_text=int(text.answer_status.eq('abstain').sum()),
                      zero_marker_text=int(text.distinct_markers.eq(0).sum()),
                      word_total=int(text.word_count.sum()), median_words=text.word_count.median(),
                      mean_distinct_markers=text.distinct_markers.mean(),
                      median_distinct_markers=text.distinct_markers.median(),
                      raw_marker_count=int(text.raw_marker_count.sum()),
                      mean_marker_density=text.marker_density.mean())
        for status in ['committed', 'abstain', 'missing']:
            record[status] = int(group.answer_status.eq(status).sum())
            record[('commit' if status == 'committed' else status) + '_rate'] = record[status] / len(group)
        summaries.append(record)
        for count in range(marker_count + 1):
            hits = int(text.distinct_markers.eq(count).sum())
            distribution.append(dict(key, distinct_markers=count, responses=hits, linguistic_available=n,
                                     percent=100 * hits / n if n else np.nan,
                                     cumulative_pct=100 * text.distinct_markers.le(count).sum() / n if n else np.nan))
        for band in LENGTH_BANDS:
            subset = text[text.length_band.eq(band)]
            lengths.append(dict(key, length_band=band, responses=len(subset),
                                mean_distinct_markers=subset.distinct_markers.mean(),
                                median_distinct_markers=subset.distinct_markers.median(),
                                raw_marker_count=int(subset.raw_marker_count.sum()),
                                mean_marker_density=subset.marker_density.mean()))
    return pd.DataFrame(summaries), pd.DataFrame(distribution), pd.DataFrame(lengths)


def audit_svg(path):
    root = ET.parse(path).getroot()
    sizes, families = [], set()
    for element in root.iter():
        style = element.attrib.get('style', '')
        sizes.extend(float(s) for s in re.findall(r'font-size:\s*([\d.]+)px', style))
        for family in re.findall(r'font-family:\s*([^;]+)', style):
            families.add(family.strip("'\" "))
    if not sizes:
        raise ValueError('SVG must preserve auditable text')
    return dict(width_cm=float(root.attrib['width'].removesuffix('pt')) / 72 * 2.54,
                height_cm=float(root.attrib['height'].removesuffix('pt')) / 72 * 2.54,
                min_font_pt=min(sizes), max_font_pt=max(sizes), font_families=sorted(families))


def title_and_footer(fig, title, footer, handles=None):
    fig.suptitle(textwrap.fill(title, 59), x=.04, y=.98, ha='left',
                 fontsize=12, fontfamily=FONT, weight='bold', color='#0f172a')
    if handles:
        fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .057),
                   ncol=2, frameon=False, fontsize=9, handlelength=1.1, columnspacing=1.2)
    fig.text(.04, .014, textwrap.fill(footer, 94), va='bottom', fontsize=8.5,
             fontfamily=FONT, color='#475569')


def save_print(fig, output, figure_id):
    """Fail on clipped text; save at the intended physical size, not tight-cropped."""
    fig.canvas.draw()
    bounds = fig.bbox
    skipped = set()
    for ax in fig.axes:
        for axis in [ax.xaxis, ax.yaxis]:
            lo, hi = sorted(axis.get_view_interval())
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if not lo - 1e-9 <= tick.get_loc() <= hi + 1e-9:
                    skipped.update([id(tick.label1), id(tick.label2)])
    for text in fig.findobj(Text):
        if id(text) not in skipped and text.get_visible() and text.get_text():
            box = text.get_window_extent(fig.canvas.get_renderer())
            if box.x0 < -1 or box.y0 < -1 or box.x1 > bounds.x1 + 1 or box.y1 > bounds.y1 + 1:
                raise ValueError(f'Clipped text in {figure_id}: {text.get_text()!r}')
    for extension in ['png', 'svg']:
        fig.savefig(Path(output) / 'figures' / f'{figure_id}.{extension}', dpi=300, facecolor='white')
    plt.close(fig)


class PrintRenderer(Renderer):
    """Preserve plotted artists and data; change only pagination and typography."""
    def finish(self, fig, figure_id, title, table, caption, section, legend=True, median=False):
        table.to_csv(self.output / f'figure_tables/{figure_id}.csv', index=False)
        active = [i for i, ax in enumerate(fig.axes) if ax.get_visible()]
        pages = [active[i:i + 2] for i in range(0, len(active), 2)]
        for page_number, selected in enumerate(pages, 1):
            page = copy.deepcopy(fig)
            for i, ax in list(enumerate(page.axes)):
                if i not in selected:
                    ax.remove()
            for old in list(page.texts):
                old.remove()
            dense = any(len(ax.get_yticks()) > 7 for ax in page.axes)
            height = (25 if dense else 21) if len(selected) == 2 else (16 if dense else 14)
            page.set_size_inches(WIDTH_CM / 2.54, height / 2.54)
            long_labels = any(len(t.get_text()) > 12 for ax in page.axes for t in ax.get_yticklabels())
            left = .37 if dense or long_labels else .16
            bottom = .22 if legend else .13
            top = .85
            gap = .16 if len(selected) == 2 else 0
            axis_height = (top - bottom - gap) / len(selected)
            for index, ax in enumerate(page.axes):
                ax.set_position([left, top - (index + 1) * axis_height - index * gap,
                                 .96 - left, axis_height])
                for text in ax.findobj(Text):
                    text.set_fontfamily(FONT)
                    text.set_fontsize(9)
                ax.title.set_fontsize(10)
                ax.title.set_text(textwrap.fill(ax.title.get_text(), 47, replace_whitespace=False))
                if not dense:
                    ax.set_ylabel(textwrap.fill(ax.get_ylabel(), 31), fontsize=9)
                for line in ax.lines:
                    if line.get_marker() not in [None, 'None', '']:
                        line.set_markersize(3)
                for collection in ax.collections:
                    if hasattr(collection, 'get_sizes') and len(collection.get_sizes()):
                        collection.set_sizes(np.minimum(collection.get_sizes(), 10))
            handles = None
            if legend:
                handles = [Line2D([], [], marker='o', linestyle='', color=MODEL_COLORS[m],
                                  markersize=4, label=MODEL_LABELS[m]) for m in MODELS]
                if median:
                    handles.append(Line2D([], [], marker='D', linestyle='', color='#17202e',
                                          markersize=4, label='Median 6 model'))
            suffix = f' | {page_number}/{len(pages)}' if len(pages) > 1 else ''
            title_and_footer(page, title + suffix,
                             'ETHICS · final berteks termasuk abstain · marker kandidat', handles)
            identifier = figure_id + f'_p{page_number:02d}'
            save_print(page, self.output, identifier)
            self.index.append(dict(figure_id=identifier, parent_id=figure_id, page=page_number,
                                   panels=len(selected), title=title, section=section,
                                   png=f'figures/{identifier}.png', svg=f'figures/{identifier}.svg',
                                   table=f'figure_tables/{figure_id}.csv', caption=caption, rows=len(table)))
            print('  Cetak:', identifier, flush=True)
        plt.close(fig)

    def coverage(self, table):
        """Transpose the wide 17-scenario overview instead of shrinking its labels."""
        pivot = table.pivot(index='persona', columns='model_id', values='coverage_pct').reindex(index=PERSONAS, columns=MODELS)
        fig, ax = plt.subplots(figsize=(WIDTH_CM / 2.54, 18 / 2.54))
        fig.subplots_adjust(left=.25, right=.97, top=.82, bottom=.18)
        ax.imshow(pivot, aspect='auto', cmap='YlGnBu', vmin=0, vmax=100)
        labels = ['N'] + [f'{IDENTITY_LABELS[i]} · {CONDITION_LABELS[c]}' for i in IDENTITIES for c in CONDITIONS]
        ax.set_yticks(range(17), labels, fontsize=9)
        short = ['Llama 3.1\n8B', 'Llama 3.3\n70B', 'Ministral\n8B', 'DeepSeek\nR1 32B', 'GPT-OSS\n20B', 'Qwen3\nNext 80B']
        ax.set_xticks(range(6), short, fontsize=8.5)
        for y in range(17):
            for x in range(6):
                value = pivot.iloc[y, x]
                ax.text(x, y, f'{value:.1f}', ha='center', va='center', fontsize=8.5,
                        color='white' if value > 60 else '#0f172a')
        title = 'Cakupan teks pada 17 skenario persona (%)'
        title_and_footer(fig, title, 'Skala warna 0–100%. Seluruh respons menjadi denominator; nol berbeda dari teks tidak tersedia.')
        identifier = '01_coverage_model_persona_p01'
        save_print(fig, self.output, identifier)
        table.to_csv(self.output / 'figure_tables/01_coverage_model_persona.csv', index=False)
        self.index.append(dict(figure_id=identifier, parent_id='01_coverage_model_persona', page=1, panels=1,
                               title=title, section='coverage', png=f'figures/{identifier}.png',
                               svg=f'figures/{identifier}.svg', table='figure_tables/01_coverage_model_persona.csv',
                               caption='Persentase respons berteks dari seluruh respons. Label-only, kosong dan technical-only tetap tercatat.',
                               rows=len(table)))


def record_figure(output, identifier, parent, title, section, table, caption, panels, page=1):
    table.to_csv(Path(output) / 'figure_tables' / f'{identifier}.csv', index=False)
    return dict(figure_id=identifier, parent_id=parent, page=page, panels=panels, title=title,
                section=section, png=f'figures/{identifier}.png', svg=f'figures/{identifier}.svg',
                table=f'figure_tables/{identifier}.csv', rows=len(table), caption=caption)


def render_expression_print(output, table, cards, identities=IDENTITIES):
    plt.rcParams.update({'font.family': FONT, 'font.size': 9, 'svg.fonttype': 'none'})
    index = []
    for identity in identities:
        personas = ['neutral'] + [identity + '_' + c for c in CONDITIONS]
        for start in range(0, len(MODELS), 2):
            models = MODELS[start:start + 2]
            selected = table[table.persona.isin(personas) & table.model_id.isin(models)].copy()
            fig, axes = plt.subplots(2, 1, figsize=(WIDTH_CM / 2.54, 25 / 2.54))
            fig.subplots_adjust(left=.385, right=.97, top=.86, bottom=.205, hspace=.46)
            cmap = plt.get_cmap('YlGnBu').copy()
            cmap.set_bad('#d1d5db')
            for ax, model in zip(axes, models):
                group = selected[selected.model_id.eq(model)]
                values = group.pivot(index='marker_id', columns='persona', values='prevalence_pct').reindex(
                    index=RELIGION_MARKERS, columns=personas)
                im = ax.imshow(values, vmin=0, vmax=100, cmap=cmap, aspect='auto')
                ax.set_title(MODEL_LABELS[model], fontsize=10, color=MODEL_COLORS[model], weight='bold', pad=9)
                ns = group.groupby('persona').linguistic_available.first().reindex(personas)
                ax.set_xticks(range(5), [f'{c}\nn={int(n):,}' for c, n in zip(['N','E','I1','I2','I3'], ns)], fontsize=8.5)
                ax.set_yticks(range(20), [LABELS[m] for m in RELIGION_MARKERS], fontsize=9)
                ax.tick_params(length=0)
                for y in range(20):
                    for x in range(5):
                        v = values.iloc[y, x]
                        label = 'NA' if not np.isfinite(v) else ('<0.1' if 0 < v < .1 else f'{v:.1f}')
                        ax.text(x, y, label, ha='center', va='center', fontsize=8.5,
                                color='white' if v >= 60 else '#0f172a')
                ax.set_xticks(np.arange(-.5, 5, 1), minor=True)
                ax.set_yticks(np.arange(-.5, 20, 1), minor=True)
                ax.grid(which='minor', color='white', linewidth=.4)
                ax.tick_params(which='minor', length=0)
            title = f'Profil 20 marker agama | {IDENTITY_LABELS[identity]} | {start // 2 + 1}/3'
            title_and_footer(fig, title,
                             'Kamus kandidat; bukan seluruh kekayaan bahasa. N diulang. Termasuk abstain berteks; '
                             'label-only/kosong tidak masuk denominator. Nol ≠ NA; multi-label.')
            fig.text(.04, .915, 'N: netral · E: explicit · I1–I3: implicit', fontsize=9, color='#475569')
            cax = fig.add_axes([.385, .105, .585, .015])
            bar = fig.colorbar(im, cax=cax, orientation='horizontal')
            bar.set_label('Respons berteks dengan marker (%)', fontsize=9)
            bar.ax.tick_params(labelsize=8.5)
            parent = f'27_profile_20_markers_{identity}'
            identifier = parent + f'_p{start // 2 + 1:02d}'
            save_print(fig, output, identifier)
            index.append(record_figure(output, identifier, parent, title, 'expression_heatmaps', selected,
                                       'Prevalence deskriptif dengan skala 0–100% yang sama pada semua halaman. '
                                       'CSV menyimpan raw count, denominator dan density per 100 kata; angka sumber tidak diubah.',
                                       2, start // 2 + 1))
            print('  Cetak:', identifier, flush=True)
    for card_id in dict.fromkeys(c['card_id'] for c in cards):
        triplet = [r for r in cards if r['card_id'] == card_id]
        first = triplet[0]
        if len(triplet) != 3:
            raise ValueError('A context card requires exactly three responses')
        fig = plt.figure(figsize=(WIDTH_CM / 2.54, 25 / 2.54))
        title = f'Bahasa agama dalam konteks | {MODEL_LABELS[first["model_id"]]}'
        title_and_footer(fig, title,
                         'Contoh purposif, bukan sampel representatif. Warna = pola kamus kandidat, bukan endorsement. '
                         '[…] = teks di luar cuplikan; spasi tampilan dirapikan.')
        fig.text(.04, .923, f'{IDENTITY_LABELS[first["card_identity"]]} · {first["subset"]} · {first["question_id"]}',
                 fontsize=9, color='#475569')
        records = []
        for n, card in enumerate(triplet):
            top = .89 - n * .247
            persona = card['persona']
            condition = 'N' if persona == 'neutral' else CONDITION_LABELS[persona.split('_', 1)[1]]
            text, hits = card['final_output'], card['hits']
            fig.text(.04, top, f'{condition} | {persona}', fontsize=10, weight='bold')
            fig.text(.04, top - .021,
                     f'{card["word_count"]:,} kata · {len({h["marker_id"] for h in hits})} jenis marker · {card["answer_status"]}',
                     fontsize=8.5, color='#475569')
            ax = fig.add_axes([.025, top - .209, .95, .17])
            ax.set_axis_off()
            fig.canvas.draw()
            start, end = excerpt_window(text, hits, max_words=100)
            _draw_highlighted_text(ax, text, hits, start, end, fontsize=9, line_spacing=1.4)
            fig.text(.04, top - .226, f'Karakter [{start}:{end}] / {len(text):,}; teks utuh dan provenance tersedia di HTML.',
                     fontsize=8.5, color='#64748b')
            row = {k: card[k] for k in ['card_id','model_id','persona','subset','question_id','response_key',
                                         'answer_status','surface_status','word_count','raw_final_sha256','source_file','source_row']}
            row.update(excerpt_start=start, excerpt_end=end, excerpt=text[start:end], raw_marker_hits=len(hits),
                       distinct_markers=len({h['marker_id'] for h in hits}))
            records.append(row)
        handles = [Patch(facecolor=color, label=label) for label, color in GROUPS.values()]
        fig.legend(handles=handles, loc='lower left', bbox_to_anchor=(.025, .063), ncol=2, fontsize=8.5,
                   frameon=False, columnspacing=.8, handlelength=1.1)
        parent = '28_' + card_id + '_' + first['card_identity']
        identifier = parent + '_p01'
        save_print(fig, output, identifier)
        index.append(record_figure(output, identifier, parent, title, 'expression_cards', pd.DataFrame(records),
                                   'Triplet dan cuplikan sama dengan versi terdahulu, pada soal/model/subset yang sama. '
                                   'Tidak disorot bukan berarti tidak mengandung bahasa agama. Teks utuh, offset dan hash tetap tersedia.', 3))
        print('  Cetak:', identifier, flush=True)
    return index


def render_diversity(output, distribution, marker_count=20, identities=IDENTITIES):
    plt.rcParams.update({'font.family': FONT, 'font.size': 9, 'svg.fonttype': 'none'})
    colors = ['#475569', '#2563eb', '#d97706', '#059669', '#7c3aed']
    styles = ['-', '-', '--', '-.', ':']
    index = []
    for identity in identities:
        personas = ['neutral'] + [identity + '_' + c for c in CONDITIONS]
        selected = distribution[distribution.persona.isin(personas)].copy()
        fig, axes = plt.subplots(3, 2, figsize=(WIDTH_CM / 2.54, 23 / 2.54))
        fig.subplots_adjust(left=.12, right=.97, top=.84, bottom=.23, hspace=.55, wspace=.38)
        for ax, model in zip(axes.flat, MODELS):
            for persona, color, style in zip(personas, colors, styles):
                group = selected[selected.model_id.eq(model) & selected.persona.eq(persona)]
                ax.step(group.distinct_markers, group.cumulative_pct, where='post', color=color, ls=style, lw=1.3)
            ax.set_title(MODEL_LABELS[model], color=MODEL_COLORS[model], fontsize=10, weight='bold')
            ax.set_xlim(0, marker_count)
            ax.set_ylim(0, 103)
            ax.set_xticks([0, 5, 10, 15, 20] if marker_count == 20 else range(marker_count + 1))
            ax.set_yticks([0, 50, 100])
            ax.tick_params(labelsize=8.5)
            ax.grid(color='#e2e8f0')
            ax.set_axisbelow(True)
        fig.text(.50, .174, 'Jumlah jenis marker berbeda per respons', ha='center', fontsize=9)
        fig.text(.023, .54, 'Respons dengan jumlah jenis ≤ x (%)', rotation=90, va='center', fontsize=9)
        title = f'Keragaman kategori marker agama | {IDENTITY_LABELS[identity]}'
        handles = [Line2D([], [], color=c, ls=s, label=l) for c,s,l in zip(colors, styles, ['N','E','I1','I2','I3'])]
        title_and_footer(fig, title,
                         'Deskriptif, bukan uji efek persona. Dipengaruhi panjang respons; ringkasan per rentang panjang '
                         'tersedia di CSV. Nol hit termasuk; tanpa teks tidak diberi nilai keragaman.', handles)
        fig.text(.04, .905, 'Kurva kumulatif; lebih ke kanan = lebih banyak jenis terdeteksi.', fontsize=9)
        parent = '29_diversity_' + identity
        identifier = parent + '_p01'
        save_print(fig, output, identifier)
        index.append(record_figure(output, identifier, parent, title, 'diversity', selected,
                                   'Jumlah kategori berbeda dari 20 marker kandidat, bukan keragaman kosakata atau kualitas bahasa. '
                                   'Satu respons = satu observasi, termasuk abstain berteks dan nol hit. '
                                   'N diulang. Tidak ada pengujian signifikansi baru; panjang respons dan availability dapat berbeda.', 6))
        print('  Cetak:', identifier, flush=True)
    return index
