"""Render descriptive profiles and exact-span examples, without new detectors."""
from pathlib import Path
import html
import re
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from src.benchmarks.ethics.religion_ethics_gsm8k_style_data import (
    MODELS, MODEL_LABELS, MODEL_COLORS, IDENTITIES, IDENTITY_LABELS, CONDITIONS, CONDITION_LABELS, LABELS,
)
from src.benchmarks.ethics.religion_ethics_expression import (
    RELIGION_MARKERS, GROUPS, highlight_segments, highlight_html, excerpt_window,
)


def _save(fig, output, figure_id):
    for extension in ['png', 'svg']:
        fig.savefig(output / 'figures' / f'{figure_id}.{extension}', dpi=170, facecolor='white')
    plt.close(fig)


def draw_heatmap(table, identity, output):
    personas = ['neutral'] + [identity + '_' + c for c in CONDITIONS]
    selected = table[table.persona.isin(personas)].copy()
    fig, axes = plt.subplots(2, 3, figsize=(23, 17))
    fig.subplots_adjust(left=.105, right=.95, top=.89, bottom=.135, wspace=.78, hspace=.30)
    fig.suptitle(f'Profil lengkap bahasa agama | {IDENTITY_LABELS[identity]}',
                 x=.04, y=.976, ha='left', fontsize=25, weight='bold', color='#0f172a')
    fig.text(.04, .944, '20 marker kandidat · persentase respons berteks yang memuat setiap marker',
             fontsize=15, color='#475569')
    cmap = plt.get_cmap('YlGnBu').copy()
    cmap.set_bad('#d1d5db')
    for ax, model in zip(axes.flat, MODELS):
        group = selected[selected.model_id.eq(model)]
        values = group.pivot(index='marker_id', columns='persona', values='prevalence_pct').reindex(
            index=RELIGION_MARKERS, columns=personas)
        im = ax.imshow(values, vmin=0, vmax=100, cmap=cmap, aspect='auto')
        ax.set_title(MODEL_LABELS[model], color=MODEL_COLORS[model], fontsize=17, pad=13, weight='bold')
        denominators = group.groupby('persona').linguistic_available.first().reindex(personas)
        ax.set_xticks(range(5), [f'{c}\nn={int(n):,}' for c, n in zip(['N', 'E', 'I1', 'I2', 'I3'], denominators)], fontsize=10)
        ax.set_yticks(range(20), [LABELS[m] for m in RELIGION_MARKERS], fontsize=10)
        ax.tick_params(length=0)
        for y in range(20):
            for x in range(5):
                value = values.iloc[y, x]
                label = 'NA' if not np.isfinite(value) else ('<0.1' if 0 < value < .1 else f'{value:.1f}')
                ax.text(x, y, label, ha='center', va='center', fontsize=8.5,
                        color='white' if value >= 60 else '#0f172a')
        ax.set_xticks(np.arange(-.5, 5, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 20, 1), minor=True)
        ax.grid(which='minor', color='white', linewidth=.6)
        ax.tick_params(which='minor', length=0)
    cax = fig.add_axes([.30, .072, .44, .014])
    bar = fig.colorbar(im, cax=cax, orientation='horizontal')
    bar.set_label('Prevalence pada respons berteks (%) · skala identik pada semua panel', fontsize=12)
    fig.text(.04, .025, 'N diulang sebagai pembanding. Termasuk abstain berteks; label-only/kosong tidak masuk denominator. '
             'Nol ≠ NA. Multi-label, bukan komposisi 100%.', fontsize=11, color='#475569')
    figure_id = f'27_profile_20_markers_{identity}'
    _save(fig, output, figure_id)
    selected.to_csv(output / 'figure_tables' / f'{figure_id}.csv', index=False)
    return dict(figure_id=figure_id, section='expression_heatmaps',
                title=f'Profil 20 marker — {IDENTITY_LABELS[identity]}',
                png=f'figures/{figure_id}.png', svg=f'figures/{figure_id}.svg',
                table=f'figure_tables/{figure_id}.csv', rows=len(selected),
                caption='Deskriptif, bukan uji signifikansi. Angka source v3 dipertahankan. CSV memuat raw_count, '
                        'linguistic_available, total, abstain_text, word_total dan mean_response_density per 100 kata. '
                        'Satu respons dapat memuat beberapa marker; nama marker bukan bukti keyakinan model.')


def _draw_highlighted_text(ax, text, hits, start, end, fontsize=12, line_spacing=1.65):
    """Wrap proportional-font text while preserving the exact span color boundaries."""
    segments = highlight_segments(text, hits)
    renderer = ax.figure.canvas.get_renderer()
    width, height = ax.get_window_extent(renderer).width, ax.get_window_extent(renderer).height
    font = FontProperties(family='DejaVu Sans', size=fontsize)
    line_height = fontsize * ax.figure.dpi / 72 * line_spacing / height
    x, y = .025, .98
    pieces = []
    if start:
        pieces.append(('[…] ', None))
    for segment in segments:
        left, right = max(start, segment['start']), min(end, segment['end'])
        if left < right:
            pieces.append((text[left:right], segment['group']))
    if end < len(text):
        pieces.append((' […]', None))
    for value, group in pieces:
        for token in re.findall(r'\S+|\s+', value):
            token = ' ' if token.isspace() else token
            token_width = renderer.get_text_width_height_descent(token, font, False)[0] / width
            if x + token_width > .97:
                x, y = .025, y - line_height
            if token == ' ' and x == .025:
                continue
            if token_width > .945:
                raise ValueError('Unbreakable excerpt token exceeds the card width')
            if y < .025:
                raise ValueError('Excerpt exceeds the card height')
            ax.text(x, y, token, fontproperties=font, va='top', color='#17202e', transform=ax.transAxes,
                    bbox=None if group is None else dict(facecolor=GROUPS[group][1], edgecolor='none', pad=.3),
                    parse_math=False)
            x += token_width


def draw_card(triplet, output):
    first = triplet[0]
    model, identity = first['model_id'], first['card_identity']
    fig = plt.figure(figsize=(19.5, 9.5))
    fig.suptitle(f'Bahasa agama dalam konteks | {MODEL_LABELS[model]}', x=.045, y=.974,
                 ha='left', fontsize=23, weight='bold', color=MODEL_COLORS[model])
    fig.text(.045, .92, f"{IDENTITY_LABELS[identity]} · {first['subset']} · {first['question_id']} · soal sama, tiga skenario",
             fontsize=14, color='#475569')
    records = []
    for index, record in enumerate(triplet):
        x = .045 + index * .314
        persona, text, hits = record['persona'], record['final_output'], record['hits']
        condition = 'N' if persona == 'neutral' else CONDITION_LABELS[persona.split('_', 1)[1]]
        fig.text(x, .855, f'{condition}  |  {persona}', fontsize=14, weight='bold', color='#0f172a')
        fig.text(x, .817, f"{record['word_count']:,} kata penjelasan · {len({h['marker_id'] for h in hits})} marker agama",
                 fontsize=11, color='#475569')
        fig.text(x, .785, f"{record['answer_status']} · {record['surface_status']}", fontsize=10.5, color='#64748b')
        start, end = excerpt_window(text, hits, max_words=100)
        ax = fig.add_axes([x, .225, .292, .525])
        ax.set_axis_off()
        fig.canvas.draw()
        _draw_highlighted_text(ax, text, hits, start, end)
        fig.text(x, .192, f'Cuplikan karakter [{start}:{end}] / {len(text):,}; teks utuh ada di HTML.', fontsize=10, color='#64748b')
        records.append({key: record[key] for key in ['card_id', 'model_id', 'persona', 'subset', 'question_id',
                                                     'response_key', 'answer_status', 'surface_status', 'word_count',
                                                     'source_file', 'source_row', 'raw_final_sha256']})
        records[-1].update(excerpt_start=start, excerpt_end=end, excerpt=text[start:end],
                           raw_marker_hits=len(hits), distinct_markers=len({h['marker_id'] for h in hits}))
    handles = [Patch(facecolor=color, label=label) for label, color in GROUPS.values()]
    fig.legend(handles=handles, loc='lower left', bbox_to_anchor=(.04, .101), ncol=3, fontsize=11, frameon=False)
    fig.text(.045, .060, 'Warna = kategori pola kandidat. Fungsi/referen belum terverifikasi; bukan bukti keyakinan atau endorsement model.',
             fontsize=11, color='#475569')
    fig.text(.045, .030, 'Contoh purposif dengan marker beragam, bukan sampel representatif. […] = teks di luar cuplikan; spasi tampilan dirapikan.',
             fontsize=11, color='#475569')
    figure_id = '28_' + first['card_id'] + '_' + identity
    _save(fig, output, figure_id)
    pd.DataFrame(records).to_csv(output / 'figure_tables' / f'{figure_id}.csv', index=False)
    return dict(figure_id=figure_id, section='expression_cards', title=f'Kartu konteks — {MODEL_LABELS[model]} / {identity}',
                png=f'figures/{figure_id}.png', svg=f'figures/{figure_id}.svg',
                table=f'figure_tables/{figure_id}.csv', rows=len(records),
                caption='N, E dan implicit memakai model, subset, dan question_id yang sama. Pemilihan purposif '
                        'memaksimalkan ragam marker E/implicit yang terdeteksi, bukan berdasarkan GT atau akurasi. '
                        'Cuplikan dipilih untuk menampilkan ragam marker; respons dan occurrence utuh tersedia di HTML/JSON/CSV. '
                        'Tidak disorot bukan berarti tidak mengandung unsur agama.')


def full_cards_html(cards):
    legend = ' · '.join(f'<span style="background:{color};padding:4px">{label}</span>' for label, color in GROUPS.values())
    parts = ['<!doctype html><html lang="id"><head><meta charset="utf-8"><title>Kartu bahasa agama ETHICS</title>',
             '<style>.expression-cards{font:16px/1.65 system-ui,sans-serif;color:#17202e;max-width:1500px;margin:30px auto}',
             '.expression-cards summary{cursor:pointer;font-weight:700}.expression-cards article{border:1px solid #cbd5e1;padding:20px;margin:14px 0}',
             '.expression-cards .response{white-space:pre-wrap;overflow-wrap:anywhere}.expression-cards .provenance{font-size:12px;overflow-wrap:anywhere}',
             '.expression-cards mark{color:inherit;padding:0}.expression-cards table{border-collapse:collapse;font-size:12px}',
             '.expression-cards td,.expression-cards th{border:1px solid #cbd5e1;padding:5px}</style></head><body>',
             '<div class="expression-cards"><h1>Respons utuh dan sorotan marker kandidat</h1>',
             '<p>Contoh purposif, bukan estimasi frekuensi. Teks final asli; tanpa reasoning tersembunyi. '
             'Warna mengikuti kategori kamus, bukan fungsi wacana terverifikasi. Arahkan penunjuk pada sorotan untuk melihat marker.</p>',
             '<p>' + legend + '</p>']
    for record in cards:
        heading = ' | '.join(str(record[k]) for k in ['card_id', 'model_id', 'persona', 'subset', 'question_id'])
        parts.append('<article><details><summary>' + html.escape(heading) + '</summary>')
        provenance = f"{record['source_file']} | source_row (0-based)={record['source_row']} | sha256={record['raw_final_sha256']}"
        parts.append('<p class="provenance">' + html.escape(provenance) + '</p>')
        parts.append('<p>' + html.escape(record['answer_status'] + ' | ' + record['surface_status']) + '</p>')
        parts.append('<div class="response">' + highlight_html(record['final_output'], record['hits']) + '</div>')
        parts.append('<details><summary>Provenance seluruh sorotan</summary>')
        if record['hits']:
            columns = [c for c in ['marker_id', 'rule_id', 'raw_start', 'raw_end', 'raw_matched_text',
                                   'candidate_status', 'speaker_role', 'referent', 'discourse_scope', 'polarity']
                       if c in record['hits'][0]]
            parts.append(pd.DataFrame(record['hits'])[columns].to_html(index=False, escape=True))
        else:
            parts.append('<p>Tidak ada hit pada 20 marker agama. Ini bukan bukti tidak adanya bahasa agama.</p>')
        parts.append('</details></details></article>')
    parts.append('</div></body></html>')
    return ''.join(parts)


def render_expression(output, table, cards, identities=IDENTITIES):
    output = Path(output)
    (output / 'figures').mkdir()
    (output / 'figure_tables').mkdir()
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none'})
    index = []
    for identity in identities:
        index.append(draw_heatmap(table, identity, output))
        print('  Heatmap:', identity, flush=True)
    for card_id in dict.fromkeys(record['card_id'] for record in cards):
        index.append(draw_card([r for r in cards if r['card_id'] == card_id], output))
        print('  Kartu:', card_id, flush=True)
    with (output / 'response_cards.html').open('x', encoding='utf-8') as handle:
        handle.write(full_cards_html(cards))
    result = pd.DataFrame(index)
    result.to_csv(output / 'figure_index.csv', index=False)
    return result
