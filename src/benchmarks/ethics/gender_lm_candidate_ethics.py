"""Question-led ETHICS presentation; reuses frozen statistics and extraction."""
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from .gender_lm_candidate_common import PRINT_STYLE, save_panel
from .gender_lm_reference_data import (
    ROOT, SOURCE, MODELS, MODEL_LABELS, COLORS, PERSONAS, SCENARIOS,
)
from .gender_lm_reference_plots import LABELS
from .gender_lm_presentation_v2_data import (
    MARKER_GROUPS, GROUP_LABELS, source_table, view_data, coverage_table,
)
from .gender_lm_presentation_v2_plots import plot_coverage, plot_view

PACKAGE = ROOT / 'result/ETHICS/reports/gender_lm_presentation_v2_bundle_20260920_r2'
OVERVIEW_DENSITY = [
    'organization_explicit_step', 'stance_hedge', 'connective_causal',
    'self_reference_first_person', 'framing_deontological',
    'moral_evaluation_negative', 'gender_role_framing',
]
OVERVIEW_PRESENCE = [
    'ai_identity_statement', 'response_stance_refusal',
    'gender_first_person_identity_reference', 'gender_identity_relevance_denial',
]
PROFILE_MARKERS = [
    'organization_explicit_step', 'stance_hedge', 'connective_causal',
    'self_reference_first_person', 'response_stance_refusal',
    'framing_deontological', 'moral_evaluation_negative',
    'gender_first_person_identity_reference', 'gender_role_framing',
]


def input_paths():
    return [SOURCE / name for name in ['comparisons.csv', 'marker_summary.csv',
            'response_length_summary.csv', 'report_status_coverage.csv', 'registry.json']] + [
        PACKAGE / name for name in ['marker_catalog.csv', 'expression_forms.csv',
            'example_responses.csv', 'example_spans.csv', 'examples.html', 'explorer.html',
            'visual_coverage.csv']]


def implicit_zoom(rows):
    """Select implicit panels without transforming any effect or interval."""
    if set(rows.panel_id) != set(SCENARIOS):
        raise ValueError('Zoom requires the four original scenarios')
    return rows.loc[rows.panel_id.isin(SCENARIOS[1:])].copy()


def profile_selection(summary, markers):
    """Use a common descriptive prevalence scale, with raw counts alongside."""
    keys = ['scope', 'model_id', 'persona', 'marker_id']
    selected = summary.loc[summary.scope.eq('overall') & summary.marker_id.isin(markers)].copy()
    rows = selected.loc[selected.measurement.eq('presence')].copy()
    for metric, column, output in [('count', 'raw_occurrences', 'raw_count_total'),
                                   ('density', 'value', 'density_per_100_words')]:
        lookup = selected.loc[selected.measurement.eq(metric), keys + [column]]
        rows = rows.merge(lookup.rename(columns={column: output}), on=keys,
                          how='left', validate='one_to_one')
    rows['display_value'] = rows.value
    rows['display_unit'] = 'percent'
    return rows


def _legend(fig, y):
    handles = [Line2D([], [], color=COLORS[m], marker='o', linestyle='',
                      markersize=3.5, label=MODEL_LABELS[m]) for m in MODELS]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, y),
               ncol=3, frameon=False, columnspacing=.8, handletextpad=.4)


def plot_effects(rows, markers, panels, title, note, profile=False):
    """Separate model positions; a shared native-unit axis within each figure."""
    keys = ['panel_id', 'marker_id', 'model_id']
    expected = {(p, marker, m) for p in panels for marker in markers for m in MODELS}
    if rows.duplicated(keys).any() or set(rows[keys].itertuples(index=False, name=None)) != expected:
        raise ValueError('Incomplete candidate plot grid')
    value = 'display_value' if profile else 'display_effect'
    columns = [value] if profile else [value, 'display_ci_low', 'display_ci_high']
    if rows.display_unit.nunique() != 1:
        raise ValueError('Mixed native units cannot share an axis')
    numeric = rows[columns].to_numpy(dtype=float)
    if np.isinf(numeric).any():
        raise ValueError('Infinite plot coordinate')
    if not profile and (rows.display_ci_low > rows.display_ci_high).any():
        raise ValueError('Invalid interval')
    finite = numeric[np.isfinite(numeric)]
    bound = max(float(np.max(np.abs(finite))) * 1.12, .05) if len(finite) else 1.
    labels = dict(LABELS, stance_self_correction='Koreksi diri', stance_distancing='Distancing')
    height = min(26., max(18., 10. + 2.1 * len(markers)))
    with plt.rc_context(PRINT_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(16 / 2.54, height / 2.54))
        fig.subplots_adjust(left=.285, right=.945, bottom=.24, top=.88,
                            wspace=.28, hspace=.31)
        for index, ax in enumerate(axes.flat):
            if index >= len(panels):
                ax.axis('off')
                continue
            panel = panels[index]
            ax.set_gid('data-panel:' + panel)
            ax.set_title(panel, fontsize=10, fontweight='bold', pad=9)
            ax.set_axisbelow(True)
            ax.grid(axis='x', color='#e2e8f0', linewidth=.6)
            ax.axvline(0, color='#64748b', linewidth=.8)
            for boundary in np.arange(.5, len(markers) - .5):
                ax.axhline(boundary, color='#f1f5f9', linewidth=.7)
            missing = False
            for i, model in enumerate(MODELS):
                data = rows.loc[rows.panel_id.eq(panel) & rows.model_id.eq(model)].set_index('marker_id').loc[markers]
                y = np.arange(len(markers)) + (i - 2.5) * .12
                available = data[value].notna().to_numpy()
                if not profile:
                    valid_ci = available & data.display_ci_low.notna() & data.display_ci_high.notna()
                    intervals = ax.hlines(y[valid_ci], data.display_ci_low[valid_ci],
                                         data.display_ci_high[valid_ci], colors=COLORS[model], linewidth=.6)
                    intervals.set_gid(f'ci:{panel}:{model}')
                dots = ax.scatter(data[value][available], y[available], color=COLORS[model],
                                  s=10, linewidths=0, zorder=3, clip_on=False)
                dots.set_gid(f'model:{panel}:{model}')
                if (~available).any():
                    missing = True
                    ax.scatter(np.full((~available).sum(), 1.035), y[~available],
                               transform=ax.get_yaxis_transform(), color=COLORS[model],
                               marker='x', s=9, clip_on=False)
            if missing:
                ax.text(1.035, 1.025, 'NA', transform=ax.transAxes, ha='center', fontsize=9)
            ax.set_yticks(range(len(markers)), [textwrap.fill(labels[m], 18) for m in markers])
            ax.tick_params(axis='y', length=0, pad=7, labelleft=index % 2 == 0)
            ax.set_ylim(len(markers) - .5, -.5)
            ax.set_xlim((0, 100) if profile else (-bound, bound))
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3, prune=None))
            if profile:
                ax.set_xticks([0, 50, 100])
            else:
                ticks = ax.get_xticks()
                ax.set_xticks(ticks[(ticks >= -bound) & (ticks <= bound)])
            for side in ['left', 'top', 'right']:
                ax.spines[side].set_visible(False)
            ax.spines['bottom'].set_color('#94a3b8')
        fig.suptitle(title, fontsize=12, fontweight='bold', y=.98)
        fig.text(.5, .937, 'Gender ETHICS · candidate · overall', ha='center', fontsize=9)
        unit = rows.display_unit.iloc[0]
        axis_title = ('Prevalence (% penjelasan tersedia)' if profile else
                      'Man − Woman (poin persentase)' if unit == 'percentage_points' else
                      'Man − Woman (kemunculan/100 kata)')
        fig.text(.59, .185, axis_title, ha='center', fontsize=9)
        _legend(fig, .113)
        footer = note + ('\nProfil deskriptif, bukan efek berpasangan.' if profile else
                         '\nGaris = CI 95% marginal; keputusan Holm dan dukungan: CSV.')
        fig.text(.035, .025, footer, fontsize=9, va='bottom', linespacing=1.35)
    return fig


def plot_verbosity(rows):
    """Plot source medians and IQR, not inferential confidence intervals."""
    with plt.rc_context(PRINT_STYLE):
        fig, ax = plt.subplots(figsize=(16 / 2.54, 20 / 2.54))
        fig.subplots_adjust(left=.20, right=.95, top=.84, bottom=.26)
        for i, model in enumerate(MODELS):
            data = rows.loc[rows.model_id.eq(model)].set_index('persona').loc[PERSONAS]
            y = np.arange(9) + (i - 2.5) * .12
            ax.hlines(y, data.q25, data.q75, color=COLORS[model], linewidth=.75)
            ax.scatter(data['median'], y, s=12, color=COLORS[model], clip_on=False, zorder=3)
        ax.set_yticks(range(9), ['N', 'Man E', 'Woman E', 'Man I1', 'Woman I1',
                               'Man I2', 'Woman I2', 'Man I3', 'Woman I3'])
        ax.invert_yaxis()
        ax.set_xlim(left=0)
        ticks = ax.get_xticks()
        low, high = ax.get_xlim()
        ax.set_xticks(ticks[(ticks >= low) & (ticks <= high)])
        ax.grid(axis='x', color='#e2e8f0')
        ax.set_axisbelow(True)
        ax.set_xlabel('Jumlah kata: titik = median; garis = IQR')
        for side in ['top', 'right', 'left']:
            ax.spines[side].set_visible(False)
        fig.suptitle('E02 · Seberapa panjang penjelasannya?', fontsize=12, fontweight='bold', y=.97)
        fig.text(.5, .916, 'Gender ETHICS · candidate · overall', ha='center', fontsize=9)
        _legend(fig, .105)
        fig.text(.035, .03, 'Populasi: penjelasan substantif + verdict-only (0 kata).\n'
                 'Missing/technical-only tidak masuk; bukan denominator marker.\n'
                 'IQR adalah variasi panjang respons, bukan CI efek.', fontsize=9, linespacing=1.4)
    return fig


def evidence_excerpt(response, hit=None):
    text = response['explanation_text']
    if hit is None:
        return text[:240] + ('…' if len(text) > 240 else '')
    start, end = int(hit['start']), int(hit['end'])
    if text[start:end] != hit['text']:
        raise ValueError('Evidence span does not match source text')
    lo, hi = max(0, start - 85), min(len(text), end + 100)
    return ('…' if lo else '') + text[lo:hi] + ('…' if hi < len(text) else '')


def plot_example(responses, spans):
    with plt.rc_context(PRINT_STYLE):
        fig = plt.figure(figsize=(16 / 2.54, 22 / 2.54))
        fig.suptitle('E07 · Seperti apa perubahan bahasanya?', fontsize=12, fontweight='bold', y=.975)
        fig.text(.05, .92, 'DeepSeek R1 32B · commonsense · soal 0092\n'
                 'Ilustrasi purposif dari satu soal, bukan hasil yang representatif.', fontsize=9, linespacing=1.4)
        export = []
        for i, row in enumerate(responses.to_dict('records')):
            hits = spans.loc[spans.response_id.eq(row['response_id'])].sort_values('start')
            hit = hits.iloc[0].to_dict() if len(hits) else None
            excerpt = evidence_excerpt(row, hit)
            y = .82 - .235 * i
            fig.text(.05, y, row['persona'], fontsize=10, fontweight='bold', color=COLORS[MODELS[3]])
            fig.text(.05, y - .035, textwrap.fill(excerpt, 79), fontsize=9, va='top', linespacing=1.35)
            marker = hit['marker_id'] if hit else 'Tidak ada span gender candidate.'
            fig.text(.05, y - .172, marker, fontsize=9, color='#475569')
            export.append(dict(row, excerpt=excerpt, example_marker=marker))
        fig.text(.05, .045, 'Referen gender belum diaudit: dapat mengacu pada tokoh soal.\n'
                 'Konteks utuh, rule ID, dan offset tersimpan pada tabel contoh.\n'
                 'Contoh menjelaskan marker; tidak menilai benar/salah jawaban.', fontsize=9, linespacing=1.4)
    return fig, pd.DataFrame(export)


def build(root: Path):
    root = Path(root)
    summary, comparisons = source_table('marker_summary.csv'), source_table('comparisons.csv')
    rows_all = pd.concat([view_data(group, 'direct', comparisons=comparisons)
                          for group in MARKER_GROUPS], ignore_index=True)
    all_markers = [marker for markers in MARKER_GROUPS.values() for marker in markers]
    profile = profile_selection(summary, all_markers)
    profile.to_csv(root / 'tables/ethics_all_marker_profiles.csv', index=False)
    rows_all.to_csv(root / 'tables/ethics_all_direct_effects.csv', index=False)
    other = []
    for group in MARKER_GROUPS:
        for mode in ['neutral', 'implicit']:
            for gender in ['Man', 'Woman']:
                other.append(view_data(group, mode, persona=gender, comparisons=comparisons).assign(display_mode=mode))
    pd.concat(other, ignore_index=True).to_csv(root / 'tables/ethics_other_contrasts.csv', index=False)
    pd.read_csv(SOURCE / 'report_status_coverage.csv').to_csv(root / 'tables/ethics_text_status.csv', index=False)
    pd.read_csv(PACKAGE / 'marker_catalog.csv').to_csv(root / 'tables/ethics_marker_catalog.csv', index=False)
    pd.read_csv(PACKAGE / 'expression_forms.csv').to_csv(root / 'tables/ethics_expression_forms.csv', index=False)
    figures = []
    def save(stem, title, question, caption, fig, table, section='main'):
        result = save_panel(root, stem, title, question, caption, fig, table, section)
        figures.append(result)
        return result
    coverage = coverage_table(summary, comparisons)
    save('ethics_E01_coverage', 'E01 · Cakupan dan pasangan',
         'Berapa respons yang benar-benar mempunyai penjelasan?',
         'Seluruh respons tetap diinventarisasi, termasuk abstain. Marker hanya tersedia '
         'jika ada penjelasan substantif; verdict-only/missing bukan nol marker. '
         'Panel pasangan memakai soal yang sama, bukan semua respons secara terpisah.',
         plot_coverage(coverage), coverage)
    length = pd.read_csv(SOURCE / 'response_length_summary.csv')
    length = length.loc[length.scope.eq('overall')].copy()
    save('ethics_E02_verbosity', 'E02 · Panjang penjelasan', 'Apakah panjang respons ikut berubah?',
         'Median/IQR sumber mempertahankan verdict-only sebagai 0 kata. Karena itu denominator '
         'panjang respons berbeda dari profil marker; n_available dan n_substantive disimpan bersama.',
         plot_verbosity(length), length)
    profile_main = profile.loc[profile.marker_id.isin(PROFILE_MARKERS) & profile.persona.isin(
        ['neutral', 'man_explicit_1', 'woman_explicit_1'])].copy()
    profile_main['panel_id'] = profile_main.persona.map(
        {'neutral': 'N', 'man_explicit_1': 'Man E', 'woman_explicit_1': 'Woman E'})
    save('ethics_E03_absolute_profile', 'E03 · Profil absolut lintas kelompok',
         'Seberapa sering marker muncul, sebelum melihat perbedaannya?',
         'Satu indikator per kelompok dipilih di muka, bukan berdasarkan efek terbesar atau p-value. '
         'Prevalence adalah % penjelasan yang memuat marker, bukan density. '
         'Profil lengkap 41 marker × 9 persona × 6 model, density, raw count dan denominator: '
         'tables/ethics_all_marker_profiles.csv.',
         plot_effects(profile_main, PROFILE_MARKERS, ['N', 'Man E', 'Woman E'],
                      'E03 · Profil sebelum melihat perbedaan',
                      '9 indikator terpilih; seluruh 41 marker tersedia di tabel.\nN, Man E, Woman E; satu respons = satu observasi.', profile=True), profile_main)
    for number, markers in [('E04', OVERVIEW_DENSITY), ('E05', OVERVIEW_PRESENCE)]:
        data = rows_all.loc[rows_all.marker_id.isin(markers)].copy()
        title = f'{number} · Ringkasan ' + ('density' if number == 'E04' else 'prevalence')
        save(f'ethics_{number}_direct_overview', title,
             'Marker mana yang membedakan Man dan Woman pada tiap skenario?',
             'Indikator ringkasan dipilih di muka; bukan daftar temuan signifikan. Semua titik model '
             'tetap ditampilkan, termasuk nol. CI 95% marginal tidak menggantikan adjusted_p Holm. '
             'Kolom n_changed/n_pairs memisahkan perbedaan yang jarang dari yang tersebar. '
             'Seluruh 41 marker ada di sembilan gambar detail dan CSV lengkap.',
             plot_effects(data, markers, SCENARIOS, title,
                          'Titik = model; posisi vertikal dipisah tanpa menggeser nilai.\n'
                          'E = eksplisit; I1–I3 = narasi implisit, bukan urutan waktu.'), data)
    b08 = rows_all.loc[rows_all.marker_id.isin(MARKER_GROUPS['B08'])].copy()
    zoom = implicit_zoom(b08)
    maximum_e = b08.loc[b08.panel_id.eq('E'), 'display_effect'].abs().max()
    maximum_i = zoom.display_effect.abs().max()
    save('ethics_E06_implicit_zoom', 'E06 · Detail implisit B08',
         'Apakah perubahan kecil tertutup skala eksplisit?',
         f'Skala diperbesar khusus I1–I3; ketiga panel implisit memakai batas yang sama. '
         f'Efek absolut terbesar E = {maximum_e:.3f} pp, implisit = {maximum_i:.3f} pp. '
         'Tidak ada nilai atau CI yang diubah. Bandingkan gambar detail B08 untuk skala bersama E/I. '
         f'{int(zoom.display_effect.ne(0).sum())}/{len(zoom)} efek implisit bukan nol; '
         'bukan hitungan efek signifikan.',
         plot_effects(zoom, MARKER_GROUPS['B08'], SCENARIOS[1:], 'E06 · B08: pembesaran panel implisit',
                      'SKALA DIPERBESAR: hanya I1–I3, sama antar-panel ini.\n'
                      'Jangan bandingkan jarak titik dengan E04/E05/B08.'), zoom)
    responses = pd.read_csv(PACKAGE / 'example_responses.csv', keep_default_na=False)
    spans = pd.read_csv(PACKAGE / 'example_spans.csv', keep_default_na=False)
    responses.to_csv(root / 'tables/ethics_example_responses.csv', index=False)
    spans.to_csv(root / 'tables/ethics_example_spans.csv', index=False)
    selected = responses.loc[responses.example_id.eq('contoh_1')]
    selected_spans = spans.loc[spans.example_id.eq('contoh_1')]
    fig, evidence = plot_example(selected, selected_spans)
    save('ethics_E07_expression_example', 'E07 · Contoh bahasa nyata',
         'Bagaimana kategori abstrak muncul di dalam kalimat?',
         'Triplet nyata: soal dan model sama, persona berbeda. Dipilih secara purposif oleh paket '
         'sumber; bukan sampel acak atau bukti kausal satu marker. Teks lengkap dan offset '
         'disertakan di tables/ethics_example_responses.csv dan ethics_example_spans.csv.', fig, evidence)
    for group, markers in MARKER_GROUPS.items():
        data = rows_all.loc[rows_all.marker_id.isin(markers)].copy()
        save(f'ethics_detail_{group}', f'{group} · {GROUP_LABELS[group]}',
             'Bagaimana rincian semua marker dalam kelompok ini?',
             'Detail lengkap dengan satu skala bersama E/I1/I2/I3, CI 95% marginal, '
             'Holm dan dukungan pasangan pada CSV. Kelengkapan visual bukan validasi kamus.',
             plot_view(data, group, 'direct'), data, section='detail')
    catalog = pd.DataFrame([
        dict(marker_id=m, group_id=group, main_overview=m in OVERVIEW_DENSITY + OVERVIEW_PRESENCE,
             detail_figure=f'figures/ethics_detail_{group}.png',
             profile_table='tables/ethics_all_marker_profiles.csv')
        for group, markers in MARKER_GROUPS.items() for m in markers])
    catalog.to_csv(root / 'tables/ethics_visual_coverage.csv', index=False)
    return figures
