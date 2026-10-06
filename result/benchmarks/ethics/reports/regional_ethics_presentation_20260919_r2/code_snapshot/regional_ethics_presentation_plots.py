"""Print-sized Regional figures, with explicit units and source tables."""
import json
from pathlib import Path
import re
import textwrap
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager, ft2font
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from .regional_ethics_presentation_data import (POLICY, IDENTITIES, IDENTITY_LABELS, CONDITIONS, CONDITION_LABELS,
    MODELS, MODEL_LABELS, MODEL_COLORS, persona_parts, sha256, write_json)
from .regional_ethics_presentation_stats import presentation_effect
from .religion_ethics_print import save_print, audit_svg

LABELS = dict(zip([
    'culture_region_north_american', 'culture_region_east_asian', 'culture_region_south_asian', 'culture_region_sub_saharan_african',
    'culture_geocultural_broader_reference', 'culture_general_reference', 'culture_meta_framing', 'culture_relativist_framing',
    'regional_persona_direct_address', 'regional_perspective_adoption', 'culture_collectivist_communal_orientation',
    'culture_individual_autonomy_orientation', 'culture_family_obligation_orientation', 'culture_community_obligation_orientation',
    'culture_reciprocity_interdependence_orientation', 'culture_hospitality_generosity_orientation', 'culture_elder_respect_orientation',
    'culture_harmony_face_orientation', 'culture_hierarchy_authority_orientation', 'culture_named_concept_reference',
    'culture_named_practice_reference', 'culture_warmup_name_echo', 'culture_identity_irrelevance'], [
    'Label Amerika Utara', 'Label Asia Timur', 'Label Asia Selatan', 'Label Afrika Sub-Sahara', 'Rujukan geokultural lain',
    'Rujukan budaya umum', 'Framing meta-budaya', 'Framing relativis', 'Sapaan persona regional', 'Adopsi perspektif regional',
    'Orientasi komunal', 'Otonomi individu', 'Kewajiban keluarga', 'Kewajiban komunitas', 'Resiprositas/interdependensi',
    'Hospitalitas/kemurahan hati', 'Hormat kepada tetua', 'Harmoni/face', 'Hierarki/otoritas', 'Konsep bernama',
    'Praktik bernama', 'Pengulangan nama persona', 'Identitas tidak relevan']))
LABELS.update({'stance_hedge': 'Hedge', 'stance_booster': 'Booster', 'stance_contextual_hedge': 'Batas konteks',
               'stance_distancing': 'Distancing', 'framing_deontological': 'Framing kewajiban',
               'framing_consequentialist': 'Framing konsekuensi', 'framing_virtue_ethics': 'Framing kebajikan',
               'framing_cultural': 'Framing budaya', 'framing_religious': 'Framing agama',
               'affect_empathy_validation': 'Empati/validasi', 'word_count': 'Panjang respons'})
GENERAL = ['stance_hedge', 'stance_booster', 'stance_contextual_hedge', 'stance_distancing',
           'framing_deontological', 'framing_consequentialist', 'framing_virtue_ethics', 'framing_cultural', 'framing_religious', 'affect_empathy_validation']
SCENARIOS = ['neutral'] + CONDITIONS
FOOTER = 'ETHICS · final berpenjelasan termasuk abstain · candidate, belum audit manusia'


def paginate(items, size):
    return [items[i:i + size] for i in range(0, len(items), size)]


def load_tables(bundle):
    bundle = Path(bundle)
    names = ['coverage', 'regional_profiles', 'all_marker_profiles', 'diversity_summary', 'diversity_distribution', 'diversity_by_length']
    tables = {name: pd.read_csv(bundle / 'tables' / f'{name}.csv') for name in names}
    for name in ['paired_contrasts', 'old_new_comparison']:
        tables[name] = pd.read_csv(bundle / 'tables/statistics' / f'{name}.csv.gz')
    for name in ['subset_weights', 'pair_composition']:
        tables[name] = pd.read_csv(bundle / 'tables/statistics' / f'{name}.csv')
    tables['context_overlap'] = pd.read_csv(bundle / 'evidence/context_overlap.csv')
    tables['named_terms'] = pd.read_csv(bundle / 'evidence/named_terms.csv')
    tables['context_response_summary'] = pd.read_csv(bundle / 'evidence/context_response_summary.csv')
    tables['cards'] = json.loads((bundle / 'evidence/response_cards.json').read_text(encoding='utf-8'))
    return tables


def figure(title, n=1, height=20, left=.20, footer=FOOTER):
    fig, axes = plt.subplots(n, 1, figsize=(16 / 2.54, height / 2.54), squeeze=False)
    fig.subplots_adjust(left=left, right=.94, top=.87, bottom=.16, hspace=.48)
    fig.suptitle(textwrap.fill(title, 58), x=.04, y=.98, ha='left', fontsize=12, weight='bold', color='#0f172a')
    fig.text(.04, .025, textwrap.fill(footer, 90), fontsize=8.5, va='bottom', color='#475569')
    return fig, axes[:, 0]


class Renderer:
    def __init__(self, output):
        plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.labelsize': 9,
                             'xtick.labelsize': 9, 'ytick.labelsize': 9, 'svg.fonttype': 'none', 'text.parse_math': False})
        self.output = Path(output)
        for name in ['figures', 'figure_tables']:
            (self.output / name).mkdir(parents=True, exist_ok=True)
        self.index = []

    def save(self, fig, identifier, table, title, section, metric, panels, page=1, parent=None, source='tables'):
        frame = table.copy()
        frame['source_row'] = table.index
        frame['analysis_policy_version'] = POLICY
        relative = f'figure_tables/{identifier}.csv'
        frame.to_csv(self.output / relative, index=False)
        with warnings.catch_warnings():
            warnings.filterwarnings('error', message='Glyph .* missing from font')
            save_print(fig, self.output, identifier)
        audit = audit_svg(self.output / f'figures/{identifier}.svg')
        self.index.append(dict(figure_id=identifier, parent_id=parent or identifier, page=page, panels=json.dumps(panels),
                               title=title, section=section, metric=metric, policy=POLICY, source_file=source,
                               source_row='see keyed CSV', data_file=relative, png=f'figures/{identifier}.png', svg=f'figures/{identifier}.svg',
                               png_sha256=sha256(self.output / f'figures/{identifier}.png'),
                               svg_sha256=sha256(self.output / f'figures/{identifier}.svg'), min_font_pt=audit['min_font_pt']))
        print('[figure]', identifier, flush=True)

    def profiles(self, table, section='regional_profile', marker_ids=None):
        markers = marker_ids or list(table.marker_id.drop_duplicates())
        models = [m for m in MODELS if m in set(table.model_id)] or list(table.model_id.unique())
        for identity in IDENTITIES:
            if not table.persona.str.startswith(identity + '_').any():
                continue
            personas = ['neutral'] + [identity + '_' + c for c in CONDITIONS]
            for page, selected in enumerate(paginate(models, 2), 1):
                fig, axes = figure(f'{IDENTITY_LABELS[identity]} — profil marker', len(selected), 25, .43,
                                   FOOTER + '\nAngka = prevalensi (%); abu-abu/NA = tanpa penjelasan. N diulang hanya untuk tampilan.')
                used = table[table.model_id.isin(selected) & table.persona.isin(personas) & table.marker_id.isin(markers)]
                for ax, model in zip(axes, selected):
                    part = used[used.model_id.eq(model)]
                    values = part.pivot(index='marker_id', columns='persona', values='prevalence_pct').reindex(index=markers, columns=personas).to_numpy(float)
                    cmap = plt.get_cmap('Blues').copy()
                    cmap.set_bad('#d1d5db')
                    ax.imshow(values, aspect='auto', vmin=0, vmax=100, cmap=cmap)
                    ax.set_xticks(range(5), [CONDITION_LABELS[c] for c in SCENARIOS])
                    ax.set_yticks(range(len(markers)), [LABELS.get(m, m.replace('_', ' ')) for m in markers])
                    ax.tick_params(axis='both', labelsize=8.5, length=0)
                    ax.set_title(MODEL_LABELS.get(model, model), fontsize=10, color=MODEL_COLORS.get(model, '#17202e'))
                    for i in range(len(markers)):
                        for j in range(5):
                            value = values[i, j]
                            ax.text(j, i, 'NA' if not np.isfinite(value) else f'{value:.1f}', ha='center', va='center', fontsize=8.5,
                                    color='white' if np.isfinite(value) and value > 55 else '#17202e')
                self.save(fig, f'{section}_{identity}_p{page:02d}', used, 'Profil prevalensi marker', section, 'prevalence_pct', selected,
                          page, f'{section}_{identity}', 'tables/regional_profiles.csv' if section == 'regional_profile' else 'tables/all_marker_profiles.csv')

    def coverage(self, table):
        for kind, columns, labels, colors in [
            ('coverage', ['n_text', 'surface_label_only', 'surface_technical_only', 'surface_missing'], ['Penjelasan', 'Label saja', 'Teknis', 'Missing'], ['#2563eb', '#f59e0b', '#9ca3af', '#dc2626']),
            ('answer_status', ['committed', 'abstain', 'missing'], ['Committed', 'Abstain', 'Missing'], ['#059669', '#f59e0b', '#dc2626'])]:
            for page, models in enumerate(paginate(MODELS, 2), 1):
                fig, axes = figure('Ketersediaan penjelasan' if kind == 'coverage' else 'Status jawaban — seluruh respons', 2, 25, .25)
                for ax, model in zip(axes, models):
                    part = table[table.model_id.eq(model)].copy()
                    order = ['neutral'] + [i + '_' + c for i in IDENTITIES for c in CONDITIONS]
                    part = part.set_index('persona').reindex(order)
                    left = np.zeros(len(part))
                    for col, label, color in zip(columns, labels, colors):
                        values = 100 * part[col] / part.total
                        ax.barh(range(len(part)), values, left=left, label=label, color=color)
                        left += values
                    short = ['N'] + [s + ' ' + CONDITION_LABELS[c] for s in ['Am.Utara', 'As.Timur', 'As.Selatan', 'Afr.SubS.'] for c in CONDITIONS]
                    ax.set_yticks(range(len(part)), short)
                    ax.invert_yaxis()
                    ax.set_xlim(0, 100)
                    ax.set_xlabel('% dari seluruh respons')
                    ax.set_title(MODEL_LABELS[model], fontsize=10)
                fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', bbox_to_anchor=(.5, .072), ncol=2, frameon=False, fontsize=9)
                self.save(fig, f'{kind}_p{page:02d}', table[table.model_id.isin(models)], kind, kind, 'percent_total', models, page, kind, 'tables/coverage.csv')

    def condition_lines(self, table, metric, title, section, ylabel):
        for identity in IDENTITIES:
            personas = ['neutral'] + [identity + '_' + c for c in CONDITIONS]
            fig, axes = figure(f'{IDENTITY_LABELS[identity]} — {title}', height=16)
            ax = axes[0]
            selected = table[table.persona.isin(personas)]
            for model in MODELS:
                part = selected[selected.model_id.eq(model)].set_index('persona').reindex(personas)
                ax.plot(range(5), part[metric], 'o-', ms=3, lw=1, color=MODEL_COLORS[model], label=MODEL_LABELS[model])
            ax.set_xticks(range(5), ['N', 'E', 'I1', 'I2', 'I3'])
            ax.set_ylabel(ylabel)
            ax.grid(alpha=.2)
            fig.legend(*ax.get_legend_handles_labels(), loc='lower center', bbox_to_anchor=(.5, .07), ncol=2, frameon=False, fontsize=8.5)
            fig.subplots_adjust(bottom=.30)
            source = 'tables/coverage.csv' if section == 'verbosity' else 'tables/diversity_summary.csv'
            self.save(fig, f'{section}_{identity}', selected, title, section, metric, [identity], source=source)

    def forest(self, table):
        focus = table[table.population.eq('text_all') & table.outcome_id.eq('culture_general_reference')]
        for identity in IDENTITIES:
            fig, axes = figure(f'{IDENTITY_LABELS[identity]} — efek rujukan budaya', 2, 25, .35,
                                   FOOTER + '\nTitik = median delta; garis = CI 95% median; ● Holm < .05, ○ lainnya. Arah kiri − kanan.')
            selected = focus[focus.identity.eq(identity)]
            for ax, suffix, title in zip(axes, ['_vs_neutral', '_vs_explicit'], ['E/I − N', 'I1/I2/I3 − E']):
                part = selected[selected.contrast_id.str.endswith(suffix)].sort_values(['model_id', 'contrast_id'])
                self.draw_forest(ax, part, title, lambda r: MODEL_LABELS[r.model_id] + ' ' + CONDITION_LABELS[r.left_condition])
            self.save(fig, f'forest_{identity}', selected, 'Efek rujukan budaya', 'paired_effects', 'median_delta_density_and_CI', [identity], source='tables/statistics/paired_contrasts.csv.gz')
        direct = focus[focus.multiplicity_family.eq('direct_identity_24')]
        pairs = list(direct.identity_contrast.dropna().unique())
        for page, pair_ids in enumerate(paginate(pairs, 2), 1):
            fig, axes = figure('Antaridentitas — efek rujukan budaya', len(pair_ids), 25, .35,
                              FOOTER + '\nMedian delta/100 kata + CI 95%; ● Holm < .05, ○ lainnya. Bukan CI rank-biserial.')
            used = direct[direct.identity_contrast.isin(pair_ids)]
            for ax, pair in zip(axes, pair_ids):
                part = used[used.identity_contrast.eq(pair)].sort_values(['model_id', 'condition'])
                first = part.iloc[0]
                title = IDENTITY_LABELS[first.left_identity] + ' − ' + IDENTITY_LABELS[first.right_identity]
                self.draw_forest(ax, part, textwrap.fill(title, 35), lambda r: MODEL_LABELS[r.model_id] + ' ' + CONDITION_LABELS[r.condition])
            self.save(fig, f'forest_direct_p{page:02d}', used, 'Antaridentitas', 'paired_effects', 'median_delta_density_and_CI', pair_ids, page, 'forest_direct', 'tables/statistics/paired_contrasts.csv.gz')

    @staticmethod
    def draw_forest(ax, table, title, label):
        display = [presentation_effect(row) for _, row in table.iterrows()]
        if len({row['unit'] for row in display}) > 1:
            raise ValueError('Forest cannot mix outcome units')
        for y, row in enumerate(table.itertuples(index=False)):
            values = display[y]
            if np.isfinite(values['effect']):
                ax.plot([values['ci_low'], values['ci_high']], [y, y], color=MODEL_COLORS[row.model_id], lw=1)
                ax.plot(values['effect'], y, 'o', ms=3, color=MODEL_COLORS[row.model_id], mfc=MODEL_COLORS[row.model_id] if row.p_holm < .05 else 'white')
            else:
                ax.text(0, y, 'NA', fontsize=8.5)
        ax.set_yticks(range(len(table)), [label(row) for row in table.itertuples(index=False)], fontsize=8.5)
        ax.invert_yaxis()
        ax.axvline(0, color='#94a3b8', lw=.8)
        ax.set_xlabel('Delta proporsi (pp)' if display and display[0]['unit'] == 'pp' else 'Median delta / 100 kata')
        ax.set_title(title, fontsize=10)

    def ecdf(self, table):
        for identity in IDENTITIES:
            fig, axes = plt.subplots(3, 2, figsize=(16 / 2.54, 22 / 2.54))
            fig.subplots_adjust(left=.14, right=.96, top=.88, bottom=.15, hspace=.5, wspace=.33)
            fig.suptitle(IDENTITY_LABELS[identity] + ' — keragaman kategori', x=.04, y=.98, ha='left', fontsize=12, weight='bold')
            colors = ['#64748b', '#2563eb', '#059669', '#d97706', '#7c3aed']
            personas = ['neutral'] + [identity + '_' + c for c in CONDITIONS]
            used = table[table.persona.isin(personas)]
            for ax, model in zip(axes.flat, MODELS):
                for persona, condition, color in zip(personas, SCENARIOS, colors):
                    part = used[used.model_id.eq(model) & used.persona.eq(persona)].sort_values('distinct_markers')
                    ax.step(part.distinct_markers, part.ecdf, where='post', label=CONDITION_LABELS[condition], color=color)
                ax.set_title(MODEL_LABELS[model], fontsize=10)
                ax.set_xlim(0, 23)
                ax.set_ylim(0, 1.02)
                ax.set_xticks([0, 5, 10, 15, 20, 23])
                ax.set_ylabel('Proporsi kumulatif')
                ax.set_xlabel('Kategori berbeda (0–23)')
            fig.legend(*axes.flat[0].get_legend_handles_labels(), loc='lower center', bbox_to_anchor=(.5, .08), ncol=5, frameon=False, fontsize=9)
            fig.text(.04, .025, textwrap.fill('Kategori kamus yang count > 0, bukan richness semantik. Bergantung pada panjang respons; nonteks dikeluarkan, abstain berteks masuk.', 88), fontsize=8.5)
            self.save(fig, f'diversity_ecdf_{identity}', used, 'Keragaman kategori', 'diversity', 'ecdf', list(MODELS), source='tables/diversity_distribution.csv')

    def cards(self, records):
        available = {f.name for f in font_manager.fontManager.ttflist}
        families = ['DejaVu Sans'] + [name for name in ['Microsoft YaHei', 'Malgun Gothic', 'Segoe UI Symbol', 'Segoe UI Emoji'] if name in available]
        fonts = [ft2font.FT2Font(font_manager.findfont(FontProperties(family=name))) for name in families]
        font = FontProperties(family=families, size=9)
        for model in MODELS:
            triplet = [r for r in records if r['model_id'] == model]
            first = triplet[0]
            fig = plt.figure(figsize=(16 / 2.54, 26 / 2.54))
            fig.text(.04, .975, MODEL_LABELS[model] + ' — N/E/I pada soal sama', fontsize=12, weight='bold', va='top')
            fig.text(.04, .94, IDENTITY_LABELS[first['identity']] + ' · ' + first['question_id'], fontsize=9)
            used = []
            for i, row in enumerate(triplet):
                top = .895 - i * .27
                condition = persona_parts(row['persona'])[1]
                fig.text(.04, top, f"{CONDITION_LABELS[condition]} · {row['analysis_surface_status']} · {row['word_count']} kata", fontsize=10, weight='bold')
                fig.text(.04, top - .025, f"Lexical [{row['excerpt_start']}:{row['excerpt_end']}] / {len(row['lexical'])} karakter", fontsize=8.5, color='#475569')
                ax = fig.add_axes([.035, top - .235, .93, .195])
                ax.set_axis_off()
                missing = sorted({c for c in row['excerpt'] if not c.isspace() and not any(f.get_char_index(ord(c)) for f in fonts)})
                if missing:
                    ax.text(.02, .85, 'Cuplikan memuat glyph tanpa font lokal.\nBuka teks utuh dan sorotan di response_cards.html.', fontsize=9, va='top')
                else:
                    fig.canvas.draw()
                    draw_excerpt(ax, row, font)
                used.append(dict(response_key=row['response_key'], excerpt=row['excerpt'], excerpt_start=row['excerpt_start'],
                                 excerpt_end=row['excerpt_end'], glyph_fallback='HTML' if missing else '|'.join(families),
                                 missing_glyphs=''.join(missing), raw_final_sha256=row['raw_final_sha256']))
            fig.text(.04, .03, textwrap.fill('Kuning = hit marker regional pada lexical surface. Contoh purposif, bukan sampel representatif. Teks utuh, soal, hit dan raw final terpisah tersedia di HTML.', 90), fontsize=8.5, va='bottom')
            self.save(fig, f'card_{MODELS.index(model) + 1:02d}', pd.DataFrame(used), 'Kartu respons', 'response_cards', 'surface_relative_spans', [model], source='evidence/response_cards.json')
        write_json(self.output / 'font_fallback.json', dict(card_font_families=families, missing_glyph_policy='HTML fallback, never broken PNG'))

    def diagnostics(self, tables):
        bands = ['1–49', '50–99', '100–199', '200–399', '400+']
        for identity in IDENTITIES:
            personas = ['neutral'] + [identity + '_' + c for c in CONDITIONS]
            table = tables['diversity_by_length']
            used = table[table.persona.isin(personas)]
            fig, axes = figure(IDENTITY_LABELS[identity] + ' — kategori menurut panjang', height=25, left=.39,
                               footer=FOOTER + '\nRerata kategori berbeda per rentang kata. n per sel tersedia di CSV; NA = rentang kosong.')
            order = pd.MultiIndex.from_product([MODELS, personas], names=['model_id', 'persona'])
            values = used.pivot(index=['model_id', 'persona'], columns='length_band', values='mean_distinct_markers').reindex(index=order, columns=bands).to_numpy(float)
            ax = axes[0]
            cmap = plt.get_cmap('YlGnBu').copy(); cmap.set_bad('#d1d5db')
            ax.imshow(values, aspect='auto', cmap=cmap, vmin=0, vmax=23)
            ax.set_xticks(range(5), bands)
            ax.set_xlabel('Rentang panjang respons (kata)')
            ax.set_yticks(range(30), [MODEL_LABELS[m] + ' ' + CONDITION_LABELS[persona_parts(p)[1]] for m, p in order], fontsize=8.5)
            for y in range(30):
                for x in range(5):
                    value = values[y, x]
                    ax.text(x, y, 'NA' if not np.isfinite(value) else f'{value:.1f}', ha='center', va='center', fontsize=8.5, color='white' if value > 12 else '#17202e')
            self.save(fig, f'length_{identity}', used, 'Keragaman menurut panjang', 'length_diagnostic', 'mean_categories_by_length', [identity], source='tables/diversity_by_length.csv')
            context = tables['context_response_summary']
            used = context[context.persona.isin(personas)]
            fig, axes = figure(IDENTITY_LABELS[identity] + ' — overlap literal', 2, 22,
                               footer=FOOTER + '\nPersentase respons berpenjelasan dengan ≥1 hit cocok literal. Kedua flag dapat true; bukan asal-usul kausal.')
            for ax, flag, title in zip(axes, ['persona_literal_overlap', 'scenario_literal_overlap'], ['Dengan prompt pengguna persona', 'Dengan teks skenario soal']):
                for model in MODELS:
                    part = used[used.model_id.eq(model) & used.flag.eq(flag)].set_index('persona').reindex(personas)
                    ax.plot(range(5), part.prevalence_pct, 'o-', ms=3, lw=1, color=MODEL_COLORS[model], label=MODEL_LABELS[model])
                ax.set_xticks(range(5), ['N', 'E', 'I1', 'I2', 'I3']); ax.set_ylabel('% respons berpenjelasan')
                ax.set_title(title, fontsize=10); ax.grid(alpha=.2)
            fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', bbox_to_anchor=(.5, .065), ncol=2, frameon=False, fontsize=8.5)
            fig.subplots_adjust(bottom=.23)
            self.save(fig, f'context_{identity}', used, 'Overlap literal', 'context_diagnostic', 'response_prevalence_pct', [identity], source='evidence/context_response_summary.csv')
        contrasts = tables['paired_contrasts']
        keys = ['model_id', 'left_persona', 'right_persona', 'outcome_id', 'outcome_scale']
        main = contrasts[contrasts.population.eq('text_all')]
        sensitivity = contrasts[contrasts.population.eq('text_both_committed')].merge(main, on=keys, suffixes=('_sensitivity', '_main'), validate='one_to_one')
        # Keep native scales separate; scatter focuses on one prespecified density outcome.
        sensitive_focus = sensitivity[sensitivity.outcome_id.eq('culture_general_reference')]
        old = tables['old_new_comparison']
        old_focus = old[old.outcome_id.eq('culture_general_reference') & old.population.eq('text_all')]
        fig, axes = figure('Sensitivitas — rujukan budaya umum', 2, 22,
                           footer=FOOTER + '\nSetiap titik = kontras; tabel lengkap memuat nol, NA, n pasangan dan perubahan Holm. Garis putus = kesamaan efek.')
        for ax, data, x, y, title in [(axes[0], sensitive_focus, 'effect_main', 'effect_sensitivity', 'Semua teks vs both-committed (I−E)'),
                                      (axes[1], old_focus, 'effect_old', 'effect_new', 'Pooled v2 lama vs pooled r5 policy baru')]:
            for model in MODELS:
                part = data[data.model_id.eq(model)]
                ax.scatter(part[x], part[y], s=12, alpha=.7, color=MODEL_COLORS[model], label=MODEL_LABELS[model])
            limits = data[[x, y]].to_numpy(float); finite = limits[np.isfinite(limits)]
            lo, hi = (float(finite.min()), float(finite.max())) if len(finite) else (-1, 1)
            if lo == hi: lo, hi = lo - .1, hi + .1
            ax.plot([lo, hi], [lo, hi], '--', color='#94a3b8', lw=.8)
            ax.set_xlabel(x + ' (median delta/100 kata)'); ax.set_ylabel(y + ' (delta/100 kata)')
            ax.set_title(title, fontsize=10)
        fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', bbox_to_anchor=(.5, .065), ncol=2, frameon=False, fontsize=8.5)
        fig.subplots_adjust(bottom=.23)
        self.save(fig, 'sensitivity_effects', pd.concat([sensitive_focus.assign(comparison='both_committed'), old_focus.assign(comparison='old_new')], ignore_index=True),
                  'Sensitivitas efek', 'sensitivity', 'median_delta_density', ['both_committed', 'old_new'], source='tables/statistics/paired_contrasts.csv.gz;tables/statistics/old_new_comparison.csv.gz')
        weights = tables['subset_weights']
        rows = weights.groupby(['model_id', 'source_subset']).n_text.sum().unstack().reindex(MODELS)
        fractions = rows.div(rows.sum(axis=1), axis=0) * 100
        fig, axes = figure('Komposisi subset pada penjelasan eligible', height=16, left=.29)
        bottom = np.zeros(6)
        for subset, color in zip(fractions.columns, ['#2563eb', '#059669', '#d97706', '#7c3aed']):
            axes[0].barh(range(6), fractions[subset], left=bottom, label=subset, color=color); bottom += fractions[subset]
        axes[0].set_yticks(range(6), [MODEL_LABELS[m] for m in MODELS]); axes[0].set_xlabel('% penjelasan eligible'); axes[0].set_xlim(0, 100)
        fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', bbox_to_anchor=(.5, .075), ncol=2, frameon=False, fontsize=9)
        fig.subplots_adjust(bottom=.3)
        self.save(fig, 'subset_composition', weights, 'Komposisi subset', 'sensitivity', 'eligible_response_share', list(MODELS), source='tables/statistics/subset_weights.csv')


def draw_excerpt(ax, record, font):
    text, start, end = record['lexical'], record['excerpt_start'], record['excerpt_end']
    boundaries = {start, end}
    for hit in record['hits']:
        if hit['end'] > start and hit['start'] < end:
            boundaries.update([max(start, hit['start']), min(end, hit['end'])])
    renderer = ax.figure.canvas.get_renderer()
    box = ax.get_window_extent(renderer)
    x, y = .02, .98
    line_height = 9 * ax.figure.dpi / 72 * 1.45 / box.height
    points = sorted(boundaries)
    for left, right in zip(points, points[1:]):
        highlighted = any(h['start'] < right and h['end'] > left for h in record['hits'])
        for token in re.findall(r'\S+|\s+', text[left:right]):
            token = ' ' if token.isspace() else token
            width = renderer.get_text_width_height_descent(token, font, False)[0] / box.width
            if x + width > .98:
                x, y = .02, y - line_height
            if y < .02 or width > .96:
                raise ValueError('Card text does not fit print layout')
            ax.text(x, y, token, va='top', fontproperties=font, transform=ax.transAxes, parse_math=False,
                    bbox=dict(facecolor='#fef08a', edgecolor='none', pad=.15) if highlighted else None)
            x += width


def render_all(tables, output):
    render = Renderer(output)
    if 'coverage' in tables:
        render.coverage(tables['coverage'])
        render.condition_lines(tables['coverage'], 'median_words', 'panjang penjelasan', 'verbosity', 'Median jumlah kata')
    if 'regional_profiles' in tables:
        render.profiles(tables['regional_profiles'])
        if 'all_marker_profiles' in tables:
            render.profiles(tables['all_marker_profiles'], 'general_ethical', [m for m in GENERAL if m in set(tables['all_marker_profiles'].marker_id)])
            focus_ids = ['culture_region_' + i for i in IDENTITIES] + ['regional_persona_direct_address', 'regional_perspective_adoption', 'culture_identity_irrelevance']
            render.profiles(tables['regional_profiles'], 'labels_accommodation', focus_ids)
    if 'paired_contrasts' in tables:
        render.forest(tables['paired_contrasts'])
    if 'diversity_distribution' in tables:
        render.ecdf(tables['diversity_distribution'])
        render.condition_lines(tables['diversity_summary'], 'mean_distinct_markers', 'rerata kategori berbeda', 'diversity_mean', 'Rerata kategori (0–23)')
    if 'cards' in tables:
        render.cards(tables['cards'])
    if 'context_response_summary' in tables:
        render.diagnostics(tables)
    index = pd.DataFrame(render.index)
    index.to_csv(Path(output) / 'figure_index.csv', index=False)
    return index
