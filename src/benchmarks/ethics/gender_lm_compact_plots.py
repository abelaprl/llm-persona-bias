"""Native 16 cm presentation, sharing axes instead of shrinking old figures."""
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import numpy as np
import pandas as pd

from .restyle_gender_lm_gsm8k import LABELS

MODELS = {
    'meta-llama_llama-3.1-8b-instruct': 'Llama 3.1 8B',
    'meta-llama_llama-3.3-70b-instruct': 'Llama 3.3 70B',
    'mistralai_ministral-8b-2512': 'Ministral 8B',
    'deepseek_deepseek-r1-distill-qwen-32b': 'DeepSeek R1 32B',
    'openai_gpt-oss-20b': 'GPT-OSS 20B',
    'qwen_qwen3-next-80b-a3b-thinking': 'Qwen3 Next 80B',
}
CODES = dict(zip(MODELS, ['L31', 'L33', 'MIN', 'DSR1', 'GPT', 'QWN']))
SHORT = LABELS | {
    'self_reference_first_person': 'Rujukan diri: saya',
    'framing_consequentialist': 'Konsekuensialisme',
    'gender_persona_acknowledgment': 'Pengakuan gender',
    'gender_identity_neutralization': 'Netralisasi gender',
    'gender_identity_relevance_denial': 'Penolakan relevansi',
    'gender_perspective_adoption': 'Perspektif gender',
    'gender_identity_discourse': 'Wacana gender',
    'gendered_normative_evaluation': 'Normatif gender',
}
EFFECT = LinearSegmentedColormap.from_list('gender_effect', ['#2166ac', '#fafafa', '#b21860'])
PERSONA_CODES = ['N', 'ME', 'WE', 'MI1', 'WI1', 'MI2', 'WI2', 'MI3', 'WI3']


def style():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5,
                         'axes.titlesize': 10, 'axes.labelsize': 9,
                         'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
                         'svg.fonttype': 'none', 'savefig.dpi': 300,
                         'figure.facecolor': 'white', 'axes.spines.top': False,
                         'axes.spines.right': False})


def matrix(data, rows, columns, x, value, row_key='marker_id'):
    return data.pivot(index=row_key, columns=x, values=value).reindex(index=rows, columns=columns)


def heatmap(ax, values, row_labels, column_labels, *, limit, diverging=False,
            cmap=None, annotate=False, symbols=None):
    colors = (cmap or (EFFECT if diverging else plt.get_cmap('Blues'))).copy()
    colors.set_bad('#d1d5db')
    image = ax.imshow(values, aspect='auto', cmap=colors,
                      vmin=-limit if diverging else 0, vmax=limit, interpolation='nearest')
    ax.set_xticks(range(len(column_labels)), column_labels)
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.tick_params(length=0, pad=3)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for (i, j), number in np.ndenumerate(np.asarray(values, float)):
        text = 'NA' if not np.isfinite(number) else (f'{number:.0f}' if annotate else '')
        if symbols is not None and pd.notna(symbols[i, j]):
            text = str(symbols[i, j])
        if text:
            red, green, blue, _ = colors(image.norm(number))
            dark = (.2126 * red + .7152 * green + .0722 * blue) < .48
            ax.text(j, i, text, ha='center', va='center', fontsize=8.5,
                    color='white' if dark else '#111827')
    return image


def model_panels(data, rows, columns, x, value, title, unit, *, models=MODELS,
                 labels=SHORT, limit=100, diverging=False, column_labels=None,
                 row_key='marker_id', stars=False):
    style()
    nrows = math.ceil(len(models) / 2)
    height = min(26, max(16, nrows * (len(rows) * .40 + 1.3) + 3))
    fig, axes = plt.subplots(nrows, 2, figsize=(16 / 2.54, height / 2.54), squeeze=False)
    fig.subplots_adjust(left=.255, right=.985, bottom=.12, top=.92, hspace=.40, wspace=.12)
    for i, (model, name) in enumerate(models.items()):
        ax = axes.flat[i]
        part = data.loc[data.model_id.eq(model)]
        values = matrix(part, rows, columns, x, value, row_key)
        symbols = None
        if stars:
            p = matrix(part, rows, columns, x, 'adjusted_p', row_key).to_numpy(float)
            symbols = np.where(p < .05, '*', '').astype(object)
            symbols[~np.isfinite(values)] = 'NA'
        image = heatmap(ax, values, [labels.get(k, k) for k in rows] if i % 2 == 0 else [''] * len(rows),
                        column_labels or columns, limit=limit, diverging=diverging, symbols=symbols)
        ax.set_title(name, loc='left', pad=8)
        if len(columns) > 6:
            ax.tick_params(axis='x', labelrotation=90)
    for ax in list(axes.flat)[len(models):]:
        ax.set_visible(False)
    fig.suptitle(title, fontsize=12, y=.975)
    bar = fig.add_axes([.30, .062, .62, .014])
    fig.colorbar(image, cax=bar, orientation='horizontal')
    fig.text(.5, .017, unit, ha='center', fontsize=8.5)
    return fig


def grid_figure(data, rows, conditions, value, title, unit, *, models=MODELS,
                labels=SHORT, limit=4, diverging=False, cmap=None, symbols_column=None):
    style()
    height = max(13, len(rows) * .52 + 4)
    fig, ax = plt.subplots(figsize=(16 / 2.54, height / 2.54))
    fig.subplots_adjust(left=.26, right=.985, bottom=.17, top=.84)
    columns = [(m, c) for m in models for c in conditions]
    part = data.assign(cell=list(zip(data.model_id, data.condition)))
    values = matrix(part, rows, columns, 'cell', value)
    symbols = None
    if symbols_column:
        symbols = matrix(part, rows, columns, 'cell', symbols_column).to_numpy()
    image = heatmap(ax, values, [labels.get(k, k) for k in rows],
                    [c for _, c in columns], limit=limit, diverging=diverging,
                    cmap=cmap, annotate=not diverging, symbols=symbols)
    for i, (model, name) in enumerate(models.items()):
        center = i * len(conditions) + (len(conditions) - 1) / 2
        ax.text(center, 1.025, CODES.get(model, name), transform=ax.get_xaxis_transform(),
                ha='center', fontsize=9)
        if i:
            ax.axvline(i * len(conditions) - .5, color='#111827', lw=.8)
    fig.suptitle(title, fontsize=12, y=.97)
    bar = fig.add_axes([.31, .085, .61, .018])
    fig.colorbar(image, cax=bar, orientation='horizontal')
    fig.text(.5, .024, unit, ha='center', fontsize=8.5)
    return fig


def overview_figure(coverage, lengths, personas):
    style()
    fig, axes = plt.subplots(2, 1, figsize=(16 / 2.54, 17 / 2.54))
    fig.subplots_adjust(left=.13, right=.85, bottom=.10, top=.89, hspace=.42)
    configs = [(coverage, 'available_percent', 'Teks tersedia (%)', 100),
               (lengths, 'median', 'Median panjang respons (kata)', None)]
    for ax, (table, value, title, bound) in zip(axes, configs):
        values = table.pivot(index='persona', columns='model_id', values=value).reindex(
            index=personas, columns=MODELS)
        image = heatmap(ax, values, PERSONA_CODES, list(CODES.values()),
                        limit=bound or max(1, np.nanmax(values)), annotate=True)
        ax.set_title(title, loc='left')
        fig.colorbar(image, ax=ax, fraction=.025, pad=.03)
    fig.suptitle('Cakupan teks dan panjang respons', fontsize=12, y=.96)
    fig.text(.5, .025, 'Label verdict saja = 0 kata; missing / teknis = NA.', ha='center')
    return fig


def support_figure(data, conditions):
    style()
    fig, axes = plt.subplots(2, 1, figsize=(16 / 2.54, 15 / 2.54))
    fig.subplots_adjust(left=.30, right=.86, bottom=.10, top=.87, hspace=.5)
    for ax, value, title, bound in zip(axes, ['pair_percent', 'low_support_markers'],
                                      ['Pasangan tersedia (%)', 'Penanda dengan n_changed < 10'], [100, 41]):
        values = data.pivot(index='model_id', columns='condition', values=value).reindex(index=MODELS, columns=conditions)
        im = heatmap(ax, values, list(MODELS.values()), conditions, limit=bound, annotate=True)
        ax.set_title(title, loc='left')
        fig.colorbar(im, ax=ax, fraction=.035, pad=.03)
    fig.suptitle('Dukungan pasangan Woman−Man', fontsize=12, y=.96)
    fig.text(.5, .025, 'Denominator: 1.514 pasangan / skenario; 41 penanda.', ha='center')
    return fig


NORMALIZATION_COLORS = ListedColormap(['#dbeafe', '#fb7185', '#fcd34d', '#e5e7eb'])
