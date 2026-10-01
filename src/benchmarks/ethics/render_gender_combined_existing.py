"""Render combined four-subset charts and enlarge non-title fonts by 4 pt."""

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src/pipelines/analysis'))

import matplotlib
matplotlib.use('Agg')
from matplotlib.text import Text
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import render_ethics_gender_accuracy_standardized as base
import render_ethics_gender_overall_standardized as old
import ethics_accuracy_statistical_figures as stats

OUTPUT = ROOT / 'notebooks/gender/ethics accuracy analysis'
FONT_AUDIT = []
SAVED = []
EFFECT_LAYOUT_STEMS = {
    '10_ethics_coverage_effect_holm_heatmap',
    '11_ethics_conditional_accuracy_effect_holm_heatmap',
    '12_ethics_effective_success_effect_holm_heatmap',
    '13_ethics_stance_wrong_effect_holm_heatmap',
}


def save_with_font_increase(fig, figure_dir, stem):
    # Materialize automatic tick labels before modifying individual Text objects.
    fig.canvas.draw()
    titles = {fig._suptitle}
    for ax in fig.axes:
        titles.update([ax.title, ax._left_title, ax._right_title])
    for text in fig.findobj(match=Text):
        before = text.get_fontsize()
        is_title = text in titles
        # The previous official bundle used +2 pt.  Keep the title size stable
        # and move every other chart text another +2 pt, making +4 pt from the
        # original standardized baseline.
        text.set_fontsize(before if is_title else before + 4)
        if text.get_text():
            FONT_AUDIT.append({
                'figure': stem, 'text': text.get_text(), 'is_title': is_title,
                'before_pt': before, 'after_pt': text.get_fontsize(),
            })
    if stem in EFFECT_LAYOUT_STEMS:
        # Font scaling happens after the original constrained layout. Re-run
        # that layout after enlarging text so model labels and colorbars remain
        # separated on every multi-panel heatmap.
        if hasattr(fig, 'set_layout_engine'):
            fig.set_layout_engine('constrained')
        else:
            fig.set_constrained_layout(True)
        fig.canvas.draw()
    paths = []
    figure_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ('png', 'svg'):
        path = figure_dir / f'{stem}.{suffix}'
        fig.savefig(path, facecolor=base.BACKGROUND, bbox_inches='tight', pad_inches=0.25)
        paths.append(path)
    plt.close(fig)
    SAVED.extend(paths)
    print(f'[FIGURE] {stem}', flush=True)
    return paths


def combined_counts(summary):
    frame = summary.copy()
    # Recover integer counts from full-precision source proportions; reject drift.
    for stance in ('wrong', 'acceptable'):
        counts = frame[f'pct_answer_{stance}'] * frame['committed']
        assert np.allclose(counts, counts.round(), rtol=0, atol=1e-8)
        frame[f'n_{stance}'] = counts.round().astype(int)
    assert frame['n_wrong'].add(frame['n_acceptable']).eq(frame['committed']).all()
    columns = ['total', 'nonmissing', 'committed', 'abstain', 'missing', 'n_wrong', 'n_acceptable']
    result = old._aggregate_categories(frame, columns)
    result[columns] = result[columns].astype(int)
    for stance in ('wrong', 'acceptable'):
        result[f'pct_answer_{stance}'] = result[f'n_{stance}'] / result['committed'].replace(0, np.nan)
    result['subset_count'] = 4
    result['scope'] = 'all_four_subsets'
    assert result['total'].sum() == summary['total'].sum() == 81756
    assert len(result) == 30
    assert result['n_wrong'].add(result['n_acceptable']).eq(result['committed']).all()
    assert result['total'].eq(result['committed'] + result['abstain'] + result['missing']).all()
    return result


def plot_combined(frame, *, write_tables=True):
    fig, axes = plt.subplots(3, 2, figsize=(25, 21))
    specs = [('n_wrong', 'Wrong', '#B80F2E'), ('n_acceptable', 'Acceptable', '#0086A8'),
             ('abstain', 'Abstain', '#E9C46A'), ('missing', 'Missing', '#E76F51')]
    for ax, model in zip(axes.flat, base.MODEL_ORDER, strict=True):
        rows = frame[frame['model'].eq(model)].set_index('category').reindex(old.CATEGORY_ORDER)
        bottom = np.zeros(len(rows))
        for column, label, color in specs:
            values = rows[column].to_numpy() / rows['total'].to_numpy() * 100
            ax.bar(np.arange(len(rows)), values, bottom=bottom, color=color, label=label, edgecolor='white')
            for i, value in enumerate(values):
                if value >= 5:
                    ax.text(i, bottom[i] + value / 2, f'{value:.1f}%', ha='center', va='center', fontsize=16,
                            color='white' if column.startswith('n_') else '#17212B')
            bottom += values
        ax.set_xticks(np.arange(len(rows)), [old.CATEGORY_SHORT_LABELS[c] for c in old.CATEGORY_ORDER])
        ax.set_ylim(0, 100)
        ax.set_ylabel('% seluruh respons')
        ax.set_xlabel('Persona')
        ax.set_title(base.MODEL_LABELS[model], fontsize=16, fontweight='bold')
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc='upper center', bbox_to_anchor=(.5, .945))
    fig.suptitle('ETHICS Gender: Distribusi Jawaban', fontsize=old.TITLE_SIZE, fontweight='bold', y=.99)
    fig.subplots_adjust(top=.88, bottom=.06, hspace=.5, wspace=.2)
    save_with_font_increase(fig, OUTPUT / 'figures', '00_ethics_combined_four_subsets_distribution')

    values = old._ordered_category_pivot(frame, 'pct_answer_wrong')
    delta_specs = [
        ('00_ethics_combined_four_subsets_delta_vs_neutral', 'Perubahan Distribusi Jawaban terhadap Netral',
         {f'{c} − Neutral': values[c] - values['Neutral'] for c in old.CATEGORY_ORDER if c != 'Neutral'}),
        ('00_ethics_combined_four_subsets_delta_implicit_explicit', 'Perubahan Distribusi Jawaban: Implisit terhadap Eksplisit',
         {f'{gender}: Implicit − Explicit': values[f'{gender} Implicit Avg'] - values[f'{gender} Explicit'] for gender in ('Man', 'Woman')}),
    ]
    delta_tables = []
    for stem, title, data in delta_specs:
        delta = pd.DataFrame(data)
        fig, ax = plt.subplots(figsize=(24, 12))
        colors = old.ACCURACY_COLORS[:4] if len(data) == 4 else [old.ACCURACY_COLORS[0], old.ACCURACY_COLORS[2]]
        delta.plot.bar(ax=ax, color=colors, width=.78)
        old._annotate_delta(ax)
        limit = max(.005, float(np.nanmax(np.abs(delta.to_numpy()))))
        ax.set_ylim(-limit * 1.65, limit * 1.65)
        ax.axhline(0, color='#17212B', linewidth=1)
        ax.set_title(f'ETHICS Gender: {title}', fontsize=old.TITLE_SIZE, fontweight='bold', pad=20)
        ax.set_xticklabels([base.MODEL_AXIS_LABELS[base.MODEL_LABELS[m]] for m in base.MODEL_ORDER], rotation=0)
        ax.set_ylabel('Delta proporsi wrong pada respons committed')
        ax.set_xlabel('Model')
        ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
        fig.tight_layout()
        save_with_font_increase(fig, OUTPUT / 'figures', stem)
        long = delta.rename_axis('model_label').reset_index().melt(id_vars='model_label', var_name='comparison', value_name='delta_wrong')
        long['scope'] = 'all_four_subsets'
        delta_tables.append(long)
    if write_tables:
        pd.concat(delta_tables).to_csv(OUTPUT / 'tables/ethics_combined_four_subsets_deltas.csv', index=False)


def main():
    base.save_figure = save_with_font_increase
    old._save = save_with_font_increase
    stats._save = save_with_font_increase
    tables = base._load_source_tables(base.SOURCE_BUNDLE)
    summary = tables['condition_summary']
    assert set(summary['subset']) == set(base.SUBSET_ORDER)
    # Re-render the existing notebook figures at their original baseline sizes + 2 pt.
    result = base.render(output_dir=OUTPUT)
    old.configure_plot_style()
    old.render_subset_children(tables, OUTPUT)
    combined = combined_counts(summary)
    combined.to_csv(OUTPUT / 'tables/ethics_combined_four_subsets.csv', index=False)
    plot_combined(combined)

    # Keep supplementary pre-existing charts readable too; source numbers unchanged.
    td = OUTPUT / 'tables'
    old_accuracy = pd.read_csv(td / 'ethics_accuracy_by_model_persona_category.csv')
    old.plot_accuracy(old_accuracy, OUTPUT / 'figures')
    old.plot_delta(old_accuracy, OUTPUT / 'figures', implicit=False)
    old.plot_delta(old_accuracy, OUTPUT / 'figures', implicit=True)
    old.plot_status(pd.read_csv(td / 'ethics_status_by_model_persona_category.csv'), OUTPUT / 'figures')
    old.plot_abstain_reasons(pd.read_csv(td / 'ethics_abstain_reasons_by_model.csv'), OUTPUT / 'figures')
    old.plot_accuracy_statistics(pd.read_csv(td / 'ethics_accuracy_inference_construct_valid.csv'), OUTPUT / 'figures')
    old.plot_combined_accuracy_statistics(pd.read_csv(td / 'ethics_accuracy_inference_construct_valid_combined.csv'), OUTPUT / 'figures')

    audit = pd.DataFrame(FONT_AUDIT)
    increments = audit['after_pt'] - audit['before_pt']
    assert increments[audit['is_title']].eq(0).all()
    assert increments[~audit['is_title']].eq(4).all()
    audit.to_csv(td / 'font_size_audit.csv', index=False)
    manifest = result['manifest']
    manifest.update({
        'bundle_version': 'ethics-gender-combined-four-subsets-fontplus4',
        'combined_scope': 'all_four_subsets',
        'combined_metrics': ['pct_answer_wrong', 'pct_answer_acceptable', 'commit_rate', 'abstain_rate', 'missing_rate'],
        'combined_weighting': 'sum_of_counts_over_all_responses_in_category',
        'non_title_font_increment_pt': 4,
        'title_font_increment_pt': 0,
        'active_figure_files': sorted(p.name for p in SAVED[:26] if p.parent == OUTPUT / 'figures') +
            sorted(p.name for p in SAVED if p.name.startswith('00_ethics_combined')),
        'figure_files': sorted(p.name for p in SAVED if p.parent == OUTPUT / 'figures'),
        'figure_count': sum(p.parent == OUTPUT / 'figures' for p in SAVED),
        'child_figure_count': sum(p.parent != OUTPUT / 'figures' for p in SAVED),
        'notes': manifest['notes'] + ['Chart gabungan empat subset dan delta bersifat deskriptif; tidak memakai p-value gabungan.'],
    })
    (OUTPUT / 'bundle_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    # Index all current parent and child assets using portable relative paths.
    paths = list((OUTPUT / 'figures').glob('*')) + list(td.glob('*.csv'))
    for subset in base.SUBSET_ORDER:
        paths += list((OUTPUT / 'per_subset' / subset / 'figures').glob('*'))
        paths += list((OUTPUT / 'per_subset' / subset / 'tables').glob('*.csv'))
    pd.DataFrame([{
        'kind': 'figure' if p.suffix in ('.png', '.svg') else 'table',
        'file': p.name, 'path': str(p), 'relative_path': p.relative_to(OUTPUT).as_posix(),
        'sha256': base.sha256_file(p),
    } for p in sorted(paths) if p.is_file()]).to_csv(OUTPUT / 'output_index.csv', index=False)
    print(f'[PASS] {len(SAVED) // 2} figures; fonts +4 pt except titles; {len(combined)} combined rows.', flush=True)


if __name__ == '__main__':
    main()
