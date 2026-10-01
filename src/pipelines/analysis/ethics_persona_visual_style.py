"""Shared presentation rules; no scoring or inferential calculations."""
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import seaborn as sns

# Exact category order from the user's GSM8K Regional reference.
PAIRED_COLORS = ['#0047AB', '#0086A8', '#B80F2E', '#E45700', '#007A4D',
                 '#65A30D', '#5B21B6', '#A0006D', '#30343B']


def religion_colors(labels):
    identities = {'Ateis': 0, 'Atheist': 0, 'Kristen': 1, 'Christian': 1,
                  'Islam': 2, 'Yahudi': 3, 'Jewish': 3}
    colors = []
    for label in labels:
        if label == 'Neutral':
            colors.append(PAIRED_COLORS[-1])
            continue
        identity = label.split()[0].rstrip(':')
        implicit = 'Implicit' in label
        colors.append(PAIRED_COLORS[2 * identities[identity] + int(implicit)])
    return colors


def plot_regional_four_panels(frame, figure_dir, *, row_order, stems, title_domain,
                             identity_groups=None, expected_counts=(14, 14, 12, 12)):
    """Keep all 52 comparisons, regrouping them by identity/pair, not position."""
    from ethics_accuracy_statistical_figures import _save
    singles = ['north_american', 'east_asian', 'south_asian', 'sub_saharan_african']
    pair_ids = frame.loc[frame.identity.str.contains('_vs_'), 'identity'].drop_duplicates().tolist()
    north_pairs = sorted(p for p in pair_ids if 'north_american' in p)
    other_pairs = sorted(p for p in pair_ids if p not in north_pairs)
    groups = [
        ('Amerika Utara dan Asia Timur', singles[:2]),
        ('Asia Selatan dan Afrika Sub-Sahara', singles[2:]),
        ('Perbandingan yang melibatkan Amerika Utara', north_pairs),
        ('Perbandingan antarkawasan lainnya', other_pairs),
    ]
    if identity_groups is not None:
        groups = identity_groups
    panels = []
    for title, identities in groups:
        columns = []
        for identity in identities:
            part = frame[frame.identity.eq(identity)].sort_values('display_order', kind='stable')
            labels = part.comparison_short.drop_duplicates().tolist()
            assert len(labels) == (4 if '_vs_' in identity else 7), identity
            columns.extend(labels)
        panels.append((title, columns))
    columns = [c for _, part in panels for c in part]
    assert [len(part) for _, part in panels] == list(expected_counts)
    assert len(set(columns)) == sum(expected_counts) and set(columns) == set(frame.comparison_short)
    assert not frame.duplicated(['row_label', 'comparison_short']).any()
    assert len(frame) == len(row_order) * sum(expected_counts)
    decisions = frame.significant_holm_0_05.astype(str).str.lower().map({'true': True, 'false': False})
    assert decisions.notna().all()
    assert np.array_equal(decisions, frame.holm_adjusted_p_value.lt(.05))
    effect_cmap = LinearSegmentedColormap.from_list('ethics_readable',
        ['#0B3C78', '#4A98C9', '#DCEAF1', '#FCFBF8', '#F7C7AE', '#E76F51', '#9B002F'])
    limit = max(.1, np.ceil(frame.effect_size.abs().max() / .05) * .05)
    specs = [
        ('effect_size', 'Ukuran Efek', effect_cmap, -limit, limit, "Cohen's h (A − B)"),
        ('exact_mcnemar_p_value', 'p-value Exact McNemar', 'rocket', 0, .05, 'p-value mentah'),
        ('holm_adjusted_p_value', 'p-value Holm', 'rocket', 0, .05, 'p-value Holm'),
    ]
    result = []
    for stem, (metric, metric_title, cmap, low, high, color_label) in zip(stems, specs, strict=True):
        count = len(row_order)
        panel_count = len(panels)
        fig = plt.figure(figsize=(20 if count > 6 else 18, (33 if count > 6 else 23)*panel_count/4))
        grid = fig.add_gridspec(panel_count+1, 1, height_ratios=[count]*panel_count + [.42],
                               left=.26 if count > 6 else .225, right=.985,
                               top=.91, bottom=.055, hspace=.82 if count == 6 else .48)
        color_ax = fig.add_subplot(grid[panel_count])
        # Preserve the original Regional raw-p star convention.
        selected_decisions = frame.exact_mcnemar_p_value.lt(.05) if metric == 'exact_mcnemar_p_value' else decisions
        enriched = frame.assign(display_significant=selected_decisions)
        drawn = 0
        for index, (panel_title, part) in enumerate(panels):
            ax = fig.add_subplot(grid[index])
            values = frame.pivot(index='row_label', columns='comparison_short', values=metric).reindex(index=row_order, columns=part)
            significant = enriched.pivot(index='row_label', columns='comparison_short', values='display_significant').reindex(index=row_order, columns=part)
            assert values.notna().all().all()
            annotations = values.copy().astype(object)
            for row in row_order:
                for col in part:
                    value = values.loc[row, col]
                    if metric == 'effect_size':
                        label = '0,00' if abs(value) < .005 else f'{value:+.2f}'.replace('.', ',')
                    else:
                        label = '<0,001' if value < .001 else f'{value:.3f}'.replace('.', ',')
                    annotations.loc[row, col] = label + ('*' if significant.loc[row, col] else '')
            sns.heatmap(values, ax=ax, cmap=cmap, vmin=low, vmax=high,
                        center=0 if metric == 'effect_size' else None,
                        annot=annotations, fmt='', annot_kws={'fontsize': 17.5 if metric == 'effect_size' else 15},
                        linewidths=1, linecolor='#FCFBF8', cbar=index == panel_count-1,
                        cbar_ax=color_ax if index == panel_count-1 else None, cbar_kws={'orientation': 'horizontal'})
            for text, sig in zip(ax.texts, significant.to_numpy().ravel(), strict=True):
                text.set_fontweight('bold' if sig else 'normal')
            drawn += len(ax.texts)
            ax.set_xticklabels([c.replace(':', '\n') for c in part], rotation=0, fontsize=15)
            ax.set_yticklabels(row_order, rotation=0, fontsize=16.5)
            ax.tick_params(axis='both', length=0, pad=8)
            ax.set_xlabel('Perbandingan skenario', fontsize=17, labelpad=10)
            ax.set_ylabel('Subset dan model' if count > 6 else 'Model', fontsize=17, labelpad=12)
            ax.set_title(f'Panel {index+1} — {panel_title}', fontsize=19, fontweight='bold', pad=14)
            for boundary in ([7] if index < 2 else [4, 8]):
                if boundary < len(part):
                    ax.axvline(boundary, color='#606060', linewidth=1.5)
            if count > 6:
                ax.axhline(6, color='#606060', linewidth=1.5)
        assert drawn == len(frame)
        color_ax.tick_params(labelsize=14.5)
        color_ax.set_xlabel(color_label, fontsize=16.5, labelpad=9)
        fig.suptitle(f'ETHICS {title_domain}: {metric_title}', fontsize=23, fontweight='bold', y=.98)
        fig.canvas.draw()
        for ax, (_, part) in zip(fig.axes[1:], panels, strict=True):
            width = ax.get_window_extent().width / len(part)
            assert all(t.get_window_extent().width < width*.98 for t in ax.texts), 'Annotation overlaps adjacent cell'
            label_boxes = [t.get_window_extent() for t in ax.get_yticklabels()]
            assert all(top.y0 > bottom.y1 for top, bottom in zip(label_boxes, label_boxes[1:])), 'Overlapping row labels'
        result += _save(fig, figure_dir, stem)
        print(f'[HEATMAP] {stem}: {panel_count} panels; {drawn} source values preserved', flush=True)
    return result
