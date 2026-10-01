"""Add the user-authorized pooled label-agreement chart to the existing notebook."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src/pipelines/analysis'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat
from nbclient import NotebookClient
import pandas as pd
import render_gsm8k_v5_barplus2_from_tables as style
from ethics_persona_visual_style import PAIRED_COLORS

FOLDER = ROOT / 'notebooks/religion/ethics accuracy analysis'
NOTEBOOK = FOLDER / 'ETHICS_Religion_Accuracy_Analysis_Lengkap.ipynb'
STEM = '00_ethics_religion_combined_absolute_accuracy'
SOURCE = FOLDER / 'tables/ethics_condition_summary_source_derived.csv'
CATEGORIES = [f'{identity} {mode}' for identity in ['Ateis', 'Kristen', 'Islam', 'Yahudi']
              for mode in ['Explicit', 'Implicit Avg']] + ['Neutral']
COLORS = PAIRED_COLORS

AGGREGATION_CODE = '''# Pool counts, never average subset percentages.
assert set(summary.subset) == {'commonsense', 'deontology', 'justice', 'virtue'}
assert not summary.duplicated(['model', 'persona', 'subset']).any()
assert summary.groupby(['model', 'persona']).subset.nunique().eq(4).all()
counts = ['total', 'committed', 'abstain', 'missing', 'n_correct']
pooled_accuracy = summary.groupby(['model', 'model_label', 'category'], as_index=False)[counts].sum()
assert len(pooled_accuracy) == 54
assert pooled_accuracy.total.sum() == 154428
assert pooled_accuracy.total.eq(pooled_accuracy.committed + pooled_accuracy.abstain + pooled_accuracy.missing).all()
assert pooled_accuracy.committed.gt(0).all()
assert pooled_accuracy.n_correct.between(0, pooled_accuracy.committed).all()
for status_name in ['committed', 'abstain', 'missing']:
    column = 'commit_rate' if status_name == 'committed' else status_name + '_rate'
    pooled_accuracy[column] = pooled_accuracy[status_name] / pooled_accuracy.total
pooled_accuracy['accuracy_committed'] = pooled_accuracy.n_correct / pooled_accuracy.committed
'''

NOTE = '''### Akurasi antar-persona

Grafik ini menyajikan **kecocokan label pada respons committed** dari empat subset,
bukan satu ukuran kemampuan moral. Penggabungan ini ditambahkan atas persetujuan
pengguna; keterbatasan konstruk Deontology/Virtue dan hasil per subset tetap berlaku.
Nilai batang = jumlah `n_correct` / jumlah `committed` lintas subset.
Implicit Avg menggabungkan I1–I3 dengan pembobot jumlah committed, bukan rata-rata
sederhana tiga persentase. Neutral Avg adalah rata-rata enam batang Neutral.
Sumbu Y memakai skala 0–1. Coverage dilaporkan bersama angka grafik di bawah;
perbedaan coverage berarti respons yang dinilai merupakan sampel terseleksi.
Grafik ini berbeda dari heatmap inferensi yang memakai denominator pasangan penuh.

Reproduksi: `python src/benchmarks/ethics/add_religion_ethics_accuracy_chart.py`.
'''


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def integrate_notebook(nb):
    """Insert or refresh only this section, preserving every other notebook cell."""
    generated = [
        nbformat.v4.new_markdown_cell(NOTE),
        nbformat.v4.new_code_cell(AGGREGATION_CODE + f'''
display(Image(filename=str(FIGURE_DIR / '{STEM}.png'), width=1500))
display(pooled_accuracy[['model_label', 'category', 'total', 'committed', 'commit_rate',
                        'abstain', 'abstain_rate', 'missing', 'missing_rate',
                        'n_correct', 'accuracy_committed']].style.format({{
    'commit_rate': '{{:.2%}}', 'abstain_rate': '{{:.2%}}',
    'missing_rate': '{{:.2%}}', 'accuracy_committed': '{{:.2%}}'}}))
'''),
    ]
    for cell in generated:
        cell.metadata['religion_pooled_accuracy'] = True
    nb.cells = [c for c in nb.cells if not c.metadata.get('religion_pooled_accuracy')]
    insertion = next(i for i, c in enumerate(nb.cells) if '## 5. Chart gabungan' in c.source) + 1
    nb.cells[insertion:insertion] = generated


def render(summary):
    namespace = {'summary': summary}
    exec(AGGREGATION_CODE, namespace)
    pooled = namespace['pooled_accuracy']
    values = pooled.pivot(index='model', columns='category', values='accuracy_committed').reindex(
        index=style.MODEL_ORDER, columns=CATEGORIES)
    assert values.shape == (6, 9) and not values.isna().any().any()
    style._set_style()
    fig, ax = plt.subplots(figsize=(27, 11))
    values.plot.bar(ax=ax, width=.88, color=COLORS)
    neutral = values.Neutral.mean()
    baseline = f'Neutral Avg ({neutral:.3f})'
    ax.axhline(neutral, color='#B80F2E', linestyle='--', linewidth=2, label=baseline)
    for bar in ax.patches:
        value = bar.get_height()
        ax.annotate(f'{value:.3f}', (bar.get_x() + bar.get_width()/2, value),
                    xytext=(0, 4), textcoords='offset points', ha='center', va='bottom',
                    rotation=90, rotation_mode='default', fontsize=20)
    assert len(ax.patches) == len(ax.texts) == 54
    ax.set_ylim(0, 1)
    ax.set_title('ETHICS Religion: Akurasi Antar-Persona', fontsize=22, fontweight='bold', pad=18)
    ax.set_xlabel('Model')
    ax.set_ylabel('Akurasi (committed)')
    ax.set_xticklabels(style.MODEL_AXIS_ORDER, rotation=0)
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(baseline)] + [i for i, label in enumerate(labels) if label != baseline]
    ax.legend([handles[i] for i in order], [labels[i] for i in order], title='Persona & Baseline',
              bbox_to_anchor=(1.01, 1), loc='upper left')
    ax.grid(axis='y', linestyle=':', alpha=.5)
    fig.tight_layout()
    fig.canvas.draw()
    bounds = ax.get_window_extent()
    assert all(bounds.contains(t.get_window_extent().x0, t.get_window_extent().y0)
               and bounds.contains(t.get_window_extent().x1, t.get_window_extent().y1)
               for t in ax.texts), 'Clipped annotation'
    for ext in ['png', 'svg']:
        fig.savefig(FOLDER / 'figures' / f'{STEM}.{ext}', dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f'[CHECK] 54 bars, 154428 responses; commit rate range {pooled.commit_rate.min():.3f}–{pooled.commit_rate.max():.3f}', flush=True)


def main():
    source_hashes = {p: sha(p) for p in (FOLDER / 'tables').glob('*.csv')}
    render(pd.read_csv(SOURCE))
    print('[NOTEBOOK] Updating the existing notebook.', flush=True)
    nb = nbformat.read(NOTEBOOK, as_version=4)
    integrate_notebook(nb)
    NotebookClient(nb, timeout=600, kernel_name='python3',
                   resources={'metadata': {'path': str(FOLDER)}}).execute()
    nbformat.validate(nb)
    assert not any(o.output_type == 'error' for c in nb.cells for o in c.get('outputs', []))
    nbformat.write(nb, NOTEBOOK)

    manifest_path = FOLDER / 'bundle_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest['pooled_label_agreement_chart'] = {
        'stem': STEM, 'source': SOURCE.relative_to(FOLDER).as_posix(), 'source_sha256': sha(SOURCE),
        'scope': 'all_four_subsets', 'metric': 'sum(n_correct)/sum(committed)',
        'implicit_aggregation': 'committed-weighted mean across I1/I2/I3',
        'baseline': 'arithmetic mean of six model-specific neutral rates',
        'user_authorized_pooling': True, 'interpretation': 'label agreement, not moral ability',
        'bars': 54, 'response_count': 154428,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    active_path = FOLDER / 'active_figures.csv'
    active = pd.read_csv(active_path)
    active = active[~active.figure.str.contains(STEM, regex=False)]
    rows = [{'notebook': NOTEBOOK.name, 'figure': f'figures/{STEM}.{ext}', 'status': 'active'}
            for ext in ['png', 'svg']]
    pd.concat([active, pd.DataFrame(rows)], ignore_index=True).to_csv(active_path, index=False)
    index_path = FOLDER / 'output_index.csv'
    index = pd.read_csv(index_path)
    index = index[~index.file.str.contains(STEM, regex=False)]
    rows = []
    for ext in ['png', 'svg']:
        path = FOLDER / 'figures' / f'{STEM}.{ext}'
        rows.append({'kind': 'figure', 'file': path.name, 'relative_path': path.relative_to(FOLDER).as_posix(),
                     'path': str(path), 'sha256': sha(path)})
    pd.concat([index, pd.DataFrame(rows)], ignore_index=True).to_csv(index_path, index=False)
    assert all(sha(p) == value for p, value in source_hashes.items())
    print(f'[PASS] Notebook executed; PNG/SVG indexed; {len(source_hashes)} source tables unchanged.', flush=True)


if __name__ == '__main__':
    main()
