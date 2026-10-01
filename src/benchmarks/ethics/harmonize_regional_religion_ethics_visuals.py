"""Harmonize active visuals without changing any source tables or data scopes."""
from pathlib import Path
import hashlib
import json
import sys
import re

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src/pipelines/analysis'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat
from nbclient import NotebookClient
import numpy as np
import pandas as pd
import render_ethics_regional_overall_standardized as regional
import add_religion_ethics_accuracy_chart as religion_accuracy
import update_religion_ethics_existing as religion
import render_gender_combined_existing as gender
from ethics_persona_visual_style import PAIRED_COLORS, plot_regional_four_panels

FOLDERS = {domain: ROOT / 'notebooks' / domain / 'ethics accuracy analysis'
           for domain in ['regional', 'religion', 'gender']}
GENDER_HOLM_STEM = '09_ethics_gender_combined_holm_pvalue_heatmap'
NOTE = '''## Konsistensi tampilan

Palet persona mengikuti referensi GSM8K Regional: biru, teal, merah, oranye,
hijau tua, hijau muda, ungu, magenta; Neutral abu-abu. Gender menggunakan empat
warna pertama dan abu-abu karena hanya memiliki dua identitas.
Grafik akurasi memakai 3 desimal, anotasi 20 pt,
dan sumbu Y 0–1. Angka lengkap tetap berada di tabel.

Heatmap Regional memakai empat panel dengan 14/14/12/12 perbandingan, masing-masing
menjaga seluruh skenario satu persona atau pasangan tetap bersama. File `07` dan
`09` dengan akhiran `part1` kini berisi **seluruh p-value Exact McNemar**;
`part2` berisi **seluruh p-value Holm**. File `06` dan `08` tetap ukuran efek.
Nama file dipertahankan supaya tautan lama tidak terputus.

Heatmap utama Gender memakai tiga panel: Man (7 perbandingan), Woman (7),
dan Man–Woman (4). P-value mentah dan Holm disajikan terpisah; file `09` lama
menampilkan p-value mentah dan `09_ethics_gender_combined_holm_pvalue_heatmap`
menampilkan Holm. Angka efek menggunakan palet biru–netral–merah yang sama.

Konvensi tanda signifikan tidak diubah: pada Regional, bintang p-value mentah
berarti p mentah < 0,05; pada efek dan Holm berarti p Holm < 0,05. Religion tetap
memakai keputusan Holm pada ketiga heatmap. Warna efek biru–netral–merah berpusat
di nol, dengan rentang simetris yang disesuaikan terhadap data masing-masing grafik.
Jangan membandingkan besar efek hanya dari intensitas warna antargrafik.

Cakupan tidak berubah: akurasi Regional dan Gender memakai Commonsense + Justice;
Religion memakai kecocokan label empat subset pada committed, dengan coverage.
Delta Regional adalah delta akurasi; delta Religion dan Gender adalah delta stance wrong.
'''


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def integrate_note(nb):
    nb.cells = [c for c in nb.cells if not c.metadata.get('harmonized_visuals')]
    cell = nbformat.v4.new_markdown_cell(NOTE)
    cell.metadata['harmonized_visuals'] = True
    nb.cells.insert(1, cell)


def refresh_index(folder):
    path = folder / 'output_index.csv'
    index = pd.read_csv(path)
    for i, row in index.iterrows():
        if row['kind'] != 'figure':
            continue
        relative = row.get('relative_path', row['file'])
        file = folder / relative
        assert file.exists(), file
        index.loc[i, 'sha256'] = sha(file)
    index.to_csv(path, index=False)
    manifest_path = folder / 'bundle_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if folder == FOLDERS['gender'] and (folder / 'figures' / f'{GENDER_HOLM_STEM}.png').exists():
        for key in ['figure_files', 'active_figure_files']:
            manifest[key] = sorted(set(manifest[key]) | {f'{GENDER_HOLM_STEM}.{ext}' for ext in ['png', 'svg']})
        manifest['figure_count'] = len(manifest['figure_files'])
        manifest['active_parent_figure_count'] = len(manifest['active_figure_files']) // 2
    manifest['visual_harmonization'] = {
        'palette': PAIRED_COLORS, 'accuracy_decimals': 3, 'accuracy_annotation_pt': 20,
        'palette_reference': 'notebooks/regional/akurasi analysis/figures/12_regional_absolute_accuracy_explicit_implicit.svg',
        'accuracy_ylim': [0, 1], 'data_scope_changed': False,
        'regional_heatmap_comparisons_per_panel': [14, 14, 12, 12],
        'gender_heatmap_comparisons_per_panel': [7, 7, 4],
        'regional_pvalue_part1': 'exact_mcnemar_p_value',
        'regional_pvalue_part2': 'holm_adjusted_p_value',
        'significance_conventions_changed': False,
        'reproduce': 'python src/benchmarks/ethics/harmonize_regional_religion_ethics_visuals.py',
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')


def render_gender():
    folder = FOLDERS['gender']
    gender.old.configure_plot_style()
    accuracy = pd.read_csv(folder / 'tables/ethics_accuracy_by_model_persona_category.csv')
    original = gender.old._save
    def styled_save(fig, directory, stem):
        ax = fig.axes[0]
        values = gender.old._ordered_category_pivot(accuracy, 'accuracy_committed').to_numpy().T.ravel()
        assert len(ax.patches) == len(ax.texts) == 30
        assert np.allclose([b.get_height() for b in ax.patches], values)
        assert ax.get_ylim() == (0, 1)
        ax.set_xlabel('Model', fontsize=19)
        ax.set_ylabel('Akurasi (committed)', fontsize=19)
        ax.tick_params(axis='both', labelsize=17)
        for text in ax.get_legend().get_texts():
            text.set_fontsize(17)
        ax.get_legend().get_title().set_fontsize(18)
        fig.tight_layout()
        return original(fig, directory, stem)
    gender.old._save = styled_save
    gender.old.plot_accuracy(accuracy, folder / 'figures')
    gender.old._save = original
    save = gender.save_with_font_increase
    def delta_only(fig, directory, stem):
        if '_delta_' in stem:
            return save(fig, directory, stem)
        plt.close(fig)
        return []
    gender.save_with_font_increase = delta_only
    gender.plot_combined(pd.read_csv(folder / 'tables/ethics_combined_four_subsets.csv'), write_tables=False)
    gender.save_with_font_increase = save
    inference = pd.read_csv(folder / 'tables/ethics_accuracy_inference_construct_valid_combined.csv')
    assert len(inference) == 108
    plot_regional_four_panels(inference, folder / 'figures',
        row_order=[gender.old.MODEL_LABELS[m] for m in gender.old.MODEL_ORDER],
        stems=['08_ethics_gender_combined_effect_holm_heatmap',
               '09_ethics_gender_combined_exact_mcnemar_and_holm_pvalue_heatmap', GENDER_HOLM_STEM],
        title_domain='Gender', identity_groups=[('Man', ['man']), ('Woman', ['woman']),
                                              ('Man–Woman', ['man_vs_woman'])],
        expected_counts=(7, 7, 4))
    path = folder / 'output_index.csv'
    index = pd.read_csv(path)
    index = index[~index.file.str.contains(GENDER_HOLM_STEM, regex=False)]
    rows = []
    for ext in ['png', 'svg']:
        file = folder / 'figures' / f'{GENDER_HOLM_STEM}.{ext}'
        rows.append({'kind': 'figure', 'file': file.name, 'path': str(file),
                     'relative_path': file.relative_to(folder).as_posix(), 'sha256': sha(file)})
    pd.concat([index, pd.DataFrame(rows)], ignore_index=True).to_csv(path, index=False)
    path = folder / 'active_figures.csv'
    active = pd.read_csv(path)
    active = active[~active.figure.str.contains(GENDER_HOLM_STEM, regex=False)]
    notebook, = folder.glob('*.ipynb')
    extra = [{'notebook': notebook.name, 'figure': f'figures/{GENDER_HOLM_STEM}.{ext}', 'status': 'active'}
             for ext in ['png', 'svg']]
    pd.concat([active, pd.DataFrame(extra)], ignore_index=True).to_csv(path, index=False)


def integrate_gender(nb):
    folder = FOLDERS['gender']
    for cell in nb.cells:
        if 'five_category_hashes = ' in cell.source:
            paths = ['figures/01_ethics_absolute_accuracy_five_categories.png',
                     'figures/01_ethics_absolute_accuracy_five_categories.svg',
                     'tables/ethics_accuracy_by_model_persona_category.csv']
            mapping = {p: sha(folder / p) for p in paths}
            cell.source = re.sub(r'five_category_hashes = [^\n]+', 'five_category_hashes = ' + repr(mapping), cell.source)
        if 'GROUPS = {' in cell.source and GENDER_HOLM_STEM not in cell.source:
            old = '"09_ethics_gender_combined_exact_mcnemar_and_holm_pvalue_heatmap",'
            cell.source = cell.source.replace(old, old + f'\n        "{GENDER_HOLM_STEM}",')
            cell.source = cell.source.replace('UTAMA — gabungan empat subset tanpa pemisahan subset',
                                               'UTAMA — ringkasan tanpa pemisahan subset')
            cell.source = cell.source.replace('5 chart utama, 13 chart pendukung, dan 8 chart detail ditampilkan.',
                                               '6 chart utama, 13 chart pendukung, dan 8 chart detail ditampilkan.')


def main():
    if '--metadata-only' in sys.argv:
        for folder in FOLDERS.values():
            refresh_index(folder)
        print('[PASS] Existing metadata/index entries refreshed.', flush=True)
        return
    if '--regional-subset-only' in sys.argv:
        folder = FOLDERS['regional']
        before = {p: sha(p) for p in (folder / 'tables').glob('*.csv')}
        regional.configure_plot_style()
        regional.plot_accuracy_statistics(pd.read_csv(folder / 'tables/ethics_accuracy_inference_construct_valid.csv'), folder / 'figures')
        refresh_index(folder)
        notebook_path, = folder.glob('*.ipynb')
        nb = nbformat.read(notebook_path, as_version=4)
        integrate_note(nb)
        print('[NOTEBOOK] Refresh Regional after row-label refinement', flush=True)
        NotebookClient(nb, timeout=600, kernel_name='python3',
                       resources={'metadata': {'path': str(folder)}}).execute()
        nbformat.validate(nb)
        assert not any(o.output_type == 'error' for c in nb.cells for o in c.get('outputs', []))
        nbformat.write(nb, notebook_path)
        assert all(sha(p) == value for p, value in before.items())
        print('[PASS] Regional row-label checks, notebook execution, and source hashes passed.', flush=True)
        return
    before = {p: sha(p) for folder in FOLDERS.values() for p in (folder / 'tables').glob('*.csv')}
    reference = ROOT / 'notebooks/regional/akurasi analysis/figures/12_regional_absolute_accuracy_explicit_implicit.svg'
    reference_text = reference.read_text(encoding='utf-8').lower()
    assert all(color.lower() in reference_text for color in PAIRED_COLORS)
    folder = FOLDERS['regional']
    regional.configure_plot_style()
    accuracy = pd.read_csv(folder / 'tables/ethics_accuracy_by_model_persona_category.csv')
    old_save = regional._save
    def checked_save(fig, directory, stem):
        if stem == '01_ethics_regional_absolute_accuracy_nine_categories':
            ax = fig.axes[0]
            expected = regional._category_pivot(accuracy, 'accuracy_committed').to_numpy().T.ravel()
            assert np.allclose([b.get_height() for b in ax.patches], expected)
            assert tuple(ax.get_ylim()) == (0, 1)
            assert len(ax.texts) == 54
            assert all(t.get_fontsize() == 20 and t.get_text() == f'{v:.3f}'
                       for t, v in zip(ax.texts, expected, strict=True))
            ax.set_xlabel('Model', fontsize=19)
            ax.set_ylabel('Akurasi (committed)', fontsize=19)
            ax.tick_params(axis='both', labelsize=17)
            for text in ax.get_legend().get_texts():
                text.set_fontsize(17)
            ax.get_legend().get_title().set_fontsize(18)
            fig.tight_layout()
        return old_save(fig, directory, stem)
    regional._save = checked_save
    regional.plot_accuracy(accuracy, folder / 'figures')
    regional.plot_delta(accuracy, folder / 'figures', implicit=False)
    regional.plot_delta(accuracy, folder / 'figures', implicit=True)
    print('[BARS] Regional accuracy: 54 unchanged values, 3 decimals, 20 pt, scale 0–1', flush=True)
    regional.plot_accuracy_statistics(pd.read_csv(folder / 'tables/ethics_accuracy_inference_construct_valid.csv'), folder / 'figures')
    if '--finalize-palette' not in sys.argv:
        regional.plot_combined_accuracy_statistics(pd.read_csv(folder / 'tables/ethics_accuracy_inference_construct_valid_combined.csv'), folder / 'figures')
    regional._save = old_save

    folder = FOLDERS['religion']
    religion_accuracy.render(pd.read_csv(religion_accuracy.SOURCE))
    religion.renderer.configure_engine()
    regional.configure_plot_style()
    original_save = religion.save_plus_two
    def delta_only(fig, directory, stem):
        if '_delta_' in stem:
            return original_save(fig, directory, stem)
        plt.close(fig)
        return []
    religion.save_plus_two = delta_only
    religion.plot_combined(pd.read_csv(folder / 'tables/ethics_religion_combined_four_subsets.csv'), write_tables=False)
    religion.save_plus_two = original_save
    render_gender()

    for domain, folder in FOLDERS.items():
        refresh_index(folder)
        notebook_path, = folder.glob('*.ipynb')
        nb = nbformat.read(notebook_path, as_version=4)
        integrate_note(nb)
        if domain == 'gender':
            integrate_gender(nb)
        print(f'[NOTEBOOK] Execute existing {domain} notebook', flush=True)
        NotebookClient(nb, timeout=600, kernel_name='python3',
                       resources={'metadata': {'path': str(folder)}}).execute()
        nbformat.validate(nb)
        assert not any(o.output_type == 'error' for c in nb.cells for o in c.get('outputs', []))
        nbformat.write(nb, notebook_path)
    assert all(sha(path) == digest for path, digest in before.items())
    print(f'[PASS] Three notebooks executed; all {len(before)} source CSV hashes unchanged.', flush=True)


if __name__ == '__main__':
    main()
