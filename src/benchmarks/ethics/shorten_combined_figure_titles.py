"""Refresh combined ETHICS figure titles from existing numeric tables."""

import base64
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src/pipelines/analysis'))
import pandas as pd
import nbformat
import render_gender_combined_existing as gender
import update_religion_ethics_existing as religion
import render_ethics_regional_overall_standardized as regional
import ethics_accuracy_statistical_figures as stats


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    folders = [ROOT / 'notebooks' / d / 'ethics accuracy analysis'
               for d in ['gender', 'religion', 'regional']]
    old_images = {p: digest(p) for folder in folders for p in (folder / 'figures').glob('*.png')}
    old_assets = {p: digest(p) for folder in folders for p in (folder / 'figures').glob('*') if p.is_file()}
    numeric_hashes = {p: digest(p) for folder in folders for p in (folder / 'tables').glob('*.csv')}
    original_save = stats._save

    # Render the Regional engine before the Religion adapter changes its globals.
    folder = folders[2]
    regional.configure_plot_style()
    regional.plot_combined_accuracy_statistics(
        pd.read_csv(folder / 'tables/ethics_accuracy_inference_construct_valid_combined.csv'), folder / 'figures')
    print('[RENDER] Regional titles refreshed', flush=True)

    folder = folders[0]
    gender.old.configure_plot_style()
    stats._save = gender.save_with_font_increase
    gender.old.plot_combined_accuracy_statistics(
        pd.read_csv(folder / 'tables/ethics_accuracy_inference_construct_valid_combined.csv'), folder / 'figures')
    # Only the distribution title changed; leave the two delta assets intact.
    gender_save = gender.save_with_font_increase
    def save_distribution(fig, figure_dir, stem):
        if stem.endswith('_distribution'):
            return gender_save(fig, figure_dir, stem)
        gender.plt.close(fig)
        return []
    gender.save_with_font_increase = save_distribution
    gender.plot_combined(pd.read_csv(folder / 'tables/ethics_combined_four_subsets.csv'), write_tables=False)
    stats._save = original_save

    folder = folders[1]
    religion.renderer.configure_engine()
    regional.configure_plot_style()
    # Archived legacy figures must not be recreated in the active figures folder.
    if (folder / 'figures/08_ethics_religion_combined_effect_holm_heatmap.png').exists():
        regional.plot_combined_accuracy_statistics(
            pd.read_csv(folder / 'tables/ethics_accuracy_inference_construct_valid_combined.csv'), folder / 'figures')
    religion.plot_combined_inference(pd.read_csv(folder / 'tables/ethics_religion_combined_paired_effects_all.csv'))
    religion_save = religion.save_plus_two
    def save_religion_distribution(fig, figure_dir, stem):
        if stem.endswith('_distribution'):
            return religion_save(fig, figure_dir, stem)
        religion.plt.close(fig)
        return []
    religion.save_plus_two = save_religion_distribution
    religion.plot_combined(pd.read_csv(folder / 'tables/ethics_religion_combined_four_subsets.csv'), write_tables=False)

    changed = {p: digest(p) for p, old in old_assets.items() if digest(p) != old}
    replacements = {old_assets[p]: new for p, new in changed.items()}
    image_replacements = {old_images[p]: base64.b64encode(p.read_bytes()).decode('ascii')
                          for p in changed if p.suffix == '.png'}
    for folder in folders:
        # Update asset checksums without changing scope or statistical metadata.
        for name in ['output_index.csv', 'bundle_manifest.json']:
            path = folder / name
            if path.exists():
                text = path.read_text(encoding='utf-8')
                for old, new in replacements.items():
                    text = text.replace(old, new)
                path.write_text(text, encoding='utf-8')
        notebook_path = next(folder.glob('*Lengkap.ipynb'))
        notebook = nbformat.read(notebook_path, as_version=4)
        refreshed = 0
        for cell in notebook.cells:
            # Keep existing executed results and all code except embedded hashes.
            for old, new in replacements.items():
                cell.source = cell.source.replace(old, new)
            for output in cell.get('outputs', []):
                payload = output.get('data', {}).get('image/png')
                if payload:
                    key = hashlib.sha256(base64.b64decode(payload)).hexdigest()
                    if key in image_replacements:
                        output.data['image/png'] = image_replacements[key]
                        refreshed += 1
        nbformat.validate(notebook)
        nbformat.write(notebook, notebook_path)
        print(f'[NOTEBOOK] {notebook_path.name}: {refreshed} embedded figures refreshed', flush=True)

    assert all(digest(p) == old for p, old in numeric_hashes.items())
    for folder in folders:
        for path in (folder / 'figures').glob('*combined*.svg'):
            text = path.read_text(encoding='utf-8')
            assert 'Gabungan Empat Subset' not in text
            assert 'Gabungan Commonsense dan Justice' not in text
    print(f'[PASS] {len(changed)} PNG/SVG assets refreshed; numeric tables unchanged.', flush=True)


if __name__ == '__main__':
    main()
