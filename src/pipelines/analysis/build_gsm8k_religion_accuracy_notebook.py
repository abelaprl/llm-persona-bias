"""Build the standalone GSM8K Religion accuracy notebook."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import nbformat as nbf

REPO_ROOT = Path(__file__).resolve().parents[3]
NOTEBOOK_DIR = REPO_ROOT / "notebooks" / "religion" / "akurasi analysis"
NOTEBOOK_PATH = NOTEBOOK_DIR / "GSM8K_Religion_Accuracy_Analysis_Lengkap.ipynb"
FIGURE_DIR = NOTEBOOK_DIR / "figures"
TABLE_DIR = (REPO_ROOT / "result" / "GMS8K" / "reports"
             / "persona_publication_analysis_v5_paired_significance_six_models_bar3_compact_smalllabels_heatmap3_sigbold"
             / "tables")
FIGURE_STEMS = (
    "05_religion_absolute_accuracy_explicit_implicit",
    "06_religion_delta_implicit_vs_explicit",
    "26_religion_all_52_effect_holm_stacked_26_26",
)
TABLE_NAMES = (
    "accuracy_by_persona_six_models.csv",
    "neutral_persona_accuracy_six_models.csv",
    "religion_explicit_implicit_and_deltas.csv",
    "religion_paired_effects_significance.csv",
    "cochran_q_omnibus_all_groups.csv",
    "token_category_summary_six_models.csv",
    "religion_completion_token_delta_implicit_vs_explicit_pct.csv",
    "religion_reasoning_token_delta_implicit_vs_explicit_pct.csv",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def py(text: str):
    return nbf.v4.new_code_cell(text.strip())


def build_notebook(
    figure_hashes: dict[str, str],
    table_hashes: dict[str, str],
    manifest_hash: str,
):
    cells = [
        md("""
# Analisis Akurasi Persona Agama pada GSM8K

Notebook tunggal ini memindahkan visual akurasi Religion yang dipakai Bab VI ke satu
lokasi. Bentuk figure, urutan model/persona, warna seri, label, dan 52 perbandingan
heatmap mengikuti figure Bab VI. Standardisasi hanya menyentuh tipografi angka dan judul.

Notebook ini tidak memanggil API, inference, parsing ulang, rescoring, atau menghitung
ulang uji statistik. Statistik dibaca dari bundle v5.
"""),
        md("""
## 1. Provenance

Bab VI memanggil Figure 05, 06, dan 26. Sumber statistiknya adalah folder `tables`
pada bundle v5. Figure lokal dibuat dari tabel yang sama dan disimpan di folder
`figures` yang berdampingan dengan notebook ini.

Rata-rata I1–I3 pada Figure 05/06 adalah agregasi deskriptif. Hasil inferensial tetap
berasal dari kolom `significant_holm_0_05`, `decision_significant_0_05`, dan
`supports_primary_claim` yang sudah tersimpan.
"""),
        py(f"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import pandas as pd
from IPython.display import Image, Markdown, display

EXPECTED_FIGURE_HASHES = {json.dumps(figure_hashes, indent=2)}
EXPECTED_TABLE_HASHES = {json.dumps(table_hashes, indent=2)}
EXPECTED_MANIFEST_HASH = {manifest_hash!r}

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def find_repo_root() -> Path:
    for path in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:
        if (path / 'result' / 'GMS8K').is_dir() and (path / 'notebooks').is_dir():
            return path
    raise FileNotFoundError('Root D:/TugasAkhir tidak ditemukan.')

REPO_ROOT = find_repo_root()
TABLE_DIR = REPO_ROOT / 'result' / 'GMS8K' / 'reports' / 'persona_publication_analysis_v5_paired_significance_six_models_bar3_compact_smalllabels_heatmap3_sigbold' / 'tables'
FIGURE_DIR = REPO_ROOT / 'notebooks' / 'religion' / 'akurasi analysis' / 'figures'
MANIFEST_PATH = FIGURE_DIR / 'render_manifest.json'
print('Statistical source:', TABLE_DIR)
print('Local figure source:', FIGURE_DIR)
"""),
        md("## 2. Validation gate"),
        py("""
accuracy_all = pd.read_csv(TABLE_DIR / 'accuracy_by_persona_six_models.csv', low_memory=False)
religion_accuracy = accuracy_all[accuracy_all['dataset'].eq('Religion')].copy()
neutral_accuracy = pd.read_csv(TABLE_DIR / 'neutral_persona_accuracy_six_models.csv')
religion_pairs = pd.read_csv(TABLE_DIR / 'religion_paired_effects_significance.csv', low_memory=False)
omnibus_all = pd.read_csv(TABLE_DIR / 'cochran_q_omnibus_all_groups.csv', low_memory=False)
religion_omnibus = omnibus_all[omnibus_all['dataset'].eq('Religion')].copy()

for name, expected in EXPECTED_TABLE_HASHES.items():
    assert sha256_file(TABLE_DIR / name) == expected, name
for name, expected in EXPECTED_FIGURE_HASHES.items():
    path = FIGURE_DIR / name
    assert path.is_file() and path.stat().st_size > 0, path
    assert sha256_file(path) == expected, path
assert MANIFEST_PATH.is_file()
assert sha256_file(MANIFEST_PATH) == EXPECTED_MANIFEST_HASH
figure_manifest = json.loads(MANIFEST_PATH.read_text(encoding='utf-8'))
manifest_figures = {item['file']: item['sha256'] for item in figure_manifest['figures']}
assert manifest_figures == EXPECTED_FIGURE_HASHES
assert figure_manifest['heatmap_footnote'] is False

assert religion_accuracy['model'].nunique() == 6
assert religion_accuracy['persona'].nunique() == 16
assert len(religion_accuracy) == 96
assert religion_accuracy['raw_total'].eq(1319).all()
assert len(neutral_accuracy) == 6
assert len(religion_omnibus) == 48
assert religion_omnibus['analysis_scope'].value_counts().to_dict() == {'primary': 24, 'sensitivity': 24}
assert len(religion_pairs) == 312
assert religion_pairs.groupby('model').size().eq(52).all()
assert religion_pairs['family_size'].astype(int).value_counts().to_dict() == {24: 144, 4: 96, 3: 72}

validation = pd.DataFrame([
    ('6 model', True, religion_accuracy['model'].nunique()),
    ('16 persona + 1 neutral', religion_accuracy['persona'].nunique() == 16 and len(neutral_accuracy) == 6, 17),
    ('96 baris persona', len(religion_accuracy) == 96, len(religion_accuracy)),
    ('1.319 soal pada persona', religion_accuracy['raw_total'].eq(1319).all(), religion_accuracy['raw_total'].unique().tolist()),
    ("48 kelompok Cochran's Q", len(religion_omnibus) == 48, len(religion_omnibus)),
    ('312 paired comparisons', len(religion_pairs) == 312, len(religion_pairs)),
    ('Holm family 3/4/24', True, '72/96/144'),
    ('Hash tabel v5', True, len(EXPECTED_TABLE_HASHES)),
    ('Hash figure PNG/SVG', True, len(EXPECTED_FIGURE_HASHES)),
    ('Hash render manifest', True, 1),
], columns=['gate', 'passed', 'observed'])
assert validation['passed'].all()
display(validation)
print('SEMUA VALIDATION GATE LULUS')
"""),
        md("## 3. Akurasi, coverage, dan neutral"),
        py("""
accuracy_columns = ['model_label','persona','raw_total','correct','total','accuracy_pct',
                    'committed','commit_rate_pct','abstain','missing','technical_failures',
                    'accuracy_committed_pct']
display(religion_accuracy[accuracy_columns].sort_values(['model_label','persona']).reset_index(drop=True))
display(Markdown('### Neutral — tabel baseline terpisah'))
display(neutral_accuracy)
display(Markdown('### Ringkasan Explicit dan Implicit'))
display(pd.read_csv(TABLE_DIR / 'religion_explicit_implicit_and_deltas.csv'))
"""),
        md("## 4. Visual akurasi dan delta"),
        py("""
def show_figure(name: str, width: int = 1400) -> None:
    path = FIGURE_DIR / name
    assert path.is_file()
    display(Markdown(f'### `{name}`'))
    display(Image(filename=str(path), width=width))

show_figure('05_religion_absolute_accuracy_explicit_implicit.png', 1550)
show_figure('06_religion_delta_implicit_vs_explicit.png', 1500)
"""),
        md("""
## 5. Paired tests, effect size, dan Holm

Family Holm dipertahankan terpisah: family 3 untuk I1–I3, family 4 untuk perbandingan
terhadap N, dan family 24 untuk antarpersona. Figure 26 menampilkan seluruh 52
perbandingan per model. `*` dan cetak tebal pada sel berarti Holm signifikan.
"""),
        py("""
family_audit = (religion_pairs.groupby(['model_label','analysis_scope','family','family_size'], dropna=False)
                .agg(comparisons=('comparison_label','size'),
                     holm_significant=('significant_holm_0_05','sum'),
                     decision_significant=('decision_significant_0_05','sum'))
                .reset_index())
assert (family_audit['comparisons'].astype(int) == family_audit['family_size'].astype(int)).all()
display(family_audit)

paired_columns = ['model_label','identity','analysis_scope','family','family_size','comparison_label',
                  'n_pairs','accuracy_a_pct','accuracy_b_pct','risk_difference_pp','cohen_h',
                  'mcnemar_exact_p_value','holm_adjusted_p_value','omnibus_cochran_q_p_value',
                  'significant_holm_0_05','decision_significant_0_05','supports_primary_claim']
display(religion_pairs[paired_columns].sort_values(['analysis_scope','model_label','family','comparison_label']).reset_index(drop=True))
show_figure('26_religion_all_52_effect_holm_stacked_26_26.png', 1450)
"""),
        md("## 6. Token dan batas interpretasi"),
        py("""
token_table = pd.read_csv(TABLE_DIR / 'token_category_summary_six_models.csv')
display(token_table[token_table['dataset'].eq('Religion')].reset_index(drop=True))
display(pd.read_csv(TABLE_DIR / 'religion_completion_token_delta_implicit_vs_explicit_pct.csv'))
display(pd.read_csv(TABLE_DIR / 'religion_reasoning_token_delta_implicit_vs_explicit_pct.csv'))
"""),
        md("""
## 7. Checklist

- Figure 05, 06, dan 26 mempertahankan bentuk Bab VI.
- Judul utama 23 pt bold; judul panel heatmap 19 pt bold.
- Label sumbu, legenda, dan angka batang normal; angka delta 16 pt normal dan berada di luar batang.
- Angka heatmap hanya bold serta bertanda `*` jika signifikan setelah Holm.
- Tidak ada API call, inference, parsing ulang, rescoring, atau perubahan tabel/raw data.
- Metrik dibaca sebagai sensitivitas performa terhadap prompt persona, bukan kemampuan kelompok agama.
"""),
        py("""
summary = pd.Series({
    'models': religion_accuracy['model'].nunique(),
    'religion_persona_conditions': religion_accuracy['persona'].nunique(),
    'neutral_conditions': len(neutral_accuracy),
    'paired_comparisons': len(religion_pairs),
    'holm_significant': int(religion_pairs['significant_holm_0_05'].sum()),
    'primary_claim_significant': int(religion_pairs['supports_primary_claim'].sum()),
    'png': 3, 'svg': 3, 'api_calls': 0, 'rescoring': False,
}, name='value').to_frame()
display(summary)
print('NOTEBOOK SELESAI DIJALANKAN TANPA ERROR')
"""),
    ]
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook['metadata'] = {
        'kernelspec': {'display_name': 'TA .venv', 'language': 'python', 'name': 'ta-venv'},
        'language_info': {'name': 'python', 'version': '3'},
    }
    return notebook


def main() -> int:
    figure_hashes = {}
    for stem in FIGURE_STEMS:
        for extension in ('png', 'svg'):
            path = FIGURE_DIR / f'{stem}.{extension}'
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(path)
            figure_hashes[path.name] = sha256(path)
    table_hashes = {}
    for name in TABLE_NAMES:
        path = TABLE_DIR / name
        if not path.is_file():
            raise FileNotFoundError(path)
        table_hashes[name] = sha256(path)
    manifest_path = FIGURE_DIR / "render_manifest.json"
    if not manifest_path.is_file() or manifest_path.stat().st_size == 0:
        raise FileNotFoundError(manifest_path)
    manifest_hash = sha256(manifest_path)
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    nbf.write(build_notebook(figure_hashes, table_hashes, manifest_hash), NOTEBOOK_PATH)
    readme = """# GSM8K Religion Accuracy Analysis

Visual dan notebook ini mempertahankan bentuk Figure 05, 06, dan 26 yang dirujuk
oleh `D:/ProposalTA-18222008/Bab VI - Evaluasi.tex` pada baris 750, 775, dan 842.
Renderer membaca tabel read-only dari bundle v5 pada folder `tables`. Figure 05/06
menggunakan `religion_explicit_implicit_and_deltas.csv`; Figure 26 menggunakan
`religion_paired_effects_significance.csv`.

`neutral` berasal dari `neutral_persona_accuracy_six_models.csv` karena baseline
tersebut disimpan terpisah dari 16 kondisi persona agama. Rata-rata I1–I3 adalah
agregasi deskriptif, bukan uji statistik baru. Uji inferensial tetap membaca hasil
v5: Cochran's Q, Exact McNemar, Cohen's h, dan koreksi Holm dalam family 3/4/24.

Standardisasi mengikuti notebook Gender/Regional: judul utama 23 pt bold, judul panel
heatmap 19 pt bold, angka batang 17 pt normal, angka delta 16 pt normal di luar batang,
dan label/legend normal. Keluarga warna identitas stabil: Atheist biru, Christian
merah, Islam ungu, dan Jewish oranye. Figure 26 tidak memuat footnote di dalam gambar.
`render_manifest.json` menyimpan hash figure dan seluruh tabel yang dibaca notebook.
Tidak ada API call, inference, parsing ulang, rescoring, atau perubahan output lama.
"""
    (NOTEBOOK_DIR / 'README.md').write_text(readme, encoding='utf-8')
    print(f'[CREATED] {NOTEBOOK_PATH}')
    print(f'[HASHES] {len(figure_hashes)} figures, {len(table_hashes)} tables')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
