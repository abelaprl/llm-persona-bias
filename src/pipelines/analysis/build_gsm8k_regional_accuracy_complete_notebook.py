"""Build the Regional Cultural GSM8K accuracy presentation notebook.

The notebook is a read-only presentation layer over the validated v5 tables
and the additive standardized Regional figures.  It does not rerun inference,
parsing, scoring, or statistical tests.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import nbformat as nbf


REPO_ROOT = Path(__file__).resolve().parents[3]
NOTEBOOK_DIR = REPO_ROOT / "notebooks" / "regional" / "akurasi analysis"
NOTEBOOK_PATH = NOTEBOOK_DIR / "GSM8K_Regional_Cultural_Accuracy_Analysis_Lengkap.ipynb"
FIGURE_DIR = NOTEBOOK_DIR / "figures"
TABLE_DIR = (
    REPO_ROOT
    / "result"
    / "GMS8K"
    / "reports"
    / "persona_publication_analysis_v5_paired_significance_six_models_"
    "bar3_compact_smalllabels_heatmap3_sigbold"
    / "tables"
)
FIGURES = (
    "12_regional_absolute_accuracy_explicit_implicit",
    "13_regional_delta_implicit_vs_explicit",
    "27_regional_all_52_effect_holm_stacked_26_26",
)
TABLES = (
    "accuracy_by_persona_six_models.csv",
    "regional_accuracy_and_deltas.csv",
    "regional_paired_effects_significance.csv",
    "regional_all_52_paired_effects_significance.csv",
    "cochran_q_omnibus_all_groups.csv",
    "regional_completion_token_delta_implicit_vs_explicit.csv",
    "neutral_persona_accuracy_six_models.csv",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


def build_notebook(figure_hashes: dict[str, str], table_hashes: dict[str, str]):
    figure_literal = json.dumps(figure_hashes, indent=2, ensure_ascii=False)
    table_literal = json.dumps(table_hashes, indent=2, ensure_ascii=False)
    cells = [
        markdown(
            """
# Analisis Akurasi Persona Regional-Kultural pada GSM8K - Enam Model Utama

Notebook ini adalah lapisan presentasi read-only untuk tiga visual akurasi
Regional-Kultural yang dirujuk Bab VI: akurasi absolut, delta Implicit-Explicit,
dan heatmap ukuran efek Cohen's h dengan keputusan Holm.

Sumber statistik tetap bundle v5 yang tervalidasi. Notebook tidak memanggil API,
tidak melakukan parsing/scoring ulang, dan tidak menghitung ulang uji statistik.
"""
        ),
        markdown(
            """
## 1. SPEC dan provenance

- Full GSM8K test split: 1.319 soal.
- 6 model x 16 kondisi persona Regional (empat region, masing-masing E dan I1-I3).
- Neutral pada Figure 12 hanya baseline visual deskriptif yang dipinjam dari
  tabel Neutral Gender untuk model yang sama; sumber Regional tidak menyediakan
  pasangan Neutral dengan protokol inferensi yang sama.
- Perbandingan utama I1-I3 terhadap E tetap dibaca dari 72 pasangan tingkat soal,
  Exact McNemar, Cochran's Q, dan koreksi Holm yang sudah tersimpan.
"""
        ),
        code(
            """
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
from IPython.display import Image, Markdown, display

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 400)

EXPECTED_FIGURE_HASHES = """
            + figure_literal
            + """
EXPECTED_TABLE_HASHES = """
            + table_literal
            + """

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
    raise FileNotFoundError('Jalankan dari D:/TugasAkhir atau subfoldernya.')

REPO_ROOT = find_repo_root()
TABLE_DIR = REPO_ROOT / 'result' / 'GMS8K' / 'reports' / 'persona_publication_analysis_v5_paired_significance_six_models_bar3_compact_smalllabels_heatmap3_sigbold' / 'tables'
FIGURE_DIR = REPO_ROOT / 'notebooks' / 'regional' / 'akurasi analysis' / 'figures'
print('Tabel v5 read-only:', TABLE_DIR)
print('Gambar lokal:', FIGURE_DIR)
"""
        ),
        markdown("## 2. Validation gate"),
        code(
            """
accuracy_all = pd.read_csv(TABLE_DIR / 'accuracy_by_persona_six_models.csv', low_memory=False)
regional_accuracy = accuracy_all[accuracy_all['dataset'].eq('Regional Cultural')].copy()
regional_summary = pd.read_csv(TABLE_DIR / 'regional_accuracy_and_deltas.csv')
regional_pairs = pd.read_csv(TABLE_DIR / 'regional_paired_effects_significance.csv', low_memory=False)
regional_all = pd.read_csv(TABLE_DIR / 'regional_all_52_paired_effects_significance.csv', low_memory=False)
omnibus = pd.read_csv(TABLE_DIR / 'cochran_q_omnibus_all_groups.csv', low_memory=False)
regional_omnibus = omnibus[omnibus['dataset'].eq('Regional Cultural')].copy()

for name, expected in EXPECTED_TABLE_HASHES.items():
    assert sha256_file(TABLE_DIR / name) == expected, name
for name, expected in EXPECTED_FIGURE_HASHES.items():
    path = FIGURE_DIR / name
    assert path.is_file() and path.stat().st_size > 0, path
    assert sha256_file(path) == expected, path

validation = pd.DataFrame([
    ('Model', regional_accuracy['model'].nunique() == 6, regional_accuracy['model'].nunique()),
    ('Persona Regional', regional_accuracy['persona'].nunique() == 16, regional_accuracy['persona'].nunique()),
    ('Baris akurasi Regional', len(regional_accuracy) == 96, len(regional_accuracy)),
    ('Full test split', regional_accuracy['raw_total'].eq(1319).all(), sorted(regional_accuracy['raw_total'].unique().tolist())),
    ("Cochran's Q Regional", len(regional_omnibus) == 24, len(regional_omnibus)),
    ('Perbandingan I-E utama', len(regional_pairs) == 72, len(regional_pairs)),
    ('Seluruh 52 perbandingan', len(regional_all) == 312, len(regional_all)),
    ('Ringkasan visual', len(regional_summary) == 6, len(regional_summary)),
    ('Hash tabel v5', True, len(EXPECTED_TABLE_HASHES)),
    ('Hash PNG/SVG lokal', True, len(EXPECTED_FIGURE_HASHES)),
], columns=['gate', 'passed', 'observed'])
assert validation['passed'].all()
display(validation)
print('SEMUA VALIDATION GATE LULUS')
"""
        ),
        markdown("## 3. Tabel akurasi dan coverage seluruh kondisi"),
        code(
            """
columns = ['model_label', 'persona', 'raw_total', 'correct', 'total', 'accuracy_pct',
           'committed', 'commit_rate_pct', 'abstain', 'missing', 'technical_failures',
           'accuracy_committed_pct']
display(regional_accuracy[columns].sort_values(['model_label', 'persona']).reset_index(drop=True).style.format({
    'accuracy_pct': '{:.4f}%', 'commit_rate_pct': '{:.4f}%',
    'accuracy_committed_pct': '{:.4f}%'}))

display(Markdown('### Ringkasan yang dipakai untuk Figure 12 dan 13'))
display(regional_summary.style.format(precision=4))
"""
        ),
        markdown("## 4. Visual akurasi dan delta Regional-Kultural"),
        code(
            """
def show_figure(name: str, width: int) -> None:
    display(Markdown(f'### `{name}`'))
    display(Image(filename=str(FIGURE_DIR / name), width=width))

show_figure('12_regional_absolute_accuracy_explicit_implicit.png', 1550)
show_figure('13_regional_delta_implicit_vs_explicit.png', 1500)
"""
        ),
        markdown(
            """
## 5. Exact McNemar, Cochran's Q, Cohen's h, dan Holm

Figure 27 berisi 52 perbandingan per model: sensitivitas terhadap Neutral,
perbandingan Implicit-Explicit, dan perbandingan lintas region. Tanda bintang
dan cetak tebal pada sel hanya berasal dari `significant_holm_0_05`; arti tanda
lainnya mengikuti caption Bab VI. Neutral pada bagian sensitivitas adalah
protokol deskriptif yang berbeda, bukan uji Regional-vs-Neutral utama.
"""
        ),
        code(
            """
family_audit = (regional_all.groupby(['model_label', 'family', 'family_size'], dropna=False)
                .agg(comparisons=('comparison_label', 'size'),
                     holm_significant=('significant_holm_0_05', 'sum'))
                .reset_index())
assert (family_audit['comparisons'].astype(int) == family_audit['family_size'].astype(int)).all()
display(family_audit)

paired_columns = ['model_label', 'identity', 'analysis_scope', 'family', 'family_size',
                  'comparison_label', 'n_pairs', 'accuracy_a_pct', 'accuracy_b_pct',
                  'risk_difference_pp', 'cohen_h', 'mcnemar_exact_p_value',
                  'holm_adjusted_p_value', 'significant_holm_0_05',
                  'decision_significant_0_05', 'supports_primary_claim']
display(regional_pairs[paired_columns].sort_values(
    ['model_label', 'identity', 'comparison_label']).reset_index(drop=True))
show_figure('27_regional_all_52_effect_holm_stacked_26_26.png', 1450)
"""
        ),
        markdown("## 6. Token dan batas interpretasi"),
        code(
            """
tokens = pd.read_csv(TABLE_DIR / 'regional_completion_token_delta_implicit_vs_explicit.csv')
display(tokens)
neutral_accuracy = pd.read_csv(TABLE_DIR / 'neutral_persona_accuracy_six_models.csv')
display(Markdown('### Neutral yang menjadi baseline visual'))
display(neutral_accuracy)
print('Rerata tiga kondisi implicit adalah ringkasan deskriptif; inferensi utama tetap per I1, I2, dan I3.')
"""
        ),
        markdown(
            """
## 7. CHECKLIST dan integration

- PNG/SVG dibuat ulang dari tabel v5 tanpa mengubah tabel statistik.
- Judul utama/panel bold; angka batang, sumbu, dan legenda normal.
- Angka delta vertikal dan ditempatkan di luar batang.
- Heatmap hanya menebalkan sel yang signifikan setelah Holm; footnote visual lama
  tidak dimasukkan, sedangkan definisi tanda dijelaskan di notebook/caption.
- Tidak ada API call, inference, parsing, rescoring, atau uji statistik baru.
- Klaim dibatasi sebagai sensitivitas performa model terhadap prompt persona.
"""
        ),
        code(
            """
png = [name for name in EXPECTED_FIGURE_HASHES if name.endswith('.png')]
svg = [name for name in EXPECTED_FIGURE_HASHES if name.endswith('.svg')]
for name in [*png, *svg]:
    path = FIGURE_DIR / name
    assert path.is_file() and path.stat().st_size > 0, path

summary = pd.Series({
    'models': regional_accuracy['model'].nunique(),
    'conditions': 16,
    'accuracy_rows': len(regional_accuracy),
    'primary_paired_comparisons': len(regional_pairs),
    'all_52_comparisons': len(regional_all),
    'holm_significant_all_52': int(regional_all['significant_holm_0_05'].sum()),
    'primary_decision_significant': int(regional_pairs['supports_primary_claim'].sum()),
    'png': len(png), 'svg': len(svg), 'api_calls': 0, 'rescoring': False,
}, name='value').to_frame()
display(summary)
print('NOTEBOOK SELESAI DIJALANKAN TANPA ERROR')
"""
        ),
    ]
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "TA .venv",
            "language": "python",
            "name": "ta-venv",
        },
        "language_info": {"name": "python", "version": "3"},
    }
    return notebook


def main() -> int:
    for path in (TABLE_DIR, FIGURE_DIR):
        if not path.exists():
            raise FileNotFoundError(path)
    figure_hashes = {}
    for stem in FIGURES:
        for extension in ("png", "svg"):
            path = FIGURE_DIR / f"{stem}.{extension}"
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(path)
            figure_hashes[path.name] = sha256(path)
    table_hashes = {}
    for name in TABLES:
        path = TABLE_DIR / name
        if not path.is_file():
            raise FileNotFoundError(path)
        table_hashes[name] = sha256(path)

    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    nbf.write(build_notebook(figure_hashes, table_hashes), NOTEBOOK_PATH)
    print(f"[CREATED] {NOTEBOOK_PATH}")
    print(f"[HASHES] {len(figure_hashes)} figures, {len(table_hashes)} tables")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
