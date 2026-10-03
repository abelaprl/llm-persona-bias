"""Update the existing ETHICS Religion notebook and figures in place."""

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src/pipelines/analysis"))

import matplotlib
matplotlib.use("Agg")
from matplotlib.text import Text
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nbformat
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import binomtest
from nbclient import NotebookClient
import render_ethics_religion_overall_standardized as renderer
import render_ethics_regional_overall_standardized as engine
from ethics_persona_visual_style import religion_colors

NOTEBOOK_DIR = ROOT / "notebooks/religion/ethics accuracy analysis"
FIGURE_DIR = NOTEBOOK_DIR / "figures"
TABLE_DIR = NOTEBOOK_DIR / "tables"
TARGET_NOTEBOOK = NOTEBOOK_DIR / "ETHICS_Religion_Accuracy_Analysis_Lengkap.ipynb"
FONT_AUDIT = []
SAVED = []
INFERENCE_STEMS = [
    '00_ethics_religion_combined_inference_effect_holm',
    '00_ethics_religion_combined_exact_mcnemar_four_panels',
    '00_ethics_religion_combined_holm_pvalue_four_panels',
]
PANEL_GROUPS = [
    ('Ateis dan Kristen', ['atheist', 'christian']),
    ('Islam dan Yahudi', ['islam', 'jewish']),
    ('Ateis dibandingkan identitas agama lain',
     ['atheist_vs_christian', 'atheist_vs_islam', 'atheist_vs_jewish']),
    ('Perbandingan lintas agama lainnya',
     ['christian_vs_islam', 'christian_vs_jewish', 'islam_vs_jewish']),
]


def plot_combined_inference(out):
    """Render existing statistics as four identity-aligned panels per metric."""
    model_order = [engine.MODEL_LABELS[m] for m in engine.MODEL_ORDER]
    assert len(out) == 312 and out.model_label.nunique() == 6
    assert out.subset.eq('all_four_subsets').all()
    assert not out.duplicated(['model_label', 'condition_a', 'condition_b']).any()
    source_h = (2 * np.arcsin(np.sqrt(out.accuracy_b_pct / 100))
                - 2 * np.arcsin(np.sqrt(out.accuracy_a_pct / 100)))
    # The pooled source uses B-A, while comparison_label uses A-B.
    # Negation is a display convention, not a new statistical test.
    assert np.allclose(out.cohens_h, source_h, atol=1e-12)
    frame = out.assign(display_h=-out.cohens_h)
    decisions = frame.significant_holm_0_05.astype(str).str.lower().map(
        {'true': True, 'false': False})
    assert decisions.notna().all()
    frame = frame.assign(display_significant=decisions)
    assert np.array_equal(decisions.to_numpy(), frame.holm_adjusted_p_value.lt(.05).to_numpy())
    assert frame.holm_adjusted_p_value.ge(frame.exact_mcnemar_p_value - 1e-12).all()
    panels = []
    for title, identities in PANEL_GROUPS:
        columns = []
        for identity in identities:
            group = frame.loc[frame.identity.eq(identity)].sort_values('display_order')
            labels = group.comparison_label.drop_duplicates().tolist()
            assert len(labels) == (4 if '_vs_' in identity else 7)
            columns.extend(labels)
        panels.append((title, columns))
    columns = [c for _, part in panels for c in part]
    assert len(columns) == len(set(columns)) == 52
    assert set(columns) == set(frame.comparison_label)
    effect_limit = max(.1, np.ceil(frame.display_h.abs().max() / .05) * .05)
    effect_cmap = LinearSegmentedColormap.from_list(
        'religion_readable', ['#0B3C78', '#4A98C9', '#DCEAF1', '#FCFBF8',
                              '#F7C7AE', '#E76F51', '#9B002F'])
    specs = [
        ('display_h', "Ukuran Efek dan Koreksi Holm", "Cohen's h (A − B)",
         effect_cmap, -effect_limit, effect_limit),
        ('exact_mcnemar_p_value', 'p-value Exact McNemar', 'p-value mentah (warna dibatasi 0–0,05)',
         'rocket', 0, .05),
        ('holm_adjusted_p_value', 'p-value Holm', 'p-value Holm (warna dibatasi 0–0,05)',
         'rocket', 0, .05),
    ]
    for stem, (metric, title, color_label, cmap, low, high) in zip(INFERENCE_STEMS, specs):
        fig = plt.figure(figsize=(18, 23))
        grid = fig.add_gridspec(5, 1, height_ratios=[6, 6, 6, 6, .42],
                               left=.225, right=.985, top=.91, bottom=.055, hspace=.82)
        color_ax = fig.add_subplot(grid[4])
        for index, (panel_title, part) in enumerate(panels):
            ax = fig.add_subplot(grid[index])
            values = frame.pivot(index='model_label', columns='comparison_label', values=metric).reindex(
                index=model_order, columns=part)
            significant = frame.pivot(index='model_label', columns='comparison_label',
                                      values='display_significant').reindex(index=model_order, columns=part)
            assert values.notna().all().all()
            annotations = values.copy().astype(object)
            for model in model_order:
                for comparison in part:
                    value = values.loc[model, comparison]
                    if metric == 'display_h':
                        label = ('0,00' if abs(value) < .005 else f'{value:+.2f}'.replace('.', ','))
                    else:
                        label = '<0,001' if value < .001 else f'{value:.3f}'.replace('.', ',')
                    annotations.loc[model, comparison] = label + ('*' if significant.loc[model, comparison] else '')
            sns.heatmap(values, ax=ax, cmap=cmap, vmin=low, vmax=high,
                        center=0 if metric == 'display_h' else None,
                        annot=annotations, fmt='',
                        annot_kws={'fontsize': 15.5 if metric == 'display_h' else 13},
                        linewidths=1, linecolor='#FCFBF8', cbar=index == 3,
                        cbar_ax=color_ax if index == 3 else None,
                        cbar_kws={'orientation': 'horizontal'})
            for artist, sig in zip(ax.texts, significant.to_numpy().ravel()):
                artist.set_fontweight('bold' if sig else 'normal')
            labels = []
            for comparison in part:
                label = comparison
                for full, short in [('Ateis', 'Ath'), ('Kristen', 'Chr'), ('Islam', 'Isl'), ('Yahudi', 'Jew')]:
                    label = label.replace(full, short)
                if ' − ' in label and label.split(' − ')[0] in ['Ath', 'Chr', 'Isl', 'Jew']:
                    pair, scenario = label.rsplit(' ', 1)
                    label = pair.replace(' − ', '−') + '\n' + scenario
                else:
                    identity, scenario = label.split(' ', 1)
                    label = identity + '\n' + scenario.replace(' ', '')
                labels.append(label)
            ax.set_xticklabels(labels, rotation=0, fontsize=13)
            ax.set_yticklabels(model_order, rotation=0, fontsize=14.5)
            ax.tick_params(axis='both', length=0, pad=8)
            ax.set_xlabel('Perbandingan skenario', fontsize=15, labelpad=10)
            ax.set_ylabel('Model', fontsize=15, labelpad=12)
            ax.set_title(f'Panel {index + 1} — {panel_title}', fontsize=19, fontweight='bold', pad=14)
            ax._inference_column_count = len(part)
            # Keep the identity boundary visible without splitting its scenarios.
            for boundary in ([7] if index < 2 else [4, 8]):
                ax.axvline(boundary, color='#606060', linewidth=1.5)
        color_ax.tick_params(labelsize=12.5)
        color_ax.set_xlabel(color_label, fontsize=14.5, labelpad=9)
        fig.suptitle(f'ETHICS Religion: {title}',
                     fontsize=23, fontweight='bold', y=.98)
        with plt.rc_context({'savefig.dpi': 180}):
            save_plus_two(fig, FIGURE_DIR, stem)
        print(f'[RENDER] {stem}: 4 panels, 52 comparisons, 6 models', flush=True)


def save_plus_two(fig, directory, stem):
    fig.canvas.draw()
    title_nodes = {fig._suptitle}
    for ax in fig.axes:
        title_nodes.update({ax.title, ax._left_title, ax._right_title})
    for text in fig.findobj(match=Text):
        before = text.get_fontsize()
        is_title = text in title_nodes
        text.set_fontsize(before if is_title else before + 2)
        if text.get_text():
            FONT_AUDIT.append({"figure": stem, "is_title": is_title,
                               "before_pt": before, "after_pt": text.get_fontsize(),
                               "text": text.get_text()})
    fig.canvas.draw()
    for ax in fig.axes:
        if hasattr(ax, '_inference_column_count'):
            cell_width = ax.get_window_extent().width / ax._inference_column_count
            assert all(t.get_window_extent().width < cell_width * .98 for t in ax.texts), 'Heatmap annotations exceed cell width'
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ("png", "svg"):
        path = directory / f"{stem}.{suffix}"
        fig.savefig(path, facecolor=engine.BACKGROUND, bbox_inches="tight", pad_inches=0.25)
        paths.append(path)
    plt.close(fig)
    SAVED.extend(paths)
    return paths


def aggregate_combined(summary):
    frame = summary.copy()
    for name in ("wrong", "acceptable"):
        values = frame[f"n_answer_{name}"]
        frame[f"n_{name}"] = pd.to_numeric(values, errors="raise").round().astype(int)
    counts = ["total", "nonmissing", "committed", "abstain", "missing", "n_wrong", "n_acceptable"]
    out = engine._aggregate_categories(frame, counts)
    out[counts] = out[counts].round().astype(int)
    out["pct_answer_wrong"] = out["n_wrong"] / out["committed"].replace(0, np.nan)
    out["pct_answer_acceptable"] = out["n_acceptable"] / out["committed"].replace(0, np.nan)
    out["scope"] = "all_four_subsets"
    out["subset_count"] = 4
    assert len(out) == 54
    assert int(out["total"].sum()) == 154428
    assert out["total"].eq(out["committed"] + out["abstain"] + out["missing"]).all()
    assert out["n_wrong"].add(out["n_acceptable"]).eq(out["committed"]).all()
    return out


def plot_combined(frame, *, write_tables=True):
    fig, axes = plt.subplots(3, 2, figsize=(29, 23))
    specs = [("n_wrong", "Wrong", "#B80F2E"), ("n_acceptable", "Acceptable", "#0086A8"),
             ("abstain", "Abstain", "#E9C46A"), ("missing", "Missing", "#E76F51")]
    for ax, model in zip(axes.flat, engine.MODEL_ORDER, strict=True):
        row = frame[frame["model"].eq(model)].set_index("category").reindex(engine.CATEGORY_ORDER)
        bottom = np.zeros(len(row))
        x = np.arange(len(row))
        for col, label, color in specs:
            vals = row[col].to_numpy() / row["total"].to_numpy() * 100
            ax.bar(x, vals, bottom=bottom, color=color, label=label, edgecolor="white")
            for i, value in enumerate(vals):
                if value >= 5:
                    ax.text(i, bottom[i] + value / 2, f"{value:.1f}%", ha="center", va="center", fontsize=16,
                            color="white" if col.startswith("n_") else "#17212B")
            bottom += vals
        ax.set_xticks(x, [engine.CATEGORY_SHORT_LABELS[c] for c in engine.CATEGORY_ORDER], rotation=35, ha="right")
        ax.set_ylim(0, 100)
        ax.set_xlabel("Persona agama")
        ax.set_ylabel("% seluruh respons")
        ax.set_title(engine.MODEL_LABELS[model], fontsize=engine.MODEL_TITLE_SIZE, fontweight="bold")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(.5, .95))
    fig.suptitle("ETHICS Religion: Distribusi Jawaban", fontsize=engine.TITLE_SIZE, fontweight="bold", y=.995)
    fig.subplots_adjust(top=.89, bottom=.10, hspace=.65, wspace=.20)
    save_plus_two(fig, FIGURE_DIR, "00_ethics_religion_combined_four_subsets_distribution")

    values = frame.pivot(index="model_label", columns="category", values="pct_answer_wrong").reindex(
        index=[engine.MODEL_LABELS[m] for m in engine.MODEL_ORDER], columns=engine.CATEGORY_ORDER)
    deltas = {
        "00_ethics_religion_combined_delta_vs_neutral": (
            "Delta Stance Wrong terhadap Persona Netral",
            {f"{c} − Neutral": values[c] - values["Neutral"] for c in engine.CATEGORY_ORDER if c != "Neutral"},
        ),
        "00_ethics_religion_combined_delta_implicit_explicit": (
            "Delta Stance Wrong: Implicit − Explicit",
            {f"{renderer.IDENTITY_LABELS[i]}: I − E": values[f"{renderer.IDENTITY_LABELS[i]} Implicit Avg"] - values[f"{renderer.IDENTITY_LABELS[i]} Explicit"] for i in renderer.IDENTITY_ORDER},
        ),
    }
    delta_rows = []
    for stem, (title, data) in deltas.items():
        delta = pd.DataFrame(data)
        fig, ax = plt.subplots(figsize=(26, 13))
        delta.plot.bar(ax=ax, width=.82, color=religion_colors(delta.columns))
        engine._annotate_bars(ax)
        limit = max(.005, float(np.nanmax(np.abs(delta.to_numpy()))))
        ax.set_ylim(-limit * 1.6, limit * 1.6)
        ax.axhline(0, color="#17212B", linewidth=1)
        ax.set_title(f"ETHICS Religion: {title}", fontsize=engine.TITLE_SIZE, fontweight="bold", pad=20)
        ax.set_xlabel("Model")
        ax.set_ylabel("Delta proporsi wrong pada respons committed")
        ax.set_xticklabels([engine.MODEL_AXIS_LABELS[engine.MODEL_LABELS[m]] for m in engine.MODEL_ORDER], rotation=0)
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
        fig.tight_layout()
        save_plus_two(fig, FIGURE_DIR, stem)
        delta_rows.append(delta.rename_axis("model_label").reset_index().melt(id_vars="model_label", var_name="comparison", value_name="delta_wrong"))
    if write_tables:
        pd.concat(delta_rows, ignore_index=True).assign(scope="all_four_subsets").to_csv(TABLE_DIR / "ethics_religion_combined_four_subsets_deltas.csv", index=False)

def make_combined_inference():
    src = pd.read_csv(TABLE_DIR / 'ethics_paired_effects_all.csv')
    keys = ['comparison_family','comparison_category','model','model_label','identity','identity_label','scenario_a','scenario_b','condition_a','condition_b','comparison_label','display_order']
    rows=[]
    for key,g in src.groupby(keys, dropna=False, sort=False):
        r=dict(zip(keys,key)); r.update(subset='all_four_subsets', subset_label='All four subsets')
        r['n_pairs']=int(g.n_pairs.sum()); r['discordant_a_correct_b_wrong']=int(g.discordant_a_correct_b_wrong.sum()); r['discordant_a_wrong_b_correct']=int(g.discordant_a_wrong_b_correct.sum()); r['discordant_total']=r['discordant_a_correct_b_wrong']+r['discordant_a_wrong_b_correct']
        r['accuracy_a_pct']=float((g.accuracy_a_pct*g.n_pairs).sum()/r['n_pairs']); r['accuracy_b_pct']=float((g.accuracy_b_pct*g.n_pairs).sum()/r['n_pairs']); r['delta_accuracy_pp']=r['accuracy_b_pct']-r['accuracy_a_pct']
        r['exact_mcnemar_p_value']=float(binomtest(min(r['discordant_a_correct_b_wrong'],r['discordant_a_wrong_b_correct']),r['discordant_total'],0.5).pvalue) if r['discordant_total'] else 1.0; r['cohens_h']=float(2*np.arcsin(np.sqrt(r['accuracy_b_pct']/100))-2*np.arcsin(np.sqrt(r['accuracy_a_pct']/100))); rows.append(r)
    out=pd.DataFrame(rows); out['holm_adjusted_p_value']=np.nan; out['holm_family_size']=0; out['significant_holm_0_05']=False
    for fam,idx in out.groupby('comparison_family').groups.items():
        p=out.loc[idx,'exact_mcnemar_p_value'].to_numpy(); order=np.argsort(p); adj=np.empty(len(p)); adj[order]=np.maximum.accumulate((len(p)-np.arange(len(p)))*p[order]); out.loc[idx,'holm_adjusted_p_value']=np.minimum(adj,1); out.loc[idx,'holm_family_size']=len(idx); out.loc[idx,'significant_holm_0_05']=out.loc[idx,'holm_adjusted_p_value']<0.05
    out.to_csv(TABLE_DIR/'ethics_religion_combined_paired_effects_all.csv',index=False)
    plot_combined_inference(out)


def update_notebook():
    nb = nbformat.read(TARGET_NOTEBOOK, as_version=4)
    # Keep the existing notebook file, but replace its cells with a reproducible
    # reader for the updated bundle and the Bab VI-aligned metric definitions.
    cells = []
    add_md = lambda s: cells.append(nbformat.v4.new_markdown_cell(s))
    add_code = lambda s: cells.append(nbformat.v4.new_code_cell(s))
    add_md("""# Analisis Persona Religion pada ETHICS — Gabungan dan Empat Subset

Notebook ini menyajikan chart gabungan empat subset sebagai bagian utama, lalu
rincian Commonsense, Deontology, Justice, dan Virtue. Cakupan sumber adalah
154.428 respons dari enam model dan 17 persona. I1, I2, dan I3 tetap tersedia
terpisah pada tabel inferensi sesuai Bab VI.

Untuk akurasi Religion, definisi Bab VI dipertahankan: akurasi penuh memakai
seluruh respons sebagai penyebut sehingga abstain dan missing dihitung sebagai
tidak sesuai ground truth. Akurasi committed ditampilkan sebagai metrik
pendamping bersama commit_rate. Chart gabungan memakai distribusi jawaban dan
status, bukan satu skor akurasi moral lintas subset.""")
    add_md("""## 1. SPEC

Tujuan: menampilkan tiga chart gabungan empat subset, status dan coverage, serta
rincian per subset dalam notebook yang sama. Semua hasil dibaca dari bundle
Religion v3/v4 yang tervalidasi; tidak ada API call, parsing ulang, atau rescoring.

Keterbatasan konstruk Deontology dan Virtue tetap dicatat. Chart per subset
menampilkan akurasi penuh dan committed sesuai naskah, sedangkan distribusi
stance disediakan sebagai informasi tambahan agar perbandingan persona tetap
dapat diperiksa.""")
    add_md("""## 2. DESIGN PLAN

- Bagian utama: distribusi jawaban gabungan, delta terhadap Neutral, dan delta I–E.
- Bagian berikutnya: tabel serta figure per subset dengan I1, I2, I3 terpisah.
- Status dan coverage selalu ditampilkan bersama akurasi.
- Inferensi antarpersona, antarskenario, Exact McNemar, Q Cochran, Cohen's h,
  dan Holm dibaca dari tabel sumber.
- Semua label figure selain judul dinaikkan 2 pt; judul tetap.""")
    add_md("## 3. IMPLEMENTATION — Sumber, tabel, dan bundle revisi")
    add_code("""import json, hashlib
from pathlib import Path
import pandas as pd
from IPython.display import Image, Markdown, display

ROOT = Path.cwd()
while not (ROOT / 'result/benchmarks/ethics/runs').is_dir(): ROOT = ROOT.parent
NOTEBOOK_DIR = ROOT / 'notebooks/religion/ethics accuracy analysis'
TABLE_DIR = NOTEBOOK_DIR / 'tables'
FIGURE_DIR = NOTEBOOK_DIR / 'figures'
SOURCE = ROOT / 'result/benchmarks/ethics/reports/Religion_Persona/_combined/persona_publication_analysis_v3_accuracy_paired_holm_six_models'
manifest = json.loads((NOTEBOOK_DIR / 'bundle_manifest.json').read_text(encoding='utf-8'))
summary = pd.read_csv(TABLE_DIR / 'ethics_condition_summary_source_derived.csv')
status = pd.read_csv(TABLE_DIR / 'ethics_status_by_model_persona_category.csv')
accuracy = pd.read_csv(TABLE_DIR / 'ethics_accuracy_by_model_persona_category.csv')
paired_all = pd.read_csv(TABLE_DIR / 'ethics_paired_effects_all.csv')
combined_inference = pd.read_csv(TABLE_DIR / 'ethics_religion_combined_paired_effects_all.csv')
reasons = pd.read_csv(TABLE_DIR / 'ethics_abstain_reasons_by_model.csv')
combined = pd.read_csv(TABLE_DIR / 'ethics_religion_combined_four_subsets.csv')
combined_deltas = pd.read_csv(TABLE_DIR / 'ethics_religion_combined_four_subsets_deltas.csv')
SUBSETS = ['commonsense', 'deontology', 'justice', 'virtue']
print('Rows:', int(summary.total.sum()), '| Models:', summary.model.nunique(), '| Personas:', summary.persona.nunique())
display(summary.groupby('subset').total.agg(['count','sum']))""")
    add_md("## 4. CHECKLIST — Cakupan dan denominator")
    add_code("""assert int(summary.total.sum()) == 154428
assert set(summary.subset) == set(SUBSETS)
assert len(summary) == 408
assert len(combined) == 54 and combined.subset_count.eq(4).all()
assert combined.total.eq(combined.committed + combined.abstain + combined.missing).all()
assert combined.n_wrong.add(combined.n_acceptable).eq(combined.committed).all()
assert manifest['api_calls'] == 0 and manifest['rescore'] is False
if 'multiple_comparison_method' in paired_all.columns:
    assert set(paired_all['multiple_comparison_method'].dropna()) <= {'Holm'}
print('[x] 154.428 respons; empat subset; 54 baris gabungan model × kategori persona.')""")
    add_md("## 5. Chart gabungan ETHICS Religion")
    add_code("""def show_figure(stem):
    path = FIGURE_DIR / f'{stem}.png'
    assert path.is_file() and path.stat().st_size > 0
    display(Markdown(f'### `{stem}`'))
    display(Image(filename=str(path), width=1500))

for stem in ['00_ethics_religion_combined_four_subsets_distribution',
             '00_ethics_religion_combined_delta_vs_neutral',
             '00_ethics_religion_combined_delta_implicit_explicit']:
    show_figure(stem)
display(combined.style.format({'commit_rate':'{:.2%}','abstain_rate':'{:.2%}','missing_rate':'{:.2%}','pct_answer_wrong':'{:.2%}','pct_answer_acceptable':'{:.2%}'}))
display(combined_deltas.style.format({'delta_wrong':'{:.4f}'}))""")
    add_md("""## 6. Heatmap gabungan: ukuran efek, Exact McNemar, dan Holm

Setiap figure memuat 52 perbandingan × 6 model dalam empat panel vertikal,
dengan urutan identik: Ateis/Kristen; Islam/Yahudi; Ateis versus identitas lain;
dan pasangan lintas agama lainnya. E/I1/I2/I3 dalam kelompok yang sama tidak
terpotong antar-panel. Ath = Ateis, Chr = Kristen, Isl = Islam, Jew = Yahudi;
N = netral, E = explicit, I1–I3 = tiga varian implicit.

**Cara membaca:** angka tebal dan * menandai p-value Holm < 0,05 pada ketiga
figure, termasuk figure p-value mentah. Cohen's h ditampilkan dengan arah A−B
sesuai label kolom; tanda pada kolom sumber `cohens_h` (B−A) dibalik untuk
penyajian ini. Angka p-value tidak berubah. Warna p-value dibatasi pada 0–0,05;
anotasi tetap menunjukkan nilai aslinya (nilai < 0,001 ditulis <0,001).

Gabungan ini memakai kecocokan label empat subset dari tabel gabungan yang
sudah tersedia, bukan distribusi stance pada tiga chart sebelumnya. Perbedaan
konstruk Deontology/Virtue membatasi penafsirannya sebagai kemampuan moral;
hasil per subset tetap diperlukan untuk interpretasi tesis. Holm pada tabel
gabungan menggunakan family lintas model (168 uji skenario dan 144 uji pasangan
agama), bukan family 3/4/24 dari contoh GSM8K. Layout tidak mengubah family ini.

Reproduksi tampilan: `python src/benchmarks/ethics/update_religion_ethics_existing.py --layout-only`.
""")
    add_code("""inference_stems = [
    '00_ethics_religion_combined_inference_effect_holm',
    '00_ethics_religion_combined_exact_mcnemar_four_panels',
    '00_ethics_religion_combined_holm_pvalue_four_panels',
]
assert len(combined_inference) == 312
assert combined_inference.comparison_label.nunique() == 52
for stem in inference_stems:
    show_figure(stem)
print('[x] Tiga heatmap, masing-masing empat panel dan 312 nilai.')""")
    add_md("## 7. Tabel dan figure per subset")
    add_code("""for subset in SUBSETS:
    display(Markdown(f'### {subset.title()}'))
    frame = summary[summary.subset.eq(subset)]
    display(frame.style.format({'commit_rate':'{:.2%}','abstain_rate':'{:.2%}','missing_rate':'{:.2%}','accuracy_committed':'{:.2%}','pct_answer_wrong':'{:.2%}','pct_answer_acceptable':'{:.2%}'}))
    for stem in [f'01_{subset}_performance_by_persona', f'02_{subset}_status_by_persona']:
        path = NOTEBOOK_DIR / 'per_subset' / subset / 'figures' / f'{stem}.png'
        assert path.is_file(), str(path)
        display(Image(filename=str(path), width=1450))""")
    add_md("## 8. Rincian sumber dan status")
    add_code("""display(reasons)
display(combined_inference)
display(paired_all)
display(pd.read_csv(SOURCE / 'tables/persona_status_independence.csv') if (SOURCE / 'tables/persona_status_independence.csv').is_file() else pd.DataFrame())""")
    add_md("""## 9. TESTING dan INTEGRATION

Notebook ini mempertahankan I1, I2, dan I3 secara terpisah untuk inferensi dan
tabel Bab VI. Tiga chart awal merangkum distribusi/status empat subset;
tiga heatmap berikutnya menampilkan statistik kecocokan label gabungan.
Jalankan **Restart Kernel and Run All** untuk mengulang pemeriksaan. Akurasi
full dan committed dibaca bersama status; hasil signifikan tetap dibaca bersama
effect size, koreksi Holm, dan keterbatasan konstruk Deontology/Virtue.""")
    nb.cells = cells
    if (FIGURE_DIR / '00_ethics_religion_combined_absolute_accuracy.png').exists():
        from add_religion_ethics_accuracy_chart import integrate_notebook
        integrate_notebook(nb)
    nbformat.validate(nb)
    client = NotebookClient(nb, timeout=600, kernel_name='python3', resources={'metadata': {'path': str(NOTEBOOK_DIR)}})
    client.execute()
    assert not any(o.output_type == 'error' for c in nb.cells if c.cell_type == 'code' for o in c.outputs)
    nbformat.write(nb, TARGET_NOTEBOOK)


def main():
    renderer.configure_engine()
    engine._save = save_plus_two
    renderer.engine._save = save_plus_two
    layout_only = '--layout-only' in sys.argv
    source_hashes = {p: engine.sha256_file(p) for p in TABLE_DIR.glob('*.csv') if p.name != 'font_size_audit.csv'}
    if layout_only:
        plot_combined_inference(pd.read_csv(TABLE_DIR / 'ethics_religion_combined_paired_effects_all.csv'))
    else:
        source_summary = pd.read_csv(TABLE_DIR / 'ethics_condition_summary_source_derived.csv')
        combined = aggregate_combined(source_summary)
        combined.to_csv(TABLE_DIR / 'ethics_religion_combined_four_subsets.csv', index=False)
        plot_combined(combined)
        make_combined_inference()
    audit = pd.DataFrame(FONT_AUDIT)
    if (TABLE_DIR / 'font_size_audit.csv').exists():
        previous = pd.read_csv(TABLE_DIR / 'font_size_audit.csv')
        audit = pd.concat([previous[~previous.figure.isin(audit.figure.unique())], audit], ignore_index=True)
    audit.to_csv(TABLE_DIR / 'font_size_audit.csv', index=False)
    inc = audit.after_pt - audit.before_pt
    assert inc[audit.is_title.eq(True)].eq(0).all() and inc[audit.is_title.eq(False)].eq(2).all()
    manifest = json.loads((NOTEBOOK_DIR / 'bundle_manifest.json').read_text(encoding='utf-8'))
    manifest.update({'bundle_version': 'ethics-religion-combined-four-subsets-fontplus2',
                     'combined_scope': 'all_four_subsets', 'non_title_font_increment_pt': 2,
                     'title_font_increment_pt': 0, 'combined_metrics': ['wrong','acceptable','abstain','missing'],
                     'font_audit_file': 'tables/font_size_audit.csv'})
    manifest['combined_inference_layout'] = {
        'panels_per_figure': 4, 'comparisons_per_panel': [14, 14, 12, 12],
        'model_count': 6, 'figure_stems': INFERENCE_STEMS,
        'display_effect_direction': 'A-B; source cohens_h is B-A',
        'display_effect_sign_multiplier': -1,
        'star_definition': 'Holm-adjusted p < 0.05 for all three figures',
        'source_table': 'tables/ethics_religion_combined_paired_effects_all.csv',
        'source_sha256': engine.sha256_file(TABLE_DIR / 'ethics_religion_combined_paired_effects_all.csv'),
    }
    (NOTEBOOK_DIR / 'bundle_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    paths = list(FIGURE_DIR.glob('*')) + list(TABLE_DIR.glob('*.csv'))
    pd.DataFrame([{'kind':'figure' if p.suffix in ('.png','.svg') else 'table', 'file':p.name,
                   'relative_path':p.relative_to(NOTEBOOK_DIR).as_posix(), 'path':str(p),
                   'sha256':engine.sha256_file(p)} for p in sorted(paths) if p.is_file()]).to_csv(NOTEBOOK_DIR / 'output_index.csv', index=False)
    update_notebook()
    if layout_only:
        assert all(engine.sha256_file(p) == digest for p, digest in source_hashes.items())
        print('[CHECK] All numerical source table hashes unchanged.', flush=True)
    print(f'[PASS] Religion notebook updated; {len(SAVED)//2} figure stems rendered; font audit passed; notebook executed.', flush=True)


if __name__ == '__main__':
    main()
