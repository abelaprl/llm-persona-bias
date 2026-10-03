"""Reorder the existing ETHICS notebook and present frozen direct comparisons.

No extraction, scoring, statistical testing, or multiplicity correction runs here.
Use a fresh output directory. The notebook is a generated presentation artifact.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import nbformat
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
REPORTS = ROOT / "result/benchmarks/ethics/reports/Gender_Persona/linguistic_markers/analysis"
BASE = REPORTS / "gender_ethics_lm_workspace_v1_5_20260909_r2"
POOL = REPORTS / "gender_ethics_lm_gabungan_20260909_r4"
NOTEBOOK = ROOT / "notebooks/gender/linguistic marker/ETHICS_Gender_Linguistic_Markers_Analysis_v1_5_candidate.ipynb"
SCOPES = ("commonsense", "deontology", "justice", "virtue", "ethics_all")
CONDITIONS = ("explicit", "implicit_1", "implicit_2", "implicit_3")
CONDITION_LABELS = dict(zip(CONDITIONS, ("E", "I1", "I2", "I3")))
MODEL_LABELS = {
    "deepseek_deepseek-r1-distill-qwen-32b": "DeepSeek R1 32B",
    "meta-llama_llama-3.1-8b-instruct": "Llama 3.1 8B",
    "meta-llama_llama-3.3-70b-instruct": "Llama 3.3 70B",
    "mistralai_ministral-8b-2512": "Ministral 8B",
    "openai_gpt-oss-20b": "GPT-OSS 20B",
    "qwen_qwen3-next-80b-a3b-thinking": "Qwen3 Next 80B",
}
MARKER_LABELS = dict(zip((
    "word_count", "avg_sentence_length", "ttr_300", "organization_explicit_step",
    "organization_bullet", "organization_section_header", "stance_hedge", "stance_booster",
    "stance_assumption", "stance_self_correction", "connective_causal", "connective_contrastive",
    "connective_conditional", "self_reference_first_person", "self_reference_inclusive_we",
    "engagement_reader_directive", "stance_contextual_hedge", "stance_distancing",
    "connective_additive", "connective_concessive", "affect_empathy_validation",
    "gender_persona_direct_address", "gender_persona_acknowledgment",
    "gender_second_person_identity_reference", "gender_hypothetical_identity_reference",
    "gender_first_person_identity_reference", "gender_perspective_adoption",
    "gender_identity_relevance_denial", "gender_identity_neutralization", "gender_identity_discourse",
    "gender_role_framing", "gendered_normative_evaluation",
), (
    "Jumlah kata", "Panjang kalimat", "TTR-300", "Langkah eksplisit", "Daftar berpoin",
    "Subjudul", "Hedging", "Kepastian", "Asumsi", "Koreksi diri", "Sebab-akibat",
    "Pertentangan", "Syarat", "Orang pertama", "We inklusif", "Arahan pembaca",
    "Hedging kontekstual", "Pengambilan jarak", "Penambahan", "Konsesi", "Empati / validasi",
    "Sapaan langsung", "Pengakuan persona", "Identitas orang kedua", "Identitas hipotetis",
    "Identitas orang pertama", "Adopsi perspektif", "Penyangkalan relevansi gender",
    "Netralisasi gender", "Pembahasan gender", "Peran gender", "Evaluasi normatif bergender",
)))
FAMILY_LABELS = {"numeric": "Ukuran umum", "gender_presence": "Marker gender: presence",
                 "gender_density": "Marker gender: density"}


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def hashes(folder):
    return {p.relative_to(folder).as_posix(): sha256(p) for p in sorted(folder.rglob("*")) if p.is_file()}


def write_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def summarize(frame, family):
    """Describe all frozen tests, separating effect direction from significance."""
    value = "prevalence_delta" if family == "gender_presence" else "rank_biserial"
    data = frame.copy()
    keys = ["subset", "model_id", "condition", "outcome_id"]
    if data.duplicated(keys).any():
        raise ValueError("Duplicate direct comparison")
    if not data.left_identity.eq("man").all() or not data.right_identity.eq("woman").all():
        raise ValueError("Expected Man minus Woman")
    if not data.condition.isin(CONDITIONS).all():
        raise ValueError("Unexpected scenario")
    for row in data.itertuples():
        suffix = "explicit_1" if row.condition == "explicit" else row.condition
        if row.left_persona != "man_" + suffix or row.right_persona != "woman_" + suffix:
            raise ValueError("Unmatched presentation scenarios")
    data["family"] = family
    data["effect_metric"] = value
    data["effect_value"] = data[value]
    data["significant"] = data.p_holm.notna() & data.p_holm.lt(0.05)
    data["direction"] = np.select(
        [data[value].gt(0), data[value].lt(0), data[value].eq(0)],
        ["Man lebih tinggi", "Woman lebih tinggi", "Selisih nol"], default="Tidak tersedia")
    data["model"] = data.model_id.map(MODEL_LABELS)
    data["marker"] = data.outcome_id.map(MARKER_LABELS)
    data["scenario"] = data.condition.map(CONDITION_LABELS)
    if data[["model", "marker"]].isna().any().any():
        raise ValueError("Missing presentation label")
    data["n_changed"] = data["discordant" if family == "gender_presence" else "n_nonzero"]
    summary, highlights = [], []
    for (subset, model), group in data.groupby(["subset", "model_id"], sort=False):
        sig = group.loc[group.significant]
        summary.append({
            "subset": subset, "model_id": model, "model": MODEL_LABELS[model], "family": family,
            "comparisons": len(group), "tested": int(group.p_holm.notna().sum()),
            "significant": len(sig), "man_higher": int(sig.effect_value.gt(0).sum()),
            "woman_higher": int(sig.effect_value.lt(0).sum()),
            "zero_effect_significant": int(sig.effect_value.eq(0).sum()),
            "n_pairs_min": int(group.n_pairs.min()), "n_pairs_max": int(group.n_pairs.max()),
            "scenarios_significant": ", ".join(CONDITION_LABELS[c] for c in CONDITIONS if c in set(sig.condition)),
        })
        for condition in CONDITIONS:
            selected = sig.loc[sig.condition.eq(condition)].copy()
            if len(selected):
                # One example per model/scenario, never a new inferential threshold.
                selected["abs_effect"] = selected.effect_value.abs()
                highlights.append(selected.sort_values(
                    ["abs_effect", "n_changed", "outcome_id"], ascending=[False, False, True]).iloc[0])
    highlights_frame = (pd.DataFrame(highlights).drop(columns="abs_effect", errors="ignore")
                        if highlights else data.iloc[:0].copy())
    return data, pd.DataFrame(summary), highlights_frame


def consistency(frame):
    rows = []
    for (subset, family, condition, marker), group in frame.groupby(
            ["subset", "family", "condition", "outcome_id"], sort=False):
        rows.append({"subset": subset, "family": family, "condition": condition, "outcome_id": marker,
                     "models": len(group), "man_higher": int(group.effect_value.gt(0).sum()),
                     "woman_higher": int(group.effect_value.lt(0).sum()),
                     "zero": int(group.effect_value.eq(0).sum()),
                     "unavailable": int(group.effect_value.isna().sum()),
                     "significant_man_higher": int((group.significant & group.effect_value.gt(0)).sum()),
                     "significant_woman_higher": int((group.significant & group.effect_value.lt(0)).sum())})
    return pd.DataFrame(rows)


def plot_consistency(frame, subset, family, path, presence_bound):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    selected = frame.loc[frame.subset.eq(subset) & frame.family.eq(family)]
    markers = selected.outcome_id.drop_duplicates().tolist()
    models = list(MODEL_LABELS)
    # Reserve physical space for rotated model labels and the three-line note.
    fig, axes = plt.subplots(1, 4, figsize=(18, max(10.5, 0.36 * len(markers) + 3.2)), sharey=True)
    bound = presence_bound if family == "gender_presence" else 1
    cmap = plt.get_cmap("RdBu").copy()
    cmap.set_bad("#dddddd")
    for ax, condition in zip(axes, CONDITIONS):
        panel = selected.loc[selected.condition.eq(condition)]
        values = panel.pivot(index="outcome_id", columns="model_id", values="effect_value").reindex(index=markers, columns=models)
        significance = panel.pivot(index="outcome_id", columns="model_id", values="significant").reindex(index=markers, columns=models)
        scale = 100 if family == "gender_presence" else 1
        artist = ax.imshow(values.to_numpy() * scale, cmap=cmap, norm=Normalize(-bound, bound), aspect="auto")
        for y, x in zip(*np.where(significance.fillna(False).to_numpy(dtype=bool))):
            val = values.iloc[y, x] * scale
            ax.text(x, y, "*", ha="center", va="center", color="white" if abs(val) > bound * .6 else "black", fontsize=12)
        ax.set_title(CONDITION_LABELS[condition], fontsize=14)
        ax.set_xticks(range(len(models)), [MODEL_LABELS[m] for m in models], rotation=60, ha="right", fontsize=9)
        ax.set_yticks(range(len(markers)), [MARKER_LABELS[m] for m in markers], fontsize=10)
        ax.set_xticks(np.arange(-.5, len(models), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(markers), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=.5)
        ax.tick_params(which="minor", bottom=False, left=False)
    scope = "ETHICS Gabungan" if subset == "ethics_all" else "ETHICS " + subset
    label = "Marker Gender" if family == "gender_presence" else "Ukuran Umum"
    fig.suptitle(f"{scope} — Laki-laki dan Perempuan\n{label}: Arah Efek per Model", fontsize=17, y=.97)
    fig.subplots_adjust(left=.22, right=.91, top=.87, bottom=.23, wspace=.07)
    color_ax = fig.add_axes([.93, .34, .012, .39])
    colorbar = fig.colorbar(artist, cax=color_ax)
    colorbar.set_label("Selisih prevalensi (poin persentase)" if family == "gender_presence" else "Rank-biserial", fontsize=10)
    fig.text(.22, .035, "Man − Woman: biru = Man lebih tinggi; merah = Woman lebih tinggi.  * p Holm < 0,05.\n"
             "Putih = selisih nol; abu-abu = tidak tersedia. Baca dukungan pasangan pada tabel.\n"
             "Konsistensi arah bersifat deskriptif, bukan uji gabungan antarmodel. Hasil candidate.", fontsize=10)
    for suffix in (".png", ".svg"):
        fig.savefig(path.with_suffix(suffix), dpi=160, facecolor="white")
    plt.close(fig)


def new_code(source, cell_id):
    return nbformat.v4.new_code_cell(source, id=cell_id)


def rewrite_notebook(original, output, local_figures):
    """Move complete sections and retain every original cell ID."""
    if len(original.cells) != 38 or any(c.id == "direct-summary-setup" for c in original.cells):
        raise ValueError("Expected the unrevised 38-cell notebook; use its backup for reproduction")
    cells = copy.deepcopy(original.cells)
    cells[0].source += "\n\nPerbandingan utama adalah laki-laki dan perempuan pada E, I1, I2, dan I3. Perbandingan E/I terhadap N dan I terhadap E memberi konteks pendukung. Perbedaan signifikansi terhadap N bukan bukti langsung perbedaan Man–Woman."
    cells[12].source = cells[12].source.replace("## Laki-laki terhadap perempuan", "## Laki-laki dan Perempuan") + "\n\nIni hasil utama antarpersona. Ringkasan mencakup semua model; contoh per skenario dipilih berdasarkan |effect size| terbesar di antara hasil dengan p Holm < 0,05. Contoh bukan keseluruhan hasil."
    cells[29].source += "\n\nIni hasil utama gabungan. Ringkasan dan grafik berikut tetap memisahkan model dan empat skenario."
    cells[8].source += "\n\nPerbandingan pendukung: perubahan setelah persona diberikan, bukan uji langsung perbedaan laki-laki dan perempuan."
    cells[10].source += "\n\nPerbandingan pendukung: perbedaan cara penyajian persona, bukan perbedaan identitas gender."
    cells[25].source += "\n\nPerbandingan ini menjadi konteks pendukung untuk hasil Man–Woman."
    cells[27].source += "\n\nPerbandingan ini menguji cara penyajian persona dan menjadi pendukung hasil Man–Woman."
    cells[31].source = cells[31].source.replace("### Marker Gender", "### Marker Gender: Laki-laki dan Perempuan")
    # Move supporting gender plots into their matching scenario sections.
    cells[32].source = 'pooled_results("gender_presence_direct_all_text")\npooled_results("gender_density_direct_all_text")\npooled_figure("06_gender_direct.png")\n'
    cells[26].source += '\npooled_results("gender_presence_vs_neutral_all_text")\npooled_results("gender_density_vs_neutral_all_text")\npooled_figure("04_gender_vs_neutral.png")\n'
    cells[28].source += '\npooled_results("gender_presence_vs_explicit_all_text")\npooled_results("gender_density_vs_explicit_all_text")\npooled_figure("05_gender_vs_explicit.png")\n'
    # Opening comparisons retain both displays, with the direct result first.
    cells[17].source = cells[17].source.replace(
        '"images/15_opening_composition_within_identity.png", "images/16_opening_composition_direct_man_woman.png"',
        '"images/16_opening_composition_direct_man_woman.png", "images/15_opening_composition_within_identity.png"')
    cells[34].source = 'pooled_results("opening_direct_all_text")\npooled_results("opening_within_all_text")\n'
    relative = output.relative_to(ROOT).as_posix()
    setup = new_code(f'''DIRECT_PRESENTATION = ROOT / {relative!r}
direct_index = json.loads((DIRECT_PRESENTATION / "presentation_hashes.json").read_text(encoding="utf-8"))
for direct_name, direct_expected in direct_index.items():
    with (DIRECT_PRESENTATION / direct_name).open("rb") as direct_handle:
        assert hashlib.file_digest(direct_handle, "sha256").hexdigest() == direct_expected, direct_name
direct_summary = pd.read_csv(DIRECT_PRESENTATION / "direct_summary.csv").fillna({{"scenarios_significant": "—"}})
direct_highlights = pd.read_csv(DIRECT_PRESENTATION / "direct_highlights.csv")
direct_index_figures = pd.read_csv(DIRECT_PRESENTATION / "figure_index.csv")

def show_direct_summary(scope, families):
    for family in families:
        label = {{"numeric": "Ukuran umum", "gender_presence": "Marker gender: presence", "gender_density": "Marker gender: density"}}[family]
        selected = direct_summary.query("subset == @scope and family == @family")
        display(Markdown(f"**{{label}} — {{scope}}**"))
        display(selected[["model", "comparisons", "tested", "significant", "man_higher", "woman_higher", "n_pairs_min", "n_pairs_max", "scenarios_significant"]])
        examples = direct_highlights.query("subset == @scope and family == @family")
        if len(examples):
            display(Markdown("**Contoh hasil per model dan skenario**"))
            display(examples[["model", "scenario", "marker", "direction", "effect_metric", "effect_value", "median_paired_delta", "ci_low", "ci_high", "p_holm", "n_pairs", "n_changed"]].style.format(
                {{"effect_value": "{{:.3f}}", "median_paired_delta": "{{:.3f}}", "ci_low": "{{:.3f}}", "ci_high": "{{:.3f}}", "p_holm": "{{:.3g}}", "n_pairs": "{{:.0f}}", "n_changed": "{{:.0f}}"}}, na_rep="—"))
        else:
            display(Markdown("Tidak ada hasil dengan p Holm < 0,05 untuk kelompok ini."))
        display(Markdown(f"CSV lengkap: `{{DIRECT_PRESENTATION / 'direct_results.csv'}}`. Ringkasan tidak mengubah uji atau koreksi Holm."))
        print(f"{{len(selected)}} model | {{int(selected.comparisons.sum())}} perbandingan | {{int(selected.significant.sum())}} p Holm < 0,05")

def show_direct_consistency(scope, family):
    rows = direct_index_figures.query("subset == @scope and family == @family and kind == 'consistency'")
    for row in rows.itertuples():
        display(Image(filename=str(DIRECT_PRESENTATION / row.figure), width=1600))
''', "direct-summary-setup")
    note = nbformat.v4.new_markdown_cell(
        "### Membaca ringkasan\n\n`man_higher` dan `woman_higher` menghitung hasil signifikan menurut arah Man − Woman, bukan seluruh hasil searah. `n_changed` adalah jumlah selisih nonzero untuk ukuran numerik atau pasangan diskordan untuk presence. Semua pasangan tetap masuk uji asal; marker langka perlu dibaca bersama dukungan tersebut.\n\nCI numerik berlaku untuk median selisih berpasangan, **bukan** rank-biserial. CI presence berlaku untuk selisih prevalensi dalam proporsi (grafik memakai poin persentase). Tabel contoh memilih satu hasil signifikan dengan |effect size| terbesar per model dan skenario, secara terpisah untuk tiap jenis ukuran; nilai besar pada marker langka tidak otomatis kuat.\n\nGrafik arah menampilkan seluruh marker dan semua model, termasuk hasil tidak signifikan. Warna yang sama antarmodel bukan pengujian kesamaan efek atau bukti kausal umum. Semua hasil tetap *candidate*; keterbatasan audit dan metode di bagian akhir tetap berlaku.", id="direct-summary-guide")
    def presentation(scope, families, cell_id):
        code = f"for direct_scope in {scope!r}:\n    show_direct_summary(direct_scope, {families!r})\n"
        for family in families:
            if family != "gender_density":
                code += f"    show_direct_consistency(direct_scope, {family!r})\n"
        return new_code(code, cell_id)
    # Keep every original section; change order only where requested.
    ordered = cells[:8] + [setup, cells[12], note,
        presentation(list(SCOPES[:4]), ["numeric"], "direct-subset-summary"), cells[13],
        cells[14], presentation(list(SCOPES[:4]), ["gender_presence", "gender_density"], "direct-subset-gender-summary"), cells[15]]
    ordered += cells[8:12] + cells[16:25]
    ordered += [cells[29], presentation(["ethics_all"], ["numeric"], "direct-pooled-summary"), cells[30],
                cells[31], presentation(["ethics_all"], ["gender_presence", "gender_density"], "direct-pooled-gender-summary"), cells[32]]
    ordered += cells[25:29] + cells[33:]
    notebook = copy.deepcopy(original)
    notebook.cells = ordered
    if len({c.id for c in ordered}) != len(ordered) or not {c.id for c in original.cells}.issubset({c.id for c in ordered}):
        raise ValueError("Original cell lost or duplicate ID")
    notebook.metadata["direct_presentation"] = {"bundle": relative, "local_figures": local_figures.relative_to(ROOT).as_posix(), "status": "candidate"}
    nbformat.validate(notebook)
    return notebook


def execute(notebook, path):
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpec
    manager = KernelManager(kernel_name="python3")
    manager._kernel_spec = KernelSpec(argv=[sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"], display_name="Python 3", language="python")
    NotebookClient(notebook, km=manager, timeout=600, resources={"metadata": {"path": str(path.parent)}}).execute()


def save_notebook(notebook, path, expected_hash):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".ipynb", prefix=".direct-", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            nbformat.write(notebook, handle)
            handle.flush()
            os.fsync(handle.fileno())
        reread = nbformat.read(temporary, as_version=4)
        nbformat.validate(reread)
        if reread != notebook or sha256(path) != expected_hash:
            raise ValueError("Staged notebook or target changed")
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def build(output, notebook_path, source_notebook=None):
    if output.exists():
        raise FileExistsError("Use a new output directory")
    local_figures = notebook_path.parent / "figures" / output.name
    if local_figures.exists():
        raise FileExistsError("Local figure directory already exists")
    original_hash = sha256(notebook_path)
    source_notebook = source_notebook or notebook_path
    original = nbformat.read(source_notebook, as_version=4)
    # Reject an unexpected notebook before creating output artifacts.
    rewrite_notebook(original, output, local_figures)
    print("[1/5] Checking frozen inputs and saving notebook backup", flush=True)
    old_hashes = {str(p.relative_to(ROOT)): hashes(p) for p in (BASE, POOL, notebook_path.parent / "figures")}
    for folder, name in ((BASE, "workspace_artifact_hashes.json"), (POOL, "artifact_hashes.json")):
        for relative, digest in json.loads((folder / name).read_text(encoding="utf-8")).items():
            if sha256(folder / relative) != digest:
                raise ValueError(f"Frozen source changed: {folder / relative}")
    output.mkdir(parents=True)
    shutil.copy2(notebook_path, output / "notebook_before.ipynb")
    shutil.copy2(source_notebook, output / "presentation_source.ipynb")
    images = output / "images"
    images.mkdir()
    source_files, details, summaries, examples = [], [], [], []
    filenames = {
        "numeric": ("paired_numeric_direct_man_woman_all_text.csv", "numeric_direct_all_text.csv"),
        "gender_presence": ("paired_gender_marker_presence_direct_man_woman_all_text.csv", "gender_presence_direct_all_text.csv"),
        "gender_density": ("paired_gender_marker_density_direct_man_woman_all_text.csv", "gender_density_direct_all_text.csv"),
    }
    for family, names in filenames.items():
        parts = []
        for folder, name in zip((BASE, POOL), names):
            path = folder / name
            parts.append(pd.read_csv(path).assign(source_file=path.relative_to(ROOT).as_posix()))
            source_files.append({"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)})
        data, summary, highlights = summarize(pd.concat(parts, ignore_index=True), family)
        details.append(data)
        summaries.append(summary)
        examples.append(highlights)
    results = pd.concat(details, ignore_index=True)
    summary = pd.concat(summaries, ignore_index=True)
    highlights = pd.concat(examples, ignore_index=True)
    for family, size in (("numeric", 504), ("gender_presence", 264), ("gender_density", 96)):
        counts = results.loc[results.family.eq(family)].groupby("subset").size()
        if set(counts.index) != set(SCOPES) or not counts.eq(size).all():
            raise ValueError(f"Unexpected coverage for {family}")
    for name, data in (("direct_results", results), ("direct_summary", summary),
                       ("direct_highlights", highlights), ("direction_consistency", consistency(results))):
        data.to_csv(output / f"{name}.csv", index=False)
    print("[2/5] Rendering direction matrices; no statistical tests rerun", flush=True)
    presence_bound = max(1, float(np.ceil(results.loc[results.family.eq("gender_presence"), "effect_value"].abs().max() * 100)))
    figure_rows = []
    for scope in SCOPES:
        for family in ("numeric", "gender_presence"):
            name = f"arah_{'umum' if family == 'numeric' else 'gender'}_{scope}"
            plot_consistency(results, scope, family, images / name, presence_bound)
            figure_rows.append({"subset": scope, "family": family, "kind": "consistency", "figure": f"images/{name}.png", "source": "direct_results.csv"})
    # Copy existing principal plots without replacing their original artifacts.
    originals = [(BASE / f"images/{i:02d}_effects_direct_man_woman_{scope}.png", scope, "numeric") for i, scope in enumerate(SCOPES[:4], 10)]
    originals += [(POOL / "images/03_laki_laki_perempuan.png", "ethics_all", "numeric"),
                  (POOL / "images/06_gender_direct.png", "ethics_all", "gender_presence")]
    originals += [(BASE / f"supplementary_images/{i:02d}_direct_gender_markers_{condition}.png", "per_subset_" + condition, "gender_presence") for i, condition in enumerate(CONDITIONS, 13)]
    for source, scope, family in originals:
        name = f"man_woman_{family}_{scope}"
        for extension in (".png", ".svg"):
            shutil.copy2(source.with_suffix(extension), images / (name + extension))
        figure_rows.append({"subset": scope, "family": family, "kind": "existing_direct", "figure": f"images/{name}.png", "source": source.relative_to(ROOT).as_posix()})
    pd.DataFrame(figure_rows).to_csv(output / "figure_index.csv", index=False)
    write_json(output / "manifest.json", {"result_status": "candidate", "statistical_tests_rerun": False,
        "sources": source_files, "script_sha256": sha256(Path(__file__)), "notebook_before_sha256": original_hash,
        "presentation_source_sha256": sha256(source_notebook),
        "old_artifact_hashes": old_hashes, "numeric_ci": "median paired delta, not rank-biserial",
        "highlight_selection": "Largest absolute significant effect per scope/family/model/scenario; no inferential filter",
        "direction_consistency": "Descriptive per model; no pooled model test", "presence_scale_limit_pp": presence_bound})
    write_json(output / "presentation_hashes.json", {p.relative_to(output).as_posix(): sha256(p) for p in sorted(output.rglob("*")) if p.is_file() and p.suffix != ".ipynb"})
    print("[3/5] Reordering and executing the complete notebook", flush=True)
    notebook = rewrite_notebook(original, output, local_figures)
    execute(notebook, notebook_path)
    errors = sum(o.output_type == "error" for c in notebook.cells if c.cell_type == "code" for o in c.outputs)
    if errors:
        raise ValueError("Notebook output contains errors")
    # Verify all original artifacts before publishing the revised notebook.
    for relative, expected in old_hashes.items():
        if hashes(ROOT / relative) != expected:
            raise ValueError(f"Existing artifacts changed: {relative}")
    print("[4/5] Publishing figures and atomically saving notebook", flush=True)
    local_figures.mkdir(parents=True)
    for path in images.iterdir():
        shutil.copy2(path, local_figures / path.name)
    if hashes(local_figures) != hashes(images):
        raise ValueError("Local figure copy mismatch")
    save_notebook(notebook, notebook_path, original_hash)
    saved = nbformat.read(notebook_path, as_version=4)
    if saved != notebook:
        raise ValueError("Notebook readback mismatch")
    original_by_id = {c.id: c for c in original.cells}
    changed_sources = [c.id for c in saved.cells if c.id in original_by_id and c.source != original_by_id[c.id].source]
    validation = {"valid": True, "original_cells_retained": len(original.cells), "total_cells": len(saved.cells),
        "changed_source_cell_ids": changed_sources, "executed_code_cells": sum(c.cell_type == "code" and c.execution_count is not None for c in saved.cells),
        "errors": errors, "direct_rows": len(results), "summary_rows": len(summary), "highlight_rows": len(highlights),
        "new_direction_figures": 10, "copied_original_figures": 10, "local_figure_files": len(hashes(local_figures)),
        "old_artifacts_unchanged": True, "notebook_sha256": sha256(notebook_path), "local_figures": str(local_figures)}
    write_json(output / "validation.json", validation)
    print("[5/5] Complete\n" + json.dumps(validation, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--notebook", type=Path, default=NOTEBOOK)
    parser.add_argument("--source-notebook", type=Path, help="Read an unrevised source backup while saving the requested target")
    args = parser.parse_args()
    build(args.output_dir.resolve(), args.notebook.resolve(), args.source_notebook.resolve() if args.source_notebook else None)
