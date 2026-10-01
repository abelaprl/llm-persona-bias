"""Render frozen ETHICS persona-minus-neutral contrasts in the Bab VI GSM8K layout."""
from __future__ import annotations

import argparse
from datetime import datetime
import html
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.benchmarks.ethics import gender_lm_reference_data as contract
from src.benchmarks.ethics.build_gender_ethics_lm_bab6_viz import sha256, _style

ROOT = contract.ROOT
SOURCE = contract.SOURCE
DEFAULT_OUTPUT = ROOT / "result/ETHICS/reports/gender_lm_neutral_volcano_20261002_r1"
FAMILIES = {
    "reasoning_organization": ("Organisasi", "#B8860B"),
    "epistemic_stance": ("Sikap epistemik", "#E76F51"),
    "logical_connective": ("Konektif logis", "#6A4C93"),
    "self_reference": ("Rujukan diri", "#1982C4"),
    "engagement": ("Keterlibatan pembaca", "#568A17"),
    "textual_elaboration": ("Elaborasi", "#008579"),
    "affective_language": ("Empati / validasi", "#008579"),
    "ethical_content_framing": ("Pembingkaian etika", "#1982C4"),
    "moral_evaluation_language": ("Evaluasi moral", "#B8860B"),
    "gender_identity_discourse": ("Wacana identitas", "#6A4C93"),
    "gender_role_discourse": ("Peran gender", "#008579"),
    "gendered_normative_discourse": ("Evaluasi normatif gender", "#B8860B"),
}
GROUPS = [
    ("general_ethics", "N01_bahasa_umum_vs_netral", "Penanda bahasa umum terhadap skenario netral"),
    ("ethics_specific", "N02_bahasa_etika_vs_netral", "Penanda bahasa etika terhadap skenario netral"),
    ("gender_specific", "N03_wacana_gender_vs_netral", "Penanda wacana gender terhadap skenario netral"),
]


def prepare(comparisons, registry, registry_scope):
    marker_ids = registry.loc[registry.registry_scope.eq(registry_scope) & registry.measurement.eq("density"), "marker_id"].tolist()
    rows = contract.select_contrasts(comparisons, marker_ids, "neutral", "density")
    expected = {(model, marker, gender, scenario) for model in contract.MODELS for marker in marker_ids
                for gender in ("Man", "Woman") for scenario in contract.SCENARIOS}
    actual = set(rows[["model_id", "marker_id", "panel_gender", "panel_scenario"]].itertuples(index=False, name=None))
    if actual != expected or len(rows) != len(expected):
        raise ValueError("Incomplete neutral contrast grid")
    if not rows.family.isin(FAMILIES).all():
        raise ValueError("Unknown marker family")
    p = rows.adjusted_p
    if ((p.notna()) & (~p.between(0, 1))).any():
        raise ValueError("Adjusted p-values outside [0, 1]")
    if ((rows.plot_value.notna()) & (~rows.plot_value.between(-1, 1))).any():
        raise ValueError("Rank-biserial outside [-1, 1]")
    rows["display_effect"] = rows.matched_rank_biserial
    rows["display_direction"] = "persona minus neutral"
    rows["display_neg_log10_p"] = -np.log10(p.clip(lower=np.finfo(float).tiny))
    rows["display_y"] = rows.display_neg_log10_p.clip(upper=50)
    rows["display_y_capped"] = rows.display_neg_log10_p.gt(50)
    rows["display_available"] = rows.display_effect.notna() & rows.adjusted_p.notna()
    return rows


def plot_volcano(rows, title):
    _style()
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), sharex=True, sharey=True)
    fig.subplots_adjust(left=.065, right=.985, top=.86, bottom=.18, hspace=.38, wspace=.12)
    present_families = [family for family in FAMILIES if family in set(rows.family)]
    for row_index, gender in enumerate(("Man", "Woman")):
        for column_index, scenario in enumerate(contract.SCENARIOS):
            ax = axes[row_index, column_index]
            block = rows.loc[rows.panel_gender.eq(gender) & rows.panel_scenario.eq(scenario) & rows.display_available]
            ax.axvline(0, color="#64748b", linewidth=.9)
            ax.axhline(-np.log10(.05), color="#64748b", linestyle="--", linewidth=.9)
            for family in present_families:
                for capped, marker in ((False, "o"), (True, "o")):
                    points = block.loc[block.family.eq(family) & block.display_y_capped.eq(capped)]
                    if points.empty:
                        continue
                    sizes = 25 + 9 * np.log10(points.n_changed.clip(lower=0) + 1)
                    ax.scatter(points.display_effect, points.display_y, s=sizes,
                               color=FAMILIES[family][1], alpha=1, marker=marker,
                               edgecolors="white", linewidths=.4, zorder=3)
            ax.set_title(f"{gender} · {scenario} − N", fontweight="bold")
            ax.set_xlim(-1.04, 1.04)
            ax.set_ylim(-1.5, 52)
            ax.set_xticks([-1, -.5, 0, .5, 1])
            ax.set_yticks([0, 10, 20, 30, 40, 50])
            ax.tick_params(labelbottom=True)
            ax.grid(color="#e2e8f0", linewidth=.7)
            ax.set_axisbelow(True)
    fig.suptitle(title, fontsize=19, fontweight="bold", y=.965)
    fig.text(.015, .52, "−log₁₀(p Holm), dibatasi pada 50", rotation=90, va="center", fontsize=12)
    fig.text(.52, .11, "Matched rank-biserial · persona − netral", ha="center", fontsize=12)
    handles = [Line2D([], [], marker="o", linestyle="", color=FAMILIES[family][1],
                      label=FAMILIES[family][0], markersize=7) for family in present_families]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.52, .035), ncol=len(handles), frameon=False)
    return fig


def build(output):
    output = Path(output).resolve()
    output.relative_to(ROOT)
    if output.exists():
        raise FileExistsError(output)
    source_hashes = {path.relative_to(ROOT).as_posix(): sha256(path)
                     for path in [SOURCE / "comparisons.csv", SOURCE / "registry.json"]}
    comparisons = pd.read_csv(SOURCE / "comparisons.csv", float_precision="round_trip")
    registry = pd.DataFrame(json.loads((SOURCE / "registry.json").read_text(encoding="utf-8")))
    figures = output / "figures"
    figures.mkdir(parents=True)
    index = []
    for scope, figure_id, title in GROUPS:
        rows = prepare(comparisons, registry, scope)
        fig = plot_volcano(rows, title)
        png, csv = figures / f"{figure_id}.png", figures / f"{figure_id}.csv"
        fig.savefig(png, dpi=300)
        plt.close(fig)
        rows.to_csv(csv, index=False)
        caption = (f"{title}. Setiap titik adalah satu marker pada satu model; warna menunjukkan kelompok marker. "
                   "Ukuran titik mengikuti jumlah pasangan dengan selisih bukan nol. Garis putus-putus: p Holm = 0,05. "
                   "Nilai −log10(p Holm) di atas 50 ditempatkan pada batas atas; p-value asli tetap ada di CSV. "
                   f"{int(rows.display_available.sum())} dari {len(rows)} estimasi tersedia; sisanya tetap tercatat pada CSV. "
                   "Keluarga koreksi Holm mengikuti sumber, bukan dihitung ulang per gambar.")
        index.append(dict(figure_id=figure_id, caption=caption, rows=len(rows), panels=8,
                          direction="persona minus neutral", png_sha256=sha256(png), csv_sha256=sha256(csv)))
        print(f"Rendered {len(index)}/3: {figure_id}; rows={len(rows)}, available={int(rows.display_available.sum())}", flush=True)
    pd.DataFrame(index).to_csv(output / "figure_index.csv", index=False)
    manifest = dict(status="candidate_pending_independent_human_audit", source_hashes=source_hashes,
                    display_direction="persona minus neutral", new_inferential_tests=False,
                    image_formats=["png"], figure_count=3, density_marker_count=30,
                    style_reference="D:/ProposalTA-18222008/image/BabVI/gsm8k/gender/03_numeric_effect_volcano_vs_neutral.png")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output / "README.md").write_text("# ETHICS terhadap skenario netral\n\nTiga grafik memakai susunan GSM8K Bab VI: Man/Woman × E/I1/I2/I3. "
        "Seluruh 30 marker density × 6 model × 8 persona = 1.440 baris sumber dipertahankan, tanpa sampling maupun uji baru. "
        "Tidak ada pembalikan tanda: perbandingan adalah persona − netral. Warna menunjukkan kelompok marker, bukan model. "
        "Biner tidak dicampur dengan density; profil biner tetap tersedia di notebook. Holm mengikuti keluarga sumber. "
        "Nilai NA tidak diplot, tetapi tetap ada di CSV. Batas sumbu vertikal 50 tidak mengubah p-value pada CSV. "
        "Titik dengan −log10(p Holm) di atas 50 ditempatkan pada batas atas. "
        "Status tetap candidate; bukan validasi kamus atau hasil audit manusia.\n", encoding="utf-8")
    for relative, expected in source_hashes.items():
        assert sha256(ROOT / relative) == expected
    return output


def install(bundle):
    import nbformat
    from nbclient import NotebookClient
    active = contract.ACTIVE
    new_index = pd.read_csv(bundle / "figure_index.csv")
    old_index = pd.read_csv(active / "figure_index.csv")
    if set(new_index.figure_id) & set(old_index.figure_id):
        raise ValueError("Neutral figures already installed")
    manifest = json.loads((active / "manifest.json").read_text(encoding="utf-8"))
    new_manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    for relative, expected in {**manifest["source_hashes"], **new_manifest["source_hashes"]}.items():
        assert sha256(ROOT / relative) == expected
    for row in new_index.itertuples():
        for ext in ("png", "csv"):
            assert sha256(bundle / "figures" / f"{row.figure_id}.{ext}") == getattr(row, f"{ext}_sha256")
            if (active / "figures" / f"{row.figure_id}.{ext}").exists():
                raise FileExistsError(row.figure_id)
    notebook = nbformat.read(active / contract.NOTEBOOK, as_version=4)
    assert notebook.cells[4].source.startswith("## 2. Hasil utama: perbandingan langsung")
    for cell in notebook.cells:
        if cell.cell_type == "markdown":
            for section in range(7, 1, -1):
                if cell.source.startswith(f"## {section}. "):
                    cell.source = cell.source.replace(f"## {section}. ", f"## {section + 1}. ", 1)
                    break
    notebook.cells[4].source = notebook.cells[4].source.replace("Hasil utama: perbandingan langsung", "Perbandingan langsung")
    notebook.cells[0].source = notebook.cells[0].source.replace("Visual utama diperbarui:", "Visual perbandingan langsung diperbarui:")
    notebook.cells[0].source = notebook.cells[0].source.replace("Titik warna mengidentifikasi **model**, bukan persona.", "Pada grafik langsung, titik warna mengidentifikasi **model**, bukan persona.")
    notebook.cells[0].source += "\n\nGrafik terhadap netral mengikuti GSM8K Bab VI utama; grafik langsung mengikuti GSM8K Lampiran D. Warna pada grafik terhadap netral menunjukkan kelompok marker, sedangkan pada grafik langsung menunjukkan model."
    introduction = """## 2. Hasil utama: penanda linguistik terhadap skenario netral

Grafik berikut mengikuti grafik GSM8K yang dipakai dalam Bab VI utama (`03_numeric_effect_volcano_vs_neutral.png`): dua baris Man/Woman dan empat kolom E/I1/I2/I3. Sumbu horizontal adalah matched rank-biserial untuk **persona − netral**, bukan Man − Woman. Nilai positif menunjukkan density marker lebih tinggi daripada netral. Sumbu vertikal adalah −log10(p Holm), bukan besar efek.

Warna menunjukkan kelompok marker; satu titik tetap satu marker pada satu model, bukan estimasi gabungan enam model. Ukuran titik mengikuti `n_changed`. Garis putus-putus menandai p Holm = 0,05. Nilai −log10(p Holm) di atas 50 ditempatkan pada batas atas; p-value asli tetap tersedia pada CSV. Pada grafik ini signifikansi dibaca dari sumbu vertikal, bukan bentuk titik.

Seluruh 30 marker density ditampilkan melalui tiga kelompok: 14 bahasa umum, 12 bahasa etika, dan 4 wacana gender. Sebelas marker biner tidak dicampurkan ke sumbu rank-biserial. Semua perbandingan sumber dipertahankan tanpa pemilihan signifikansi; estimasi tidak tersedia tidak digambar sebagai nol. Holm tetap memakai keluarga pengujian sumber. Angka ini bersifat candidate dan tidak menggantikan audit manusia.
"""
    notebook.cells[4:4] = [nbformat.v4.new_markdown_cell(introduction),
                           nbformat.v4.new_code_cell("\n".join(f"show('{name}')" for name in new_index.figure_id))]
    notebook.cells[-1].source += "\n\nReproduksi grafik terhadap netral ke direktori baru:\n\n```powershell\n.\\.venv\\Scripts\\python.exe -X utf8 src/benchmarks/ethics/build_gender_ethics_lm_neutral_volcano.py --output result/ETHICS/reports/gender_lm_neutral_volcano_reproduction_01\n```"
    index = pd.concat([old_index.iloc[:2], new_index, old_index.iloc[2:]], ignore_index=True)
    backup = active / "_arsip" / ("sebelum_netral_bab6_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    backup.mkdir(parents=True, exist_ok=False)
    originals = [contract.NOTEBOOK, "figure_index.csv", "manifest.json", "gallery.html", "presentation_source.json", "README.md"]
    for name in originals:
        shutil.copy2(active / name, backup / name)
        assert sha256(active / name) == sha256(backup / name)
    print(f"Backup verified: {backup}", flush=True)
    try:
        for row in new_index.itertuples():
            for ext in ("png", "csv"):
                shutil.copy2(bundle / "figures" / f"{row.figure_id}.{ext}", active / "figures" / f"{row.figure_id}.{ext}")
        index.to_csv(active / "figure_index.csv", index=False)
        manifest.update(figure_count=len(index), neutral_figure_count=3, neutral_bundle=bundle.relative_to(ROOT).as_posix(),
                        source_hashes={**manifest["source_hashes"], **new_manifest["source_hashes"]},
                        previous_presentation=backup.relative_to(active).as_posix(), main_figure_count=8,
                        display_direction="per figure: direct Man minus Woman; neutral persona minus neutral",
                        point_encoding="direct: model colors and significance shapes; neutral: family colors, significance on y-axis")
        (active / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        provenance = json.loads((active / "presentation_source.json").read_text(encoding="utf-8"))
        provenance.update(neutral_figure_bundle=bundle.relative_to(ROOT).as_posix(), neutral_style_reference=new_manifest["style_reference"])
        (active / "presentation_source.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        sections = [f'<section><h2>{html.escape(row.figure_id)}</h2><img src="figures/{row.figure_id}.png"><p>{html.escape(row.caption)}</p>'
                    + " · ".join(f'<a href="figures/{row.figure_id}.{ext}">{ext.upper()}</a>' for ext in ("png", "csv")) + '</section>' for row in index.itertuples()]
        (active / "gallery.html").write_text('<!doctype html><html lang="id"><meta charset="utf-8"><title>Gender ETHICS</title><style>body{font-family:Arial,sans-serif;margin:32px}img{max-width:100%}section{margin-bottom:48px}</style><h1>Gender ETHICS — linguistic marker</h1><p>Candidate. Perbandingan terhadap netral dan perbandingan langsung disajikan terpisah.</p>' + "\n".join(sections) + '</html>', encoding="utf-8")
        for cell in notebook.cells:
            if cell.cell_type == "code":
                cell.outputs, cell.execution_count = [], None
        print("Executing notebook with 16 figures...", flush=True)
        NotebookClient(notebook, timeout=180, kernel_name="python3", resources={"metadata": {"path": str(active)}}).execute()
        assert sum("image/png" in o.get("data", {}) for c in notebook.cells if c.cell_type == "code" for o in c.outputs) == 16
        nbformat.validate(notebook)
        for relative, expected in manifest["source_hashes"].items():
            assert sha256(ROOT / relative) == expected
        nbformat.write(notebook, active / contract.NOTEBOOK)
    except Exception:
        for name in originals:
            shutil.copy2(backup / name, active / name)
        raise
    print("Installed 3 neutral PNG/CSV pairs; 16 figures embedded; source hashes unchanged.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--install-only", action="store_true")
    args = parser.parse_args()
    if args.install_only:
        install(args.output.resolve())
    else:
        build(args.output)
