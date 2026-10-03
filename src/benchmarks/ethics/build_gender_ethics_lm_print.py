"""Reflow existing marker figures for a 14 cm thesis column; never recompute tests."""
from __future__ import annotations

import argparse
from datetime import datetime
import html
import json
import math
from pathlib import Path
import shutil
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D
from matplotlib.text import Text
import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.benchmarks.ethics import gender_lm_reference_data as data
from src.benchmarks.ethics import gender_lm_reference_plots as old
from src.benchmarks.ethics import build_gender_ethics_lm_bab6_viz as direct
from src.benchmarks.ethics import build_gender_ethics_lm_neutral_volcano as neutral
from src.benchmarks.ethics.gender_lm_terminology import display_text, model_display

ROOT, ACTIVE = data.ROOT, data.ACTIVE
DEFAULT_OUTPUT = ROOT / "result/benchmarks/ethics/reports/gender_lm_print_20261002_r8"
WIDTH_CM = 14
LABELS = {**old.LABELS,
    "organization_bullet": "Daftar berpoin", "organization_section_header": "Subjudul",
    "stance_hedge": "Keraguan", "stance_booster": "Penegas", "stance_self_correction": "Koreksi diri",
    "stance_contextual_hedge": "Keraguan kontekstual", "stance_distancing": "Pengambilan jarak",
    "connective_causal": "Penghubung sebab-akibat", "connective_contrastive": "Penghubung kontras",
    "connective_conditional": "Penghubung kondisi", "connective_additive": "Penghubung tambahan",
    "connective_concessive": "Penghubung konsesif", "self_reference_first_person": "Kata ganti orang pertama",
    "self_reference_inclusive_we": "Kata ganti inklusif", "engagement_reader_directive": "Sapaan atau ajakan",
    "elaboration_code_gloss": "Penjelasan tambahan", "response_stance_committed": "Pernyataan keputusan",
    "response_stance_conditional": "Keputusan bersyarat", "response_stance_refusal": "Penolakan menjawab",
    "framing_deontological": "Kerangka deontologis", "framing_consequentialist": "Kerangka konsekuensialis",
    "framing_virtue_ethics": "Kerangka etika kebajikan", "framing_religious": "Kerangka keagamaan",
    "framing_cultural": "Kerangka budaya", "gender_role_framing": "Kerangka peran gender",
    "any_gender": "Penanda gender", "explicit_user": "Rujukan pengguna eksplisit",
    "quoted_or_hypothetical": "Kutipan / hipotetis", "unknown": "Rujukan belum pasti",
    "scenario_overlap": "Istilah juga ada di soal"}
TITLES = {
    "00_cakupan": "Ketersediaan penjelasan dan pasangan analisis",
    "02_panjang_penjelasan": "Panjang penjelasan menurut skenario",
    "N01_bahasa_umum_vs_netral": "Ukuran efek penanda bahasa umum\nterhadap skenario netral",
    "N02_bahasa_etika_vs_netral": "Ukuran efek penanda bahasa etika\nterhadap skenario netral",
    "N03_wacana_gender_vs_netral": "Ukuran efek penanda wacana gender\nterhadap skenario netral",
    "01_bahasa_umum_man_minus_woman": "Perbedaan penanda bahasa umum\nantarpersona gender",
    "02_bahasa_etika_man_minus_woman": "Perbedaan penanda bahasa etika\nantarpersona gender",
    "03_bahasa_gender_man_minus_woman": "Perbedaan penanda wacana gender\nantarpersona gender",
    "04_marker_biner_man_minus_woman": "Perbedaan prevalensi penanda biner\nantarpersona gender",
    "03_profil_umum_explicit": "Prevalensi penanda bahasa umum\npada skenario eksplisit",
    "04_profil_etika_explicit": "Prevalensi penanda bahasa etika\npada skenario eksplisit",
    "05_profil_gender_semua_skenario": "Prevalensi penanda gender",
    "09_gender_terhadap_netral": "Perbedaan prevalensi penanda gender\nterhadap skenario netral",
    "10_gender_implicit_vs_explicit": "Perbedaan prevalensi penanda gender\nantara skenario implisit dan eksplisit",
    "11_keragaman_kategori": "Distribusi jumlah kategori penanda gender",
    "12_konteks_rujukan": "Konteks penyebutan gender dalam penjelasan",
}
TITLES = {key: display_text(value) for key, value in TITLES.items()}


def wrap_label(value, width=26):
    return textwrap.fill(LABELS.get(value, value), width, break_long_words=False, break_on_hyphens=False)


def round_bound(maximum):
    """Choose a symmetric readable bound without excluding any point or CI."""
    maximum = max(1., float(maximum))
    power = 10 ** math.floor(math.log10(maximum))
    for step in (1, 2, 3, 5, 10):
        if step * power >= maximum:
            return step * power
    raise AssertionError(maximum)


def decorate(fig, axes, title, xlabel, height_cm, columns=2, categorical=False,
             marker_ids=None, legend="models", median=False, significance=False,
             ylabel=None, family_rows=None):
    """Use physical margins, final-size fonts and a compact multirow legend."""
    fig.set_size_inches(WIDTH_CM / 2.54, height_cm / 2.54)
    for item in list(fig.texts):
        item.remove()
    for item in list(fig.legends):
        item.remove()
    for ax in fig.axes:
        ax.set_visible(ax in axes)
    rows = math.ceil(len(axes) / columns)
    left_cm = 4.35 if categorical else 1.55
    right_cm = .35
    top_cm, bottom_cm = 2.05, 3.5
    gap_x_cm, gap_y_cm = .55 if categorical else 1., 1.05
    plot_width = (WIDTH_CM - left_cm - right_cm - gap_x_cm * (columns - 1)) / columns
    plot_height = (height_cm - top_cm - bottom_cm - gap_y_cm * (rows - 1)) / rows
    for i, ax in enumerate(axes):
        row, col = divmod(i, columns)
        ax.set_position([(left_cm + col * (plot_width + gap_x_cm)) / WIDTH_CM,
                         (height_cm - top_cm - row * (plot_height + gap_y_cm) - plot_height) / height_cm,
                         plot_width / WIDTH_CM, plot_height / height_cm])
        ax.set_title(display_text(ax.get_title()), fontsize=9, fontweight="bold", pad=7)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="both", labelsize=8, pad=2)
        if marker_ids:
            ax.set_yticks(range(len(marker_ids)), [wrap_label(m) for m in marker_ids])
            ax.tick_params(labelleft=(col == 0))
        else:
            ax.tick_params(labelleft=(col == 0), labelbottom=True)
        for text in ax.texts:
            text.set_fontsize(8)
        for collection in ax.collections:
            if isinstance(collection, PathCollection):
                collection.set_alpha(1)
                sizes = collection.get_sizes()
                if len(sizes):
                    collection.set_sizes(np.clip(sizes * .5, 9, 22))
    fig.suptitle(display_text(title), x=.5, y=1 - .25 / height_cm, fontsize=11, fontweight="bold", color="#162032")
    fig.text((left_cm + (WIDTH_CM - left_cm - right_cm) / 2) / WIDTH_CM,
             (bottom_cm - 1.0) / height_cm, display_text(xlabel), ha="center", va="center", fontsize=9)
    if ylabel:
        fig.text(.025, .53, ylabel, rotation=90, va="center", ha="center", fontsize=9)
    handles = []
    if legend == "models":
        handles = [Line2D([], [], marker="o", linestyle="", color=data.COLORS[m],
                          label=display_text(data.MODEL_LABELS[m]), markersize=3.5) for m in data.MODELS]
        if median:
            handles.append(Line2D([], [], marker="D", linestyle="", color="#172033", label="Median enam model", markersize=3.5))
    elif legend == "families":
        handles = [Line2D([], [], marker="o", linestyle="", color=neutral.FAMILIES[f][1],
                          label=display_text(neutral.FAMILIES[f][0]), markersize=3.5)
                   for f in neutral.FAMILIES if f in set(family_rows.family)]
    if significance:
        handles.extend([Line2D([], [], marker="^", linestyle="", color="#475569", label="p Holm < 0,05", markersize=4),
                        Line2D([], [], marker="o", linestyle="", color="#475569", label="Lainnya", markersize=3.5)])
    if handles:
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .13 / height_cm),
                   ncol=2, fontsize=8, frameon=False, columnspacing=.9, handletextpad=.35, handlelength=.7,
                   labelspacing=.4, borderaxespad=0)
    return fig


def configure_effect_axes(fig, table, marker_ids, direct_mode=False):
    """Retain source estimates; simplify units and round prevalence ticks."""
    prefix = "display" if direct_mode else "plot"
    unit = table.display_unit.iloc[0] if direct_mode else table.plot_unit.iloc[0]
    numeric = unit == "matched_rank_biserial"
    if not numeric:
        values = table[[f"{prefix}_effect" if direct_mode else "plot_value", f"{prefix}_ci_low", f"{prefix}_ci_high"]].to_numpy(float)
        finite = values[np.isfinite(values)]
        bound = round_bound(np.abs(finite).max() if len(finite) else 1)
        for ax in fig.axes:
            ax.set_xlim(-bound * 1.05, bound * 1.05)
            ax.set_xticks([-bound, 0, bound], [f"{-bound:g}", "0", f"{bound:g}"])
    else:
        for ax in fig.axes:
            ax.set_xticks([-1, 0, 1])
    # The legacy prevalence plots dimmed non-significant estimates. Rebuild their
    # markers with the same positions and CIs, using shape instead of opacity.
    if not direct_mode:
        settings = list(dict.fromkeys(table.panel_scenario))
        settings = [s for s in data.SCENARIOS if s in settings]
        for ai, ax in enumerate(fig.axes):
            gender, scenario = ("Man", "Woman")[ai // len(settings)], settings[ai % len(settings)]
            for collection in list(ax.collections):
                collection.remove()
            for line in ax.lines:
                line.set_alpha(1)
            block = table.loc[table.panel_gender.eq(gender) & table.panel_scenario.eq(scenario)]
            for mi, model in enumerate(data.MODELS):
                for yi, marker_id in enumerate(marker_ids):
                    row = block.loc[block.model_id.eq(model) & block.marker_id.eq(marker_id)].iloc[0]
                    if pd.notna(row.plot_value):
                        ax.scatter(row.plot_value, yi + (mi - 2.5) * .065, color=data.COLORS[model],
                                   marker="^" if row.adjusted_p < .05 else "o", s=23, alpha=1,
                                   edgecolors="white", linewidths=.3, zorder=3)
    return "Rank-biserial" if numeric else "Selisih prevalensi (poin persentase)"


def coverage_figure(table):
    status = table.loc[table.record_type.eq("text_status")]
    support = table.loc[table.record_type.eq("paired_support")]
    fig = old.coverage(status, support)
    axes = fig.axes[:2]
    height = 12.5
    decorate(fig, axes, TITLES["00_cakupan"], "", height, columns=1, legend=None)
    for i, ax in enumerate(axes):
        ax.set_position([.31, .56 if i == 0 else .13, .565, .30])
        ax.set_title("Penjelasan tersedia (%)" if i == 0 else "Pasangan laki-laki–perempuan tersedia (%)", fontsize=9, pad=7)
        ax.set_yticks(range(6), [model_display(data.MODEL_LABELS[m]) for m in data.MODELS], fontsize=8)
        if i == 0:
            ax.set_xticks(range(9), ["N", "L·E", "P·E", "L·I1", "P·I1", "L·I2", "P·I2", "L·I3", "P·I3"])
        ax.tick_params(labelleft=True, labelsize=8)
        for annotation in ax.texts:
            annotation.set_fontsize(8)
            try:
                value = float(annotation.get_text())
                # White text is clearer on medium and dark blue cells.
                annotation.set_color("white" if value >= 50 else "#172033")
            except ValueError:
                pass
    colorbar = fig.axes[2]
    colorbar.set_visible(True)
    colorbar.set_position([.9, .13, .018, .73])
    colorbar.set_ylabel("")
    colorbar.set_title("%", fontsize=8)
    colorbar.tick_params(labelsize=8)
    return fig


def build(output):
    output = Path(output).resolve()
    output.relative_to(ROOT)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    (output / "figures").mkdir()
    (output / "inputs").mkdir()
    input_directory = ACTIVE / "figures"
    index_path = ACTIVE / "figure_index.csv"
    original_index = pd.read_csv(index_path)
    if "parent_figure_id" in original_index:
        provenance = json.loads((ACTIVE / "presentation_source.json").read_text(encoding="utf-8"))
        input_directory = ROOT / provenance["print_bundle"] / "inputs"
        index_path = input_directory / "figure_index.csv"
        original_index = pd.read_csv(index_path)
    if len(original_index) != 16:
        raise ValueError("Expected the 16-figure pre-print presentation")
    source_manifest = json.loads((ACTIVE / "manifest.json").read_text(encoding="utf-8"))
    for relative, expected in source_manifest["source_hashes"].items():
        assert direct.sha256(ROOT / relative) == expected
    shutil.copy2(index_path, output / "inputs/figure_index.csv")
    input_hashes, result, audits = {}, [], []
    registry = pd.DataFrame(json.loads((data.SOURCE / "registry.json").read_text(encoding="utf-8")))

    def save(fig, source_row, page=1, page_scope="all", extra_caption=""):
        parent = source_row.figure_id
        fid = parent if page == 1 else f"{parent}_p{page:02d}"
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        text_candidates = list(fig.texts)
        text_candidates += [t for legend in fig.legends for t in legend.findobj(Text)]
        text_candidates += [t for ax in fig.axes if ax.get_visible() for t in ax.findobj(Text)]
        visible_texts = [t for t in text_candidates if t.get_visible() and t.get_text()]
        smallest = min(t.get_fontsize() for t in visible_texts)
        if smallest < 8:
            raise ValueError(f"Font below 8 pt in {fid}: {smallest}")
        outside = []
        for t in visible_texts:
            box = t.get_window_extent(renderer)
            if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.width + 1 or box.y1 > fig.bbox.height + 1:
                outside.append(t.get_text())
        if outside:
            raise ValueError(f"Text outside canvas in {fid}: {outside}")
        for legend in fig.legends:
            legend_box = legend.get_window_extent(renderer)
            for label in fig.texts:
                if label.get_text() and label.get_window_extent(renderer).overlaps(legend_box):
                    raise ValueError(f"Figure label overlaps legend in {fid}: {label.get_text()}")
        for ax in fig.axes:
            if not ax.get_visible():
                continue
            for tick_labels in (ax.get_xticklabels(), ax.get_yticklabels()):
                boxes = [t.get_window_extent(renderer) for t in tick_labels if t.get_visible() and t.get_text()]
                if any(a.overlaps(b) for i, a in enumerate(boxes) for b in boxes[i+1:]):
                    raise ValueError(f"Tick labels overlap in {fid}: {ax.get_title()}")
        png, csv = output / "figures" / f"{fid}.png", output / "figures" / f"{fid}.csv"
        fig.savefig(png, dpi=300)
        shutil.copy2(output / "inputs" / f"{parent}.csv", csv)
        # Captions retain methodological notes without cluttering the image.
        caption = str(source_row.caption)
        if caption.startswith("**Gambar "):
            caption = caption.split("**", 2)[-1].strip()
        caption = caption.replace("Titik pekat menandai p Holm < 0,05", "Segitiga menandai p Holm < 0,05; lingkaran menandai hasil lainnya")
        caption += " " + extra_caption + " Lebar cetak 14 cm; jangan diperkecil. CSV memuat tabel sumber lengkap, termasuk panel pada halaman lain."
        caption = display_text(caption)
        result.append(dict(figure_id=fid, parent_figure_id=parent, page=page, page_scope=page_scope,
                           title=TITLES[parent], caption=caption.strip(), rows=source_row.rows,
                           panels=sum(ax.get_visible() for ax in fig.axes), width_cm=WIDTH_CM,
                           height_cm=round(fig.get_figheight()*2.54, 2), min_font_pt=smallest,
                           direction=getattr(source_row, "direction", ""), png_sha256=direct.sha256(png), csv_sha256=direct.sha256(csv)))
        audits.append(dict(figure_id=fid, min_font_pt=smallest, outside_text_count=len(outside)))
        print(f"Rendered {fid}: 14 × {fig.get_figheight()*2.54:.1f} cm; min font {smallest:g} pt", flush=True)

    for record in original_index.itertuples():
        fid = record.figure_id
        source = input_directory / f"{fid}.csv"
        assert direct.sha256(source) == record.csv_sha256
        shutil.copy2(source, output / "inputs" / source.name)
        input_hashes[source.relative_to(ROOT).as_posix()] = record.csv_sha256
        table = pd.read_csv(source, float_precision="round_trip")
        ids = [m for m in registry.marker_id if "marker_id" in table and m in set(table.marker_id)]
        if fid == "00_cakupan":
            fig = coverage_figure(table)
            save(fig, record, extra_caption="L = laki-laki; P = perempuan. Skala warna tetap 0–100%.")
        elif fid.startswith("N0"):
            fig = neutral.plot_volcano(table, TITLES[fid])
            ordered = [fig.axes[i] for pair in zip(range(4), range(4, 8)) for i in pair]
            decorate(fig, ordered, TITLES[fid], "Rank-biserial (persona − netral)", 20.5,
                     ylabel="−log₁₀(p Holm)", legend="families", family_rows=table)
            for ax in ordered:
                ax.set_xticks([-1, 0, 1])
                ax.set_yticks([0, 25, 50])
            save(fig, record)
        elif "man_minus_woman" in fid:
            measurement = "density" if table.display_unit.iloc[0] == "matched_rank_biserial" else "presence"
            fig = direct.plot_direct_effects(table, ids, TITLES[fid], measurement)
            unit = configure_effect_axes(fig, table, ids, True)
            height = max(12., min(22., len(ids) * 1.10 + 7.4))
            decorate(fig, fig.axes[:], TITLES[fid], f"{unit}\nMan − Woman", height,
                     categorical=True, marker_ids=ids, median=True, significance=True)
            save(fig, record, extra_caption="Positif: Man lebih tinggi; negatif: Woman lebih tinggi. Tanda silang menunjukkan estimasi tidak tersedia.")
        elif fid in {"09_gender_terhadap_netral", "10_gender_implicit_vs_explicit"}:
            comparison = "neutral" if fid.startswith("09") else "implicit"
            fig = old.effects(table, ids, TITLES[fid], comparison, "presence")
            axes = fig.axes[:]
            unit = configure_effect_axes(fig, table, ids)
            panel_count = len(axes)//2
            for page, gender in enumerate(("Man", "Woman"), 1):
                shown = axes[(page-1)*panel_count:page*panel_count]
                decorate(fig, shown, TITLES[fid] + f" — {gender}", unit, 19.5,
                         categorical=True, marker_ids=ids, significance=True)
                save(fig, record, page, gender, "Garis menunjukkan CI 95% marginal.")
        elif fid in {"05_profil_gender_semua_skenario", "12_konteks_rujukan"}:
            if fid.startswith("12"):
                ids = ["any_gender", "explicit_user", "quoted_or_hypothetical", "unknown", "scenario_overlap"]
            fig = old.profile(table, ids, TITLES[fid])
            axes = fig.axes[:]
            for page, scenario in enumerate(["N", *data.SCENARIOS], 1):
                shown = [axes[0]] if scenario == "N" else [axes[page-1], axes[page+4]]
                if scenario == "N":
                    shown[0].set_title("Netral")
                height = max(9.5, len(shown) * len(ids) * .65 + 5.)
                decorate(fig, shown, TITLES[fid] + f" — {scenario}", "Respons berpenjelasan dengan penanda (%)",
                         height, columns=1, categorical=True, marker_ids=ids, median=True)
                for ax in shown:
                    ax.set_xticks([0, 25, 50, 75, 100])
                save(fig, record, page, scenario, "Skenario netral ditampilkan sekali; median hanya untuk enam estimasi tersedia.")
        elif fid in {"03_profil_umum_explicit", "04_profil_etika_explicit"}:
            fig = old.profile(table, ids, TITLES[fid], ["E"])
            decorate(fig, fig.axes[:], TITLES[fid], "Respons berpenjelasan dengan penanda (%)", 20.,
                     columns=1, categorical=True, marker_ids=ids, median=True)
            save(fig, record)
        elif fid == "02_panjang_penjelasan":
            fig = old.verbosity(table)
            decorate(fig, fig.axes[:], TITLES[fid], "Skenario", 13., columns=1, ylabel="Jumlah kata")
            save(fig, record, extra_caption="Titik: median; garis vertikal: IQR.")
        elif fid == "11_keragaman_kategori":
            fig, axes = plt.subplots(5, 2, figsize=(14/2.54, 21/2.54))
            for row, scenario in enumerate(["N", *data.SCENARIOS]):
                for col, gender in enumerate(("Man", "Woman")):
                    ax = axes[row, col]
                    for model in data.MODELS:
                        block = table.loc[table.model_id.eq(model) & table.persona.eq(data.persona(gender, scenario))].sort_values("threshold")
                        ax.step(block.threshold, block.percent, where="post", color=data.COLORS[model], lw=1.0)
                    ax.set_title(f"{gender} · {scenario}")
                    ax.set_xlim(-.2, 11.2); ax.set_ylim(-2, 102)
                    ax.set_xticks([0, 5, 10]); ax.set_yticks([0, 50, 100])
                    old.tidy(ax)
            decorate(fig, list(axes.ravel()), TITLES[fid], "Jumlah kategori berbeda (0–11)", 22.,
                     ylabel="Proporsi kumulatif respons (%)")
            save(fig, record)
        else:
            raise ValueError(fid)
        plt.close(fig)
    index = pd.DataFrame(result)
    index.to_csv(output / "figure_index.csv", index=False)
    pd.DataFrame(audits).to_csv(output / "layout_checks.csv", index=False)
    manifest = {**source_manifest, "figure_count": len(index), "parent_figure_count": len(original_index),
                "print_width_cm": 14, "min_font_pt": 8, "max_height_cm": float(index.height_cm.max()),
                "image_formats": ["png"], "source_table_hashes": input_hashes,
                "new_inferential_tests": False, "presentation_version": "print14cm-v1"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for relative, expected in {**source_manifest["source_hashes"], **input_hashes}.items():
        assert direct.sha256(ROOT / relative) == expected
    (output / "README.md").write_text("# Grafik linguistic marker ETHICS untuk cetak\n\nLebar gambar 14 cm, font minimal 8 pt, PNG 300 dpi. "
        "Gunakan lebar 14 cm tanpa batas tinggi yang memperkecil gambar. Grafik padat dibagi menjadi beberapa halaman. "
        "File tanpa akhiran p02/p03 adalah halaman pertama; halaman lain wajib disertakan untuk cakupan lengkap. "
        "CSV disalin utuh dan identik dengan sumber, bukan dihitung ulang. Tiap CSV halaman memuat tabel seluruh halaman kelompoknya. "
        "Penjelasan statistik tetap di caption notebook. Status candidate tidak berubah.\n", encoding="utf-8")
    print(f"Built {len(index)} print images from {len(original_index)} unchanged tables.", flush=True)


def install(bundle):
    import nbformat
    from nbclient import NotebookClient
    bundle = Path(bundle).resolve()
    index = pd.read_csv(bundle / "figure_index.csv")
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    old_index = pd.read_csv(ACTIVE / "figure_index.csv")
    # Reinstalling a newer presentation is safe after backing up the active set.
    for relative, expected in {**manifest["source_hashes"], **manifest["source_table_hashes"]}.items():
        assert direct.sha256(ROOT / relative) == expected
    for row in index.itertuples():
        for ext in ("png", "csv"):
            assert direct.sha256(bundle / "figures" / f"{row.figure_id}.{ext}") == getattr(row, f"{ext}_sha256")
            if row.figure_id not in set(old_index.figure_id) and (ACTIVE / "figures" / f"{row.figure_id}.{ext}").exists():
                raise FileExistsError(row.figure_id)
    backup = ACTIVE / "_arsip" / ("sebelum_cetak14cm_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    backup.mkdir(parents=True)
    originals = [data.NOTEBOOK, "figure_index.csv", "manifest.json", "gallery.html", "presentation_source.json", "README.md"]
    originals += [f"figures/{fid}.{ext}" for fid in old_index.figure_id for ext in ("png", "csv")]
    for relative in originals:
        target = backup / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ACTIVE / relative, target)
        assert direct.sha256(target) == direct.sha256(ACTIVE / relative)
    (backup / "backup_hashes.json").write_text(json.dumps({r: direct.sha256(backup/r) for r in originals}, indent=2), encoding="utf-8")
    notebook = nbformat.read(ACTIVE / data.NOTEBOOK, as_version=4)
    source = notebook.cells[1].source
    start, end = source.index("def show(figure_id):"), source.index("def table_link(")
    source = source[:start] + '''def show(figure_id):
    pages = index.loc[index.parent_figure_id.eq(figure_id)].sort_values('page')
    if pages.empty:
        raise KeyError(figure_id)
    for row in pages.itertuples():
        path = FIGURES / (row.figure_id + '.png')
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row.png_sha256
        display(Image(filename=str(path), width=800))
        display(Markdown(row.caption))
        links = [f'<a href="figures/{row.figure_id}.{ext}">{ext.upper()}</a>' for ext in ['png','csv']]
        display(HTML(' · '.join(links)))

''' + source[end:]
    notebook.cells[1].source = source
    if "**Versi cetak:**" not in notebook.cells[0].source:
        notebook.cells[0].source += "\n\n**Versi cetak:** gambar selebar 14 cm, font minimal 8 pt. Gambar padat dibagi menjadi beberapa halaman; semua halaman ditampilkan otomatis. Jangan memasang batas tinggi LaTeX yang mengecilkan gambar. Warna grafik terhadap netral menunjukkan kelompok marker; warna grafik lainnya menunjukkan model."
    for cell in notebook.cells:
        if cell.cell_type == "markdown":
            cell.source = display_text(cell.source)
            cell.source = cell.source.replace("Gambar utama memakai format landscape.", "Gambar disusun untuk lebar teks TA 14 cm.")
            cell.source = cell.source.replace("Jangan memperkecil panel lebar ke 16 cm tanpa memeriksa keterbacaan.", "Pertahankan lebar cetak 14 cm.")
            cell.source = cell.source.replace("dua baris Man/Woman dan empat kolom E/I1/I2/I3", "empat baris E/I1/I2/I3 dan dua kolom Man/Woman agar terbaca pada lebar 14 cm")
            cell.source = cell.source.replace("Empat panel E/I1/I2/I3 dan enam warna model mengikuti Gender GSM8K.", "Empat panel E/I1/I2/I3 disusun 2 × 2; enam warna model mengikuti Gender GSM8K.")
    notebook.cells[-1].source += "\n\nBuilder versi cetak: `src/benchmarks/ethics/build_gender_ethics_lm_print.py`. Bundle cetak menyimpan seluruh CSV masukan dan hash sebelum perubahan."
    print(f"Verified backup: {backup}", flush=True)
    try:
        for row in index.itertuples():
            for ext in ("png", "csv"):
                shutil.copy2(bundle / "figures" / f"{row.figure_id}.{ext}", ACTIVE / "figures" / f"{row.figure_id}.{ext}")
        shutil.copy2(bundle / "figure_index.csv", ACTIVE / "figure_index.csv")
        manifest["previous_presentation"] = backup.relative_to(ACTIVE).as_posix()
        (ACTIVE / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        provenance = json.loads((ACTIVE / "presentation_source.json").read_text(encoding="utf-8"))
        provenance.update(print_bundle=bundle.relative_to(ROOT).as_posix(), style_version="print14cm-v1")
        (ACTIVE / "presentation_source.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        sections = [f'<section><h2>{html.escape(r.figure_id)}</h2><img src="figures/{r.figure_id}.png"><p>{html.escape(r.caption)}</p><a href="figures/{r.figure_id}.png">PNG</a> · <a href="figures/{r.figure_id}.csv">CSV</a></section>' for r in index.itertuples()]
        (ACTIVE / "gallery.html").write_text('<!doctype html><html lang="id"><meta charset="utf-8"><title>ETHICS — grafik cetak</title><style>body{font:16px Arial;margin:2rem}img{width:14cm;max-width:100%}section{break-after:page;margin-bottom:2rem}</style><h1>Penanda linguistik ETHICS — versi cetak</h1><p>Lebar 14 cm. Sertakan semua halaman setiap kelompok grafik. Candidate.</p>' + "\n".join(sections) + '</html>', encoding="utf-8")
        for cell in notebook.cells:
            if cell.cell_type == "code":
                cell.outputs, cell.execution_count = [], None
        print(f"Executing notebook with {len(index)} print figures...", flush=True)
        NotebookClient(notebook, timeout=180, kernel_name="python3", resources={"metadata": {"path": str(ACTIVE)}}).execute()
        assert sum("image/png" in o.get("data", {}) for c in notebook.cells if c.cell_type == "code" for o in c.outputs) == len(index)
        nbformat.validate(notebook)
        nbformat.write(notebook, ACTIVE / data.NOTEBOOK)
        for relative, expected in {**manifest["source_hashes"], **manifest["source_table_hashes"]}.items():
            assert direct.sha256(ROOT / relative) == expected
    except Exception:
        for relative in originals:
            shutil.copy2(backup / relative, ACTIVE / relative)
        raise
    print(f"Installed {len(index)} images, unchanged source statistics, PNG only.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--install-only", action="store_true")
    args = parser.parse_args()
    install(args.output) if args.install_only else build(args.output)
