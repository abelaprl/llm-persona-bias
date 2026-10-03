"""Redraw frozen GSM8K tables with shared gender terminology and PNG output.

This presentation adapter calls plotting functions only, never statistical builders.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import textwrap
import types

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
# The historical plotters predate the directory rename. Resolve their package
# imports to the current, unchanged source directory without copying modules.
package = types.ModuleType("src.core.linguistic_markers")
package.__path__ = [str(ROOT / "src/core/linguistic_markers")]
sys.modules[package.__name__] = package
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.text import Text
import numpy as np
import pandas as pd
from src.pipelines.analysis.gender_visuals import revision_v2 as v2
from src.pipelines.analysis.gender_visuals import revision_v3 as v3
from src.pipelines.analysis.gender_visuals import lexicon_analysis as lex
from src.benchmarks.ethics.gender_lm_terminology import display_text, model_display, MODEL_NAMES
from src.benchmarks.ethics import build_gender_ethics_lm_print as ethics

SOURCE = ROOT / "result/analyses/linguistic_markers/runs/gender_gsm8k_bab4_visual_revision_v3_direct_gender_r2_20260817"
OUTPUT = ROOT / "result/benchmarks/ethics/reports/gender_gsm8k_lm_consistent_20261002_r4"
TITLES = {
    "01": "Ketersediaan teks respons menurut persona",
    "02": "Panjang respons menurut skenario",
    "03": "Ukuran efek penanda linguistik numerik\nterhadap skenario netral",
    "04": "Ukuran efek penanda linguistik numerik\nantara skenario implisit dan eksplisit",
    "05": "Perbedaan prevalensi penanda struktur dan logika\nterhadap skenario netral",
    "06": "Perbedaan prevalensi penanda sikap dan kata ganti\nterhadap skenario netral",
    "07": "Sensitivitas ukuran efek terhadap ketepatan jawaban",
    "08": "Komposisi kategori pembuka respons",
    "09": "Perbedaan penggunaan kata terhadap skenario netral",
    "10": "Perbedaan penggunaan kata\nantara skenario implisit dan eksplisit",
    "11": "Istilah gender dalam respons menurut persona",
    "12": "Prevalensi pronomina gender menurut skenario",
    "13": "Perbedaan prevalensi pronomina gender\nantara persona laki-laki dan perempuan",
    "14": "Ukuran efek penanda linguistik numerik\nantara persona laki-laki dan perempuan",
    "15": "Perbedaan penanda linguistik numerik\nantara persona laki-laki dan perempuan",
    "16": "Perbedaan ketersediaan respons\nantara persona laki-laki dan perempuan",
    "17": "Perbedaan komposisi pembuka respons\nantara persona laki-laki dan perempuan",
}
OPENING = {"deliberative": "Pertimbangan", "answer first": "Jawaban langsung",
           "document heading": "Subjudul", "step first": "Langkah penyelesaian",
           "bullet first": "Daftar berpoin", "algebraic": "Aljabar",
           "instructional": "Instruksional", "problem restatement": "Pengulangan soal",
           "narrative": "Naratif", "other": "Lainnya", "empty": "Kosong"}


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def translate(value):
    return OPENING.get(value, display_text(value))


def labels_and_contrast(fig, fid):
    """Retain every plotted estimate; encode direct significance by shape."""
    for ax in fig.axes:
        for collection in ax.collections:
            if isinstance(collection, PathCollection):
                alpha = collection.get_alpha()
                if fid == "15" and np.isscalar(alpha) and alpha is not None and alpha > .9:
                    marker = MarkerStyle("^")
                    collection.set_paths([marker.get_path().transformed(marker.get_transform())])
                collection.set_alpha(np.ones_like(alpha) if alpha is not None and not np.isscalar(alpha) else 1)
        for line in ax.lines:
            if fid == "16" and line.get_marker() == "o" and (line.get_alpha() or 1) > .9:
                line.set_marker("^")
            line.set_alpha(1)
        for axis in (ax.xaxis, ax.yaxis):
            tick_labels = axis.get_ticklabels()
            if any(translate(t.get_text()) != t.get_text() for t in tick_labels):
                axis.set_ticks(axis.get_ticklocs(), [translate(t.get_text()) for t in tick_labels])
    for text in fig.findobj(Text):
        text.set_text(translate(text.get_text()))
    if fig._suptitle:
        fig._suptitle.set_text(TITLES[fid])


def native_layout(fig, axes, fid, height, columns, handles, labels, xlabel,
                  categorical=False, ylabel=None, title=None):
    """Lay out selected thesis graphics at their actual 14 cm print width."""
    fig.set_size_inches(14/2.54, height/2.54)
    for text in list(fig.texts):
        text.remove()
    for legend in list(fig.legends):
        legend.remove()
    for ax in fig.axes:
        ax.set_visible(ax in axes)
    rows = math.ceil(len(axes)/columns)
    left, right = (4.4 if categorical else 1.3), .35
    top, bottom, gap_x, gap_y = 2.1, 3.7, .6, 1.1
    if fid == "03":
        gap_y = 1.6
    if fid == "08":
        gap_y, top = 1.25, 1.75
    aw = (14-left-right-(columns-1)*gap_x)/columns
    ah = (height-top-bottom-(rows-1)*gap_y)/rows
    for i, ax in enumerate(axes):
        row, col = divmod(i, columns)
        ax.set_position([(left+col*(aw+gap_x))/14,
                         (height-top-row*(ah+gap_y)-ah)/height, aw/14, ah/height])
        panel_title = ax.get_title() or ax.get_title(loc="left")
        ax.set_title("", loc="left")
        panel_title = translate(panel_title)
        if fid == "03":
            panel_title = panel_title.replace(" · ", "\n")
        if fid == "08":
            panel_title = panel_title.replace(" · ", "\n")
        ax.set_title(panel_title, fontsize=8 if fid == "08" else 9, pad=6, fontweight="bold")
        ax.set_xlabel(""); ax.set_ylabel("")
        ax.tick_params(axis="both", labelsize=8, labelbottom=True)
        if fid == "08" and row < rows-1:
            ax.tick_params(labelbottom=False)
        if categorical:
            ticks = ax.get_yticks()
            texts = [model_display(t.get_text()) for t in ax.get_yticklabels()]
            # Shared axes keep the same formatter; reveal labels on each new left panel.
            if fid == "15":
                texts = [textwrap.fill(v2.OUTCOME_LABELS[o], 25) for o in DIRECT_OUTCOMES]
                ticks = list(range(15, -1, -1))
            if texts:
                ax.set_yticks(ticks, texts)
            ax.tick_params(labelleft=col == 0)
        elif fid == "03":
            ax.tick_params(labelleft=col == 0)
        for collection in ax.collections:
            if isinstance(collection, PathCollection):
                collection.set_sizes(collection.get_sizes()*.22)
                collection.set_linewidth(.3)
        for line in ax.lines:
            line.set_markersize(3)
            line.set_linewidth(min(line.get_linewidth(), 1))
        for text in ax.texts:
            text.set_fontsize(8)
        if fid in {"03", "15"}:
            ax.set_xticks([-1, 0, 1])
        elif fid == "08":
            ax.set_xticks([0, .5, 1], ["0", "50", "100"])
        else:
            lo, hi = ax.get_xlim()
            bound = ethics.round_bound(max(abs(lo), abs(hi)))
            ax.set_xlim(-bound*1.05 if lo < 0 else 0, bound*1.05)
            ax.set_xticks([-bound, 0, bound] if lo < 0 else [0, bound/2, bound])
    fig.suptitle(title or TITLES[fid], fontsize=11, fontweight="bold", y=1-.25/height)
    fig.text(.5, (bottom-1.05)/height, xlabel,
             ha="center", va="center", fontsize=9)
    if ylabel:
        fig.text(.025, .52, ylabel, rotation=90, ha="center", va="center", fontsize=9)
    if handles:
        for handle in handles:
            if isinstance(handle, Line2D):
                handle.set_markersize(3.5)
                handle.set_linewidth(1)
        fig.legend(handles, [translate(l) for l in labels], loc="lower center",
                   bbox_to_anchor=(.5, .05/height), ncol=2 if fid == "15" else 3,
                   fontsize=8, frameon=False, handlelength=1, columnspacing=.8,
                   handletextpad=.35, labelspacing=.4)
    fig.canvas.draw()
    return fig


DIRECT_OUTCOMES = ["word_count", "avg_sentence_length", "ttr_300", "organization_explicit_step",
                   "organization_bullet", "organization_section_header", "stance_hedge", "stance_booster",
                   "stance_assumption", "stance_self_correction", "connective_causal", "connective_contrastive",
                   "connective_conditional", "self_reference_first_person", "self_reference_inclusive_we",
                   "engagement_reader_directive"]


def build(output):
    output = Path(output).resolve()
    output.relative_to(ROOT / "result")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    images = output / "images"
    images.mkdir()
    manifest = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    run = Path(manifest["source_run"])
    paths = list(SOURCE.glob("*.csv")) + [run/f for f in ("condition_descriptives.csv", "paired_numeric.csv", "paired_binary.csv")]
    hashes = {str(p): digest(p) for p in paths}
    for p in SOURCE.glob("*.csv"):
        shutil.copy2(p, output/p.name)
    for p in SOURCE.glob("*.json"):
        shutil.copy2(p, output/p.name)
    (output/"source_tables").mkdir()
    for p in paths[-3:]:
        shutil.copy2(p, output/"source_tables"/p.name)
    for module in (v2, v3, lex):
        module.MODEL_LABELS.update(MODEL_NAMES)
    v2.OUTCOME_LABELS.update({k:v for k,v in ethics.LABELS.items() if k in v2.OUTCOME_LABELS})
    v2.OUTCOME_LABELS["ttr_300"] = "Variasi kata (TTR-300)"
    original_save = Figure.savefig
    records, audits = [], []

    def save(fig, fname, *args, **kwargs):
        target = Path(fname)
        fid = target.name[:2]
        labels_and_contrast(fig, fid)
        handles, labels = [], []
        if fig.legends:
            legend = fig.legends[0]
            handles = list(legend.legend_handles)
            labels = [t.get_text() for t in legend.get_texts()]
        pages = [(fig.axes[:], "")]
        if fid == "05":
            pages = [(fig.axes[i:i+3], v2.OUTCOME_LABELS[o]) for i,o in zip((0,3,6),
                     ("organization_explicit_step", "connective_causal", "math_latex_display"))]
        if fid == "16":
            pages = [(fig.axes[:4], "Teks tersedia"), (fig.axes[4:], "Hanya konten teknis")]
        for page, (axes, scope) in enumerate(pages, 1):
            name = target.stem + (f"_p{page:02d}" if page > 1 else "") + ".png"
            native = fid in {"03", "05", "08", "15", "16", "17"}
            if native:
                title = TITLES[fid] + (f"\n{scope}" if scope else "")
                if fid == "03":
                    native_layout(fig, axes, fid, 14, 4, handles, labels,
                                  "Rank-biserial (persona − netral)", ylabel="−log₁₀(p Holm)")
                elif fid == "15":
                    significance = [Line2D([], [], marker="^", linestyle="", color="#475569"),
                                    Line2D([], [], marker="o", linestyle="", color="#475569")]
                    native_layout(fig, axes, fid, 22, 2, handles+significance,
                                  labels+["p Holm < 0,05", "Lainnya"],
                                  "Rank-biserial\nLaki-laki − Perempuan", categorical=True)
                elif fid == "08":
                    native_layout(fig, axes, fid, 22, 2, handles, labels, "Proporsi respons (%)")
                elif fid == "05":
                    for ax in axes:
                        if not ax.get_title():
                            ax.set_title(v2.CONTRAST_LABELS[v2.PROXY_CONTRASTS[axes.index(ax)]])
                        ticks = ax.get_yticks()
                        if not len(ticks):
                            ax.set_yticks(axes[0].get_yticks(), [t.get_text() for t in axes[0].get_yticklabels()])
                    native_layout(fig, axes, fid, 18, 3, handles, labels,
                                  "Selisih prevalensi (poin persentase)\nGaris: interval kepercayaan 95%",
                                  categorical=True, title=title)
                elif fid == "16":
                    for ax, scenario in zip(axes, ("E", "I1", "I2", "I3")):
                        ax.set_title(scenario)
                        ax.set_yticks(range(5,-1,-1), [model_display(v2.MODEL_LABELS[m]) for m in v2.MODEL_ORDER])
                    sh = [Line2D([],[],marker="^",linestyle="",color="#475569"), Line2D([],[],marker="o",linestyle="",color="#475569")]
                    native_layout(fig, axes, fid, 15, 2, sh, ["p Holm < 0,05", "Lainnya"],
                                  "Laki-laki − Perempuan (poin persentase)\nGaris: interval kepercayaan 95%",
                                  categorical=True, title=title)
                elif fid == "17":
                    handles = [Line2D([],[],marker="o",linestyle="",color=c) for c in ("#7C3AED", "#94A3B8")]
                    for ax in axes:
                        ax.set_yticks(range(5,-1,-1), [model_display(v2.MODEL_LABELS[m]) for m in v2.MODEL_ORDER])
                    native_layout(fig, axes, fid, 14, 2, handles, ["p Holm < 0,05", "Lainnya"],
                                  "Variasi total marginal (poin persentase)", categorical=True)
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            visible = [t for t in fig.texts if t.get_visible() and t.get_text()]
            visible += [t for leg in fig.legends for t in leg.findobj(Text) if t.get_visible() and t.get_text()]
            visible += [t for ax in fig.axes if ax.get_visible() for t in ax.findobj(Text) if t.get_visible() and t.get_text()]
            outside = [t.get_text() for t in visible if not fig.bbox.expanded(1.005, 1.005).contains(*t.get_window_extent(renderer).get_points()[0])
                       or not fig.bbox.expanded(1.005, 1.005).contains(*t.get_window_extent(renderer).get_points()[1])]
            if native and outside:
                raise ValueError(f"Text outside {name}: {outside}")
            original_save(fig, images/name, dpi=300 if native else 230,
                          bbox_inches=None if native else "tight", facecolor="white")
            records.append(dict(figure=name, parent_figure=target.name, page=page, scope=scope,
                                description=TITLES[fid].replace("\n", " "), native_14cm=native,
                                width_cm=round(fig.get_figwidth()*2.54,2), height_cm=round(fig.get_figheight()*2.54,2),
                                png_sha256=digest(images/name)))
            audits.append(dict(figure=name, outside_text=outside if native else "legacy_layout", min_font_pt=min(t.get_fontsize() for t in visible)))
            print(f"Rendered GSM8K {name}", flush=True)

    Figure.savefig = save
    try:
        v3.refresh_v3_figures(SOURCE, images)
    finally:
        Figure.savefig = original_save
        plt.close("all")
    assert all(digest(p) == h for p,h in hashes.items())
    pd.DataFrame(records).to_csv(output/"figure_index.csv", index=False)
    (output/"layout_checks.json").write_text(json.dumps(audits, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest.update(presentation_version="gender-consistent-v1", statistics_source=str(SOURCE),
                    source_table_hashes=hashes, new_inferential_tests=False, image_formats=["png"])
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Verified {len(hashes)} unchanged source tables; {len(records)} PNG pages.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    build(args.output)
