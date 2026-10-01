"""Render standardized Regional Cultural GSM8K accuracy figures.

This renderer is an additive presentation layer over the validated v5 tables.
It does not read model responses, recompute accuracy, rerun paired tests, or
change any existing result bundle.  The visual rules follow the Gender
reference: bold titles, normal-weight axes/legends/annotations, vertical bar
labels, and Holm-only bold heatmap cells.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import seaborn as sns

from render_gsm8k_regional_all_52_effect_holm_stacked_26_26 import (
    MODEL_ORDER,
    PANEL_SPECS,
    comparison_specs,
    format_effect,
    stack_pair_label,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
REPORT_ROOT = REPO_ROOT / "result" / "GMS8K" / "reports"
TABLE_DIR = (
    REPORT_ROOT
    / "persona_publication_analysis_v5_paired_significance_six_models_"
    "bar3_compact_smalllabels_heatmap3_sigbold"
    / "tables"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "notebooks" / "regional" / "akurasi analysis" / "figures"
)

MODEL_IDS = [
    "qwen_qwen3-next-80b-a3b-thinking",
    "meta-llama_llama-3.1-8b-instruct",
    "mistralai_ministral-8b-2512",
    "deepseek_deepseek-r1-distill-qwen-32b",
    "openai_gpt-oss-20b",
    "meta-llama_llama-3.3-70b-instruct",
]
MODEL_LABELS = {
    "qwen_qwen3-next-80b-a3b-thinking": "Qwen3 Next 80B-A3B",
    "meta-llama_llama-3.1-8b-instruct": "Llama 3.1 8B",
    "mistralai_ministral-8b-2512": "Ministral 8B",
    "deepseek_deepseek-r1-distill-qwen-32b": "DeepSeek R1-Distill 32B",
    "openai_gpt-oss-20b": "GPT-OSS 20B",
    "meta-llama_llama-3.3-70b-instruct": "Llama 3.3 70B",
}
MODEL_AXIS_LABELS = {
    "Qwen3 Next 80B-A3B": "Qwen3 Next\n80B-A3B",
    "Llama 3.1 8B": "Llama 3.1\n8B",
    "Ministral 8B": "Ministral\n8B",
    "DeepSeek R1-Distill 32B": "DeepSeek R1-Distill\n32B",
    "GPT-OSS 20B": "GPT-OSS\n20B",
    "Llama 3.3 70B": "Llama 3.3\n70B",
}
MODEL_AXIS_ORDER = [MODEL_AXIS_LABELS[label] for label in MODEL_ORDER]

REGION_LABELS = (
    "North American",
    "East Asian",
    "South Asian",
    "Sub-Saharan African",
)
ACCURACY_COLORS = [
    "#0047AB",
    "#0086A8",
    "#B80F2E",
    "#E45700",
    "#007A4D",
    "#65A30D",
    "#5B21B6",
    "#A0006D",
    "#30343B",
]
DELTA_COLORS = ["#0047AB", "#B80F2E", "#007A4D", "#5B21B6"]
ANNOTATION_SIZE = 17
DELTA_ANNOTATION_SIZE = 16
TITLE_SIZE = 23
FIGURE_SCALE = 1.12
# Keep raster/vector outputs byte-stable across runs so the manifest hashes
# validate reproducibly.  Without this, Matplotlib writes the current date
# into SVG metadata even when the plotted content is unchanged.
FIGURE_METADATA = {"Date": "1970-01-01T00:00:00"}


def sha256_file(path: Path, chunk_size: int = 2**20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _style() -> None:
    sns.set_theme(
        context="notebook",
        style="whitegrid",
        palette="colorblind",
        font_scale=1.25 + 2.0 / 12.0,
    )
    mpl.rcParams.update(
        {
            "figure.dpi": 90,
            "savefig.dpi": 320,
            "savefig.bbox": "tight",
            "figure.facecolor": "#FCFBF8",
            "axes.facecolor": "#FCFBF8",
            "savefig.facecolor": "#FCFBF8",
            "text.color": "#17212B",
            "axes.labelcolor": "#17212B",
            "axes.edgecolor": "#5A6573",
            "axes.titleweight": "bold",
            "axes.titlesize": TITLE_SIZE,
            "axes.labelsize": 19,
            "xtick.labelsize": 17,
            "ytick.labelsize": 17,
            "legend.fontsize": 17,
            "legend.title_fontsize": 18,
            "svg.hashsalt": "gsm8k-regional-accuracy-standardized-v1",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "DejaVu Sans",
        }
    )


def _ordered(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["model"] = result["model"].astype(str)
    result = result.set_index("model").reindex(MODEL_IDS).reset_index()
    if result["model_label"].isna().any():
        raise ValueError("Tabel Regional tidak memuat seluruh enam model utama.")
    return result


def _save_pair(fig: mpl.figure.Figure, output_dir: Path, stem: str, dpi: int) -> None:
    fig.savefig(
        output_dir / f"{stem}.png",
        dpi=dpi,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
        metadata=FIGURE_METADATA,
    )
    fig.savefig(
        output_dir / f"{stem}.svg",
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
        metadata=FIGURE_METADATA,
    )
    plt.close(fig)


def _annotate_accuracy(ax: mpl.axes.Axes) -> None:
    for bar in ax.patches:
        value = bar.get_height()
        ax.annotate(
            f"{value:.3f}",
            (bar.get_x() + bar.get_width() / 2, value),
            ha="center",
            va="bottom",
            xytext=(0, 3),
            textcoords="offset points",
            fontsize=ANNOTATION_SIZE,
            fontweight="normal",
            rotation=90,
        )


def _annotate_delta(ax: mpl.axes.Axes) -> None:
    for bar in ax.patches:
        value = bar.get_height()
        ax.annotate(
            f"{value:+.4f}",
            (bar.get_x() + bar.get_width() / 2, value),
            ha="center",
            va="bottom" if value >= 0 else "top",
            xytext=(0, 10 if value >= 0 else -10),
            textcoords="offset points",
            fontsize=DELTA_ANNOTATION_SIZE,
            fontweight="normal",
            rotation=90,
            rotation_mode="default",
        )


def render_accuracy_and_delta(frame: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    accuracy_columns = [
        column
        for identity in REGION_LABELS
        for column in (f"{identity} Explicit", f"{identity} Implicit Avg")
    ] + ["Neutral"]
    accuracy = frame.set_index("model_label")[accuracy_columns] / 100.0
    neutral_average = float(accuracy["Neutral"].mean())

    fig, ax = plt.subplots(figsize=(23 * FIGURE_SCALE, 9.5 * FIGURE_SCALE))
    accuracy.plot(kind="bar", width=0.88, color=ACCURACY_COLORS, ax=ax)
    baseline_label = f"Global Neutral Avg ({neutral_average:.3f})"
    ax.axhline(
        neutral_average,
        color="#B80F2E",
        linestyle="--",
        linewidth=2.0,
        label=baseline_label,
        zorder=1,
    )
    _annotate_accuracy(ax)
    values = accuracy.to_numpy(dtype=float)
    regional_min = max(0.70, math.floor((float(np.nanmin(values)) - 0.02) * 20) / 20)
    ax.set_ylim(regional_min, 1.00)
    ax.set_title(
        "GSM8K Regional Cultural: Perbandingan Akurasi Antar-Persona",
        fontsize=TITLE_SIZE,
        fontweight="bold",
        pad=18,
    )
    ax.set_xlabel("Model", fontweight="normal")
    ax.set_ylabel("Accuracy", fontweight="normal")
    ax.set_xticklabels(MODEL_AXIS_ORDER, rotation=0, ha="center", fontweight="normal")
    handles, labels = ax.get_legend_handles_labels()
    baseline_index = labels.index(baseline_label)
    handles = [handles[baseline_index]] + [h for index, h in enumerate(handles) if index != baseline_index]
    labels = [labels[baseline_index]] + [label for index, label in enumerate(labels) if index != baseline_index]
    ax.legend(
        handles,
        labels,
        title="Regional Persona & Baseline",
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        frameon=True,
    )
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    _save_pair(fig, output_dir, "12_regional_absolute_accuracy_explicit_implicit", dpi)

    delta_columns = [f"{identity} Delta Imp-Exp" for identity in REGION_LABELS]
    delta = frame.set_index("model_label")[delta_columns] / 100.0
    delta.columns = [f"{identity}: Implicit Avg - Explicit" for identity in REGION_LABELS]
    fig, ax = plt.subplots(figsize=(19 * FIGURE_SCALE, 9 * FIGURE_SCALE))
    delta.plot(kind="bar", width=0.82, color=DELTA_COLORS, ax=ax)
    ax.axhline(0, color="black", linewidth=1)
    _annotate_delta(ax)
    limit = max(0.001, float(np.nanmax(np.abs(delta.to_numpy(dtype=float)))))
    ax.set_ylim(-limit * 1.40, limit * 1.40)
    ax.set_title(
        "GSM8K Regional Cultural - Analisis Delta: Persona Implicit vs Explicit",
        fontsize=TITLE_SIZE,
        fontweight="bold",
        pad=16,
    )
    ax.set_xlabel("Model", fontweight="normal")
    ax.set_ylabel("Difference on accuracy scale", fontweight="normal")
    ax.set_xticklabels(MODEL_AXIS_ORDER, rotation=0, ha="center", fontweight="normal")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    _save_pair(fig, output_dir, "13_regional_delta_implicit_vs_explicit", dpi)


def _boolean_column(frame: pd.DataFrame, column: str) -> pd.Series:
    normalized = frame[column].astype(str).str.strip().str.lower()
    if not normalized.isin({"true", "false", "1", "0"}).all():
        raise ValueError(f"Nilai boolean tidak dikenal pada kolom {column}.")
    return normalized.isin({"true", "1"})


def render_effect_heatmap(frame: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    specs = comparison_specs()
    comparison_order = [str(spec["comparison_label"]) for spec in specs]
    axis_labels = [stack_pair_label(str(spec["axis_label"])) for spec in specs]
    selected = frame[frame["comparison_label"].isin(comparison_order)].copy()
    selected["significant_holm_0_05"] = _boolean_column(selected, "significant_holm_0_05")

    assert len(selected) == len(MODEL_ORDER) * len(comparison_order)
    assert set(selected["model_label"]) == set(MODEL_ORDER)
    assert set(selected["comparison_label"]) == set(comparison_order)

    effects = selected.pivot(
        index="model_label", columns="comparison_label", values="cohen_h"
    ).reindex(index=MODEL_ORDER, columns=comparison_order)
    significant = selected.pivot(
        index="model_label", columns="comparison_label", values="significant_holm_0_05"
    ).reindex(index=MODEL_ORDER, columns=comparison_order).astype(bool)
    assert effects.shape == (6, 52)
    assert effects.notna().all().all()

    annotations = effects.copy().astype(object)
    for model in effects.index:
        for comparison in effects.columns:
            annotations.loc[model, comparison] = format_effect(
                float(effects.loc[model, comparison]),
                bool(significant.loc[model, comparison]),
            )

    cmap = LinearSegmentedColormap.from_list(
        "regional_effect_standardized",
        ["#0B3C78", "#4A98C9", "#DCEAF1", "#FCFBF8", "#F7C7AE", "#E76F51", "#9B002F"],
    )
    max_abs = float(np.nanmax(np.abs(effects.to_numpy(dtype=float))))
    limit = max(0.10, math.ceil(max_abs / 0.05) * 0.05)

    fig = plt.figure(figsize=(18, 22))
    grid = fig.add_gridspec(
        5,
        1,
        height_ratios=[6, 6, 6, 6, 0.42],
        left=0.235,
        right=0.985,
        top=0.91,
        bottom=0.055,
        hspace=0.72,
    )
    axes = [fig.add_subplot(grid[index]) for index in range(4)]
    colorbar_ax = fig.add_subplot(grid[4])
    heatmap = None

    for panel_index, (ax, spec) in enumerate(zip(axes, PANEL_SPECS)):
        start = int(spec["start"])
        stop = int(spec["stop"])
        panel_effects = effects.iloc[:, start:stop]
        panel_significant = significant.iloc[:, start:stop]
        panel_annotations = annotations.iloc[:, start:stop]
        show_colorbar = panel_index == len(PANEL_SPECS) - 1
        heatmap = sns.heatmap(
            panel_effects,
            annot=panel_annotations,
            fmt="",
            cmap=cmap,
            center=0,
            vmin=-limit,
            vmax=limit,
            linewidths=1.25,
            linecolor="#FCFBF8",
            annot_kws={"fontsize": 17.5, "fontweight": "normal"},
            cbar=show_colorbar,
            cbar_ax=colorbar_ax if show_colorbar else None,
            cbar_kws={"orientation": "horizontal"} if show_colorbar else None,
            ax=ax,
        )
        for text_artist, value, is_significant in zip(
            ax.texts,
            panel_effects.to_numpy().ravel(),
            panel_significant.to_numpy().ravel(),
        ):
            text_artist.set_color("white" if abs(float(value)) >= 0.46 * limit else "#262626")
            text_artist.set_fontweight("bold" if bool(is_significant) else "normal")

        ax.set_yticklabels(
            list(MODEL_ORDER), rotation=0, fontsize=16.5, fontweight="normal"
        )
        ax.set_xticklabels(
            axis_labels[start:stop], rotation=0, ha="center", fontsize=15, fontweight="normal"
        )
        ax.tick_params(axis="x", length=0, pad=9)
        ax.tick_params(axis="y", length=0, pad=11)
        ax.set_xlabel("Perbandingan skenario", fontsize=17, fontweight="normal", labelpad=13)
        ax.set_ylabel("Model", fontsize=17, fontweight="normal", labelpad=15)
        ax.set_title(str(spec["title"]), fontsize=19, fontweight="bold", pad=14)

    assert heatmap is not None
    colorbar = heatmap.collections[0].colorbar
    ticks = np.linspace(-limit, limit, 9)
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels([f"{value:.2f}".replace("-", "-").replace(".", ",") for value in ticks])
    colorbar.ax.tick_params(labelsize=14.5)
    colorbar.set_label("Cohen's h", fontsize=16.5, fontweight="normal", labelpad=8)
    fig.suptitle(
        "Ukuran Efek dan Hasil Koreksi Holm pada Domain Regional-Kultural",
        fontsize=23,
        fontweight="bold",
        y=0.965,
    )
    # Arti tanda *, dagger, dan double-dagger dijelaskan di caption/notebook.
    # Footnote lama sengaja tidak dimasukkan ke gambar agar konsisten dengan Fig. 24 Gender.
    _save_pair(fig, output_dir, "27_regional_all_52_effect_holm_stacked_26_26", dpi)


def _write_manifest(output_dir: Path, table_dir: Path) -> None:
    figure_files = sorted(output_dir.glob("*.png")) + sorted(output_dir.glob("*.svg"))
    table_names = [
        "accuracy_by_persona_six_models.csv",
        "regional_accuracy_and_deltas.csv",
        "regional_paired_effects_significance.csv",
        "regional_all_52_paired_effects_significance.csv",
        "cochran_q_omnibus_all_groups.csv",
    ]
    manifest = {
        "bundle_version": "gsm8k-regional-accuracy-standardized-v1.0.0",
        "source_table_dir": str(table_dir),
        "source_tables": {
            name: sha256_file(table_dir / name) for name in table_names
        },
        "figures": [
            {
                "file": path.name,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in figure_files
        ],
        "statistical_tables_unchanged": True,
        "bar_annotation_pt": ANNOTATION_SIZE,
        "delta_annotation_pt": DELTA_ANNOTATION_SIZE,
        "delta_annotation_rotation": 90,
        "heatmap_title_pt": 23,
        "heatmap_footnote": False,
    }
    (output_dir / "render_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table-dir", type=Path, default=TABLE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=320)
    args = parser.parse_args()
    if args.dpi < 300:
        raise ValueError("DPI output harus minimal 300.")

    table_dir = args.table_dir.resolve()
    output_dir = args.output_dir.resolve()
    required = {
        "regional_accuracy_and_deltas.csv",
        "regional_all_52_paired_effects_significance.csv",
    }
    missing = sorted(name for name in required if not (table_dir / name).is_file())
    if missing:
        raise FileNotFoundError(f"Tabel v5 Regional tidak lengkap: {missing}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Folder output tidak kosong; pilih folder versioned baru: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    _style()

    regional = _ordered(pd.read_csv(table_dir / "regional_accuracy_and_deltas.csv"))
    render_accuracy_and_delta(regional, output_dir, args.dpi)
    all_effects = pd.read_csv(
        table_dir / "regional_all_52_paired_effects_significance.csv",
        low_memory=False,
    )
    render_effect_heatmap(all_effects, output_dir, args.dpi)
    _write_manifest(output_dir, table_dir)
    print(f"[DONE] Regional standardized figures: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
