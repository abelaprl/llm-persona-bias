"""Build the thesis-facing ETHICS Gender accuracy v4 notebook.

This module does not rescore responses and does not call an LLM.  It reads the
validated paired-comparison tables produced by the v3 analysis, verifies their
provenance and expected totals, and creates a new gender-first bundle without
overwriting v3.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib as mpl
import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from nbclient import NotebookClient


ANALYSIS_VERSION = "ethics-gender-v4-accuracy-gender-first-20260826"
EXPECTED_V3_VERSION = "ethics-gender-v3-accuracy-paired-holm-20260821"
MODEL_ORDER = [
    "Qwen3 Next 80B-A3B",
    "Llama 3.1 8B",
    "Ministral 8B",
    "DeepSeek R1-Distill 32B",
    "GPT-OSS 20B",
    "Llama 3.3 70B",
]
SUBSET_ORDER = ["commonsense", "deontology", "justice", "virtue"]
SUBSET_LABELS = {
    "commonsense": "Commonsense",
    "deontology": "Deontology",
    "justice": "Justice",
    "virtue": "Virtue",
}
SCENARIO_ORDER = ["E", "I1", "I2", "I3"]
BOOL_MAP = {True: True, False: False, "True": True, "False": False, "true": True, "false": False}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def v3_root(root: Path) -> Path:
    return (
        root
        / "result"
        / "ETHICS"
        / "reports"
        / "Gender_Persona"
        / "_combined"
        / "persona_publication_analysis_v3_accuracy_paired_holm_six_models"
    )


def default_output_dir(root: Path) -> Path:
    return (
        root
        / "result"
        / "ETHICS"
        / "reports"
        / "Gender_Persona"
        / "_combined"
        / "persona_publication_analysis_v4_accuracy_gender_first_six_models"
    )


def source_notebook_path(root: Path) -> Path:
    return root / "notebooks" / "gender" / "ETHICS_Gender_Persona_Publication_Analysis_v4_Accuracy_Gender_First.ipynb"


def executed_notebook_path(root: Path) -> Path:
    return root / "result" / "ETHICS" / "ETHICS_Gender_Persona_Publication_Analysis_v4_Accuracy_Gender_First.ipynb"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_bool(series: pd.Series) -> pd.Series:
    result = series.map(BOOL_MAP)
    if result.isna().any():
        raise ValueError(f"Nilai boolean tidak dikenal: {sorted(series[result.isna()].astype(str).unique())}")
    return result.astype(bool)


def load_v3_tables(root: Path) -> dict[str, object]:
    base = v3_root(root)
    manifest_path = base / "metadata" / "analysis_manifest.json"
    validation_path = base / "metadata" / "validation.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if manifest.get("analysis_version") != EXPECTED_V3_VERSION:
        raise ValueError(f"Versi sumber bukan v3 kanonik: {manifest.get('analysis_version')}")
    if not validation.get("valid"):
        raise ValueError("Validasi v3 tidak berstatus valid")

    table_dir = base / "tables"
    direct = pd.read_csv(table_dir / "mcnemar_gender_comparisons.csv")
    scenario = pd.read_csv(table_dir / "mcnemar_scenario_comparisons.csv")
    cochran = pd.read_csv(table_dir / "cochran_q_by_subset_model_gender.csv")
    status = pd.read_csv(table_dir / "response_status_by_model.csv")
    committed = pd.read_csv(table_dir / "accuracy_gender_comparison_by_model.csv")
    for frame in (direct, scenario, cochran):
        for column in [name for name in frame.columns if name.startswith("significant")]:
            frame[column] = as_bool(frame[column])

    scenario = scenario.merge(
        cochran[["subset", "model", "identity", "significant_0_05"]].rename(
            columns={"significant_0_05": "cochran_q_significant_0_05"}
        ),
        on=["subset", "model", "identity"],
        how="left",
        validate="many_to_one",
    )
    scenario["supported_q_and_holm"] = (
        scenario["significant_holm_0_05"] & scenario["cochran_q_significant_0_05"]
    )
    return {
        "base": base,
        "manifest": manifest,
        "validation": validation,
        "manifest_path": manifest_path,
        "validation_path": validation_path,
        "direct": direct,
        "scenario": scenario,
        "cochran": cochran,
        "status": status,
        "committed": committed,
    }


def direct_summary(direct: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model in MODEL_ORDER:
        selected = direct[direct["model_label"].eq(model)]
        significant = selected[selected["significant_holm_0_05"]]
        largest = significant.loc[significant["delta_accuracy_pp"].abs().idxmax()] if len(significant) else None
        scenarios = [scenario for scenario in SCENARIO_ORDER if scenario in set(significant["scenario_a"])]
        rows.append(
            {
                "model": model,
                "n_comparisons": int(len(selected)),
                "n_significant": int(len(significant)),
                "n_man_higher": int((significant["delta_accuracy_pp"] > 0).sum()),
                "n_woman_higher": int((significant["delta_accuracy_pp"] < 0).sum()),
                "significant_scenarios": ", ".join(scenarios) if scenarios else "—",
                "largest_comparison": (
                    f"{largest['subset_label']}, {largest['scenario_a']}" if largest is not None else "—"
                ),
                "largest_direction": (
                    "L>P" if largest is not None and largest["delta_accuracy_pp"] > 0 else
                    "P>L" if largest is not None else "—"
                ),
                "largest_delta_pp": float(largest["delta_accuracy_pp"]) if largest is not None else np.nan,
                "largest_cohens_h": float(largest["cohens_h"]) if largest is not None else np.nan,
                "largest_p_holm": float(largest["holm_adjusted_p_value"]) if largest is not None else np.nan,
            }
        )
    return pd.DataFrame(rows)


def scenario_summary(scenario: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for category in ["implicit_vs_explicit", "persona_vs_neutral"]:
        for model in MODEL_ORDER:
            selected = scenario[
                scenario["comparison_category"].eq(category)
                & scenario["model_label"].eq(model)
            ]
            significant = selected[selected["supported_q_and_holm"]]
            largest = significant.loc[significant["delta_accuracy_pp"].abs().idxmax()] if len(significant) else None
            rows.append(
                {
                    "comparison_category": category,
                    "model": model,
                    "n_comparisons": int(len(selected)),
                    "n_supported_q_and_holm": int(len(significant)),
                    "n_higher": int((significant["delta_accuracy_pp"] > 0).sum()),
                    "n_lower": int((significant["delta_accuracy_pp"] < 0).sum()),
                    "largest_comparison": (
                        f"{largest['subset_label']}, {largest['identity_label']} {largest['scenario_a']}–{largest['scenario_b']}"
                        if largest is not None else "—"
                    ),
                    "largest_delta_pp": float(largest["delta_accuracy_pp"]) if largest is not None else np.nan,
                    "largest_cohens_h": float(largest["cohens_h"]) if largest is not None else np.nan,
                }
            )
    return pd.DataFrame(rows)


def configure_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.05)
    mpl.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "figure.facecolor": "#FCFBF8",
            "axes.facecolor": "#FCFBF8",
            "savefig.facecolor": "#FCFBF8",
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


EFFECT_CMAP = LinearSegmentedColormap.from_list(
    "gender_first_effect", ["#2166AC", "#D7E5F0", "#FCFBF8", "#F2D2CC", "#B2182B"]
)


def annotation_matrix(values: pd.DataFrame, significant: pd.DataFrame) -> pd.DataFrame:
    annotations = values.copy().astype(object)
    for row in values.index:
        for column in values.columns:
            value = values.loc[row, column]
            if pd.isna(value):
                annotations.loc[row, column] = ""
            else:
                star = "*" if bool(significant.loc[row, column]) else ""
                annotations.loc[row, column] = f"{value:+.2f}{star}"
    return annotations


def save_figure(fig: plt.Figure, figure_dir: Path, stem: str) -> list[Path]:
    paths = [figure_dir / f"{stem}.png", figure_dir / f"{stem}.svg"]
    fig.savefig(paths[0], dpi=300)
    fig.savefig(paths[1])
    plt.close(fig)
    return paths


def effect_heatmap(
    frame: pd.DataFrame,
    figure_dir: Path,
    *,
    category: str,
    stem: str,
    title: str,
    significance_column: str,
) -> list[Path]:
    selected_all = frame[frame["comparison_category"].eq(category)].copy()
    if category == "man_vs_woman":
        selected_all["column_label"] = selected_all["scenario_a"]
        columns = SCENARIO_ORDER
    elif category == "implicit_vs_explicit":
        selected_all["column_label"] = selected_all["identity"].map({"man": "L", "woman": "P"}) + ":" + selected_all["scenario_a"] + "–E"
        columns = [f"{identity}:{scenario}–E" for identity in ["L", "P"] for scenario in ["I1", "I2", "I3"]]
    else:
        selected_all["column_label"] = selected_all["identity"].map({"man": "L", "woman": "P"}) + ":" + selected_all["scenario_a"] + "–N"
        columns = [f"{identity}:{scenario}–N" for identity in ["L", "P"] for scenario in SCENARIO_ORDER]

    limit = max(0.20, float(np.ceil(selected_all["cohens_h"].abs().max() * 10) / 10))
    is_implicit_explicit = category == "implicit_vs_explicit"
    figure_size = (13.6, 10.4) if is_implicit_explicit else (13.6, 8.4)
    tick_fontsize = 10.5 if is_implicit_explicit else 9
    annotation_fontsize = 9.5 if is_implicit_explicit else 8
    panel_title_fontsize = 14 if is_implicit_explicit else 13
    note_fontsize = 10 if is_implicit_explicit else 9
    fig, axes = plt.subplots(2, 2, figsize=figure_size, constrained_layout=True)
    for ax, subset in zip(axes.flat, SUBSET_ORDER):
        selected = selected_all[selected_all["subset"].eq(subset)]
        values = selected.pivot(index="model_label", columns="column_label", values="cohens_h").reindex(
            index=MODEL_ORDER, columns=columns
        )
        significant = selected.pivot(
            index="model_label", columns="column_label", values=significance_column
        ).reindex(index=MODEL_ORDER, columns=columns).fillna(False).astype(bool)
        sns.heatmap(
            values,
            ax=ax,
            cmap=EFFECT_CMAP,
            center=0,
            vmin=-limit,
            vmax=limit,
            annot=annotation_matrix(values, significant),
            fmt="",
            linewidths=0.45,
            linecolor="#FFFFFF",
            cbar=False,
            annot_kws={"fontsize": annotation_fontsize},
        )
        ax.set_title(SUBSET_LABELS[subset], fontsize=panel_title_fontsize)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=0, labelsize=tick_fontsize)
        ax.tick_params(axis="y", rotation=0, labelsize=tick_fontsize)
    scalar = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=-limit, vmax=limit), cmap=EFFECT_CMAP)
    colorbar = fig.colorbar(scalar, ax=axes.ravel().tolist(), shrink=0.80, pad=0.015)
    colorbar.set_label("Cohen's h")
    fig.suptitle(title, fontsize=18 if is_implicit_explicit else 16, fontweight="bold")
    note = (
        "Nilai positif menunjukkan akurasi persona laki-laki lebih tinggi; nilai negatif menunjukkan akurasi persona perempuan lebih tinggi. * signifikan setelah koreksi Holm."
        if category == "man_vs_woman"
        else "Nilai positif menunjukkan akurasi kondisi pertama lebih tinggi; nilai negatif menunjukkan kondisi pembanding lebih tinggi. * signifikan setelah koreksi Holm dan didukung Q Cochran."
    )
    fig.text(0.5, -0.015, note, ha="center", fontsize=note_fontsize)
    return save_figure(fig, figure_dir, stem)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")


def artifact_index(output_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(p for p in output_dir.rglob("*") if p.is_file()):
        rows.append(
            {
                "relative_path": path.relative_to(output_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return pd.DataFrame(rows)


def load_existing_output(output: Path) -> dict[str, object]:
    """Load a completed v4 bundle without regenerating figures or statistics."""
    metadata_dir = output / "metadata"
    table_dir = output / "tables"
    validation = json.loads((metadata_dir / "validation.json").read_text(encoding="utf-8"))
    manifest = json.loads((metadata_dir / "analysis_manifest.json").read_text(encoding="utf-8"))
    if not validation.get("valid") or validation.get("analysis_version") != ANALYSIS_VERSION:
        raise ValueError(f"Bundle v4 yang ada tidak valid: {output}")
    return {
        "output_dir": output,
        "direct_summary": pd.read_csv(table_dir / "gender_direct_summary_by_model.csv"),
        "direct_significant": pd.read_csv(table_dir / "gender_direct_significant_results.csv"),
        "scenario_summary": pd.read_csv(table_dir / "scenario_summary_by_model.csv"),
        "status": pd.read_csv(table_dir / "response_status_by_model.csv"),
        "validation": validation,
        "manifest": manifest,
        "output_index": pd.read_csv(output / "output_index.csv"),
    }


def run_analysis(output_dir: Path | str | None = None) -> dict[str, object]:
    root = repo_root()
    output = Path(output_dir).resolve() if output_dir else default_output_dir(root)
    if output.exists() and any(output.iterdir()):
        return load_existing_output(output)
    table_dir = output / "tables"
    figure_dir = output / "figures"
    metadata_dir = output / "metadata"
    for directory in (table_dir, figure_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)

    source = load_v3_tables(root)
    direct = source["direct"]
    scenario = source["scenario"]
    status = source["status"]
    summary_direct = direct_summary(direct)
    summary_scenario = scenario_summary(scenario)
    significant_direct = direct[direct["significant_holm_0_05"]].copy()

    write_csv(summary_direct, table_dir / "gender_direct_summary_by_model.csv")
    write_csv(significant_direct, table_dir / "gender_direct_significant_results.csv")
    write_csv(summary_scenario, table_dir / "scenario_summary_by_model.csv")
    write_csv(status, table_dir / "response_status_by_model.csv")

    configure_plot_style()
    figures: list[Path] = []
    figures += effect_heatmap(
        direct,
        figure_dir,
        category="man_vs_woman",
        stem="01_gender_direct_effect_holm_heatmap",
        title="Perbandingan Akurasi Persona Laki-laki dan Perempuan",
        significance_column="significant_holm_0_05",
    )
    figures += effect_heatmap(
        scenario,
        figure_dir,
        category="implicit_vs_explicit",
        stem="02_implicit_vs_explicit_effect_q_holm_heatmap",
        title="Perbandingan Akurasi Skenario Implisit dan Eksplisit",
        significance_column="supported_q_and_holm",
    )
    figures += effect_heatmap(
        scenario,
        figure_dir,
        category="persona_vs_neutral",
        stem="03_persona_vs_neutral_effect_q_holm_heatmap",
        title="Perbandingan Akurasi Skenario Persona dan Netral",
        significance_column="supported_q_and_holm",
    )

    direct_sig = significant_direct
    ie = scenario[scenario["comparison_category"].eq("implicit_vs_explicit")]
    neutral = scenario[scenario["comparison_category"].eq("persona_vs_neutral")]
    validation = {
        "valid": True,
        "analysis_version": ANALYSIS_VERSION,
        "source_analysis_version": source["manifest"]["analysis_version"],
        "source_validation_valid": bool(source["validation"]["valid"]),
        "source_manifest_sha256": sha256(source["manifest_path"]),
        "source_validation_sha256": sha256(source["validation_path"]),
        "gender_comparisons": int(len(direct)),
        "gender_significant_holm": int(len(direct_sig)),
        "gender_significant_man_higher": int((direct_sig["delta_accuracy_pp"] > 0).sum()),
        "gender_significant_woman_higher": int((direct_sig["delta_accuracy_pp"] < 0).sum()),
        "models_with_significant_gender_comparison": int(direct_sig["model"].nunique()),
        "implicit_vs_explicit_comparisons": int(len(ie)),
        "implicit_vs_explicit_supported_q_holm": int(ie["supported_q_and_holm"].sum()),
        "implicit_vs_explicit_higher": int((ie["supported_q_and_holm"] & (ie["delta_accuracy_pp"] > 0)).sum()),
        "implicit_vs_explicit_lower": int((ie["supported_q_and_holm"] & (ie["delta_accuracy_pp"] < 0)).sum()),
        "persona_vs_neutral_comparisons": int(len(neutral)),
        "persona_vs_neutral_supported_q_holm": int(neutral["supported_q_and_holm"].sum()),
        "persona_vs_neutral_higher": int((neutral["supported_q_and_holm"] & (neutral["delta_accuracy_pp"] > 0)).sum()),
        "persona_vs_neutral_lower": int((neutral["supported_q_and_holm"] & (neutral["delta_accuracy_pp"] < 0)).sum()),
        "subsets_present": sorted(direct["subset"].unique().tolist()),
        "combined_four_subset_accuracy_used": False,
        "llm_api_calls": 0,
        "rescoring_performed": False,
        "v3_overwritten": False,
        "png_files": len([path for path in figures if path.suffix == ".png"]),
        "svg_files": len([path for path in figures if path.suffix == ".svg"]),
    }
    expected = {
        "gender_comparisons": 96,
        "gender_significant_holm": 16,
        "gender_significant_man_higher": 12,
        "gender_significant_woman_higher": 4,
        "models_with_significant_gender_comparison": 5,
        "implicit_vs_explicit_comparisons": 144,
        "implicit_vs_explicit_supported_q_holm": 63,
        "implicit_vs_explicit_higher": 28,
        "implicit_vs_explicit_lower": 35,
        "persona_vs_neutral_comparisons": 192,
        "persona_vs_neutral_supported_q_holm": 62,
        "persona_vs_neutral_higher": 22,
        "persona_vs_neutral_lower": 40,
    }
    mismatches = {key: (validation[key], value) for key, value in expected.items() if validation[key] != value}
    if mismatches or validation["subsets_present"] != SUBSET_ORDER:
        validation["valid"] = False
        validation["mismatches"] = mismatches
        raise ValueError(f"Validasi v4 gagal: {validation}")

    manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "created_at": datetime.now(ZoneInfo("Asia/Jakarta")).isoformat(),
        "source_bundle": str(source["base"]),
        "source_analysis_version": source["manifest"]["analysis_version"],
        "output": str(output),
        "purpose": "Penyajian akurasi ETHICS Gender yang memprioritaskan perbandingan laki-laki-perempuan, lalu I/E/N.",
        "statistics_recomputed": False,
        "response_rescoring": False,
        "llm_inference_rerun": False,
        "combined_four_subset_accuracy_used": False,
        "subsets_retained_as_analysis_context": SUBSET_ORDER,
        "validation": validation,
    }
    (metadata_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (metadata_dir / "validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    index = artifact_index(output)
    write_csv(index, output / "output_index.csv")
    return {
        "output_dir": output,
        "direct_summary": summary_direct,
        "direct_significant": significant_direct,
        "scenario_summary": summary_scenario,
        "status": status,
        "validation": validation,
        "manifest": manifest,
        "output_index": artifact_index(output),
    }


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


def build_notebook(path: Path) -> Path:
    if path.exists():
        raise FileExistsError(f"Menolak menimpa notebook sumber: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    cells = [
        md(
            """
# ETHICS Gender Persona — Analisis Akurasi v4 (Gender-First)

Notebook ini menyajikan hasil akurasi dengan urutan pertanyaan penelitian:
perbandingan persona laki-laki dan perempuan sebagai hasil utama, kemudian
perbandingan skenario implisit–eksplisit dan persona–netral. Empat subset tetap
dipertahankan sebagai konteks analisis, tetapi tidak digabung menjadi satu
angka akurasi ETHICS.
"""
        ),
        md("## 0. Provenance dan validasi"),
        code(
            """
from pathlib import Path
import sys
import pandas as pd
from IPython.display import Image, display

start = Path.cwd().resolve()
REPO_ROOT = next(path for path in [start, *start.parents] if (path / "AGENT.md").exists() and (path / "src").exists())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.benchmarks.ethics.build_gender_publication_notebook_v4_accuracy_gender_first import run_analysis

result = run_analysis()
OUTPUT_DIR = Path(result["output_dir"])
FIGURE_DIR = OUTPUT_DIR / "figures"
display(pd.DataFrame([result["validation"]]).T.rename(columns={0: "value"}))
assert result["validation"]["valid"]
"""
        ),
        md(
            """
## 1. Perbandingan persona laki-laki dan perempuan — hasil utama

Dari 96 perbandingan langsung pada soal dan skenario yang sama, 16 tetap
signifikan setelah koreksi Holm. Di antara 16 hasil tersebut, 12 mengarah pada
akurasi persona laki-laki yang lebih tinggi dan empat mengarah pada persona
perempuan. Hasil ditemukan pada lima dari enam model, tetapi arahnya tidak
seragam.
"""
        ),
        code(
            """
display(result["direct_summary"].style.format({"largest_delta_pp": "{:+.2f}", "largest_cohens_h": "{:+.2f}", "largest_p_holm": "{:.4g}"}))
display(Image(filename=str(FIGURE_DIR / "01_gender_direct_effect_holm_heatmap.png"), width=1500))
"""
        ),
        md(
            """
## 2. Perbandingan skenario implisit dan eksplisit

Sebanyak 63 dari 144 perbandingan I1–I3 terhadap E didukung Q Cochran dan
koreksi Holm. Akurasi skenario implisit lebih rendah pada 35 hasil dan lebih
tinggi pada 28 hasil. I1, I2, dan I3 tetap diuji secara terpisah.
"""
        ),
        code(
            """
display(result["scenario_summary"].query("comparison_category == 'implicit_vs_explicit'").style.format({"largest_delta_pp": "{:+.2f}", "largest_cohens_h": "{:+.2f}"}))
display(Image(filename=str(FIGURE_DIR / "02_implicit_vs_explicit_effect_q_holm_heatmap.png"), width=1500))
"""
        ),
        md(
            """
## 3. Perbandingan skenario persona dan netral

Sebanyak 62 dari 192 perbandingan E dan I1–I3 terhadap N didukung Q Cochran
dan koreksi Holm. Akurasi skenario persona lebih rendah pada 40 hasil dan lebih
tinggi pada 22 hasil.
"""
        ),
        code(
            """
display(result["scenario_summary"].query("comparison_category == 'persona_vs_neutral'").style.format({"largest_delta_pp": "{:+.2f}", "largest_cohens_h": "{:+.2f}"}))
display(Image(filename=str(FIGURE_DIR / "03_persona_vs_neutral_effect_q_holm_heatmap.png"), width=1500))
"""
        ),
        md(
            """
## 4. Akurasi dan status respons

Akurasi penuh tetap perlu dibaca bersama status respons karena abstain dan
missing masuk penyebut sebagai jawaban salah. Bagian ini digunakan untuk
menjelaskan perbedaan akurasi, bukan menggantikan hasil berpasangan utama.
"""
        ),
        code("display(result[\"status\"].style.format({\"percentage\": \"{:.2f}\"}))"),
        md(
            """
## 5. Batas interpretasi subset

Inferensi tetap dilakukan per subset. Commonsense dan Justice dapat dibaca
sebagai ketepatan keputusan biner. Pada Deontology, informasi penentu label
tidak dikirim kepada model. Pada Virtue, arti label tidak sesuai dengan bentuk
pertanyaan benar–salah. Karena itu, notebook ini tidak menggunakan satu angka
akurasi gabungan empat subset.
"""
        ),
        md("## 6. Indeks artefak"),
        code("display(result[\"output_index\"])")
    ]
    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
    )
    nbf.write(notebook, path)
    return path


def execute_notebook(source: Path, output: Path) -> Path:
    if output.exists():
        raise FileExistsError(f"Menolak menimpa notebook tereksekusi: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    notebook = nbf.read(source, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=900,
        kernel_name="python3",
        resources={"metadata": {"path": str(repo_root())}},
    )
    client.execute()
    nbf.write(notebook, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    root = repo_root()
    source = build_notebook(source_notebook_path(root))
    print(f"[NOTEBOOK] Source: {source}")
    if args.execute:
        executed = execute_notebook(source, executed_notebook_path(root))
        print(f"[NOTEBOOK] Executed: {executed}")


if __name__ == "__main__":
    main()
