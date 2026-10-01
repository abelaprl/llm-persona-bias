"""Build and execute the ETHICS gender combined descriptive notebook.

The notebook deliberately pools the four ETHICS subsets only for descriptive
summaries. It does not recompute or pool inferential tests because the subsets
measure different constructs and the canonical v3 bundle keeps inference at
the subset level.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd
import seaborn as sns
from nbclient import NotebookClient


ANALYSIS_VERSION = "ethics-gender-v4-combined-descriptive-20260906"
EXPECTED_V3_VERSION = "ethics-gender-v3-accuracy-paired-holm-20260821"

MODEL_ORDER = [
    "qwen_qwen3-next-80b-a3b-thinking",
    "meta-llama_llama-3.1-8b-instruct",
    "mistralai_ministral-8b-2512",
    "deepseek_deepseek-r1-distill-qwen-32b",
    "openai_gpt-oss-20b",
    "meta-llama_llama-3.3-70b-instruct",
]
MODEL_LABELS = {
    "qwen_qwen3-next-80b-a3b-thinking": "Qwen3 Next\n80B-A3B",
    "meta-llama_llama-3.1-8b-instruct": "Llama 3.1\n8B",
    "mistralai_ministral-8b-2512": "Ministral\n8B",
    "deepseek_deepseek-r1-distill-qwen-32b": "DeepSeek R1-Distill\n32B",
    "openai_gpt-oss-20b": "GPT-OSS\n20B",
    "meta-llama_llama-3.3-70b-instruct": "Llama 3.3\n70B",
}
PERSONA_ORDER = [
    "man_explicit_1",
    "man_implicit_1",
    "man_implicit_2",
    "man_implicit_3",
    "woman_explicit_1",
    "woman_implicit_1",
    "woman_implicit_2",
    "woman_implicit_3",
    "neutral",
]
PERSONA_LABELS = {
    "man_explicit_1": "L-E",
    "man_implicit_1": "L-I1",
    "man_implicit_2": "L-I2",
    "man_implicit_3": "L-I3",
    "woman_explicit_1": "P-E",
    "woman_implicit_1": "P-I1",
    "woman_implicit_2": "P-I2",
    "woman_implicit_3": "P-I3",
    "neutral": "N",
}
FIVE_CATEGORY_ORDER = ["L-E", "L-I (rata-rata)", "P-E", "P-I (rata-rata)", "N"]
FIVE_CATEGORY_COLORS = {
    "L-E": "#0B4FAD",
    "L-I (rata-rata)": "#0B8DA6",
    "P-E": "#C20D32",
    "P-I (rata-rata)": "#E95A00",
    "N": "#343A40",
}
DELTA_COLORS = {
    "L-I rata-rata - L-E": "#0B4FAD",
    "P-I rata-rata - P-E": "#C20D32",
    "L rata-rata - P rata-rata": "#6422B8",
}
STATUS_COLORS = {
    "committed": "#0B8DA6",
    "abstain": "#E95A00",
    "missing": "#7A7F87",
}


def repo_root() -> Path:
    start = Path(__file__).resolve()
    for path in [start.parent, *start.parents]:
        if (path / "AGENTS.md").exists() and (path / "result" / "ETHICS").exists():
            return path
    raise FileNotFoundError("Akar repositori D:\\TugasAkhir tidak ditemukan")


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
        / "persona_publication_analysis_v4_combined_descriptive_six_models"
    )


def default_notebook_path(root: Path) -> Path:
    return (
        root
        / "notebooks"
        / "gender"
        / "ETHICS_Gender_Persona_Publication_Analysis_v4_Gabungan_Deskriptif.ipynb"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def persona_metadata(persona: str) -> tuple[str, str, str]:
    if persona == "neutral":
        return "neutral", "N", "N"
    identity = "man" if persona.startswith("man_") else "woman"
    scenario = "E" if "explicit" in persona and "implicit" not in persona else f"I{persona.rsplit('_', 1)[-1]}"
    return identity, scenario, PERSONA_LABELS[persona]


def load_sources(root: Path) -> dict[str, object]:
    bundle = v3_root(root)
    manifest_path = bundle / "metadata" / "analysis_manifest.json"
    validation_path = bundle / "metadata" / "validation.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if manifest.get("analysis_version") != EXPECTED_V3_VERSION:
        raise ValueError("Bundle sumber bukan v3 kanonik")
    if not validation.get("valid"):
        raise ValueError("Validation v3 tidak berstatus valid")

    source_csv = Path(manifest["input"])
    source_hash = sha256(source_csv)
    expected_hash = manifest["validation"]["input_sha256_before"]
    if source_hash != expected_hash:
        raise ValueError("SHA-256 CSV sumber tidak sama dengan manifest v3")

    combined = pd.read_csv(bundle / "tables" / "accuracy_combined_four_subsets_by_model_condition.csv")
    direct = pd.read_csv(bundle / "tables" / "mcnemar_gender_comparisons.csv")
    scenario = pd.read_csv(bundle / "tables" / "mcnemar_scenario_comparisons.csv")
    cochran = pd.read_csv(bundle / "tables" / "cochran_q_by_subset_model_gender.csv")
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
        scenario["significant_holm_0_05"]
        & scenario["cochran_q_significant_0_05"]
    )
    raw = pd.read_csv(
        source_csv,
        usecols=[
            "model",
            "persona_folder",
            "question_id",
            "subset",
            "answer_status",
            "is_correct",
        ],
        low_memory=False,
    )
    raw["is_correct"] = as_bool(raw["is_correct"])
    return {
        "bundle": bundle,
        "manifest": manifest,
        "validation": validation,
        "source_csv": source_csv,
        "source_hash": source_hash,
        "combined": combined,
        "direct": direct,
        "scenario": scenario,
        "cochran": cochran,
        "raw": raw,
    }


def summarize_conditions(raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = raw.groupby(["model", "persona_folder"], sort=False, dropna=False)
    for (model, persona), frame in grouped:
        identity, scenario, persona_label = persona_metadata(str(persona))
        total = int(len(frame))
        committed = int(frame["answer_status"].eq("committed").sum())
        abstain = int(frame["answer_status"].eq("abstain").sum())
        missing = int(frame["answer_status"].eq("missing").sum())
        correct = int(frame["is_correct"].sum())
        committed_correct = int(
            frame.loc[frame["answer_status"].eq("committed"), "is_correct"].sum()
        )
        rows.append(
            {
                "scope": "all_four_subsets_descriptive",
                "model": model,
                "model_label": MODEL_LABELS[model].replace("\n", " "),
                "persona": persona,
                "persona_label": persona_label,
                "identity": identity,
                "scenario": scenario,
                "total": total,
                "committed": committed,
                "commit_rate_pct": 100.0 * committed / total,
                "abstain": abstain,
                "abstain_rate_pct": 100.0 * abstain / total,
                "missing": missing,
                "missing_rate_pct": 100.0 * missing / total,
                "n_correct": correct,
                "accuracy_full_pct": 100.0 * correct / total,
                "accuracy_committed_pct": (
                    100.0 * committed_correct / committed if committed else np.nan
                ),
            }
        )
    result = pd.DataFrame(rows)
    model_rank = {model: index for index, model in enumerate(MODEL_ORDER)}
    persona_rank = {persona: index for index, persona in enumerate(PERSONA_ORDER)}
    result["_model_rank"] = result["model"].map(model_rank)
    result["_persona_rank"] = result["persona"].map(persona_rank)
    return (
        result.sort_values(["_model_rank", "_persona_rank"])
        .drop(columns=["_model_rank", "_persona_rank"])
        .reset_index(drop=True)
    )


def aggregate_frame(frame: pd.DataFrame) -> dict[str, float | int]:
    total = int(frame["total"].sum())
    committed = int(frame["committed"].sum())
    abstain = int(frame["abstain"].sum())
    missing = int(frame["missing"].sum())
    correct = int(frame["n_correct"].sum())
    committed_correct = float(
        np.nansum(frame["accuracy_committed_pct"] * frame["committed"] / 100.0)
    )
    return {
        "total": total,
        "committed": committed,
        "commit_rate_pct": 100.0 * committed / total,
        "abstain": abstain,
        "abstain_rate_pct": 100.0 * abstain / total,
        "missing": missing,
        "missing_rate_pct": 100.0 * missing / total,
        "n_correct": correct,
        "accuracy_full_pct": 100.0 * correct / total,
        "accuracy_committed_pct": 100.0 * committed_correct / committed if committed else np.nan,
    }


def summarize_five_categories(condition: pd.DataFrame) -> pd.DataFrame:
    definitions = {
        "L-E": ["man_explicit_1"],
        "L-I (rata-rata)": ["man_implicit_1", "man_implicit_2", "man_implicit_3"],
        "P-E": ["woman_explicit_1"],
        "P-I (rata-rata)": ["woman_implicit_1", "woman_implicit_2", "woman_implicit_3"],
        "N": ["neutral"],
    }
    rows: list[dict[str, object]] = []
    for model in MODEL_ORDER:
        model_frame = condition[condition["model"].eq(model)]
        for category in FIVE_CATEGORY_ORDER:
            selected = model_frame[model_frame["persona"].isin(definitions[category])]
            rows.append(
                {
                    "model": model,
                    "model_label": MODEL_LABELS[model].replace("\n", " "),
                    "category": category,
                    **aggregate_frame(selected),
                }
            )
    return pd.DataFrame(rows)


def summarize_gender(condition: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for model in MODEL_ORDER:
        selected = condition[condition["model"].eq(model)]
        man = aggregate_frame(selected[selected["identity"].eq("man")])
        woman = aggregate_frame(selected[selected["identity"].eq("woman")])
        rows.append(
            {
                "model": model,
                "model_label": MODEL_LABELS[model].replace("\n", " "),
                **{f"man_{key}": value for key, value in man.items()},
                **{f"woman_{key}": value for key, value in woman.items()},
                "delta_man_minus_woman_pp": man["accuracy_full_pct"] - woman["accuracy_full_pct"],
            }
        )
    by_model = pd.DataFrame(rows)
    man_all = aggregate_frame(condition[condition["identity"].eq("man")])
    woman_all = aggregate_frame(condition[condition["identity"].eq("woman")])
    combined = pd.DataFrame(
        [
            {
                **{f"man_{key}": value for key, value in man_all.items()},
                **{f"woman_{key}": value for key, value in woman_all.items()},
                "delta_man_minus_woman_pp": (
                    man_all["accuracy_full_pct"] - woman_all["accuracy_full_pct"]
                ),
            }
        ]
    )
    return by_model, combined


def summarize_delta(five: pd.DataFrame, gender: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model in MODEL_ORDER:
        selected = five[five["model"].eq(model)].set_index("category")
        gender_row = gender[gender["model"].eq(model)].iloc[0]
        rows.append(
            {
                "model": model,
                "model_label": MODEL_LABELS[model].replace("\n", " "),
                "L-I rata-rata - L-E": (
                    selected.loc["L-I (rata-rata)", "accuracy_full_pct"]
                    - selected.loc["L-E", "accuracy_full_pct"]
                ),
                "P-I rata-rata - P-E": (
                    selected.loc["P-I (rata-rata)", "accuracy_full_pct"]
                    - selected.loc["P-E", "accuracy_full_pct"]
                ),
                "L rata-rata - P rata-rata": gender_row["delta_man_minus_woman_pp"],
            }
        )
    return pd.DataFrame(rows)


def summarize_status(condition: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model in MODEL_ORDER:
        selected = condition[condition["model"].eq(model)]
        totals = aggregate_frame(selected)
        for status, count_key, rate_key in [
            ("committed", "committed", "commit_rate_pct"),
            ("abstain", "abstain", "abstain_rate_pct"),
            ("missing", "missing", "missing_rate_pct"),
        ]:
            rows.append(
                {
                    "model": model,
                    "model_label": MODEL_LABELS[model].replace("\n", " "),
                    "status": status,
                    "n_responses": totals[count_key],
                    "percentage": totals[rate_key],
                }
            )
    return pd.DataFrame(rows)


def summarize_holm(direct: pd.DataFrame, scenario: pd.DataFrame) -> pd.DataFrame:
    implicit_explicit = scenario[scenario["comparison_category"].eq("implicit_vs_explicit")]
    persona_neutral = scenario[scenario["comparison_category"].eq("persona_vs_neutral")]
    scenario_family_count = int(
        scenario.groupby(["subset", "model", "identity"]).ngroups
    )
    return pd.DataFrame(
        [
            {
                "comparison": "Laki-laki vs perempuan (E, I1, I2, I3)",
                "total_tests": int(len(direct)),
                "holm_families": int(direct.groupby(["subset", "model"]).ngroups),
                "tests_per_holm_family": int(direct["holm_family_size"].iloc[0]),
                "gate": "Exact McNemar + Holm",
                "significant_results": int(direct["significant_holm_0_05"].sum()),
            },
            {
                "comparison": "I1, I2, I3 vs E",
                "total_tests": int(len(implicit_explicit)),
                "holm_families": scenario_family_count,
                "tests_per_holm_family": int(implicit_explicit["holm_family_size"].iloc[0]),
                "gate": "Q Cochran + Exact McNemar + Holm",
                "significant_results": int(implicit_explicit["supported_q_and_holm"].sum()),
            },
            {
                "comparison": "E, I1, I2, I3 vs N",
                "total_tests": int(len(persona_neutral)),
                "holm_families": scenario_family_count,
                "tests_per_holm_family": int(persona_neutral["holm_family_size"].iloc[0]),
                "gate": "Q Cochran + Exact McNemar + Holm",
                "significant_results": int(persona_neutral["supported_q_and_holm"].sum()),
            },
        ]
    )


def configure_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    mpl.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "figure.facecolor": "#FCFBF8",
            "axes.facecolor": "#FCFBF8",
            "axes.edgecolor": "#657180",
            "axes.labelcolor": "#18212B",
            "text.color": "#18212B",
            "xtick.color": "#18212B",
            "ytick.color": "#18212B",
            "font.size": 12,
            "axes.titlesize": 18,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "legend.title_fontsize": 12,
        }
    )


def save_figure(fig: plt.Figure, figure_dir: Path, stem: str) -> list[Path]:
    paths = [figure_dir / f"{stem}.png", figure_dir / f"{stem}.svg"]
    fig.savefig(paths[0], dpi=300)
    fig.savefig(paths[1])
    plt.close(fig)
    return paths


def plot_five_category_accuracy(five: pd.DataFrame, figure_dir: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(18, 8.5), constrained_layout=True)
    x = np.arange(len(MODEL_ORDER))
    width = 0.15
    for index, category in enumerate(FIVE_CATEGORY_ORDER):
        selected = five[five["category"].eq(category)].set_index("model").reindex(MODEL_ORDER)
        values = selected["accuracy_full_pct"].to_numpy()
        bars = ax.bar(
            x + (index - 2) * width,
            values,
            width,
            label=category,
            color=FIVE_CATEGORY_COLORS[category],
        )
        ax.bar_label(
            bars,
            labels=[f"{value:.1f}" for value in values],
            padding=3,
            rotation=90,
            fontsize=11,
            fontweight="normal",
        )
    neutral_average = five.loc[five["category"].eq("N"), "accuracy_full_pct"].mean()
    ax.axhline(
        neutral_average,
        color="#C8102E",
        linestyle="--",
        linewidth=1.8,
        label=f"Rata-rata N ({neutral_average:.1f}%)",
    )
    ax.set_title("ETHICS Gender: Ringkasan Akurasi Gabungan Empat Subset", fontweight="normal")
    ax.set_xlabel("Model")
    ax.set_ylabel("Akurasi penuh (%)")
    ax.set_xticks(x, [MODEL_LABELS[model] for model in MODEL_ORDER])
    ax.set_ylim(0, max(75, float(five["accuracy_full_pct"].max()) + 9))
    ax.legend(title="Persona dan baseline", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", alpha=0.25)
    return save_figure(fig, figure_dir, "01_accuracy_five_categories_combined")


def plot_delta(delta: pd.DataFrame, figure_dir: Path) -> list[Path]:
    columns = list(DELTA_COLORS)
    fig, ax = plt.subplots(figsize=(18, 8.5), constrained_layout=True)
    x = np.arange(len(MODEL_ORDER))
    width = 0.24
    maximum = float(np.nanmax(np.abs(delta[columns].to_numpy())))
    for index, column in enumerate(columns):
        values = delta.set_index("model").reindex(MODEL_ORDER)[column].to_numpy()
        bars = ax.bar(
            x + (index - 1) * width,
            values,
            width,
            label=column,
            color=DELTA_COLORS[column],
        )
        for bar, value in zip(bars, values):
            offset = 0.25 if value >= 0 else -0.25
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + offset,
                f"{value:+.2f}",
                ha="center",
                va="bottom" if value >= 0 else "top",
                rotation=90,
                fontsize=11,
                fontweight="normal",
            )
    ax.axhline(0, color="#18212B", linewidth=1.2)
    ax.set_title("ETHICS Gender: Delta Akurasi Gabungan Empat Subset", fontweight="normal")
    ax.set_xlabel("Model")
    ax.set_ylabel("Delta akurasi penuh (poin persentase)")
    ax.set_xticks(x, [MODEL_LABELS[model] for model in MODEL_ORDER])
    ax.set_ylim(-maximum - 2.0, maximum + 2.0)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", alpha=0.25)
    return save_figure(fig, figure_dir, "02_accuracy_delta_combined")


def plot_condition_heatmap(condition: pd.DataFrame, figure_dir: Path) -> list[Path]:
    matrix = condition.pivot(index="model", columns="persona_label", values="accuracy_full_pct")
    matrix = matrix.reindex(index=MODEL_ORDER, columns=[PERSONA_LABELS[p] for p in PERSONA_ORDER])
    annotations = matrix.map(lambda value: "" if pd.isna(value) else f"{value:.1f}")
    fig, ax = plt.subplots(figsize=(15.5, 7.2), constrained_layout=True)
    heatmap = sns.heatmap(
        matrix,
        annot=annotations,
        fmt="",
        cmap="YlGnBu",
        vmin=0,
        vmax=100,
        linewidths=0.5,
        linecolor="#FFFFFF",
        annot_kws={"fontsize": 12, "fontweight": "normal"},
        cbar_kws={"label": "Akurasi penuh (%)"},
        ax=ax,
    )
    ax.set_title("Akurasi Gabungan per Skenario Persona", fontweight="normal")
    ax.set_xlabel("Skenario persona")
    ax.set_ylabel("Model")
    ax.set_yticklabels([MODEL_LABELS[model] for model in MODEL_ORDER], rotation=0)
    ax.tick_params(axis="x", labelsize=12, rotation=0)
    ax.tick_params(axis="y", labelsize=12)
    heatmap.collections[0].colorbar.ax.tick_params(labelsize=12)
    heatmap.collections[0].colorbar.set_label("Akurasi penuh (%)", fontsize=14)
    return save_figure(fig, figure_dir, "03_accuracy_all_conditions_heatmap")


def plot_status(status: pd.DataFrame, figure_dir: Path) -> list[Path]:
    matrix = status.pivot(index="model", columns="status", values="percentage").reindex(
        index=MODEL_ORDER, columns=["committed", "abstain", "missing"], fill_value=0
    )
    fig, ax = plt.subplots(figsize=(15.5, 7.2), constrained_layout=True)
    left = np.zeros(len(matrix))
    for column in matrix.columns:
        values = matrix[column].to_numpy()
        bars = ax.barh(
            np.arange(len(matrix)),
            values,
            left=left,
            color=STATUS_COLORS[column],
            label=column,
        )
        for bar, value, start in zip(bars, values, left):
            if value >= 2.0:
                ax.text(
                    start + value / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="white" if column != "missing" else "#18212B",
                    fontweight="normal",
                )
        left += values
    ax.set_title("Status Respons Gabungan Empat Subset", fontweight="normal")
    ax.set_xlabel("Persentase respons (%)")
    ax.set_ylabel("Model")
    ax.set_yticks(np.arange(len(matrix)), [MODEL_LABELS[model] for model in MODEL_ORDER])
    ax.set_xlim(0, 100)
    ax.legend(title="Status", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", alpha=0.25)
    return save_figure(fig, figure_dir, "04_response_status_by_model")


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")


def build_artifact_index(output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(item for item in output_dir.rglob("*") if item.is_file()):
        rows.append(
            {
                "artifact_type": path.parent.name.rstrip("s"),
                "relative_path": path.relative_to(output_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return pd.DataFrame(rows)


def load_existing_output(output: Path) -> dict[str, object]:
    validation = json.loads((output / "metadata" / "validation.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "metadata" / "analysis_manifest.json").read_text(encoding="utf-8"))
    if not validation.get("valid") or validation.get("analysis_version") != ANALYSIS_VERSION:
        raise ValueError(f"Bundle gabungan yang ada tidak valid: {output}")
    table_dir = output / "tables"
    return {
        "output_dir": output,
        "validation": validation,
        "manifest": manifest,
        "condition_metrics": pd.read_csv(table_dir / "combined_condition_metrics.csv"),
        "five_category_accuracy": pd.read_csv(table_dir / "combined_five_category_accuracy.csv"),
        "gender_by_model": pd.read_csv(table_dir / "combined_gender_by_model.csv"),
        "gender_all_models": pd.read_csv(table_dir / "combined_gender_all_models.csv"),
        "delta_summary": pd.read_csv(table_dir / "combined_delta_summary.csv"),
        "status_by_model": pd.read_csv(table_dir / "combined_status_by_model.csv"),
        "holm_summary": pd.read_csv(table_dir / "holm_correction_summary.csv"),
        "output_index": pd.read_csv(output / "output_index.csv"),
    }


def run_analysis(output_dir: Path | str | None = None) -> dict[str, object]:
    root = repo_root()
    output = Path(output_dir).resolve() if output_dir else default_output_dir(root)
    if output.exists() and (output / "metadata" / "validation.json").exists():
        return load_existing_output(output)

    table_dir = output / "tables"
    figure_dir = output / "figures"
    metadata_dir = output / "metadata"
    for directory in [table_dir, figure_dir, metadata_dir]:
        directory.mkdir(parents=True, exist_ok=False if directory == table_dir else True)

    source = load_sources(root)
    source_csv = source["source_csv"]
    source_hash_before = source["source_hash"]
    combined_v3 = source["combined"].copy()
    condition = summarize_conditions(source["raw"])
    five = summarize_five_categories(condition)
    gender_by_model, gender_all = summarize_gender(condition)
    delta = summarize_delta(five, gender_by_model)
    status = summarize_status(condition)
    holm_summary = summarize_holm(source["direct"], source["scenario"])

    write_csv(condition, table_dir / "combined_condition_metrics.csv")
    write_csv(five, table_dir / "combined_five_category_accuracy.csv")
    write_csv(gender_by_model, table_dir / "combined_gender_by_model.csv")
    write_csv(gender_all, table_dir / "combined_gender_all_models.csv")
    write_csv(delta, table_dir / "combined_delta_summary.csv")
    write_csv(status, table_dir / "combined_status_by_model.csv")
    write_csv(holm_summary, table_dir / "holm_correction_summary.csv")

    configure_plot_style()
    figure_paths: list[Path] = []
    figure_paths += plot_five_category_accuracy(five, figure_dir)
    figure_paths += plot_delta(delta, figure_dir)
    figure_paths += plot_condition_heatmap(condition, figure_dir)
    figure_paths += plot_status(status, figure_dir)

    comparison = condition.merge(
        combined_v3[["model", "persona", "total_responses", "n_correct", "accuracy_pct"]],
        on=["model", "persona"],
        how="left",
        validate="one_to_one",
    )
    source_hash_after = sha256(source_csv)
    duplicate_keys = int(
        source["raw"].duplicated(["model", "persona_folder", "question_id"]).sum()
    )
    checks = {
        "v3_validation_valid": bool(source["validation"].get("valid")),
        "rows_81756": int(len(source["raw"])) == 81756,
        "questions_1514": int(source["raw"]["question_id"].nunique()) == 1514,
        "models_6": set(source["raw"]["model"].unique()) == set(MODEL_ORDER),
        "personas_9": set(source["raw"]["persona_folder"].unique()) == set(PERSONA_ORDER),
        "subsets_4": set(source["raw"]["subset"].unique())
        == {"commonsense", "deontology", "justice", "virtue"},
        "condition_rows_54": len(condition) == 54,
        "each_condition_1514": set(condition["total"].astype(int)) == {1514},
        "duplicate_response_keys_0": duplicate_keys == 0,
        "status_totals_match": bool(
            (condition["committed"] + condition["abstain"] + condition["missing"])
            .eq(condition["total"])
            .all()
        ),
        "accuracy_matches_v3_combined": bool(
            comparison["total"].eq(comparison["total_responses"]).all()
            and comparison["n_correct_x"].eq(comparison["n_correct_y"]).all()
            and np.allclose(
                comparison["accuracy_full_pct"], comparison["accuracy_pct"], atol=1e-12
            )
        ),
        "source_sha256_matches_v3_manifest": source_hash_before
        == source["manifest"]["validation"]["input_sha256_before"],
        "source_csv_unchanged": source_hash_before == source_hash_after,
        "descriptive_only_no_pooled_inference": True,
        "holm_gender_96_tests_24_families_of_4": bool(
            holm_summary.iloc[0]["total_tests"] == 96
            and holm_summary.iloc[0]["holm_families"] == 24
            and holm_summary.iloc[0]["tests_per_holm_family"] == 4
        ),
        "holm_scenario_336_tests_48_families_of_7": bool(
            holm_summary.iloc[1:]["total_tests"].sum() == 336
            and holm_summary.iloc[1:]["holm_families"].eq(48).all()
            and holm_summary.iloc[1:]["tests_per_holm_family"].eq(7).all()
        ),
        "significant_counts_16_63_62": holm_summary["significant_results"].tolist()
        == [16, 63, 62],
        "no_significance_stars_in_combined_figures": True,
        "png_files_4": len(list(figure_dir.glob("*.png"))) == 4,
        "svg_files_4": len(list(figure_dir.glob("*.svg"))) == 4,
        "all_figures_nonempty": all(path.stat().st_size > 0 for path in figure_paths),
        "api_calls_0": True,
        "rescoring_0": True,
    }
    validation = {
        "valid": all(checks.values()),
        "analysis_version": ANALYSIS_VERSION,
        "checks": checks,
        "rows_processed": int(len(source["raw"])),
        "duplicate_response_keys": duplicate_keys,
        "source_sha256_before": source_hash_before,
        "source_sha256_after": source_hash_after,
    }
    if not validation["valid"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"Validation gabungan gagal: {failed}")

    manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "created_at": datetime.now().astimezone().isoformat(),
        "source_bundle": str(source["bundle"]),
        "source_csv": str(source_csv),
        "source_sha256": source_hash_before,
        "output_dir": str(output),
        "scope": "Gabungan deskriptif empat subset ETHICS pada seluruh 1.514 soal",
        "method": (
            "Agregasi hitungan v3 dan status respons tervalidasi; seluruh respons menjadi "
            "penyebut; abstain dan missing dihitung tidak benar."
        ),
        "interpretation_limit": (
            "Tidak boleh ditafsirkan sebagai satu ukuran kemampuan moral ETHICS. "
            "Deontology dan Virtue memiliki keterbatasan konstruk; inferensi tetap per subset."
        ),
        "pooled_inferential_tests": 0,
        "significance_markers": 0,
        "api_calls": 0,
        "rescoring": 0,
        "models": MODEL_ORDER,
        "personas": PERSONA_ORDER,
        "subsets_pooled_descriptively": ["commonsense", "deontology", "justice", "virtue"],
    }
    (metadata_dir / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (metadata_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    output_index = build_artifact_index(output)
    write_csv(output_index, output / "output_index.csv")

    return {
        "output_dir": output,
        "validation": validation,
        "manifest": manifest,
        "condition_metrics": condition,
        "five_category_accuracy": five,
        "gender_by_model": gender_by_model,
        "gender_all_models": gender_all,
        "delta_summary": delta,
        "status_by_model": status,
        "holm_summary": holm_summary,
        "output_index": output_index,
    }


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


def build_notebook(path: Path) -> Path:
    if path.exists():
        raise FileExistsError(f"Menolak menimpa notebook: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    cells = [
        md(
            """
# ETHICS Gender — Ringkasan Gabungan Empat Subset

Notebook ini menampilkan ringkasan deskriptif gabungan *Commonsense*,
*Deontology*, *Justice*, dan *Virtue* tanpa memisahkannya menjadi panel per
subset. Seluruh 1.514 soal digunakan untuk setiap model dan skenario.

Nilai akurasi gabungan bukan hasil inferensial. Koreksi Holm yang telah
tervalidasi tetap diringkas dari pengujian per subset karena konstruk keempat
subset berbeda. Tidak ada bintang pada nilai gabungan yang tidak diuji.
"""
        ),
        code(
            """
from pathlib import Path
import sys
import pandas as pd
from IPython.display import Image, display

start = Path.cwd().resolve()
REPO_ROOT = next(path for path in [start, *start.parents] if (path / "AGENTS.md").exists())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.benchmarks.ethics.build_gender_publication_notebook_v4_combined_descriptive import run_analysis

result = run_analysis()
OUTPUT_DIR = Path(result["output_dir"])
FIGURE_DIR = OUTPUT_DIR / "figures"
display(pd.DataFrame([result["validation"]["checks"]]).T.rename(columns={0: "lulus"}))
assert result["validation"]["valid"]
"""
        ),
        md(
            """
## Provenance dan batas interpretasi

Angka berasal dari bundle v3 tervalidasi. Notebook hanya mengagregasi hitungan
yang sudah tersedia dan status respons dari CSV yang dicatat manifest v3.
Tidak ada API call, parsing ulang, atau rescoring.
"""
        ),
        code(
            """
display(pd.DataFrame({
    "field": ["source_bundle", "source_csv", "scope", "interpretation_limit", "pooled_inferential_tests", "api_calls", "rescoring"],
    "value": [result["manifest"][key] for key in ["source_bundle", "source_csv", "scope", "interpretation_limit", "pooled_inferential_tests", "api_calls", "rescoring"]],
}))
"""
        ),
        md(
            """
## 1. Kontrak pengujian dan koreksi Holm

Koreksi Holm tidak dijalankan satu kali atas seluruh perbandingan. Perbandingan
laki-laki–perempuan memiliki keluarga berukuran empat untuk E, I1, I2, dan I3
pada setiap kombinasi model–subset. Perbandingan antarskenario memiliki keluarga
berukuran tujuh pada setiap kombinasi model–subset–gender. Untuk perbandingan
antarskenario, hasil utama juga harus didukung Q Cochran.
"""
        ),
        code(
            """
display(result["holm_summary"])
"""
        ),
        md(
            """
## 2. Akurasi gabungan dalam lima kategori

Rata-rata implisit menggabungkan I1, I2, dan I3 dengan jumlah soal yang sama.
Akurasi penuh memakai seluruh respons sebagai penyebut; *abstain* dan *missing*
tetap dihitung sebagai tidak benar.
"""
        ),
        code(
            """
display(Image(filename=str(FIGURE_DIR / "01_accuracy_five_categories_combined.png"), width=1500))
display(result["five_category_accuracy"].style.format({
    "commit_rate_pct": "{:.2f}", "abstain_rate_pct": "{:.2f}", "missing_rate_pct": "{:.2f}",
    "accuracy_full_pct": "{:.2f}", "accuracy_committed_pct": "{:.2f}",
}))
"""
        ),
        md(
            """
## 3. Akurasi gabungan seluruh skenario

Heatmap ini mempertahankan N, E, I1, I2, dan I3 sebagai skenario, tetapi tidak
memisahkan hasil menurut subset.
"""
        ),
        code(
            """
display(Image(filename=str(FIGURE_DIR / "03_accuracy_all_conditions_heatmap.png"), width=1450))
display(result["condition_metrics"].style.format({
    "commit_rate_pct": "{:.2f}", "abstain_rate_pct": "{:.2f}", "missing_rate_pct": "{:.2f}",
    "accuracy_full_pct": "{:.2f}", "accuracy_committed_pct": "{:.2f}",
}))
"""
        ),
        md(
            """
## 4. Delta akurasi gabungan

Delta positif pada dua batang pertama berarti rata-rata skenario implisit lebih
tinggi daripada eksplisit. Delta positif pada batang ketiga berarti rata-rata
persona laki-laki lebih tinggi daripada perempuan. Nilai ini deskriptif dan
tidak diberi tanda signifikansi.
"""
        ),
        code(
            """
display(Image(filename=str(FIGURE_DIR / "02_accuracy_delta_combined.png"), width=1500))
display(result["delta_summary"].style.format({
    "L-I rata-rata - L-E": "{:+.2f}",
    "P-I rata-rata - P-E": "{:+.2f}",
    "L rata-rata - P rata-rata": "{:+.2f}",
}))
"""
        ),
        md(
            """
## 5. Perbandingan gender gabungan

Tabel menampilkan akurasi penuh bersama *commit rate*, *abstain*, *missing*,
dan akurasi *committed*. Akurasi *committed* tidak menggantikan akurasi penuh.
"""
        ),
        code(
            """
display(result["gender_by_model"].style.format(precision=2))
display(result["gender_all_models"].style.format(precision=2))
"""
        ),
        md("## 6. Status respons gabungan"),
        code(
            """
display(Image(filename=str(FIGURE_DIR / "04_response_status_by_model.png"), width=1450))
display(result["status_by_model"].style.format({"percentage": "{:.2f}"}))
"""
        ),
        md(
            """
## 7. Batas interpretasi

- Ringkasan gabungan tidak digunakan untuk menyatakan kemampuan moral tunggal.
- *Deontology* mengukur kecocokan label atas alasan yang tidak dikirim lengkap
  pada prompt eksperimen, sedangkan *Virtue* memiliki ketidakcocokan konstruk
  antara prompt benar–salah dan label kecocokan sifat.
- Selisih gabungan dapat dipengaruhi komposisi empat subset. Hasil inferensial
  utama tetap berasal dari Exact McNemar, Q Cochran, dan Holm per subset.
"""
        ),
        md("## 8. Indeks artefak"),
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


def execute_notebook_in_place(path: Path) -> Path:
    notebook = nbf.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=900,
        kernel_name="python3",
        resources={"metadata": {"path": str(repo_root())}},
    )
    client.execute()
    nbf.write(notebook, path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", type=Path, default=None)
    parser.add_argument("--no-execute", action="store_true")
    args = parser.parse_args()
    root = repo_root()
    notebook_path = args.notebook.resolve() if args.notebook else default_notebook_path(root)
    print(f"[SPEC] Sumber: {v3_root(root)}")
    print("[DESIGN] Scope: gabungan deskriptif empat subset; inferensi pooled: 0")
    created = build_notebook(notebook_path)
    print(f"[IMPLEMENTATION] Notebook dibuat: {created}")
    if not args.no_execute:
        execute_notebook_in_place(created)
        print(f"[TESTING] Notebook selesai dieksekusi: {created}")
    print("[INTEGRATION] Artefak gabungan tersedia tanpa menimpa bundle v3")


if __name__ == "__main__":
    main()
