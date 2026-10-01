"""Create additive, Bab-VI-ready ETHICS gender marker visuals.

The figures use the existing *overall ETHICS* candidate table.  They make the
same evidence dimensions visible as the GSM8K gender figures: effect direction,
Holm-adjusted significance, and the number of paired responses that changed.
They do not pool or create an ETHICS accuracy score.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ANALYSIS_DIR = (
    REPO_ROOT
    / "result/ETHICS/reports/Gender_Persona/linguistic_markers/analysis"
    / "gender_ethics_linguistic_markers_overall_candidate_r2_20260905"
)
DEFAULT_REGISTRY = (
    REPO_ROOT
    / "result/ETHICS/reports/Gender_Persona"
    / "linguistic_markers_gender_candidate_20260821_v1_6_0/marker_registry.csv"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "result/ETHICS/reports/Gender_Persona/linguistic_markers/visual"
    / "gender_ethics_bab6_overall_candidate_v1_1_20260906"
)

MODEL_ORDER = (
    "meta-llama_llama-3.1-8b-instruct",
    "meta-llama_llama-3.3-70b-instruct",
    "mistralai_ministral-8b-2512",
    "deepseek_deepseek-r1-distill-qwen-32b",
    "openai_gpt-oss-20b",
    "qwen_qwen3-next-80b-a3b-thinking",
)
MODEL_LABELS = {
    "meta-llama_llama-3.1-8b-instruct": "Llama 3.1 8B",
    "meta-llama_llama-3.3-70b-instruct": "Llama 3.3 70B",
    "mistralai_ministral-8b-2512": "Ministral 8B",
    "deepseek_deepseek-r1-distill-qwen-32b": "DeepSeek R1 32B",
    "openai_gpt-oss-20b": "GPT-OSS 20B",
    "qwen_qwen3-next-80b-a3b-thinking": "Qwen3 Next 80B",
}
MODEL_COLORS = dict(zip(MODEL_ORDER, ("#2563EB", "#7C3AED", "#059669", "#DC2626", "#D97706", "#0891B2")))
CONDITIONS = ("explicit", "implicit_1", "implicit_2", "implicit_3")
CONDITION_LABELS = {"explicit": "Eksplisit", "implicit_1": "Implisit I1", "implicit_2": "Implisit I2", "implicit_3": "Implisit I3"}
SURFACE_OUTCOMES = ("word_count", "avg_sentence_length", "ttr_300")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--include-consistency",
        action="store_true",
        help="Also render the large effect-consistency map (not needed for Bab VI).",
    )
    return parser.parse_args()


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _outcome_metadata(registry: pd.DataFrame, manifest: dict) -> tuple[list[str], dict[str, str], dict[str, str]]:
    outcomes = list(manifest["primary_surface_outcomes"]) + list(manifest["primary_marker_outcomes"])
    family = {outcome: "surface" for outcome in SURFACE_OUTCOMES}
    family.update(registry.set_index("marker_id")["family"].to_dict())
    labels = {
        "word_count": "Jumlah kata",
        "avg_sentence_length": "Panjang kalimat rata-rata",
        "ttr_300": "TTR (maks. 300 kata)",
    }
    labels.update({outcome: outcome.replace("_", " ") for outcome in outcomes if outcome not in labels})
    return outcomes, family, labels


def _load(analysis_dir: Path, registry_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    paired_path = analysis_dir / "overall_paired_numeric_21.csv"
    manifest_path = analysis_dir / "manifest.json"
    for path in (paired_path, manifest_path, registry_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    paired = pd.read_csv(paired_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    registry = pd.read_csv(registry_path)
    direct = paired[
        paired["pair_filter"].eq("all_visible_text")
        & paired["analysis_family"].eq("direct_gender")
    ].copy()
    expected = len(MODEL_ORDER) * len(CONDITIONS) * 21
    if len(direct) != expected:
        raise ValueError(f"Expected {expected} all-visible direct rows, got {len(direct)}")
    if not direct["result_status"].eq("candidate").all():
        raise ValueError("This visualizer is restricted to candidate inputs.")
    return direct, registry, manifest


def _plot_volcano(direct: pd.DataFrame, outcomes: list[str], families: dict[str, str], output_dir: Path) -> Path:
    palette = {
        "surface": "#0F766E",
        "reasoning_organization": "#2563EB",
        "epistemic_stance": "#7C3AED",
        "logical_connective": "#D97706",
        "self_reference": "#DC2626",
        "engagement": "#0891B2",
        "affective_language": "#DB2777",
    }
    direct = direct.copy()
    direct["outcome_family"] = direct["outcome_id"].map(families).fillna("other")
    direct["neglog10_p_holm"] = -np.log10(direct["p_holm"].clip(lower=1e-300))
    ymax = max(2.0, float(direct["neglog10_p_holm"].quantile(0.995)) * 1.15)
    fig, axes = plt.subplots(1, 4, figsize=(18, 6.3), sharex=True, sharey=True)
    for ax, condition in zip(axes, CONDITIONS):
        panel = direct[direct["contrast_id"].eq(f"man_minus_woman_{condition}")]
        for family, points in panel.groupby("outcome_family", sort=True):
            sizes = 30 + 16 * np.log10(points["n_nonzero"].clip(lower=0) + 1)
            ax.scatter(points["rank_biserial"], points["neglog10_p_holm"], s=sizes, color=palette.get(family, "#64748B"), alpha=0.84, edgecolors="white", linewidths=0.65)
        ax.axvline(0, color="#334155", linewidth=0.8)
        ax.axhline(-math.log10(0.05), color="#64748B", linewidth=0.8, linestyle="--")
        ax.set_xlim(-1.02, 1.02)
        ax.set_ylim(0, ymax)
        ax.set_title(CONDITION_LABELS[condition], fontweight="bold")
        ax.grid(color="#E2E8F0", linewidth=0.55)
    handles = [Line2D([0], [0], marker="o", linestyle="", color=color, label=family.replace("_", " ").title(), markersize=8) for family, color in palette.items() if family in set(direct["outcome_family"])]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.015))
    fig.supxlabel("Rank-biserial Man − Woman", y=0.12)
    axes[0].set_ylabel("−log10(p Holm)")
    fig.suptitle("ETHICS overall (candidate): uji langsung marker Man − Woman\nTitik = model × outcome; ukuran = jumlah pasangan dengan perubahan nonnol", fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0.01, 0.18, 0.99, 0.90), w_pad=1.0)
    return _save(fig, output_dir / "01_direct_gender_numeric_volcano.png")


def _plot_consistency(direct: pd.DataFrame, outcomes: list[str], labels: dict[str, str], output_dir: Path) -> Path:
    y_lookup = {outcome: len(outcomes) - 1 - index for index, outcome in enumerate(outcomes)}
    fig, axes = plt.subplots(1, 4, figsize=(18, 10.2), sharex=True, sharey=True)
    for ax, condition in zip(axes, CONDITIONS):
        panel = direct[direct["contrast_id"].eq(f"man_minus_woman_{condition}")]
        for _, point in panel.iterrows():
            size = 30 + 16 * math.log10(float(point["n_nonzero"]) + 1)
            ax.scatter(point["rank_biserial"], y_lookup[point["outcome_id"]], s=size, color=MODEL_COLORS[point["model_id"]], alpha=0.96 if point["p_holm"] < 0.05 else 0.38, edgecolors="white", linewidths=0.6)
        for outcome, value in panel.groupby("outcome_id")["rank_biserial"].median().items():
            ax.scatter(value, y_lookup[outcome], marker="D", s=64, color="#111827", zorder=4)
        ax.axvline(0, color="#334155", linewidth=0.8)
        ax.set_xlim(-1.02, 1.02)
        ax.set_title(CONDITION_LABELS[condition], fontweight="bold")
        ax.grid(axis="x", color="#E2E8F0", linewidth=0.6)
    axes[0].set_yticks([y_lookup[outcome] for outcome in outcomes], [labels[outcome] for outcome in outcomes], fontsize=10)
    handles = [Line2D([0], [0], marker="o", linestyle="", color=MODEL_COLORS[model], label=MODEL_LABELS[model]) for model in MODEL_ORDER]
    handles.append(Line2D([0], [0], marker="D", linestyle="", color="#111827", label="Median enam model"))
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.005))
    fig.supxlabel("Rank-biserial Man − Woman", y=0.075)
    fig.suptitle("ETHICS overall (candidate): konsistensi arah efek marker\nPudar = p Holm ≥ 0,05; ukuran = pasangan dengan perubahan nonnol", fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0.01, 0.14, 0.99, 0.93), w_pad=1.0)
    return _save(fig, output_dir / "02_direct_gender_effect_consistency.png")


def _write_readme(output_dir: Path, analysis_dir: Path, figures: list[Path]) -> None:
    text = f"""# Visualisasi ETHICS gender untuk Bab VI — candidate

Bundle ini bersifat additive dan tidak menimpa visual per-subset maupun bundle
overall sebelumnya.

## Dasar data

- Tabel inferensial: `{analysis_dir / 'overall_paired_numeric_21.csv'}`
- Filter utama: seluruh respons dengan visible text (`all_visible_text`), termasuk *abstain*.
- Kontras: direct Man − Woman untuk E, I1, I2, dan I3.
- Outcome: 3 surface outcomes + 18 marker primer.
- Statistik: paired rank-biserial, p-value dengan koreksi Holm per model × kondisi × outcome.

## Batas interpretasi

"Overall" adalah pooling berbobot item atas empat subset ETHICS, bukan satu
skor accuracy dan bukan estimasi meta-analitik. Konstruksi empat subset berbeda;
angka accuracy tetap per subset. Semua marker dan visual masih `candidate` sampai
audit manusia independen serta adjudikasi selesai.

## Figur

{chr(10).join(f'- `{path.name}`' for path in figures)}
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def build(
    analysis_dir: Path,
    registry: Path,
    output_dir: Path,
    *,
    include_consistency: bool = False,
) -> Path:
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    direct, marker_registry, manifest = _load(analysis_dir.resolve(), registry.resolve())
    outcomes, families, labels = _outcome_metadata(marker_registry, manifest)
    figures = [_plot_volcano(direct, outcomes, families, output_dir)]
    if include_consistency:
        figures.append(_plot_consistency(direct, outcomes, labels, output_dir))
    index = pd.DataFrame({"figure": [path.name for path in figures], "result_status": "candidate", "analysis_scope": "overall_ethics", "comparison": "direct_man_minus_woman"})
    index.to_csv(output_dir / "figure_index.csv", index=False, encoding="utf-8")
    (output_dir / "visual_manifest.json").write_text(json.dumps({"result_status": "candidate", "analysis_scope": "overall_ethics", "source_analysis_dir": str(analysis_dir.resolve()), "source_subsets": manifest["subsets"], "figures": index.to_dict(orient="records")}, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_readme(output_dir, analysis_dir.resolve(), figures)
    return output_dir


def main() -> None:
    args = parse_args()
    output_dir = build(
        args.analysis_dir,
        args.registry,
        args.output_dir,
        include_consistency=args.include_consistency,
    )
    print(f"[DONE] Candidate visual bundle written to: {output_dir}")


if __name__ == "__main__":
    main()
