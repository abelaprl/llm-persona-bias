"""Build the two agreed ETHICS metrics for all four subsets.

Metrics written for every domain and subset:
- accuracy_full = n_correct / total responses
- prop_wrong_committed = n_answer_wrong / committed responses

The script does not rerun model inference or extraction. It uses the validated
summary tables already stored in each domain notebook bundle.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[3]
SUBSETS = ["commonsense", "deontology", "justice", "virtue"]
SUBSET_LABELS = {
    "commonsense": "Commonsense",
    "deontology": "Deontology",
    "justice": "Justice",
    "virtue": "Virtue",
}
MODEL_ORDER = [
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
IDENTITY_ORDER = {
    "gender": ["man", "woman"],
    "religion": ["islam", "christian", "jewish", "atheist"],
    "regional": ["north_american", "east_asian", "south_asian", "sub_saharan_african"],
}
IDENTITY_LABELS = {
    "man": "Laki-laki", "woman": "Perempuan",
    "islam": "Islam", "christian": "Kristen", "jewish": "Yahudi", "atheist": "Ateis",
    "north_american": "NA", "east_asian": "AT", "south_asian": "AS", "sub_saharan_african": "ASS",
}


def domain_paths(domain: str):
    folder = ROOT / "notebooks" / domain / "ethics accuracy analysis"
    tables = folder / "tables"
    if domain == "gender":
        status = tables / "ethics_condition_summary_source_copy.csv"
        accuracy = None
    else:
        status = tables / f"ethics_{domain}_parity_condition_summary.csv"
        accuracy = tables / "ethics_accuracy_source_v3.csv"
    return folder, status, accuracy


def load_domain(domain: str) -> pd.DataFrame:
    folder, status_path, accuracy_path = domain_paths(domain)
    status = pd.read_csv(status_path)
    if accuracy_path is None:
        data = status.copy()
    else:
        accuracy = pd.read_csv(accuracy_path)
        accuracy = accuracy[["model", "persona", "subset", "total_responses", "n_correct"]].copy()
        data = status.drop(columns=["n_correct"], errors="ignore").merge(
            accuracy,
            on=["model", "persona", "subset"],
            how="left",
            validate="one_to_one",
        )
        data["total"] = data["total_responses"]
        data = data.drop(columns=["total_responses"])
    if "n_answer_wrong" not in data.columns:
        if "pct_answer_wrong" not in data.columns:
            raise ValueError(f"{domain}: n_answer_wrong dan pct_answer_wrong tidak tersedia")
        data["n_answer_wrong"] = (pd.to_numeric(data["pct_answer_wrong"], errors="coerce") * pd.to_numeric(data["committed"], errors="coerce")).round()
    required = ["model", "persona", "subset", "total", "committed", "n_correct", "n_answer_wrong"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"{domain}: missing columns {missing}")
    data["total"] = pd.to_numeric(data["total"], errors="coerce")
    data["n_correct"] = pd.to_numeric(data["n_correct"], errors="coerce")
    data["committed"] = pd.to_numeric(data["committed"], errors="coerce")
    data["n_answer_wrong"] = pd.to_numeric(data["n_answer_wrong"], errors="coerce")
    if data["n_correct"].isna().any():
        raise ValueError(f"{domain}: n_correct masih kosong")
    if set(data["subset"]) != set(SUBSETS):
        raise ValueError(f"{domain}: subset tidak lengkap")
    data["accuracy_full"] = data["n_correct"] / data["total"]
    data["prop_wrong_committed"] = data["n_answer_wrong"] / data["committed"]
    data["model_label"] = data["model"].map(MODEL_LABELS).fillna(data["model"])
    data["subset_label"] = data["subset"].map(SUBSET_LABELS)
    return data


def persona_order(domain: str, values: list[str]) -> list[str]:
    order = []
    for identity in IDENTITY_ORDER[domain]:
        explicit = f"{identity}_explicit"
        if explicit in values:
            order.append(explicit)
        order.extend(p for p in [f"{identity}_implicit_1", f"{identity}_implicit_2", f"{identity}_implicit_3"] if p in values)
    if "neutral" in values:
        order.append("neutral")
    return order


def persona_label(persona: str) -> str:
    if persona == "neutral":
        return "N"
    if persona.endswith("_explicit"):
        return f"{IDENTITY_LABELS.get(persona[:-9], persona[:-9])} E"
    for i in (1, 2, 3):
        suffix = f"_implicit_{i}"
        if persona.endswith(suffix):
            return f"{IDENTITY_LABELS.get(persona[:-len(suffix)], persona[:-len(suffix)])} I{i}"
    return persona


def write_outputs(domain: str, data: pd.DataFrame) -> dict[str, str]:
    folder, _, _ = domain_paths(domain)
    table_dir = folder / "tables"
    figure_dir = folder / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "model", "model_label", "persona", "persona_label", "subset", "subset_label",
        "total", "committed", "abstain", "missing", "n_correct", "accuracy_full",
        "n_answer_wrong", "prop_wrong_committed",
    ]
    out = data.copy()
    out["persona_label"] = out["persona"].map(persona_label)
    out = out[columns].sort_values(["model", "persona", "subset"], kind="stable")
    table_path = table_dir / f"ethics_{domain}_two_metrics_all_subsets.csv"
    out.to_csv(table_path, index=False)

    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 100, "savefig.dpi": 220})
    p_order = persona_order(domain, list(out["persona"].unique()))
    p_labels = [persona_label(p) for p in p_order]
    fig, axes = plt.subplots(2, 4, figsize=(max(25, len(p_order) * 1.15), 12), sharey="row")
    metrics = [("accuracy_full", "Akurasi penuh"), ("prop_wrong_committed", "Proporsi wrong")]
    for row, (metric, metric_label) in enumerate(metrics):
        for col, subset in enumerate(SUBSETS):
            ax = axes[row, col]
            part = out[out["subset"].eq(subset)]
            matrix = part.pivot(index="model_label", columns="persona", values=metric).reindex(
                index=[MODEL_LABELS[m] for m in MODEL_ORDER], columns=p_order
            ) * 100
            matrix.index = [x.replace(" ", "\n", 1) for x in matrix.index]
            matrix.columns = p_labels
            sns.heatmap(
                matrix, ax=ax, vmin=0, vmax=100, cmap="YlGnBu", annot=True, fmt=".1f",
                linewidths=.45, linecolor="white", cbar=(col == 3),
                cbar_kws={"label": "%"} if col == 3 else None,
            )
            ax.set_title(f"{metric_label}\n{SUBSET_LABELS[subset]}", fontsize=12, pad=10)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.tick_params(axis="x", rotation=55, labelsize=8)
            ax.tick_params(axis="y", rotation=0, labelsize=8)
    fig.suptitle(
        f"ETHICS {domain.title()}: Dua metrik pada empat subset",
        fontsize=17, fontweight="bold", y=1.02,
    )
    fig.text(
        0.5, -0.015,
        "Akurasi penuh = kecocokan dengan ground truth / seluruh respons; "
        "proporsi wrong = keputusan wrong / respons committed.",
        ha="center", fontsize=10,
    )
    fig.tight_layout()
    stem = f"00_ethics_{domain}_two_metrics_all_subsets_heatmap"
    png_path, svg_path = figure_dir / f"{stem}.png", figure_dir / f"{stem}.svg"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    manifest = {
        "domain": domain,
        "subsets": SUBSETS,
        "metrics": {
            "accuracy_full": "n_correct / total",
            "prop_wrong_committed": "n_answer_wrong / committed",
        },
        "table": str(table_path.relative_to(ROOT)),
        "figure_png": str(png_path.relative_to(ROOT)),
        "figure_svg": str(svg_path.relative_to(ROOT)),
        "rows": int(len(out)),
    }
    manifest_path = table_dir / f"ethics_{domain}_two_metrics_all_subsets_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"table": str(table_path), "png": str(png_path), "svg": str(svg_path), "manifest": str(manifest_path)}


def main():
    results = {}
    for domain in ("gender", "religion", "regional"):
        results[domain] = write_outputs(domain, load_domain(domain))
    print(json.dumps(results, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
