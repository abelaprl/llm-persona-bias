"""Build the thesis-facing GSM8K religion linguistic-marker visual bundle.

The builder consumes a completed source-aligned religion run.  It keeps the
four identities and the three implicit proxies separate, reports model-level
heterogeneity, and treats direct identity comparisons as paired analyses.
The output directory is write-once.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.MAIN_PARSING.LINGUISTIC_MARKERS.religion_config import (  # noqa: E402
    DIRECT_CONDITIONS,
    DIRECT_IDENTITY_PAIRS,
    IDENTITY_CONDITIONS,
    OPENING_CATEGORIES,
    PIPELINE_VERSION,
    RELIGION_PERSONAS,
)


BUNDLE_VERSION = "religion-gsm8k-bab4-visual-v1.3.0"
IDENTITIES = ("islam", "christian", "jewish", "atheist")
CONDITIONS = ("neutral", "explicit", "implicit_1", "implicit_2", "implicit_3")
PAIR_CONDITIONS = ("explicit", "implicit_1", "implicit_2", "implicit_3")
NEUTRAL_CONTRASTS = (
    "explicit_vs_neutral",
    "implicit_1_vs_neutral",
    "implicit_2_vs_neutral",
    "implicit_3_vs_neutral",
)

MODEL_LABELS = {
    "meta-llama_llama-3.1-8b-instruct": "Llama 3.1 8B",
    "meta-llama_llama-3.3-70b-instruct": "Llama 3.3 70B",
    "mistralai_ministral-8b-2512": "Ministral 8B",
    "deepseek_deepseek-r1-distill-qwen-32b": "DeepSeek R1 32B",
    "openai_gpt-oss-20b": "GPT-OSS 20B",
    "qwen_qwen3-next-80b-a3b-thinking": "Qwen3 Next 80B",
}
MODEL_ORDER = tuple(MODEL_LABELS)
MODEL_COLORS = dict(
    zip(MODEL_ORDER, ("#2563EB", "#7C3AED", "#059669", "#DC2626", "#D97706", "#0891B2"))
)
IDENTITY_LABELS = {
    "islam": "Islam",
    "christian": "Christian",
    "jewish": "Jewish",
    "atheist": "Atheist",
}
IDENTITY_COLORS = {
    "islam": "#0F9D8A",
    "christian": "#7C3AED",
    "jewish": "#2563EB",
    "atheist": "#D97706",
}
CONDITION_LABELS = {
    "neutral": "N",
    "explicit": "E",
    "implicit_1": "I1",
    "implicit_2": "I2",
    "implicit_3": "I3",
}
PAIR_LABELS = {
    f"{left}_vs_{right}": f"{IDENTITY_LABELS[left]} − {IDENTITY_LABELS[right]}"
    for left, right in DIRECT_IDENTITY_PAIRS
}
PERSONA_LABELS = {"neutral": "N"}
for _identity, _conditions in IDENTITY_CONDITIONS.items():
    _prefix = {"islam": "IS", "christian": "CH", "jewish": "JW", "atheist": "AT"}[_identity]
    for _condition, _persona in _conditions.items():
        PERSONA_LABELS[_persona] = f"{_prefix}-{CONDITION_LABELS[_condition]}"
# Neutral is a shared baseline, not an atheist-specific condition.  The
# persona loops above all point their neutral entry to the same source label.
PERSONA_LABELS["neutral"] = "N"

RELIGION_MARKERS = (
    "religion_identity_islam",
    "religion_identity_christian",
    "religion_identity_jewish",
    "religion_identity_atheist",
    "religion_generic_reference",
    "religion_deity_invocation",
    "religion_scripture_authority",
    "religion_practice_community",
    "religion_formulaic_expression",
    "religion_identity_irrelevance",
    "religion_blessing_invocation",
    "religion_respect_acknowledgement",
)
RELIGION_MARKER_LABELS = {
    "religion_identity_islam": "Identitas Islam",
    "religion_identity_christian": "Identitas Christian",
    "religion_identity_jewish": "Identitas Jewish",
    "religion_identity_atheist": "Identitas atheist",
    "religion_generic_reference": "Rujukan agama umum",
    "religion_deity_invocation": "Tuhan/deity",
    "religion_scripture_authority": "Kitab/otoritas",
    "religion_practice_community": "Praktik/komunitas",
    "religion_formulaic_expression": "Sapaan/formula agama",
    "religion_identity_irrelevance": "Identitas tidak relevan",
    "religion_blessing_invocation": "Doa/blessing",
    "religion_respect_acknowledgement": "Pengakuan/respect",
}
ALIGNED_MARKER = {
    "islam": "religion_identity_islam",
    "christian": "religion_identity_christian",
    "jewish": "religion_identity_jewish",
    "atheist": "religion_identity_atheist",
}
ACCOMMODATION_MARKERS = (
    "religion_generic_reference",
    "religion_deity_invocation",
    "religion_scripture_authority",
    "religion_practice_community",
    "religion_formulaic_expression",
    "religion_identity_irrelevance",
    "religion_blessing_invocation",
    "religion_respect_acknowledgement",
)
GENERAL_OUTCOMES = (
    "word_count",
    "avg_sentence_length",
    "ttr_300",
    "organization_explicit_step",
    "organization_bullet",
    "organization_section_header",
    "stance_hedge",
    "stance_booster",
    "stance_assumption",
    "stance_self_correction",
    "connective_causal",
    "connective_contrastive",
    "connective_conditional",
    "self_reference_first_person",
    "self_reference_inclusive_we",
    "engagement_reader_directive",
    "math_latex_display",
    "math_inline_arithmetic",
)
OUTCOME_LABELS = {
    "word_count": "Jumlah kata",
    "avg_sentence_length": "Panjang kalimat",
    "ttr_300": "TTR-300",
    "organization_explicit_step": "Langkah eksplisit",
    "organization_bullet": "Bullet",
    "organization_section_header": "Section header",
    "stance_hedge": "Hedge",
    "stance_booster": "Booster",
    "stance_assumption": "Asumsi",
    "stance_self_correction": "Self-correction",
    "connective_causal": "Konektif kausal",
    "connective_contrastive": "Konektif kontras",
    "connective_conditional": "Konektif kondisional",
    "self_reference_first_person": "Orang pertama",
    "self_reference_inclusive_we": "Inclusive we",
    "engagement_reader_directive": "Direktif pembaca",
    "math_latex_display": "LaTeX display",
    "math_inline_arithmetic": "Aritmetika inline",
}

OPENING_COLORS = {
    category: color
    for category, color in zip(
        OPENING_CATEGORIES,
        (
            "#4E79A7", "#E15759", "#59A14F", "#B07AA1", "#F28E2B",
            "#76B7B2", "#EDC948", "#9C755F", "#FF9DA7", "#BAB0AC", "#1B9E77",
        ),
    )
}

# Closed term inventory for the supplementary term display.  Context-bounded
# Christian patterns avoid the GSM8K character named Christian.
TERM_PATTERNS = (
    ("Christian", re.compile(r"\b(?:as|being)\s+(?:a\s+)?christian\b|\bchristian(?:s|ity)\b|\bchristian\s+(?:faith|identity|beliefs?|perspective)\b", re.I)),
    ("Allah", re.compile(r"\ballah\b", re.I)),
    ("Muslim", re.compile(r"\bmuslims?\b", re.I)),
    ("Islam", re.compile(r"\bislam(?:ic)?\b", re.I)),
    ("Qur'an", re.compile(r"\b(?:qur['’]?an|koran)\b", re.I)),
    ("Hadith", re.compile(r"\bhadith\b", re.I)),
    ("Ramadan", re.compile(r"\bramadan\b", re.I)),
    ("Eid", re.compile(r"\beid\b", re.I)),
    ("Jewish", re.compile(r"\b(?:jews?|jewish|judaism|judaic)\b", re.I)),
    ("Torah", re.compile(r"\btorah\b", re.I)),
    ("Talmud", re.compile(r"\btalmud\b", re.I)),
    ("Shabbat", re.compile(r"\b(?:shabbat|sabbath)\b", re.I)),
    ("Yom Kippur", re.compile(r"\byom\s+kippur\b", re.I)),
    ("Atheist", re.compile(r"\b(?:atheists?|atheism|atheistic)\b", re.I)),
    ("Agnostic", re.compile(r"\b(?:agnostics?|agnosticism)\b", re.I)),
    ("Nonreligious", re.compile(r"\b(?:non-?religious|secularism)\b", re.I)),
    ("God", re.compile(r"\bgod(?:s)?\b", re.I)),
    ("Lord", re.compile(r"\bthe\s+lord\b", re.I)),
    ("Jesus/Christ", re.compile(r"\b(?:jesus(?:\s+christ)?|christ)\b", re.I)),
    ("Bible", re.compile(r"\b(?:bible|biblical|gospels?)\b", re.I)),
    ("Scripture", re.compile(r"\bscriptures?\b", re.I)),
    ("Faith", re.compile(r"\bfaith(?:ful)?\b", re.I)),
    ("Religion", re.compile(r"\breligions?\b|\breligious(?:ly)?\b", re.I)),
    ("Spiritual", re.compile(r"\bspiritual(?:ity)?\b", re.I)),
    ("Prayer", re.compile(r"\bpray(?:er|ers|ing)?\b", re.I)),
    ("Worship", re.compile(r"\bworship(?:ping|ing)?\b", re.I)),
    ("Halal/Haram", re.compile(r"\b(?:halal|haram)\b", re.I)),
    ("Kosher", re.compile(r"\bkosher\b", re.I)),
    ("Zakat/Sharia", re.compile(r"\b(?:zakat|sharia)\b", re.I)),
    ("Mosque", re.compile(r"\b(?:mosque|masjid)\b", re.I)),
    ("Synagogue", re.compile(r"\bsynagogue\b", re.I)),
    ("Bless", re.compile(r"\bbless(?:es|ed|ing)?\b", re.I)),
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bool_mask(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.fillna(False).astype(bool)
    return values.fillna("").astype(str).str.strip().str.casefold().eq("true")


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 14.5,
            "axes.titlesize": 18,
            "axes.labelsize": 17.5,
            "axes.titlepad": 10,
            "axes.labelpad": 9,
            "xtick.labelsize": 15.5,
            "ytick.labelsize": 15.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.titlesize": 20,
            "legend.frameon": False,
            "legend.fontsize": 18,
            "legend.markerscale": 1.25,
            "legend.handlelength": 1.8,
            "legend.handletextpad": 0.6,
            "legend.columnspacing": 1.4,
        }
    )


def _save(fig: plt.Figure, image_dir: Path, filename: str) -> Path:
    path = image_dir / filename
    fig.savefig(path, dpi=235, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _read_run(run_dir: Path) -> dict[str, pd.DataFrame]:
    names = (
        "input_inventory",
        "quality_control",
        "condition_descriptives",
        "paired_numeric",
        "correctness_sensitivity_paired_numeric",
        "paired_binary",
        "paired_opening_multinomial",
        "direct_identity_numeric",
        "direct_identity_correctness_sensitivity",
        "direct_identity_binary",
        "direct_identity_availability",
        "direct_identity_opening_multinomial",
    )
    return {name: pd.read_csv(run_dir / f"{name}.csv") for name in names}


def _load_response_frame(run_dir: Path) -> pd.DataFrame:
    header = pd.read_csv(run_dir / "response_records.csv", nrows=0).columns
    marker_columns = [
        f"{marker}__present" for marker in RELIGION_MARKERS
        if f"{marker}__present" in header
    ]
    usecols = [
        "model_id", "identity", "condition", "persona", "question_id",
        "visible_text", "text_available", "word_count", "opening_category",
        *marker_columns,
    ]
    frame = pd.read_csv(run_dir / "response_records.csv", usecols=usecols, low_memory=False)
    frame["text_available"] = _bool_mask(frame["text_available"])
    return frame


def _coverage_table(inventory: pd.DataFrame, quality: pd.DataFrame) -> pd.DataFrame:
    base = inventory[[
        "model_id", "persona", "n_expected_records", "n_records_available", "n_missing_records"
    ]].rename(
        columns={
            "n_expected_records": "target",
            "n_records_available": "n_source_records",
            "n_missing_records": "n_source_gap",
        }
    )
    qc = quality[[
        "model_id", "persona", "n_text_available", "n_missing_text",
        "n_technical_only", "n_truncated",
    ]]
    result = base.merge(qc, on=["model_id", "persona"], how="left", validate="one_to_one")
    result["text_coverage_rate"] = result["n_text_available"] / result["target"]
    return result


def _plot_coverage(coverage: pd.DataFrame, image_dir: Path) -> Path:
    table = coverage.pivot(index="model_id", columns="persona", values="n_text_available")
    table = table.reindex(index=MODEL_ORDER, columns=RELIGION_PERSONAS)
    target = coverage.pivot(index="model_id", columns="persona", values="target").reindex_like(table)
    rates = table / target
    fig, ax = plt.subplots(figsize=(21, 8.2))
    image = ax.imshow(rates.to_numpy(float), cmap="YlGnBu", vmin=0.80, vmax=1.0, aspect="auto")
    for row in range(len(table.index)):
        for column in range(len(table.columns)):
            value = table.iloc[row, column]
            expected = target.iloc[row, column]
            ax.text(
                column, row, f"{int(value):,}\n/{int(expected):,}",
                ha="center", va="center", fontsize=12.5,
                color="white" if rates.iloc[row, column] >= 0.91 else "#0F172A",
            )
    ax.set_xticks(
        range(len(table.columns)),
        [PERSONA_LABELS[p] for p in table.columns],
        rotation=45,
        ha="right",
        fontsize=14.5,
    )
    ax.set_yticks(range(len(table.index)), [MODEL_LABELS[m] for m in table.index], fontsize=15)
    ax.set_title("Cakupan teks tersedia per model × 17 kondisi persona", fontweight="bold")
    ax.set_xlabel("N = Neutral; IS/CH/JW/AT = identitas; E/I1/I2/I3 = kondisi")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.018, pad=0.012)
    colorbar.set_label("Proporsi dari target")
    fig.tight_layout()
    return _save(fig, image_dir, "01_coverage_model_persona.png")


def _plot_verbosity(descriptives: pd.DataFrame, image_dir: Path) -> Path:
    data = descriptives[descriptives["outcome_id"].eq("word_count")].copy()
    x_lookup = {condition: index for index, condition in enumerate(CONDITIONS)}
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True)
    for ax, identity in zip(axes.flat, IDENTITIES):
        panel = data[data["identity"].eq(identity)]
        for model in MODEL_ORDER:
            points = panel[panel["model_id"].eq(model)]
            for _, point in points.iterrows():
                x = x_lookup[point["condition"]]
                ax.errorbar(
                    x, point["median"],
                    yerr=np.array([[point["median"] - point["q1"]], [point["q3"] - point["median"]]]),
                    fmt="o", ms=7.5, capsize=3.0, color=MODEL_COLORS[model], alpha=0.88,
                )
        medians = panel.groupby("condition")["median"].median()
        for condition, value in medians.items():
            ax.scatter(x_lookup[condition], value, marker="D", s=82, color="#111827", zorder=5)
        ax.set_title(IDENTITY_LABELS[identity], fontweight="bold")
        ax.grid(axis="y", color="#E2E8F0", linewidth=0.6)
    for ax in axes[-1]:
        ax.set_xticks(range(len(CONDITIONS)), [CONDITION_LABELS[c] for c in CONDITIONS])
    fig.supxlabel("Kondisi persona", y=0.115)
    fig.supylabel("Median jumlah kata (garis = IQR)", x=0.012)
    handles = [Line2D([0], [0], marker="o", linestyle="", color=MODEL_COLORS[m], label=MODEL_LABELS[m]) for m in MODEL_ORDER]
    handles.append(Line2D([0], [0], marker="D", linestyle="", color="#111827", label="Median 6 model"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.012), ncol=4)
    fig.suptitle("Verbosity respons untuk setiap identitas dan kondisi", y=0.995, fontweight="bold")
    fig.tight_layout(rect=(0.055, 0.18, 0.99, 0.95), h_pad=2.0, w_pad=1.5)
    return _save(fig, image_dir, "02_verbosity_by_identity_condition.png")


def _marker_prevalence(frame: pd.DataFrame) -> pd.DataFrame:
    present_columns = [f"{marker}__present" for marker in RELIGION_MARKERS]
    for column in present_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    long = frame.melt(
        id_vars=["model_id", "identity", "condition", "persona"],
        value_vars=present_columns,
        var_name="marker_id",
        value_name="present",
    )
    long["marker_id"] = long["marker_id"].str.replace("__present", "", regex=False)
    result = (
        long.groupby(["model_id", "identity", "condition", "persona", "marker_id"], dropna=False)
        .agg(n_available=("present", "count"), n_present=("present", "sum"))
        .reset_index()
    )
    result["prevalence_pp"] = 100 * result["n_present"] / result["n_available"]
    return result


def _plot_aligned_mentions(prevalence: pd.DataFrame, image_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(16, 11.5), sharex=True)
    x_lookup = {condition: index for index, condition in enumerate(PAIR_CONDITIONS)}
    for ax, identity in zip(axes.flat, IDENTITIES):
        marker = ALIGNED_MARKER[identity]
        panel = prevalence[
            prevalence["identity"].eq(identity)
            & prevalence["marker_id"].eq(marker)
            & prevalence["condition"].isin(PAIR_CONDITIONS)
        ]
        for model in MODEL_ORDER:
            group = panel[panel["model_id"].eq(model)]
            ax.scatter(
                [x_lookup[c] for c in group["condition"]], group["prevalence_pp"],
                s=70, color=MODEL_COLORS[model], alpha=0.90,
            )
        medians = panel.groupby("condition")["prevalence_pp"].median()
        for condition, value in medians.items():
            ax.scatter(x_lookup[condition], value, marker="D", s=90, color="#111827", zorder=5)
        ax.set_title(f"{IDENTITY_LABELS[identity]}: {RELIGION_MARKER_LABELS[marker]}", fontweight="bold")
        ax.grid(axis="y", color="#E2E8F0", linewidth=0.6)
    for ax in axes[-1]:
        ax.set_xticks(range(len(PAIR_CONDITIONS)), [CONDITION_LABELS[c] for c in PAIR_CONDITIONS])
    fig.supxlabel("Kondisi persona", y=0.115)
    fig.supylabel("Respons dengan penyebutan (%)", x=0.012)
    handles = [Line2D([0], [0], marker="o", linestyle="", color=MODEL_COLORS[m], label=MODEL_LABELS[m]) for m in MODEL_ORDER]
    handles.append(Line2D([0], [0], marker="D", linestyle="", color="#111827", label="Median 6 model"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.012), ncol=4)
    fig.suptitle("Penyebutan identitas yang selaras dengan persona", y=0.995, fontweight="bold")
    fig.tight_layout(rect=(0.055, 0.18, 0.99, 0.95), h_pad=2.0, w_pad=1.5)
    return _save(fig, image_dir, "03_aligned_identity_mentions.png")


def _plot_accommodation_explicit(prevalence: pd.DataFrame, image_dir: Path) -> Path:
    y_lookup = {marker: len(ACCOMMODATION_MARKERS) - 1 - index for index, marker in enumerate(ACCOMMODATION_MARKERS)}
    fig, axes = plt.subplots(2, 2, figsize=(17, 12.5), sharex=False, sharey=True)
    for ax, identity in zip(axes.flat, IDENTITIES):
        panel = prevalence[
            prevalence["identity"].eq(identity)
            & prevalence["condition"].eq("explicit")
            & prevalence["marker_id"].isin(ACCOMMODATION_MARKERS)
        ]
        for _, point in panel.iterrows():
            ax.scatter(
                point["prevalence_pp"], y_lookup[point["marker_id"]],
                s=70, color=MODEL_COLORS[point["model_id"]], alpha=0.90,
            )
        medians = panel.groupby("marker_id")["prevalence_pp"].median()
        for marker, value in medians.items():
            ax.scatter(value, y_lookup[marker], marker="D", s=90, color="#111827", zorder=5)
        ax.set_title(IDENTITY_LABELS[identity], fontweight="bold")
        ax.grid(axis="x", color="#E2E8F0", linewidth=0.6)
    axes[0, 0].set_yticks(
        [y_lookup[m] for m in ACCOMMODATION_MARKERS],
        [RELIGION_MARKER_LABELS[m] for m in ACCOMMODATION_MARKERS],
    )
    handles = [Line2D([0], [0], marker="o", linestyle="", color=MODEL_COLORS[m], label=MODEL_LABELS[m]) for m in MODEL_ORDER]
    handles.append(Line2D([0], [0], marker="D", linestyle="", color="#111827", label="Median 6 model"))
    fig.supxlabel("Prevalensi respons (%)", y=0.115)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.012), ncol=4)
    fig.suptitle("Akomodasi bahasa agama pada persona Explicit", y=0.995, fontweight="bold")
    fig.tight_layout(rect=(0.01, 0.18, 0.99, 0.95), h_pad=2.0, w_pad=1.5)
    return _save(fig, image_dir, "04_religious_accommodation_explicit.png")


def _term_table(frame: pd.DataFrame) -> pd.DataFrame:
    available = frame[frame["text_available"] & frame["visible_text"].notna()].copy()
    rows: list[dict[str, object]] = []
    for (model, persona, identity, condition), group in available.groupby(
        ["model_id", "persona", "identity", "condition"], dropna=False, sort=True
    ):
        counts: Counter[str] = Counter()
        response_counts: Counter[str] = Counter()
        for text in group["visible_text"].astype(str):
            seen: set[str] = set()
            for label, pattern in TERM_PATTERNS:
                matches = list(pattern.finditer(text))
                if matches:
                    counts[label] += len(matches)
                    seen.add(label)
            response_counts.update(seen)
        total_words = pd.to_numeric(group["word_count"], errors="coerce").sum()
        for label, _ in TERM_PATTERNS:
            rows.append(
                {
                    "model_id": model,
                    "persona": persona,
                    "identity": identity,
                    "condition": condition,
                    "term": label,
                    "n_available": len(group),
                    "responses_with_term": response_counts[label],
                    "response_prevalence_pp": 100 * response_counts[label] / len(group),
                    "occurrences": counts[label],
                    "occurrences_per_10k_words": 10_000 * counts[label] / total_words if total_words else math.nan,
                }
            )
    return pd.DataFrame(rows)


def _plot_term_lollipop(term_table: pd.DataFrame, image_dir: Path) -> Path:
    data = term_table[term_table["condition"].eq("explicit")].copy()
    aggregate = (
        data.groupby(["identity", "term"], as_index=False)
        .agg(weight=("occurrences_per_10k_words", "mean"), prevalence=("response_prevalence_pp", "mean"))
    )
    fig, axes = plt.subplots(2, 2, figsize=(17, 14))
    for ax, identity in zip(axes.flat, IDENTITIES):
        panel = aggregate[
            aggregate["identity"].eq(identity) & aggregate["weight"].gt(0)
        ].nlargest(12, "weight").sort_values("weight")
        ax.hlines(panel["term"], 0, panel["weight"], color="#CBD5E1", linewidth=2.2)
        ax.scatter(panel["weight"], panel["term"], s=85, color=IDENTITY_COLORS[identity])
        for _, point in panel.iterrows():
            ax.text(point["weight"], point["term"], f"  {point['weight']:.2f}", va="center", fontsize=14)
        ax.set_title(IDENTITY_LABELS[identity], fontweight="bold")
        ax.grid(axis="x", color="#E2E8F0", linewidth=0.6)
    fig.supxlabel("Rerata setara antar-model per 10.000 kata", y=0.025)
    fig.suptitle("Kata agama yang paling sering muncul pada persona Explicit", y=0.995, fontweight="bold")
    fig.tight_layout(rect=(0.02, 0.065, 0.99, 0.95), h_pad=2.0, w_pad=1.8)
    return _save(fig, image_dir, "05_religious_terms_ranked_explicit.png")


def _draw_cloud(ax: plt.Axes, weights: dict[str, float], title: str, seed: int) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(title, fontweight="bold")
    positive = [(term, value) for term, value in weights.items() if value > 0]
    if not positive:
        ax.text(0.5, 0.5, "Tidak ada istilah", ha="center", va="center", color="#64748B")
        return
    positive.sort(key=lambda item: item[1], reverse=True)
    positive = positive[:20]
    maximum = max(value for _, value in positive)
    positions = [
        (0.50, 0.52), (0.26, 0.72), (0.72, 0.74), (0.24, 0.36), (0.73, 0.34),
        (0.50, 0.82), (0.50, 0.22), (0.14, 0.54), (0.86, 0.55), (0.34, 0.57),
        (0.67, 0.57), (0.36, 0.88), (0.67, 0.88), (0.36, 0.15), (0.66, 0.15),
        (0.10, 0.83), (0.88, 0.84), (0.10, 0.20), (0.89, 0.20), (0.50, 0.67),
    ]
    rng = np.random.default_rng(seed)
    palette = ("#0F766E", "#2563EB", "#7C3AED", "#D97706", "#BE123C")
    for index, ((term, value), (x, y)) in enumerate(zip(positive, positions)):
        size = 13 + 30 * math.sqrt(value / maximum)
        ax.text(
            x, y, term, ha="center", va="center", fontsize=size,
            color=palette[index % len(palette)], rotation=float(rng.choice((0, 0, 0, 0, 10, -10))),
            alpha=0.94,
        )


def _plot_term_cloud(term_table: pd.DataFrame, image_dir: Path) -> Path:
    explicit = term_table[term_table["condition"].eq("explicit")]
    aggregate = explicit.groupby(["identity", "term"])["occurrences_per_10k_words"].mean().reset_index()
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    for index, (ax, identity) in enumerate(zip(axes.flat, IDENTITIES)):
        panel = aggregate[aggregate["identity"].eq(identity)]
        _draw_cloud(ax, dict(zip(panel["term"], panel["occurrences_per_10k_words"])), IDENTITY_LABELS[identity], 1822 + index)
    fig.suptitle("Istilah agama pada respons persona Explicit", y=0.995, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, image_dir, "06_religious_terms_cloud_explicit.png")


def _model_legend() -> list[Line2D]:
    return [
        Line2D([0], [0], marker="o", linestyle="", color=MODEL_COLORS[model], label=MODEL_LABELS[model])
        for model in MODEL_ORDER
    ] + [Line2D([0], [0], marker="D", linestyle="", color="#111827", label="Median 6 model")]


def _plot_within_religion_forest(binary: pd.DataFrame, image_dir: Path, identity: str) -> Path:
    data = binary[
        binary["identity"].eq(identity)
        & binary["outcome_id"].isin(RELIGION_MARKERS)
        & binary["contrast_id"].isin(NEUTRAL_CONTRASTS)
    ].copy()
    data["effect_pp"] = 100 * data["prevalence_delta"]
    data["low_pp"] = 100 * data["ci_low"]
    data["high_pp"] = 100 * data["ci_high"]
    y_lookup = {marker: len(RELIGION_MARKERS) - 1 - index for index, marker in enumerate(RELIGION_MARKERS)}
    fig, axes = plt.subplots(1, 4, figsize=(23, 12.5), sharex=True, sharey=True)
    low = float(np.nanmin(data["low_pp"].to_numpy(float))) if len(data) else 0.0
    high = float(np.nanmax(data["high_pp"].to_numpy(float))) if len(data) else 0.0
    span = max(high - low, 0.1)
    common_xlim = (min(0.0, low) - 0.05 * span, max(0.0, high) + 0.05 * span)
    for ax, contrast in zip(axes, NEUTRAL_CONTRASTS):
        panel = data[data["contrast_id"].eq(contrast)]
        for _, point in panel.iterrows():
            effect = point["effect_pp"]
            ax.errorbar(
                effect, y_lookup[point["outcome_id"]],
                xerr=np.array([[max(effect - point["low_pp"], 0)], [max(point["high_pp"] - effect, 0)]]),
                fmt="o", ms=7.5, capsize=3.0, color=MODEL_COLORS[point["model_id"]],
                alpha=0.95 if point["p_holm"] < 0.05 else 0.55,
            )
        medians = panel.groupby("outcome_id")["effect_pp"].median()
        for marker, value in medians.items():
            ax.scatter(value, y_lookup[marker], marker="D", s=75, color="#111827", zorder=5)
        ax.axvline(0, color="#334155", linewidth=0.8)
        ax.set_xlim(*common_xlim)
        condition = contrast.replace("_vs_neutral", "")
        ax.set_title(f"{CONDITION_LABELS[condition]} − N", fontweight="bold")
        ax.grid(axis="x", color="#E2E8F0", linewidth=0.6)
    axes[0].set_yticks(
        [y_lookup[m] for m in RELIGION_MARKERS],
        [RELIGION_MARKER_LABELS[m] for m in RELIGION_MARKERS],
        fontsize=15,
    )
    fig.supxlabel("Δ prevalensi (poin persentase)", y=0.105)
    fig.legend(handles=_model_legend(), loc="lower center", bbox_to_anchor=(0.5, 0.012), ncol=4)
    fig.suptitle(f"Efek persona {IDENTITY_LABELS[identity]} terhadap Neutral: marker agama", y=0.995, fontweight="bold")
    fig.tight_layout(rect=(0.01, 0.18, 0.99, 0.95), w_pad=1.2)
    return _save(fig, image_dir, f"07_{identity}_religion_marker_forest_vs_neutral.png")


def _plot_general_effects(numeric: pd.DataFrame, image_dir: Path, identity: str) -> Path:
    data = numeric[
        numeric["identity"].eq(identity)
        & numeric["outcome_id"].isin(GENERAL_OUTCOMES)
        & numeric["contrast_id"].isin(NEUTRAL_CONTRASTS)
    ].copy()
    y_lookup = {outcome: len(GENERAL_OUTCOMES) - 1 - index for index, outcome in enumerate(GENERAL_OUTCOMES)}
    fig, axes = plt.subplots(1, 4, figsize=(23, 13), sharex=True, sharey=True)
    for ax, contrast in zip(axes, NEUTRAL_CONTRASTS):
        panel = data[data["contrast_id"].eq(contrast)]
        for _, point in panel.iterrows():
            ax.scatter(
                point["rank_biserial"], y_lookup[point["outcome_id"]],
                s=42 + 22 * math.log10(float(point["n_nonzero"]) + 1),
                color=MODEL_COLORS[point["model_id"]],
                alpha=0.95 if point["p_holm"] < 0.05 else 0.50,
                edgecolors="white", linewidths=0.6,
            )
        medians = panel.groupby("outcome_id")["rank_biserial"].median()
        for outcome, value in medians.items():
            ax.scatter(value, y_lookup[outcome], marker="D", s=75, color="#111827", zorder=5)
        ax.axvline(0, color="#334155", linewidth=0.8)
        ax.axvline(-0.3, color="#CBD5E1", linestyle="--", linewidth=0.6)
        ax.axvline(0.3, color="#CBD5E1", linestyle="--", linewidth=0.6)
        condition = contrast.replace("_vs_neutral", "")
        ax.set_title(f"{CONDITION_LABELS[condition]} − N", fontweight="bold")
        ax.set_xlim(-1.03, 1.03)
        ax.grid(axis="x", color="#E2E8F0", linewidth=0.6)
    axes[0].set_yticks(
        [y_lookup[o] for o in GENERAL_OUTCOMES],
        [OUTCOME_LABELS[o] for o in GENERAL_OUTCOMES],
        fontsize=15,
    )
    fig.supxlabel("Rank-biserial", y=0.105)
    fig.legend(handles=_model_legend(), loc="lower center", bbox_to_anchor=(0.5, 0.012), ncol=4)
    fig.suptitle(f"Efek persona {IDENTITY_LABELS[identity]} terhadap Neutral: marker linguistik umum", y=0.995, fontweight="bold")
    fig.tight_layout(rect=(0.01, 0.18, 0.99, 0.95), w_pad=1.2)
    return _save(fig, image_dir, f"08_{identity}_general_effects_vs_neutral.png")


def _plot_direct_religion_forest(binary: pd.DataFrame, image_dir: Path, condition: str) -> Path:
    data = binary[
        binary["condition"].eq(condition)
        & binary["outcome_id"].isin(RELIGION_MARKERS)
    ].copy()
    data["effect_pp"] = 100 * data["prevalence_delta"]
    data["low_pp"] = 100 * data["ci_low"]
    data["high_pp"] = 100 * data["ci_high"]
    y_lookup = {marker: len(RELIGION_MARKERS) - 1 - index for index, marker in enumerate(RELIGION_MARKERS)}
    pairs = [f"{left}_vs_{right}" for left, right in DIRECT_IDENTITY_PAIRS]
    fig, axes = plt.subplots(3, 2, figsize=(18, 22), sharex=True, sharey=True)
    low = float(np.nanmin(data["low_pp"].to_numpy(float))) if len(data) else 0.0
    high = float(np.nanmax(data["high_pp"].to_numpy(float))) if len(data) else 0.0
    span = max(high - low, 0.1)
    common_xlim = (min(0.0, low) - 0.05 * span, max(0.0, high) + 0.05 * span)
    for ax, pair in zip(axes.flat, pairs):
        panel = data[data["identity_contrast"].eq(pair)]
        for _, point in panel.iterrows():
            effect = point["effect_pp"]
            ax.errorbar(
                effect, y_lookup[point["outcome_id"]],
                xerr=np.array([[max(effect - point["low_pp"], 0)], [max(point["high_pp"] - effect, 0)]]),
                fmt="o", ms=7.5, capsize=3.0, color=MODEL_COLORS[point["model_id"]],
                alpha=0.95 if point["p_holm"] < 0.05 else 0.55,
            )
        medians = panel.groupby("outcome_id")["effect_pp"].median()
        for marker, value in medians.items():
            ax.scatter(value, y_lookup[marker], marker="D", s=75, color="#111827", zorder=5)
        ax.axvline(0, color="#334155", linewidth=0.8)
        ax.set_xlim(*common_xlim)
        ax.set_title(PAIR_LABELS[pair], fontweight="bold")
        ax.grid(axis="x", color="#E2E8F0", linewidth=0.6)
    for ax in axes[:, 0]:
        ax.set_yticks(
            [y_lookup[m] for m in RELIGION_MARKERS],
            [RELIGION_MARKER_LABELS[m] for m in RELIGION_MARKERS],
            fontsize=15,
        )
    fig.supxlabel("Δ prevalensi (poin persentase)", y=0.055)
    fig.legend(handles=_model_legend(), loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=4)
    fig.suptitle(f"Perbandingan langsung antaridentitas pada kondisi {CONDITION_LABELS[condition]}", y=0.995, fontweight="bold")
    fig.tight_layout(rect=(0.01, 0.10, 0.99, 0.96), h_pad=2.0, w_pad=1.6)
    return _save(fig, image_dir, f"09_direct_identity_religion_forest_{condition}.png")


def _sensitivity_tables(primary: pd.DataFrame, sensitivity: pd.DataFrame, direct: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["model_id", "benchmark", "subset", "outcome_id", "outcome_family", "analysis_role", "contrast_id"]
    group_key = "condition" if direct else "identity"
    if group_key not in keys:
        keys.append(group_key)
    if direct:
        keys.append("identity_contrast")
    left = primary[keys + ["n_pairs", "n_nonzero", "rank_biserial", "p_holm"]].rename(
        columns={"n_pairs": "n_pairs_all", "n_nonzero": "n_nonzero_all", "rank_biserial": "effect_all", "p_holm": "p_holm_all"}
    )
    right = sensitivity[keys + ["n_pairs", "n_nonzero", "rank_biserial", "p_holm"]].rename(
        columns={"n_pairs": "n_pairs_both_correct", "n_nonzero": "n_nonzero_both_correct", "rank_biserial": "effect_both_correct", "p_holm": "p_holm_both_correct"}
    )
    comparison = left.merge(right, on=keys, how="inner", validate="one_to_one")
    comparison["pair_retention"] = comparison["n_pairs_both_correct"] / comparison["n_pairs_all"]
    comparison["supported_both"] = comparison["n_nonzero_all"].ge(20) & comparison["n_nonzero_both_correct"].ge(20)
    comparison["same_direction"] = np.sign(comparison["effect_all"]).eq(np.sign(comparison["effect_both_correct"]))
    summary_keys = [group_key]
    if not direct:
        comparison["condition"] = comparison["contrast_id"].str.replace("_vs_neutral", "", regex=False).str.replace("_vs_explicit", "", regex=False)
        summary_keys = ["identity", "condition"]
    summaries = []
    primary_rows = comparison[comparison["analysis_role"].eq("primary")]
    for values, group in primary_rows.groupby(summary_keys, sort=False):
        if not isinstance(values, tuple):
            values = (values,)
        supported = group[group["supported_both"]]
        row = dict(zip(summary_keys, values))
        row.update(
            {
                "n_primary_tests": len(group),
                "median_pair_retention": group["pair_retention"].median(),
                "n_supported_both": len(supported),
                "same_direction_rate_supported": supported["same_direction"].mean() if len(supported) else math.nan,
                "median_abs_effect_change": (group["effect_all"] - group["effect_both_correct"]).abs().median(),
            }
        )
        summaries.append(row)
    return comparison, pd.DataFrame(summaries)


def _plot_sensitivity(within: pd.DataFrame, direct: pd.DataFrame, image_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    for ax, data, title in (
        (axes[0], within, "Persona vs pembanding"),
        (axes[1], direct, "Direct antaridentitas"),
    ):
        selected = data[data["analysis_role"].eq("primary") & data["supported_both"]]
        ax.scatter(
            selected["effect_all"],
            selected["effect_both_correct"],
            s=48,
            alpha=0.50,
            color="#2563EB",
        )
        ax.axline((0, 0), slope=1, color="#111827", linewidth=0.8)
        ax.axhline(0, color="#94A3B8", linewidth=0.6)
        ax.axvline(0, color="#94A3B8", linewidth=0.6)
        ax.set_xlim(-1.02, 1.02)
        ax.set_ylim(-1.02, 1.02)
        ax.set_title(title, fontweight="bold")
        ax.grid(color="#E2E8F0", linewidth=0.5)
    fig.supxlabel("Rank-biserial seluruh pasangan", y=0.035)
    fig.supylabel("Rank-biserial kedua respons benar", x=0.01)
    fig.suptitle("Sensitivitas effect size terhadap correctness", y=0.995, fontweight="bold")
    fig.tight_layout(rect=(0.055, 0.09, 0.99, 0.93), w_pad=1.8)
    return _save(fig, image_dir, "10_correctness_sensitivity.png")


def _opening_composition(frame: pd.DataFrame) -> pd.DataFrame:
    available = frame[frame["text_available"] & frame["opening_category"].notna()].copy()
    # Neutral is stored once in response_records.  Replicate that shared
    # baseline for each identity so every small multiple displays N honestly.
    neutral = available[available["condition"].eq("neutral")].copy()
    non_neutral = available[~available["condition"].eq("neutral")].copy()
    if len(neutral):
        neutral = pd.concat(
            [neutral.assign(identity=identity) for identity in IDENTITIES],
            ignore_index=True,
        )
    available = pd.concat([non_neutral, neutral], ignore_index=True)
    counts = (
        available.groupby(["model_id", "identity", "condition", "opening_category"], dropna=False)
        .size().rename("n").reset_index()
    )
    counts["total"] = counts.groupby(["model_id", "identity", "condition"])["n"].transform("sum")
    counts["proportion"] = counts["n"] / counts["total"]
    return counts


def _plot_opening_composition(composition: pd.DataFrame, image_dir: Path) -> Path:
    aggregate = (
        composition.groupby(["identity", "condition", "opening_category"], as_index=False)["proportion"].mean()
    )
    present_categories = [
        category for category in OPENING_CATEGORIES
        if composition["opening_category"].eq(category).any()
    ]
    fig, axes = plt.subplots(2, 2, figsize=(17, 13.5), sharex=True)
    for ax, identity in zip(axes.flat, IDENTITIES):
        panel = aggregate[aggregate["identity"].eq(identity)]
        bottoms = np.zeros(len(CONDITIONS))
        for category in present_categories:
            values = []
            for condition in CONDITIONS:
                selected = panel[panel["condition"].eq(condition) & panel["opening_category"].eq(category)]
                values.append(float(selected["proportion"].iloc[0]) if len(selected) else 0.0)
            ax.bar(range(len(CONDITIONS)), values, bottom=bottoms, color=OPENING_COLORS[category], width=0.72, label=category)
            bottoms += np.asarray(values)
        ax.set_title(IDENTITY_LABELS[identity], fontweight="bold")
        ax.set_ylim(0, 1)
        ax.set_yticks(np.linspace(0, 1, 6), [f"{int(value * 100)}%" for value in np.linspace(0, 1, 6)])
        ax.set_xticks(range(len(CONDITIONS)), [CONDITION_LABELS[c] for c in CONDITIONS])
    fig.supxlabel("Kondisi", y=0.16)
    fig.supylabel("Proporsi respons", x=0.012)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=4,
        fontsize=14,
    )
    fig.suptitle("Komposisi opening respons (rerata setara antar-model)", y=0.995, fontweight="bold")
    fig.tight_layout(rect=(0.055, 0.22, 0.99, 0.95), h_pad=2.0, w_pad=1.5)
    return _save(fig, image_dir, "11_opening_composition_100pct.png")


def _plot_direct_opening(opening: pd.DataFrame, image_dir: Path) -> Path:
    pairs = [f"{left}_vs_{right}" for left, right in DIRECT_IDENTITY_PAIRS]
    y_lookup = {pair: len(pairs) - 1 - index for index, pair in enumerate(pairs)}
    fig, axes = plt.subplots(1, 4, figsize=(20, 9), sharex=True, sharey=True)
    for ax, condition in zip(axes, PAIR_CONDITIONS):
        panel = opening[opening["condition"].eq(condition)]
        for _, point in panel.iterrows():
            ax.scatter(
                100 * point["marginal_total_variation"], y_lookup[point["identity_contrast"]],
                s=75, color=MODEL_COLORS[point["model_id"]],
                alpha=0.95 if point["p_holm"] < 0.05 else 0.55,
            )
        medians = panel.groupby("identity_contrast")["marginal_total_variation"].median()
        for pair, value in medians.items():
            ax.scatter(100 * value, y_lookup[pair], marker="D", s=82, color="#111827", zorder=5)
        ax.set_title(CONDITION_LABELS[condition], fontweight="bold")
        ax.grid(axis="x", color="#E2E8F0", linewidth=0.6)
    axes[0].set_yticks([y_lookup[p] for p in pairs], [PAIR_LABELS[p] for p in pairs])
    fig.supxlabel("Marginal total variation (%)", y=0.105)
    fig.legend(handles=_model_legend(), loc="lower center", bbox_to_anchor=(0.5, 0.012), ncol=4)
    fig.suptitle("Perbedaan komposisi opening antaridentitas", y=0.995, fontweight="bold")
    fig.tight_layout(rect=(0.01, 0.18, 0.99, 0.93), w_pad=1.2)
    return _save(fig, image_dir, "12_direct_identity_opening.png")


def _plot_availability(availability: pd.DataFrame, image_dir: Path) -> Path:
    data = availability[availability["outcome_id"].eq("text_available")].copy()
    data["effect_pp"] = 100 * data["rate_delta"]
    data["low_pp"] = 100 * data["ci_low"]
    data["high_pp"] = 100 * data["ci_high"]
    pairs = [f"{left}_vs_{right}" for left, right in DIRECT_IDENTITY_PAIRS]
    y_lookup = {pair: len(pairs) - 1 - index for index, pair in enumerate(pairs)}
    fig, axes = plt.subplots(1, 4, figsize=(20, 9), sharex=True, sharey=True)
    for ax, condition in zip(axes, PAIR_CONDITIONS):
        panel = data[data["condition"].eq(condition)]
        for _, point in panel.iterrows():
            effect = point["effect_pp"]
            ax.errorbar(
                effect, y_lookup[point["identity_contrast"]],
                xerr=np.array([[max(effect - point["low_pp"], 0)], [max(point["high_pp"] - effect, 0)]]),
                fmt="o", ms=7.5, capsize=3.0, color=MODEL_COLORS[point["model_id"]],
                alpha=0.95 if point["p_holm"] < 0.05 else 0.55,
            )
        ax.axvline(0, color="#334155", linewidth=0.8)
        ax.set_title(CONDITION_LABELS[condition], fontweight="bold")
        ax.grid(axis="x", color="#E2E8F0", linewidth=0.6)
    axes[0].set_yticks([y_lookup[p] for p in pairs], [PAIR_LABELS[p] for p in pairs])
    fig.supxlabel("Δ teks tersedia (poin persentase)", y=0.105)
    fig.legend(
        handles=_model_legend()[:-1],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=3,
    )
    fig.suptitle("Differential availability pada perbandingan antaridentitas", y=0.995, fontweight="bold")
    fig.tight_layout(rect=(0.01, 0.18, 0.99, 0.93), w_pad=1.2)
    return _save(fig, image_dir, "13_direct_identity_availability.png")


def _top_tables(tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    within = tables["paired_binary"]
    within = within[within["outcome_id"].isin(RELIGION_MARKERS)].copy()
    within["abs_delta_pp"] = 100 * within["prevalence_delta"].abs()
    within.sort_values(["p_holm", "abs_delta_pp"], ascending=[True, False]).to_csv(
        output_dir / "religion_marker_within_identity_tests.csv", index=False
    )
    direct = tables["direct_identity_binary"]
    direct = direct[direct["outcome_id"].isin(RELIGION_MARKERS)].copy()
    direct["abs_delta_pp"] = 100 * direct["prevalence_delta"].abs()
    direct.sort_values(["p_holm", "abs_delta_pp"], ascending=[True, False]).to_csv(
        output_dir / "religion_marker_direct_identity_tests.csv", index=False
    )


def refresh_figures(
    bundle_dir: Path,
    image_dir: Path | None = None,
) -> tuple[Path, ...]:
    """Redraw every figure from existing result tables without rerunning tests."""

    bundle_dir = bundle_dir.resolve()
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_dir = Path(manifest["source_run"]).resolve()
    image_dir = image_dir.resolve() if image_dir is not None else bundle_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    bundle_tables = {
        "coverage": bundle_dir / "response_counts_model_persona.csv",
        "prevalence": bundle_dir / "religion_marker_prevalence_model_persona.csv",
        "terms": bundle_dir / "religion_term_prevalence_model_persona.csv",
        "opening_composition": bundle_dir / "opening_composition_model_identity_condition.csv",
        "within_sensitivity": bundle_dir / "correctness_sensitivity_within_comparison.csv",
        "direct_sensitivity": bundle_dir / "correctness_sensitivity_direct_comparison.csv",
    }
    run_table_paths = {
        "descriptives": run_dir / "condition_descriptives.csv",
        "paired_numeric": run_dir / "paired_numeric.csv",
        "paired_binary": run_dir / "paired_binary.csv",
        "direct_binary": run_dir / "direct_identity_binary.csv",
        "direct_opening": run_dir / "direct_identity_opening_multinomial.csv",
        "direct_availability": run_dir / "direct_identity_availability.csv",
    }
    missing = [
        path
        for path in (*bundle_tables.values(), *run_table_paths.values())
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Cannot refresh religion figures; missing tables: "
            + ", ".join(str(path) for path in missing)
        )

    _style()
    tables = {
        name: pd.read_csv(path)
        for name, path in {**bundle_tables, **run_table_paths}.items()
    }
    figures: list[Path] = [
        _plot_coverage(tables["coverage"], image_dir),
        _plot_verbosity(tables["descriptives"], image_dir),
        _plot_aligned_mentions(tables["prevalence"], image_dir),
        _plot_accommodation_explicit(tables["prevalence"], image_dir),
        _plot_term_lollipop(tables["terms"], image_dir),
        _plot_term_cloud(tables["terms"], image_dir),
    ]
    figures.extend(
        _plot_within_religion_forest(tables["paired_binary"], image_dir, identity)
        for identity in IDENTITIES
    )
    figures.extend(
        _plot_general_effects(tables["paired_numeric"], image_dir, identity)
        for identity in IDENTITIES
    )
    figures.extend(
        _plot_direct_religion_forest(tables["direct_binary"], image_dir, condition)
        for condition in PAIR_CONDITIONS
    )
    figures.extend(
        (
            _plot_sensitivity(
                tables["within_sensitivity"],
                tables["direct_sensitivity"],
                image_dir,
            ),
            _plot_opening_composition(tables["opening_composition"], image_dir),
            _plot_direct_opening(tables["direct_opening"], image_dir),
            _plot_availability(tables["direct_availability"], image_dir),
        )
    )
    return tuple(figures)


def build_bundle(run_dir: Path, output_dir: Path) -> Path:
    run_dir = run_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("pipeline_version") != PIPELINE_VERSION:
        raise ValueError(
            f"Expected {PIPELINE_VERSION}, found {source_manifest.get('pipeline_version')}"
        )
    if source_manifest.get("hidden_reasoning_used") is not False:
        raise ValueError("Visible response surface is required.")

    output_dir.mkdir(parents=True, exist_ok=False)
    image_dir = output_dir / "images"
    image_dir.mkdir()
    _style()
    tables = _read_run(run_dir)
    response = _load_response_frame(run_dir)

    coverage = _coverage_table(tables["input_inventory"], tables["quality_control"])
    coverage.to_csv(output_dir / "response_counts_model_persona.csv", index=False)
    prevalence = _marker_prevalence(response)
    prevalence.to_csv(output_dir / "religion_marker_prevalence_model_persona.csv", index=False)
    term_table = _term_table(response)
    term_table.to_csv(output_dir / "religion_term_prevalence_model_persona.csv", index=False)
    opening_composition = _opening_composition(response)
    opening_composition.to_csv(output_dir / "opening_composition_model_identity_condition.csv", index=False)

    within_comparison, within_summary = _sensitivity_tables(
        tables["paired_numeric"], tables["correctness_sensitivity_paired_numeric"], False
    )
    direct_comparison, direct_summary = _sensitivity_tables(
        tables["direct_identity_numeric"], tables["direct_identity_correctness_sensitivity"], True
    )
    within_comparison.to_csv(output_dir / "correctness_sensitivity_within_comparison.csv", index=False)
    within_summary.to_csv(output_dir / "correctness_sensitivity_within_summary.csv", index=False)
    direct_comparison.to_csv(output_dir / "correctness_sensitivity_direct_comparison.csv", index=False)
    direct_summary.to_csv(output_dir / "correctness_sensitivity_direct_summary.csv", index=False)
    _top_tables(tables, output_dir)

    figures: list[tuple[str, str]] = []
    def add(path: Path, description: str) -> None:
        figures.append((path.name, description))

    add(_plot_coverage(coverage, image_dir), "Cakupan teks model × 17 kondisi persona.")
    add(_plot_verbosity(tables["condition_descriptives"], image_dir), "Median dan IQR jumlah kata.")
    add(_plot_aligned_mentions(prevalence, image_dir), "Prevalence penyebutan identitas selaras.")
    add(_plot_accommodation_explicit(prevalence, image_dir), "Akomodasi bahasa agama pada kondisi Explicit.")
    add(_plot_term_lollipop(term_table, image_dir), "Ranking istilah agama Explicit dengan nilai eksak.")
    add(_plot_term_cloud(term_table, image_dir), "Word cloud tertutup istilah agama Explicit.")
    for identity in IDENTITIES:
        add(_plot_within_religion_forest(tables["paired_binary"], image_dir, identity), f"Marker agama {identity} vs Neutral.")
    for identity in IDENTITIES:
        add(_plot_general_effects(tables["paired_numeric"], image_dir, identity), f"Marker umum {identity} vs Neutral.")
    for condition in PAIR_CONDITIONS:
        add(_plot_direct_religion_forest(tables["direct_identity_binary"], image_dir, condition), f"Direct identity marker agama kondisi {condition}.")
    add(_plot_sensitivity(within_comparison, direct_comparison, image_dir), "Sensitivitas correctness.")
    add(_plot_opening_composition(opening_composition, image_dir), "Komposisi opening 100 persen.")
    add(_plot_direct_opening(tables["direct_identity_opening_multinomial"], image_dir), "Direct identity opening Stuart-Maxwell.")
    add(_plot_availability(tables["direct_identity_availability"], image_dir), "Direct identity availability.")

    pd.DataFrame(figures, columns=["figure", "description"]).to_csv(
        output_dir / "figure_index.csv", index=False
    )
    bundle_manifest = {
        "bundle_version": BUNDLE_VERSION,
        "source_run": str(run_dir),
        "source_manifest_sha256": _sha256(manifest_path),
        "source_pipeline_version": source_manifest["pipeline_version"],
        "source_response_records": source_manifest["response_records"],
        "source_gap_records": source_manifest["source_gap_records"],
        "bootstrap_resamples": source_manifest["bootstrap_resamples"],
        "hidden_reasoning_used": False,
        "correctness_primary_filter": False,
        "implicit_proxies_aggregated_for_inference": False,
        "religion_identities_pooled": False,
        "direct_identity_pairs": [list(pair) for pair in DIRECT_IDENTITY_PAIRS],
        "generated_figures": [name for name, _ in figures],
        "generated_tables": [
            "response_counts_model_persona.csv",
            "religion_marker_prevalence_model_persona.csv",
            "religion_term_prevalence_model_persona.csv",
            "opening_composition_model_identity_condition.csv",
            "correctness_sensitivity_within_comparison.csv",
            "correctness_sensitivity_within_summary.csv",
            "correctness_sensitivity_direct_comparison.csv",
            "correctness_sensitivity_direct_summary.csv",
            "religion_marker_within_identity_tests.csv",
            "religion_marker_direct_identity_tests.csv",
            "figure_index.csv",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(bundle_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[DONE] Visual bundle: {output_dir}")
    return output_dir


def main() -> int:
    args = _parser().parse_args()
    build_bundle(args.run_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
