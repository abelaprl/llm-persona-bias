"""Audit explicit gendered words in visible GSM8K model responses.

This module is intentionally separate from the open-vocabulary lexical
contrast analysis.  Pronouns such as ``he`` and ``she`` were stopwords there;
here they are the object of measurement.  The lexicon is closed, versioned,
and limited to unambiguous gender pronouns and human-denoting nouns.
"""

from __future__ import annotations

import argparse
from collections import Counter
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


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.linguistic_markers.config import (  # noqa: E402
    GENDER_PERSONAS,
    IDENTITY_CONDITIONS,
    PLANNED_CONTRASTS,
)
from src.core.linguistic_markers.statistics import (  # noqa: E402
    _bootstrap_ci,
    _mcnemar_exact,
    _stable_seed,
    holm_adjust,
)


LEXICON_VERSION = "gendered-visible-lexicon-v1"
TERM_CATEGORIES = {
    # Pronouns are the primary focus requested by the user.
    "he": "masculine_pronoun",
    "him": "masculine_pronoun",
    "his": "masculine_pronoun",
    "himself": "masculine_pronoun",
    "she": "feminine_pronoun",
    "her": "feminine_pronoun",
    "hers": "feminine_pronoun",
    "herself": "feminine_pronoun",
    # Human-denoting gender nouns. Ambiguous titles such as ``miss`` and
    # initials such as ``ms`` are deliberately excluded.
    "man": "masculine_noun",
    "men": "masculine_noun",
    "male": "masculine_noun",
    "males": "masculine_noun",
    "boy": "masculine_noun",
    "boys": "masculine_noun",
    "father": "masculine_noun",
    "fathers": "masculine_noun",
    "dad": "masculine_noun",
    "dads": "masculine_noun",
    "husband": "masculine_noun",
    "husbands": "masculine_noun",
    "son": "masculine_noun",
    "sons": "masculine_noun",
    "brother": "masculine_noun",
    "brothers": "masculine_noun",
    "gentleman": "masculine_noun",
    "gentlemen": "masculine_noun",
    "uncle": "masculine_noun",
    "uncles": "masculine_noun",
    "guy": "masculine_noun",
    "guys": "masculine_noun",
    "woman": "feminine_noun",
    "women": "feminine_noun",
    "female": "feminine_noun",
    "females": "feminine_noun",
    "girl": "feminine_noun",
    "girls": "feminine_noun",
    "mother": "feminine_noun",
    "mothers": "feminine_noun",
    "mom": "feminine_noun",
    "moms": "feminine_noun",
    "wife": "feminine_noun",
    "wives": "feminine_noun",
    "daughter": "feminine_noun",
    "daughters": "feminine_noun",
    "sister": "feminine_noun",
    "sisters": "feminine_noun",
    "lady": "feminine_noun",
    "ladies": "feminine_noun",
    "aunt": "feminine_noun",
    "aunts": "feminine_noun",
    "gal": "feminine_noun",
    "gals": "feminine_noun",
}
TERMS = tuple(TERM_CATEGORIES)
PRONOUN_CLASSES = {
    "masculine_pronoun": tuple(
        term for term, category in TERM_CATEGORIES.items() if category == "masculine_pronoun"
    ),
    "feminine_pronoun": tuple(
        term for term, category in TERM_CATEGORIES.items() if category == "feminine_pronoun"
    ),
}

MODEL_LABELS = {
    "meta-llama_llama-3.1-8b-instruct": "Llama 3.1 8B",
    "meta-llama_llama-3.3-70b-instruct": "Llama 3.3 70B",
    "mistralai_ministral-8b-2512": "Ministral 8B",
    "deepseek_deepseek-r1-distill-qwen-32b": "DeepSeek R1 32B",
    "openai_gpt-oss-20b": "GPT-OSS 20B",
    "qwen_qwen3-next-80b-a3b-thinking": "Qwen3 Next 80B",
}
MODEL_ORDER = tuple(MODEL_LABELS)
PERSONA_LABELS = {
    "neutral": "N",
    "man_explicit_1": "M-E",
    "man_implicit_1": "M-I1",
    "man_implicit_2": "M-I2",
    "man_implicit_3": "M-I3",
    "woman_explicit_1": "W-E",
    "woman_implicit_1": "W-I1",
    "woman_implicit_2": "W-I2",
    "woman_implicit_3": "W-I3",
}
CONDITION_LABELS = {
    "neutral": "N",
    "explicit": "E",
    "implicit_1": "I1",
    "implicit_2": "I2",
    "implicit_3": "I3",
}
CONTRAST_LABELS = {
    "implicit_1_vs_neutral": "I1 − N",
    "implicit_2_vs_neutral": "I2 − N",
    "implicit_3_vs_neutral": "I3 − N",
}
COLORS = {
    "masculine": "#2563EB",
    "feminine": "#D946A1",
    "neutral": "#64748B",
}
TOKEN_RE = re.compile(r"(?u)\b[a-z]+(?:'[a-z]+)?\b")


def tokenize_gender_terms(text: str) -> Counter[str]:
    """Count exact lexicon tokens; substrings such as ``he`` in ``the`` do not match."""

    tokens = TOKEN_RE.findall(str(text).casefold())
    return Counter(token for token in tokens if token in TERM_CATEGORIES)


def _bool_mask(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.fillna(False).astype(bool)
    return values.fillna("").astype(str).str.strip().str.casefold().eq("true")


def _load(run_dir: Path) -> tuple[pd.DataFrame, object, np.ndarray]:
    # Keep tokenization and paired-statistics helpers importable in minimal
    # environments; scikit-learn is only required for the full sparse-matrix
    # build performed by this loader.
    from sklearn.feature_extraction.text import CountVectorizer

    columns = [
        "model_id",
        "persona",
        "question_id",
        "visible_text",
        "text_available",
        "word_count",
    ]
    frame = pd.read_csv(run_dir / "response_records.csv", usecols=columns, low_memory=False)
    frame = frame[
        _bool_mask(frame["text_available"]) & frame["visible_text"].notna()
    ].reset_index(drop=True)
    vectorizer = CountVectorizer(
        lowercase=True,
        vocabulary=list(TERMS),
        token_pattern=r"(?u)\b[a-z]+(?:'[a-z]+)?\b",
        dtype=np.int32,
    )
    matrix = vectorizer.transform(frame["visible_text"].astype(str))
    return frame, matrix, vectorizer.get_feature_names_out()


def _term_table(frame: pd.DataFrame, matrix, vocabulary: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (model_id, persona), indices in frame.groupby(
        ["model_id", "persona"], sort=True
    ).groups.items():
        positions = np.asarray(list(indices), dtype=int)
        selected = matrix[positions]
        occurrences = np.asarray(selected.sum(axis=0)).ravel()
        response_presence = np.asarray((selected > 0).sum(axis=0)).ravel()
        total_words = float(
            pd.to_numeric(frame.loc[positions, "word_count"], errors="coerce").sum()
        )
        for column, term in enumerate(vocabulary):
            count = int(occurrences[column])
            present = int(response_presence[column])
            rows.append(
                {
                    "lexicon_version": LEXICON_VERSION,
                    "model_id": model_id,
                    "persona": persona,
                    "term": term,
                    "term_category": TERM_CATEGORIES[term],
                    "n_responses_available": len(positions),
                    "responses_with_term": present,
                    "response_prevalence_pp": 100 * present / len(positions),
                    "occurrences": count,
                    "total_words": int(total_words),
                    "occurrences_per_10k_words": (
                        10_000 * count / total_words if total_words else math.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def _class_presence(
    frame: pd.DataFrame,
    matrix,
    vocabulary: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    term_index = {term: index for index, term in enumerate(vocabulary)}
    response = frame[["model_id", "persona", "question_id"]].copy()
    for class_id, terms in PRONOUN_CLASSES.items():
        columns = [term_index[term] for term in terms]
        response[class_id] = np.asarray((matrix[:, columns].sum(axis=1) > 0)).ravel().astype(int)

    rows: list[dict[str, object]] = []
    for (model_id, persona), group in response.groupby(["model_id", "persona"], sort=True):
        for class_id in PRONOUN_CLASSES:
            rows.append(
                {
                    "model_id": model_id,
                    "persona": persona,
                    "pronoun_class": class_id,
                    "n_responses_available": len(group),
                    "responses_with_class": int(group[class_id].sum()),
                    "response_prevalence_pp": 100 * float(group[class_id].mean()),
                }
            )
    return response, pd.DataFrame(rows)


def _paired_pronoun_classes(
    response: pd.DataFrame,
    *,
    bootstrap_resamples: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_id, model in response.groupby("model_id", sort=True):
        for identity, conditions in IDENTITY_CONDITIONS.items():
            for class_id in PRONOUN_CLASSES:
                for contrast_id, left_condition, right_condition in PLANNED_CONTRASTS:
                    left_persona = conditions[left_condition]
                    right_persona = conditions[right_condition]
                    left = model[model["persona"].eq(left_persona)][
                        ["question_id", class_id]
                    ]
                    right = model[model["persona"].eq(right_persona)][
                        ["question_id", class_id]
                    ]
                    paired = left.merge(
                        right,
                        on="question_id",
                        how="inner",
                        validate="one_to_one",
                        suffixes=("_left", "_right"),
                    )
                    left_values = paired[f"{class_id}_left"].to_numpy(float)
                    right_values = paired[f"{class_id}_right"].to_numpy(float)
                    deltas = left_values - right_values
                    ci_low, ci_high = _bootstrap_ci(
                        deltas,
                        "mean",
                        bootstrap_resamples,
                        _stable_seed(
                            LEXICON_VERSION,
                            model_id,
                            identity,
                            class_id,
                            contrast_id,
                        ),
                    )
                    rows.append(
                        {
                            "model_id": model_id,
                            "identity": identity,
                            "pronoun_class": class_id,
                            "contrast_id": contrast_id,
                            "left_condition": left_condition,
                            "right_condition": right_condition,
                            "left_persona": left_persona,
                            "right_persona": right_persona,
                            "prevalence_left": float(left_values.mean()),
                            "prevalence_right": float(right_values.mean()),
                            "prevalence_delta": float(deltas.mean()),
                            "ci_low": ci_low,
                            "ci_high": ci_high,
                            **_mcnemar_exact(left_values, right_values),
                        }
                    )
    result = pd.DataFrame(rows)
    family = ["model_id", "identity", "pronoun_class"]
    result["p_holm"] = result.groupby(family, dropna=False)["p_raw"].transform(
        lambda values: holm_adjust(values.to_numpy(float))
    )
    return result


def _paired_pronoun_gender_direct(
    response: pd.DataFrame,
    *,
    bootstrap_resamples: int,
) -> pd.DataFrame:
    """Directly compare Man and Woman pronoun presence within each condition."""

    conditions = ("explicit", "implicit_1", "implicit_2", "implicit_3")
    rows: list[dict[str, object]] = []
    for model_id, model in response.groupby("model_id", sort=True):
        for class_id in PRONOUN_CLASSES:
            for condition in conditions:
                left_persona = IDENTITY_CONDITIONS["man"][condition]
                right_persona = IDENTITY_CONDITIONS["woman"][condition]
                left = model[model["persona"].eq(left_persona)][
                    ["question_id", class_id]
                ]
                right = model[model["persona"].eq(right_persona)][
                    ["question_id", class_id]
                ]
                paired = left.merge(
                    right,
                    on="question_id",
                    how="inner",
                    validate="one_to_one",
                    suffixes=("_left", "_right"),
                )
                left_values = paired[f"{class_id}_left"].to_numpy(float)
                right_values = paired[f"{class_id}_right"].to_numpy(float)
                deltas = left_values - right_values
                contrast_id = f"man_vs_woman_{condition}"
                ci_low, ci_high = _bootstrap_ci(
                    deltas,
                    "mean",
                    bootstrap_resamples,
                    _stable_seed(
                        LEXICON_VERSION,
                        model_id,
                        "man_vs_woman",
                        class_id,
                        condition,
                    ),
                )
                rows.append(
                    {
                        "model_id": model_id,
                        "identity_contrast": "man_vs_woman",
                        "condition": condition,
                        "pronoun_class": class_id,
                        "contrast_id": contrast_id,
                        "left_persona": left_persona,
                        "right_persona": right_persona,
                        "prevalence_left": float(left_values.mean()),
                        "prevalence_right": float(right_values.mean()),
                        "prevalence_delta": float(deltas.mean()),
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        **_mcnemar_exact(left_values, right_values),
                    }
                )
    result = pd.DataFrame(rows)
    family = ["model_id", "pronoun_class"]
    result["p_holm"] = result.groupby(family, dropna=False)["p_raw"].transform(
        lambda values: holm_adjust(values.to_numpy(float))
    )
    return result


def _aggregate_persona_terms(term_table: pd.DataFrame) -> pd.DataFrame:
    return (
        term_table.groupby(["persona", "term", "term_category"], as_index=False)
        .agg(
            mean_model_occurrences_per_10k_words=(
                "occurrences_per_10k_words",
                "mean",
            ),
            mean_model_response_prevalence_pp=("response_prevalence_pp", "mean"),
            total_occurrences=("occurrences", "sum"),
            n_models=("model_id", "nunique"),
        )
        .sort_values(
            ["persona", "mean_model_occurrences_per_10k_words"],
            ascending=[True, False],
            kind="stable",
        )
    )


def _term_color(category: str) -> str:
    return COLORS["masculine"] if category.startswith("masculine") else COLORS["feminine"]


def _draw_gender_cloud(
    ax: plt.Axes,
    terms: pd.DataFrame,
    *,
    title: str,
    weight_cap: float,
    seed: int,
) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(title, fontsize=18, fontweight="bold", pad=5)
    terms = terms[terms["mean_model_occurrences_per_10k_words"] > 0].nlargest(
        16, "mean_model_occurrences_per_10k_words"
    )
    if terms.empty:
        ax.text(0.5, 0.5, "tidak ada kata gender", ha="center", va="center", color="#64748B")
        return

    rng = np.random.default_rng(seed)
    boxes: list[tuple[float, float, float, float]] = []
    golden_angle = math.pi * (3 - math.sqrt(5))
    for rank, (_, row) in enumerate(terms.reset_index(drop=True).iterrows()):
        weight = min(float(row["mean_model_occurrences_per_10k_words"]), weight_cap)
        scaled = math.sqrt(weight / max(weight_cap, 1e-9))
        font_size = 9.0 + 28.0 * scaled
        word = str(row["term"])
        width = min(0.82, 0.0105 * len(word) * font_size / 9.0)
        height = min(0.22, 0.0055 * font_size)
        phase = rng.uniform(0, 2 * math.pi)
        for attempt in range(900):
            radius = 0.017 * math.sqrt(attempt)
            angle = phase + attempt * golden_angle
            x = 0.5 + radius * math.cos(angle)
            y = 0.5 + 0.76 * radius * math.sin(angle)
            candidate = (
                x - width / 2,
                y - height / 2,
                x + width / 2,
                y + height / 2,
            )
            if (
                candidate[0] < 0.025
                or candidate[1] < 0.035
                or candidate[2] > 0.975
                or candidate[3] > 0.965
            ):
                continue
            overlaps = any(
                candidate[0] < box[2] + 0.014
                and candidate[2] > box[0] - 0.014
                and candidate[1] < box[3] + 0.009
                and candidate[3] > box[1] - 0.009
                for box in boxes
            )
            if overlaps:
                continue
            ax.text(
                x,
                y,
                word,
                ha="center",
                va="center",
                fontsize=font_size,
                color=_term_color(str(row["term_category"])),
                alpha=1.0 - 0.30 * rank / max(len(terms) - 1, 1),
                fontweight="bold" if rank < 5 else "normal",
            )
            boxes.append(candidate)
            break


def _plot_gender_clouds(aggregated: pd.DataFrame, figure_dir: Path) -> Path:
    conditions = ("neutral", "explicit", "implicit_1", "implicit_2", "implicit_3")
    selected_weights: list[float] = []
    panels: dict[tuple[str, str], pd.DataFrame] = {}
    for identity in ("man", "woman"):
        for condition in conditions:
            persona = IDENTITY_CONDITIONS[identity][condition]
            panel = aggregated[aggregated["persona"].eq(persona)]
            panels[(identity, condition)] = panel
            selected_weights.extend(panel["mean_model_occurrences_per_10k_words"].tolist())
    positive = np.asarray([value for value in selected_weights if value > 0], dtype=float)
    weight_cap = float(np.quantile(positive, 0.97)) if len(positive) else 1.0

    fig, axes = plt.subplots(5, 2, figsize=(15, 21))
    for row, condition in enumerate(conditions):
        for column, identity in enumerate(("man", "woman")):
            _draw_gender_cloud(
                axes[row, column],
                panels[(identity, condition)],
                title=f"{identity.title()} · {CONDITION_LABELS[condition]}",
                weight_cap=weight_cap,
                seed=18222008 + row * 10 + column,
            )
            axes[row, column].add_patch(
                plt.Rectangle(
                    (0.005, 0.01),
                    0.99,
                    0.95,
                    transform=axes[row, column].transAxes,
                    fill=False,
                    edgecolor="#E2E8F0",
                    linewidth=0.8,
                )
            )
    handles = [
        Line2D([0], [0], marker="o", linestyle="", color=COLORS["masculine"], label="Masculine terms"),
        Line2D([0], [0], marker="o", linestyle="", color=COLORS["feminine"], label="Feminine terms"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.022),
        fontsize=18,
    )
    fig.suptitle(
        "Penyebutan gender dalam respons model per persona\n"
        "Ukuran kata = rata-rata setara antar-model dari kemunculan per 10.000 kata",
        y=0.995,
        fontsize=20,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.055,
        "Dari leksikon 52 kata gender, tiap panel menampilkan 16 yang paling sering muncul. Panel N memakai baseline neutral yang sama.",
        ha="center",
        color="#475569",
        fontsize=14,
    )
    fig.tight_layout(rect=(0.02, 0.075, 0.98, 0.955), h_pad=1.7, w_pad=1.5)
    path = figure_dir / "11_gendered_terms_wordcloud_by_persona.png"
    fig.savefig(path, dpi=230, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _plot_model_pronoun_prevalence(class_table: pd.DataFrame, figure_dir: Path) -> Path:
    fig, axes = plt.subplots(3, 2, figsize=(15, 13))
    conditions = ("neutral", "explicit", "implicit_1", "implicit_2", "implicit_3")
    x = np.arange(len(conditions))
    for ax, model_id in zip(axes.flat, MODEL_ORDER):
        model = class_table[class_table["model_id"].eq(model_id)]
        for class_id, color, class_label in (
            ("masculine_pronoun", COLORS["masculine"], "Masculine pronouns"),
            ("feminine_pronoun", COLORS["feminine"], "Feminine pronouns"),
        ):
            indexed = model[model["pronoun_class"].eq(class_id)].set_index("persona")
            for identity, linestyle in (("man", "-"), ("woman", "--")):
                personas = [
                    IDENTITY_CONDITIONS[identity][condition]
                    for condition in conditions
                ]
                values = indexed.reindex(personas)["response_prevalence_pp"].to_numpy(float)
                ax.plot(
                    x,
                    values,
                    marker="o",
                    linewidth=2,
                    markersize=7.5,
                    linestyle=linestyle,
                    color=color,
                    label=f"{class_label} · {identity.title()}",
                )
        ax.set_xticks(x, ("N", "E", "I1", "I2", "I3"), fontsize=15)
        ax.set_title(MODEL_LABELS[model_id], loc="left", fontweight="bold")
        ax.grid(axis="y", color="#E2E8F0", linewidth=0.6)
    fig.supylabel("Respons yang memuat pronoun (%)", x=0.012)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.94),
        fontsize=18,
    )
    fig.suptitle(
        "Pemakaian pronoun gender pada setiap model × persona",
        y=0.985,
        fontsize=20,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.02,
        "Garis utuh = persona Man; garis putus-putus = persona Woman. Keduanya berawal dari baseline neutral yang sama.",
        ha="center",
        color="#475569",
        fontsize=14,
    )
    fig.tight_layout(rect=(0.065, 0.055, 0.98, 0.925), h_pad=2.0, w_pad=1.5)
    path = figure_dir / "12_gender_pronoun_prevalence_by_model_persona.png"
    fig.savefig(path, dpi=230, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _plot_pronoun_gender_forest(paired: pd.DataFrame, figure_dir: Path) -> Path:
    conditions = ("explicit", "implicit_1", "implicit_2", "implicit_3")
    classes = ("masculine_pronoun", "feminine_pronoun")
    y_lookup = {
        model_id: len(MODEL_ORDER) - 1 - index
        for index, model_id in enumerate(MODEL_ORDER)
    }
    selected = paired.copy()
    selected["delta_pp"] = 100 * selected["prevalence_delta"]
    selected["ci_low_pp"] = 100 * selected["ci_low"]
    selected["ci_high_pp"] = 100 * selected["ci_high"]
    fig, axes = plt.subplots(2, 4, figsize=(20, 8.7), squeeze=False)
    for row, class_id in enumerate(classes):
        class_data = selected[selected["pronoun_class"].eq(class_id)]
        bound = float(np.nanmax(np.abs(class_data[["ci_low_pp", "ci_high_pp"]].to_numpy(float))))
        bound = max(bound * 1.15, 0.5)
        for column, condition in enumerate(conditions):
            ax = axes[row, column]
            panel = class_data[class_data["condition"].eq(condition)]
            for _, point in panel.iterrows():
                y = y_lookup[point["model_id"]]
                effect = point["delta_pp"]
                ax.errorbar(
                    effect,
                    y,
                    xerr=np.array(
                        [[max(effect - point["ci_low_pp"], 0)], [max(point["ci_high_pp"] - effect, 0)]]
                    ),
                    fmt="o",
                    markersize=8,
                    capsize=3.5,
                    elinewidth=1.45,
                    color=COLORS["masculine"] if class_id == "masculine_pronoun" else COLORS["feminine"],
                )
            ax.axvline(0, color="#334155", linewidth=0.8)
            ax.set_xlim(-bound, bound)
            ax.set_ylim(-0.8, len(MODEL_ORDER) - 0.2)
            ax.grid(axis="x", color="#E2E8F0", linewidth=0.6)
            if row == 0:
                ax.set_title(CONDITION_LABELS[condition], fontweight="bold")
            if column == 0:
                ax.set_yticks(
                    [y_lookup[model_id] for model_id in MODEL_ORDER],
                    [MODEL_LABELS[model_id] for model_id in MODEL_ORDER],
                    fontsize=15,
                )
                ax.set_ylabel(
                    "Masculine pronouns" if class_id == "masculine_pronoun" else "Feminine pronouns",
                    fontweight="bold",
                )
            else:
                ax.set_yticks([])
    fig.supxlabel("Man − Woman (pp), 95% paired bootstrap CI", y=0.03)
    fig.suptitle(
        "Perbedaan langsung pemakaian pronoun: persona Man minus Woman\n"
        "Setiap titik dipasangkan pada model dan pertanyaan yang sama",
        y=0.995,
        fontsize=20,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.01, 0.08, 0.99, 0.91), h_pad=2.0, w_pad=1.2)
    path = figure_dir / "13_gender_pronoun_direct_man_vs_woman_forest.png"
    fig.savefig(path, dpi=230, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def refresh_gendered_lexicon_figures(
    bundle_dir: Path,
    figure_dir: Path | None = None,
) -> tuple[Path, ...]:
    """Redraw the three gender-lexicon figures from their existing CSV tables."""

    bundle_dir = bundle_dir.resolve()
    figure_dir = (
        figure_dir.resolve()
        if figure_dir is not None
        else bundle_dir / "images"
    )
    required = (
        bundle_dir / "gendered_term_persona_aggregate.csv",
        bundle_dir / "gender_pronoun_class_prevalence.csv",
        bundle_dir / "paired_gender_pronoun_direct.csv",
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Cannot refresh gender figures; missing tables: "
            + ", ".join(str(path) for path in missing)
        )
    figure_dir.mkdir(parents=True, exist_ok=True)
    aggregated = pd.read_csv(required[0])
    class_table = pd.read_csv(required[1])
    paired_direct = pd.read_csv(required[2])
    return (
        _plot_gender_clouds(aggregated, figure_dir),
        _plot_model_pronoun_prevalence(class_table, figure_dir),
        _plot_pronoun_gender_forest(paired_direct, figure_dir),
    )


def build_gendered_lexicon_artifacts(
    run_dir: Path,
    bundle_dir: Path,
    *,
    bootstrap_resamples: int = 10_000,
    update_bundle_metadata: bool = True,
) -> dict[str, object]:
    run_dir = run_dir.resolve()
    bundle_dir = bundle_dir.resolve()
    figure_dir = bundle_dir / "images"
    if not (run_dir / "response_records.csv").is_file():
        raise FileNotFoundError(run_dir / "response_records.csv")
    if not figure_dir.is_dir():
        raise FileNotFoundError(f"Bundle image directory not found: {figure_dir}")

    expected_outputs = (
        bundle_dir / "gendered_term_model_persona.csv",
        bundle_dir / "gendered_term_persona_aggregate.csv",
        bundle_dir / "gender_pronoun_class_prevalence.csv",
        bundle_dir / "paired_gender_pronoun_classes.csv",
        bundle_dir / "paired_gender_pronoun_direct.csv",
        bundle_dir / "gendered_lexicon_metadata.json",
        figure_dir / "11_gendered_terms_wordcloud_by_persona.png",
        figure_dir / "12_gender_pronoun_prevalence_by_model_persona.png",
        figure_dir / "13_gender_pronoun_direct_man_vs_woman_forest.png",
    )
    existing = [path for path in expected_outputs if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing gender-lexicon artifacts: "
            + ", ".join(str(path) for path in existing)
        )

    frame, matrix, vocabulary = _load(run_dir)
    term_table = _term_table(frame, matrix, vocabulary)
    response_classes, class_table = _class_presence(frame, matrix, vocabulary)
    paired = _paired_pronoun_classes(
        response_classes,
        bootstrap_resamples=bootstrap_resamples,
    )
    paired_direct = _paired_pronoun_gender_direct(
        response_classes,
        bootstrap_resamples=bootstrap_resamples,
    )
    aggregated = _aggregate_persona_terms(term_table)

    term_table.to_csv(bundle_dir / "gendered_term_model_persona.csv", index=False)
    aggregated.to_csv(bundle_dir / "gendered_term_persona_aggregate.csv", index=False)
    class_table.to_csv(bundle_dir / "gender_pronoun_class_prevalence.csv", index=False)
    paired.to_csv(bundle_dir / "paired_gender_pronoun_classes.csv", index=False)
    paired_direct.to_csv(bundle_dir / "paired_gender_pronoun_direct.csv", index=False)

    figures = (
        _plot_gender_clouds(aggregated, figure_dir),
        _plot_model_pronoun_prevalence(class_table, figure_dir),
        _plot_pronoun_gender_forest(paired_direct, figure_dir),
    )
    metadata = {
        "status": "candidate_closed_lexicon_descriptive_and_paired",
        "lexicon_version": LEXICON_VERSION,
        "text_field": "visible_text",
        "hidden_reasoning_used": False,
        "terms": TERM_CATEGORIES,
        "ambiguous_exclusions": ["miss", "ms", "master", "queen", "king"],
        "absolute_wordcloud_weight": (
            "equal-model mean occurrences per 10,000 visible-response words"
        ),
        "pronoun_prevalence_metric": (
            "percentage of visible responses containing at least one class term"
        ),
        "paired_key": ["model_id", "question_id"],
        "bootstrap_resamples": bootstrap_resamples,
        "n_available_responses": len(frame),
        "n_terms": len(TERM_CATEGORIES),
        "generated_figures": [path.name for path in figures],
    }
    (bundle_dir / "gendered_lexicon_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if update_bundle_metadata:
        descriptions = {
            "11_gendered_terms_wordcloud_by_persona.png": (
                "Word cloud closed gender lexicon by persona; exact normalized weights in CSV."
            ),
            "12_gender_pronoun_prevalence_by_model_persona.png": (
                "Per-model response prevalence of masculine and feminine pronoun classes."
            ),
            "13_gender_pronoun_direct_man_vs_woman_forest.png": (
                "Direct paired Man-minus-Woman pronoun prevalence delta with 95% bootstrap CI."
            ),
        }
        figure_index_path = bundle_dir / "figure_index.csv"
        figure_index = pd.read_csv(figure_index_path)
        additions = pd.DataFrame(
            [
                {"figure": name, "description": description}
                for name, description in descriptions.items()
            ]
        )
        figure_index = pd.concat((figure_index, additions), ignore_index=True)
        if figure_index["figure"].duplicated().any():
            raise ValueError("Duplicate figure name while updating bundle index.")
        figure_index.to_csv(figure_index_path, index=False)

        manifest_path = bundle_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.setdefault("figures", {}).update(descriptions)
        manifest["gendered_lexicon_version"] = LEXICON_VERSION
        manifest["gendered_terms_include_pronouns"] = True
        manifest["generated_files_before_manifest"] = sorted(
            path.relative_to(bundle_dir).as_posix()
            for path in bundle_dir.rglob("*")
            if path.is_file() and path.name != "manifest.json"
        )
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        readme_path = bundle_dir / "README.md"
        readme = readme_path.read_text(encoding="utf-8")
        heading = "## Dedicated gendered-word analysis"
        if heading not in readme:
            readme += f"""

{heading}

The open-vocabulary lollipop analysis removes ordinary pronouns as stopwords.
The dedicated `{LEXICON_VERSION}` analysis therefore measures `he/him/his`,
`she/her/hers`, and a closed list of human-denoting gender nouns separately.

- Figure 11 is the requested gender-word cloud. Its size is an equal-model mean
  of occurrences per 10,000 visible-response words.
- Figure 12 shows exact pronoun-class response prevalence for every model and
  all nine personas.
- Figure 13 reports direct paired Man-minus-Woman prevalence differences for
  Explicit, I1, I2, and I3 with 95% bootstrap confidence intervals.

Prompt text and hidden reasoning are not included. Exact counts and rates are
stored in the five `gender*csv` tables.
"""
            readme_path.write_text(readme, encoding="utf-8")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    args = parser.parse_args()
    metadata = build_gendered_lexicon_artifacts(
        args.run_dir,
        args.bundle_dir,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    print(
        f"[DONE] {metadata['lexicon_version']} -> "
        f"{args.bundle_dir.resolve()} ({len(metadata['generated_figures'])} figures)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
