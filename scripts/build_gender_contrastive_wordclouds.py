"""Build descriptive, paired lexical contrast clouds for the GSM8K gender run.

The figures deliberately avoid raw token-frequency clouds.  Each token is
weighted by the difference in response-level prevalence between two personas
answering the same question.  Differences are calculated separately per model
and then averaged so that a model with more available responses cannot dominate
the result.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


PIPELINE_VERSION = "gender-lm-v1.2.1-candidate"
IDENTITIES = ("man", "woman")
CONDITIONS = ("explicit", "implicit_1", "implicit_2", "implicit_3")
CONTRASTS = (
    ("explicit_vs_neutral", "explicit", "neutral"),
    ("implicit_1_vs_neutral", "implicit_1", "neutral"),
    ("implicit_2_vs_neutral", "implicit_2", "neutral"),
    ("implicit_3_vs_neutral", "implicit_3", "neutral"),
    ("implicit_1_vs_explicit", "implicit_1", "explicit"),
    ("implicit_2_vs_explicit", "implicit_2", "explicit"),
    ("implicit_3_vs_explicit", "implicit_3", "explicit"),
)

# Closed-class words are removed, while stylistically meaningful pronouns and
# discourse connectives (I, we, you, therefore, however, because, etc.) remain.
STOP_WORDS = frozenset(
    """
    a an the and or of to in on at by for from with as is are was were be been
    being do does did have has had having this that these those it its they
    them their he him his she her hers not no yes into onto out up down over
    under again further than too very can may might must shall should will would
    could about above after before between during through while where when what
    which who whom whose why how all any both each few more most other some such
    only own same once here there also just
    """.split()
)

# These terms mostly encode GSM8K's domain or LaTeX serialization rather than
# lexical style.  They are already covered more directly by the math-format and
# organization markers in the main pipeline.
DOMAIN_STOP_WORDS = frozenset(
    """
    answer answers amount amounts begin boxed calculate calculated calculates
    calculating calculation calculations cdot determine determined determines
    determining dfrac divide divided divides dividing end equals find finding
    finds final frac given gives information math mathematical multiply
    multiplied multiplies multiplying number numbers problem problems solution
    solutions solve solved solves solving subtract subtracted subtracting
    subtracts sum sums text textboxed times total totals
    """.split()
)

COLORS = {
    "man": "#2563EB",
    "woman": "#DB2777",
    "neutral": "#475569",
    "explicit": "#7C3AED",
    "implicit": "#059669",
}


def _persona(identity: str, condition: str) -> str:
    if condition == "neutral":
        return "neutral"
    suffix = "explicit_1" if condition == "explicit" else condition
    return f"{identity}_{suffix}"


def _condition_label(condition: str) -> str:
    return {
        "neutral": "N",
        "explicit": "E",
        "implicit_1": "I1",
        "implicit_2": "I2",
        "implicit_3": "I3",
    }[condition]


def load_available_responses(run_dir: Path) -> pd.DataFrame:
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("pipeline_version") != PIPELINE_VERSION:
        raise ValueError("Unexpected linguistic-marker pipeline version.")
    if manifest.get("correctness_filter_applied") is not False:
        raise ValueError("Lexical exploration must not use a correctness-filtered run.")

    columns = [
        "model_id",
        "persona",
        "question_id",
        "visible_text",
        "text_available",
    ]
    frame = pd.read_csv(run_dir / "response_records.csv", usecols=columns)
    availability = (
        frame["text_available"].fillna(False).astype(str).str.casefold().eq("true")
    )
    available = frame[availability & frame["visible_text"].notna()].copy()
    duplicate = available.duplicated(
        ["model_id", "persona", "question_id"], keep=False
    )
    if duplicate.any():
        raise ValueError("Duplicate model-persona-question key in available responses.")
    available["matrix_row"] = np.arange(len(available), dtype=np.int64)
    return available.reset_index(drop=True)


def compute_lexical_contrasts(
    frame: pd.DataFrame,
    *,
    min_document_frequency: int = 25,
    minimum_direction_models: int = 4,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Return stable token prevalence contrasts and reproducibility metadata."""

    vectorizer = CountVectorizer(
        lowercase=True,
        strip_accents="unicode",
        token_pattern=r"(?u)\b(?:[iI]|[a-zA-Z][a-zA-Z']+)\b",
        stop_words=list(STOP_WORDS | DOMAIN_STOP_WORDS),
        binary=True,
        min_df=min_document_frequency,
        dtype=np.uint8,
    )
    matrix = vectorizer.fit_transform(frame["visible_text"].astype(str))
    vocabulary = vectorizer.get_feature_names_out()
    models = tuple(sorted(frame["model_id"].unique()))
    rows: list[pd.DataFrame] = []
    pair_inventory: list[dict[str, object]] = []

    for identity in IDENTITIES:
        for contrast_id, left_condition, right_condition in CONTRASTS:
            model_deltas: list[np.ndarray] = []
            model_left_counts: list[np.ndarray] = []
            model_right_counts: list[np.ndarray] = []
            pair_counts: list[int] = []

            for model_id in models:
                model_frame = frame[frame["model_id"].eq(model_id)]
                left = model_frame[
                    model_frame["persona"].eq(_persona(identity, left_condition))
                ][["question_id", "matrix_row"]]
                right = model_frame[
                    model_frame["persona"].eq(_persona(identity, right_condition))
                ][["question_id", "matrix_row"]]
                paired = left.merge(
                    right,
                    on="question_id",
                    how="inner",
                    validate="one_to_one",
                    suffixes=("_left", "_right"),
                )
                n_pairs = len(paired)
                if not n_pairs:
                    continue
                left_counts = np.asarray(
                    matrix[paired["matrix_row_left"].to_numpy()].sum(axis=0)
                ).ravel().astype(np.float64)
                right_counts = np.asarray(
                    matrix[paired["matrix_row_right"].to_numpy()].sum(axis=0)
                ).ravel().astype(np.float64)
                delta_pp = (left_counts - right_counts) * (100.0 / n_pairs)
                model_deltas.append(delta_pp)
                model_left_counts.append(left_counts)
                model_right_counts.append(right_counts)
                pair_counts.append(n_pairs)
                pair_inventory.append(
                    {
                        "identity": identity,
                        "contrast_id": contrast_id,
                        "model_id": model_id,
                        "n_pairs": n_pairs,
                    }
                )

            if not model_deltas:
                continue
            deltas = np.vstack(model_deltas)
            left_counts = np.vstack(model_left_counts).sum(axis=0)
            right_counts = np.vstack(model_right_counts).sum(axis=0)
            mean_delta = deltas.mean(axis=0)
            n_positive = (deltas > 0).sum(axis=0)
            n_negative = (deltas < 0).sum(axis=0)
            direction_support = np.where(mean_delta >= 0, n_positive, n_negative)
            n_models = deltas.shape[0]
            total_presence = left_counts + right_counts
            eligible = (
                (total_presence >= min_document_frequency)
                & (direction_support >= min(minimum_direction_models, n_models))
                & (np.abs(mean_delta) > 0)
            )
            selected = np.flatnonzero(eligible)
            if not len(selected):
                continue

            total_pairs = int(sum(pair_counts))
            pooled_delta = (left_counts / total_pairs - right_counts / total_pairs) * 100
            contrast_frame = pd.DataFrame(
                {
                    "identity": identity,
                    "contrast_id": contrast_id,
                    "left_condition": left_condition,
                    "right_condition": right_condition,
                    "token": vocabulary[selected],
                    "mean_model_delta_pp": mean_delta[selected],
                    "pooled_delta_pp": pooled_delta[selected],
                    "abs_mean_model_delta_pp": np.abs(mean_delta[selected]),
                    "direction_support_models": direction_support[selected],
                    "n_models": n_models,
                    "direction_agreement": direction_support[selected] / n_models,
                    "left_paired_presence": left_counts[selected],
                    "right_paired_presence": right_counts[selected],
                    "total_pairs_across_models": total_pairs,
                }
            )
            contrast_frame["more_prevalent_in"] = np.where(
                contrast_frame["mean_model_delta_pp"] > 0,
                left_condition,
                right_condition,
            )
            rows.append(contrast_frame)

    result = pd.concat(rows, ignore_index=True)
    result = result.sort_values(
        ["contrast_id", "identity", "abs_mean_model_delta_pp", "token"],
        ascending=[True, True, False, True],
        kind="stable",
    ).reset_index(drop=True)
    metadata = {
        "status": "descriptive_exploratory_candidate",
        "pipeline_version": PIPELINE_VERSION,
        "correctness_filter_applied": False,
        "text_field": "visible_text",
        "hidden_reasoning_used": False,
        "weight": (
            "Equal-model mean of paired response-level token-prevalence "
            "difference in percentage points"
        ),
        "pairing_key": ["model_id", "question_id"],
        "token_presence": "binary within each response",
        "minimum_global_document_frequency": min_document_frequency,
        "minimum_same_direction_models": minimum_direction_models,
        "excluded_domain_and_latex_terms": sorted(DOMAIN_STOP_WORDS),
        "models": list(models),
        "n_available_responses": int(len(frame)),
        "vocabulary_size_after_min_df": int(len(vocabulary)),
        "pair_inventory": pair_inventory,
        "warning": "Font size is descriptive effect size, not statistical significance.",
    }
    return result, metadata


def _draw_cloud(
    ax: plt.Axes,
    terms: pd.DataFrame,
    *,
    title: str,
    color: str,
    weight_cap: float,
    seed: int,
    max_words: int = 18,
) -> None:
    """Draw a deterministic dependency-free word cloud with approximate packing."""

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(title, fontsize=10, fontweight="bold", color=color, pad=4)
    terms = terms.nlargest(max_words, "abs_mean_model_delta_pp")
    if terms.empty:
        ax.text(0.5, 0.5, "tidak ada token stabil", ha="center", va="center")
        return

    rng = np.random.default_rng(seed)
    boxes: list[tuple[float, float, float, float]] = []
    golden_angle = math.pi * (3 - math.sqrt(5))
    ranked = terms.reset_index(drop=True)

    for rank, row in ranked.iterrows():
        weight = min(float(row["abs_mean_model_delta_pp"]), weight_cap)
        scaled = math.sqrt(weight / max(weight_cap, 1e-9))
        font_size = 7.0 + 20.0 * scaled
        word = str(row["token"])
        # Approximate text bounds in axes coordinates.  The estimate is kept
        # conservative to avoid collisions without requiring canvas redraws.
        width = min(0.90, (0.0135 * len(word) * font_size) / 10.0)
        height = min(0.22, 0.0060 * font_size)
        phase = rng.uniform(0, 2 * math.pi)
        placed = False
        for attempt in range(700):
            radius = 0.018 * math.sqrt(attempt)
            angle = phase + attempt * golden_angle
            x = 0.5 + radius * math.cos(angle)
            y = 0.5 + 0.78 * radius * math.sin(angle)
            candidate = (
                x - width / 2,
                y - height / 2,
                x + width / 2,
                y + height / 2,
            )
            if (
                candidate[0] < 0.02
                or candidate[1] < 0.03
                or candidate[2] > 0.98
                or candidate[3] > 0.97
            ):
                continue
            overlap = any(
                candidate[0] < box[2] + 0.016
                and candidate[2] > box[0] - 0.016
                and candidate[1] < box[3] + 0.010
                and candidate[3] > box[1] - 0.010
                for box in boxes
            )
            if overlap:
                continue
            alpha = 1.0 - 0.35 * (rank / max(len(ranked) - 1, 1))
            ax.text(
                x,
                y,
                word,
                ha="center",
                va="center",
                fontsize=font_size,
                color=color,
                alpha=alpha,
                fontweight="bold" if rank < 5 else "normal",
            )
            boxes.append(candidate)
            placed = True
            break
        if not placed:
            continue


def _plot_contrast_group(
    weights: pd.DataFrame,
    contrasts: tuple[tuple[str, str, str], ...],
    output_path: Path,
    *,
    figure_title: str,
) -> None:
    figure_rows = len(contrasts)
    fig, axes = plt.subplots(figure_rows, 4, figsize=(25, 5.0 * figure_rows))
    axes = np.atleast_2d(axes)
    figure_weights = weights[weights["contrast_id"].isin([c[0] for c in contrasts])]
    weight_cap = float(figure_weights["abs_mean_model_delta_pp"].quantile(0.98))

    for row_index, (contrast_id, left_condition, right_condition) in enumerate(contrasts):
        for identity_index, identity in enumerate(IDENTITIES):
            subset = figure_weights[
                figure_weights["identity"].eq(identity)
                & figure_weights["contrast_id"].eq(contrast_id)
            ]
            left_terms = subset[subset["mean_model_delta_pp"] > 0]
            right_terms = subset[subset["mean_model_delta_pp"] < 0]
            left_col = identity_index * 2
            left_color = (
                COLORS[identity]
                if right_condition == "neutral"
                else COLORS["implicit"]
            )
            right_color = (
                COLORS["neutral"]
                if right_condition == "neutral"
                else COLORS["explicit"]
            )
            _draw_cloud(
                axes[row_index, left_col],
                left_terms,
                title=(
                    f"{identity.title()} {_condition_label(left_condition)}+\n"
                    f"lebih sering daripada {_condition_label(right_condition)}"
                ),
                color=left_color,
                weight_cap=weight_cap,
                seed=100 + row_index * 10 + identity_index * 2,
            )
            _draw_cloud(
                axes[row_index, left_col + 1],
                right_terms,
                title=(
                    f"{identity.title()} {_condition_label(right_condition)}+\n"
                    f"lebih sering daripada {_condition_label(left_condition)}"
                ),
                color=right_color,
                weight_cap=weight_cap,
                seed=101 + row_index * 10 + identity_index * 2,
            )

    fig.suptitle(
        figure_title
        + "\nUkuran kata = |selisih prevalensi|; paired by question; "
        "rata-rata setara per model; arah konsisten pada ≥4/6 model",
        fontsize=17,
        fontweight="bold",
        y=0.998,
    )
    fig.text(
        0.5,
        0.003,
        "Eksploratif/deskriptif: ukuran kata bukan p-value dan bukan bukti signifikansi.",
        ha="center",
        fontsize=10,
        color="#475569",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.975))
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_wordcloud_artifacts(run_dir: Path, output_dir: Path) -> pd.DataFrame:
    run_dir = Path(run_dir).resolve()
    output_dir = Path(output_dir).resolve()
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    responses = load_available_responses(run_dir)
    weights, metadata = compute_lexical_contrasts(responses)
    weights.to_csv(
        output_dir / "lexical_contrast_token_weights.csv",
        index=False,
        encoding="utf-8",
    )
    (output_dir / "lexical_contrast_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    versus_neutral = tuple(c for c in CONTRASTS if c[2] == "neutral")
    versus_explicit = tuple(c for c in CONTRASTS if c[2] == "explicit")
    _plot_contrast_group(
        weights,
        versus_neutral,
        image_dir / "05_contrastive_wordcloud_personas_vs_neutral.png",
        figure_title="Contrastive word clouds: E/I1/I2/I3 versus neutral",
    )
    _plot_contrast_group(
        weights,
        versus_explicit,
        image_dir / "06_contrastive_wordcloud_implicit_vs_explicit.png",
        figure_title="Contrastive word clouds: implicit personas versus explicit",
    )
    return weights


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    weights = build_wordcloud_artifacts(args.run_dir, args.output_dir)
    print(f"[DONE] {len(weights):,} stable lexical token weights written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
