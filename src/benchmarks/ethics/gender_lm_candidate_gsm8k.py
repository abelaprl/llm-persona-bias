"""Question-led GSM8K figures using frozen statistics and marker counts only."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from .gender_lm_reference_data import COLORS, MODELS, MODEL_LABELS, PERSONAS, ROOT, scenario


BUNDLE = ROOT / "result/analyses/linguistic_markers/runs/gender_gsm8k_bab4_visual_revision_v3_direct_gender_r2_20260817"
SOURCE = ROOT / "result/analyses/linguistic_markers/runs/gender_gsm8k_full_source_aligned_v1_2_1_20260817"
WIDTH = 16 / 2.54
STATUS = "candidate_pending_manual_precision_recall_audit"
SCENARIOS = ["E", "I1", "I2", "I3"]
OUTCOME_LABELS = {
    "word_count": "Jumlah kata",
    "avg_sentence_length": "Panjang kalimat",
    "ttr_300": "TTR-300",
    "organization_explicit_step": "Langkah eksplisit",
    "organization_bullet": "Bullet",
    "organization_section_header": "Judul bagian",
    "stance_hedge": "Hedge",
    "stance_booster": "Booster",
    "stance_assumption": "Asumsi eksplisit",
    "stance_self_correction": "Self-correction",
    "connective_causal": "Konektif kausal",
    "connective_contrastive": "Konektif kontras",
    "connective_conditional": "Konektif kondisional",
    "self_reference_first_person": "Orang pertama",
    "self_reference_inclusive_we": "Inclusive we",
    "engagement_reader_directive": "Direktif pembaca",
}
PRIMARY = list(OUTCOME_LABELS)
MARKERS = PRIMARY[3:]
PERSONA_LABELS = {p: "N" if p == "neutral" else ("M-" if p.startswith("man_") else "W-") + scenario(p)
                  for p in PERSONAS}
BUNDLE_TABLES = [
    "response_counts_model_persona.csv", "direct_gender_paired_numeric.csv",
    "lexical_contrast_token_weights.csv", "gender_pronoun_class_prevalence.csv",
    "paired_gender_pronoun_direct.csv", "opening_composition.csv",
    "gendered_term_model_persona.csv", "direct_gender_availability.csv",
    "direct_gender_opening_multinomial.csv", "direct_gender_paired_binary.csv",
    "direct_gender_correctness_comparison.csv", "correctness_sensitivity_comparison.csv",
    "paired_opening_multinomial.csv", "manual_audit_status.csv", "visible_surface_audit.csv",
]
SOURCE_TABLES = ["condition_descriptives.csv", "marker_registry.csv", "paired_numeric.csv", "paired_binary.csv"]


def input_paths() -> list[Path]:
    """Return every frozen data file read by build for provenance hashing."""
    return ([BUNDLE / "manifest.json", SOURCE / "manifest.json"]
            + [BUNDLE / name for name in BUNDLE_TABLES]
            + [SOURCE / name for name in SOURCE_TABLES]
            + [SOURCE / "response_records.csv"])


def prepare_direct_numeric(source: pd.DataFrame) -> pd.DataFrame:
    """Preserve Man-minus-Woman rank effects; never use median-delta CIs as rank CIs."""
    data = source.loc[source.analysis_role.eq("primary")].copy()
    if not (data.left_identity.eq("man") & data.right_identity.eq("woman")).all():
        raise ValueError("Unexpected GSM8K source direction")
    if data.duplicated(["model_id", "condition", "outcome_id"]).any():
        raise ValueError("Duplicate direct numeric comparison")
    data["panel_scenario"] = data.left_persona.map(scenario)
    if not data.right_persona.map(scenario).eq(data.panel_scenario).all():
        raise ValueError("Direct contrast must use the same presentation scenario")
    data["plot_value"] = data.rank_biserial
    data["plot_ci_low"] = np.nan
    data["plot_ci_high"] = np.nan
    data["plot_unit"] = "rank_biserial_man_minus_woman"
    data["source_ci_estimand"] = "median_paired_delta"
    data["sparse_support"] = data.n_nonzero.lt(20)
    data["support_label"] = data.n_nonzero.astype(str) + "/" + data.n_pairs.astype(str)
    return data


def prepare_coverage(source: pd.DataFrame) -> pd.DataFrame:
    data = source.copy()
    if not data.target.eq(data.n_records + data.n_source_gap).all():
        raise ValueError("Coverage target does not reconcile with records and source gap")
    data["text_available_pct"] = 100 * data.n_text_available / data.target
    data["text_unavailable_pct"] = 100 * (data.target - data.n_text_available) / data.target
    data["persona_label"] = data.persona.map(PERSONA_LABELS)
    return data


def aggregate_marker_counts(frame: pd.DataFrame, markers: list[str]) -> pd.DataFrame:
    """Summarize already-computed counts without treating unavailable responses as zeros."""
    available = frame.loc[frame.text_available.eq(True)].copy()
    rows = []
    for (model_id, persona_id), group in available.groupby(["model_id", "persona"], sort=False):
        for marker in markers:
            counts = group[marker + "__count"]
            densities = group[marker + "__density"]
            if counts.isna().any() or densities.isna().any():
                raise ValueError("Available response has missing marker measurement")
            total_words = group.word_count.sum()
            rows.append({
                "model_id": model_id, "persona": persona_id, "outcome_id": marker,
                "n_available": len(group), "responses_with_marker": int(counts.gt(0).sum()),
                "raw_occurrences": int(counts.sum()), "total_words": int(total_words),
                "prevalence_pct": 100 * counts.gt(0).mean(),
                "mean_density_per_100_words": densities.mean(),
                "pooled_density_per_100_words": 100 * counts.sum() / total_words if total_words else np.nan,
            })
    return pd.DataFrame(rows)


def select_lexical_examples(source: pd.DataFrame) -> pd.DataFrame:
    """Select descriptive examples; original tests and denominators stay in the table."""
    data = source.sort_values(["identity", "contrast_id", "abs_mean_model_delta_pp", "token"],
                              ascending=[True, True, False, True]).copy()
    data["selection_rank"] = data.groupby(["identity", "contrast_id"]).cumcount() + 1
    data["selection_role"] = "descriptive_top5_not_inference"
    return data.loc[data.selection_rank.le(5)].copy()


def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 9, "figure.titlesize": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "svg.fonttype": "none",
    })


def _model_legend(fig: plt.Figure, bottom: float = 0.01) -> None:
    handles = [Line2D([0], [0], marker="o", linestyle="", color=COLORS[m], label=MODEL_LABELS[m], markersize=5)
               for m in MODELS]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, bottom),
               frameon=False, columnspacing=1, handletextpad=0.4)


def _grid(ax: plt.Axes, axis: str = "x") -> None:
    ax.grid(axis=axis, color="#E2E8F0", linewidth=0.5)
    ax.set_axisbelow(True)


def rank_page(data: pd.DataFrame, outcomes: list[str], title: str) -> plt.Figure:
    """Four scenario columns; model y offsets preserve every source x coordinate."""
    _style()
    fig, axes = plt.subplots(1, 4, figsize=(WIDTH, max(4.8, len(outcomes) * 0.98 + 1.5)), sharey=True)
    y_lookup = {outcome: len(outcomes) - 1 - i for i, outcome in enumerate(outcomes)}
    offsets = np.linspace(0.24, -0.24, len(MODELS))
    for ax, panel_scenario in zip(axes, SCENARIOS):
        panel = data.loc[data.panel_scenario.eq(panel_scenario) & data.outcome_id.isin(outcomes)]
        for model, offset in zip(MODELS, offsets):
            points = panel.loc[panel.model_id.eq(model)]
            for _, point in points.iterrows():
                ax.scatter(point.plot_value, y_lookup[point.outcome_id] + offset, s=23,
                           facecolors="none" if point.sparse_support else COLORS[model],
                           edgecolors=COLORS[model], linewidths=0.9, clip_on=False, zorder=3)
        for outcome in outcomes:
            group = panel.loc[panel.outcome_id.eq(outcome)]
            if not len(group):
                continue
            support = (f"{int(group.n_nonzero.min())}–{int(group.n_nonzero.max())}\n"
                       f"/ {int(group.n_pairs.min())}–{int(group.n_pairs.max())}")
            ax.text(0, y_lookup[outcome] - 0.36, support, ha="center", va="top", fontsize=9,
                    linespacing=1.0, color="#475569")
        ax.axvline(0, color="#94A3B8", linewidth=0.8)
        ax.set_xlim(-1, 1)
        ax.set_xticks([-1, 0, 1])
        ax.set_ylim(-0.72, len(outcomes) - 0.5)
        ax.set_title(panel_scenario, fontweight="bold")
        ax.set_yticks(list(y_lookup.values()), [OUTCOME_LABELS[o] for o in outcomes])
        ax.tick_params(axis="y", length=0)
        _grid(ax)
    fig.suptitle(title + "\nEnam model ditampilkan terpisah", y=0.985, fontsize=11)
    fig.supxlabel("Rank-biserial Man − Woman", y=0.09, fontsize=9)
    fig.text(0.5, 0.065, "Angka: rentang n_nonzero / n_pairs; titik kosong: n_nonzero < 20",
             ha="center", fontsize=9)
    _model_legend(fig)
    fig.subplots_adjust(left=0.27, right=0.985, top=0.90, bottom=0.16, wspace=0.28)
    return fig


def _coverage_figure(data: pd.DataFrame) -> plt.Figure:
    matrix = data.pivot(index="model_id", columns="persona", values="text_available_pct").reindex(index=MODELS, columns=PERSONAS)
    counts = data.pivot(index="model_id", columns="persona", values="n_text_available").reindex(index=MODELS, columns=PERSONAS)
    fig, ax = plt.subplots(figsize=(WIDTH, 4.3))
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    for row in range(6):
        for col in range(9):
            value = matrix.iloc[row, col]
            ax.text(col, row, f"{int(counts.iloc[row, col])}\n{value:.1f}%", ha="center", va="center",
                    color="white" if value > 65 else "#172554", fontsize=9)
    ax.set_xticks(range(9), [PERSONA_LABELS[p] for p in PERSONAS], rotation=45, ha="right")
    ax.set_yticks(range(6), [MODEL_LABELS[m] for m in MODELS])
    ax.tick_params(length=0)
    for label, model in zip(ax.get_yticklabels(), MODELS):
        label.set_color(COLORS[model])
    ax.set_title("Teks yang tersedia untuk pengukuran", pad=12)
    fig.subplots_adjust(left=0.24, right=0.99, top=0.86, bottom=0.19)
    return fig


def _verbosity_figure(data: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(3, 2, figsize=(WIDTH, 9.4))
    for ax, model in zip(axes.flat, MODELS):
        panel = data.loc[data.model_id.eq(model) & data.outcome_id.eq("word_count")].set_index("persona").reindex(PERSONAS)
        y = np.arange(9)[::-1]
        ax.hlines(y, panel.q1, panel.q3, color=COLORS[model], linewidth=1.3)
        ax.scatter(panel["median"], y, color=COLORS[model], s=20, zorder=3)
        ax.set_yticks(y, [PERSONA_LABELS[p] for p in PERSONAS])
        upper = float(50 * np.ceil(1.05 * panel.q3.max() / 50))
        ax.set_xlim(0, upper)
        ax.set_xticks([0, upper / 2, upper])
        ax.set_title(MODEL_LABELS[model], color=COLORS[model])
        ax.set_xlabel("Jumlah kata: median dan IQR")
        _grid(ax)
    fig.suptitle("Apakah panjang penjelasan berubah?", y=0.985)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.93, bottom=0.06, hspace=0.5, wspace=0.4)
    return fig


def _profile_figure(data: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(WIDTH, 7.7))
    positions = {marker: len(MARKERS) - 1 - index for index, marker in enumerate(MARKERS)}
    for model, offset in zip(MODELS, np.linspace(0.28, -0.28, 6)):
        panel = data.loc[data.model_id.eq(model) & data.persona.eq("neutral") & data.outcome_id.isin(MARKERS)]
        ax.scatter(panel.prevalence_pct, panel.outcome_id.map(positions) + offset, color=COLORS[model], s=20)
    ax.set_yticks(list(positions.values()), [OUTCOME_LABELS[m] for m in MARKERS])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Respons yang memuat penanda (%)")
    ax.set_title("Profil absolut pada skenario netral (N)", pad=12)
    _grid(ax)
    _model_legend(fig)
    fig.subplots_adjust(left=0.29, right=0.98, top=0.92, bottom=0.15)
    return fig


def _lexical_figure(selected: pd.DataFrame) -> plt.Figure:
    # The main page shows E-N; every I1/I2/I3 comparison remains in the full table.
    data = selected.loc[selected.contrast_id.eq("explicit_vs_neutral")]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 4.3), sharex=True)
    bound = max(1, float(np.ceil(1.1 * data.mean_model_delta_pp.abs().max())))
    for ax, identity in zip(axes, ["man", "woman"]):
        panel = data.loc[data.identity.eq(identity)].sort_values("mean_model_delta_pp")
        y = np.arange(len(panel))
        ax.hlines(y, 0, panel.mean_model_delta_pp, color="#64748B", linewidth=1.3)
        ax.scatter(panel.mean_model_delta_pp, y, color="#334155", s=30)
        ax.set_yticks(y, panel.token)
        ax.axvline(0, color="#CBD5E1", linewidth=0.8)
        ax.set_title(identity.title() + ": E − N")
        ax.set_xlabel("Delta prevalensi (pp)")
        ax.set_xlim(-bound, bound)
        ax.set_xticks([-bound, 0, bound])
        _grid(ax)
    fig.suptitle("Ungkapan aktual yang berubah pada skenario E", y=0.985)
    fig.text(0.5, 0.03, "Top 5 deskriptif per persona; rata-rata setara enam model", ha="center", fontsize=9)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.82, bottom=0.23, wspace=0.65)
    return fig


def _pronoun_profile_figure(data: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(3, 2, figsize=(WIDTH, 8.9), sharey=True)
    classes = [("masculine_pronoun", "o", "Maskulin"), ("feminine_pronoun", "^", "Feminin")]
    for ax, model in zip(axes.flat, MODELS):
        for kind, symbol, _ in classes:
            panel = data.loc[data.model_id.eq(model) & data.pronoun_class.eq(kind)].set_index("persona").reindex(PERSONAS)
            ax.plot(range(9), panel.response_prevalence_pp, marker=symbol, markersize=3.8,
                    linewidth=0.8, linestyle="-" if symbol == "o" else "--", color=COLORS[model])
        ax.set_xticks(range(9), [PERSONA_LABELS[p] for p in PERSONAS], rotation=90)
        ax.set_title(MODEL_LABELS[model], color=COLORS[model])
        ax.set_ylim(bottom=0)
        _grid(ax, "y")
    axes[1, 0].set_ylabel("Respons yang memuat pronomina (%)")
    handles = [Line2D([0], [0], marker=symbol, linestyle="-" if symbol == "o" else "--",
                      color="#475569", label=label) for _, symbol, label in classes]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Pronomina dalam penjelasan matematika", y=0.985)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.92, bottom=0.13, hspace=0.70, wspace=0.25)
    return fig


def _pronoun_direct_figure(data: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(2, 4, figsize=(WIDTH, 5.6), sharex=True, sharey=True)
    limit = max(1, float(np.ceil(data[["plot_ci_low", "plot_ci_high"]].abs().max().max())))
    for row, (kind, label) in enumerate([("masculine_pronoun", "Maskulin"), ("feminine_pronoun", "Feminin")]):
        for col, condition in enumerate(SCENARIOS):
            ax = axes[row, col]
            for i, model in enumerate(MODELS):
                point = data.loc[data.model_id.eq(model) & data.panel_scenario.eq(condition) & data.pronoun_class.eq(kind)].iloc[0]
                y = 5 - i
                ax.plot([point.plot_ci_low, point.plot_ci_high], [y, y], color=COLORS[model], linewidth=0.9)
                ax.scatter(point.plot_value, y, s=20, color=COLORS[model], zorder=3)
            ax.axvline(0, color="#94A3B8", linewidth=0.7)
            ax.set_xlim(-limit, limit)
            ax.set_xticks([-limit, 0, limit])
            ax.set_yticks(range(6))
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)
            ax.set_title(condition if row == 0 else "")
            _grid(ax)
        axes[row, 0].set_ylabel(label)
    fig.suptitle("Selisih pronomina Man − Woman", y=0.985)
    fig.supxlabel("Delta prevalensi (pp), CI bootstrap berpasangan 95%", y=0.14, fontsize=9)
    _model_legend(fig, 0.01)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.87, bottom=0.23, hspace=0.35, wspace=0.35)
    return fig


def _opening_figure(data: pd.DataFrame) -> plt.Figure:
    categories = [c for c in data.opening_category.unique() if data.loc[data.opening_category.eq(c), "n"].sum() > 0]
    palette = dict(zip(categories, plt.get_cmap("tab10").colors))
    fig, axes = plt.subplots(3, 2, figsize=(WIDTH, 9.8), sharex=True)
    for ax, model in zip(axes.flat, MODELS):
        panel = data.loc[data.model_id.eq(model)].pivot(index="persona", columns="opening_category", values="proportion").reindex(PERSONAS)
        left = np.zeros(len(PERSONAS))
        y = np.arange(len(PERSONAS))[::-1]
        for category in categories:
            width = 100 * panel[category].fillna(0).to_numpy()
            ax.barh(y, width, left=left, height=0.72, color=palette[category], label=category)
            left += width
        ax.set_yticks(y, [PERSONA_LABELS[p] for p in PERSONAS])
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 50, 100])
        ax.set_title(MODEL_LABELS[model], color=COLORS[model])
        ax.set_xlabel("Komposisi pembuka (%)")
    handles = [Line2D([0], [0], marker="s", linestyle="", color=palette[c], label=c) for c in categories]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, columnspacing=0.6, handletextpad=0.25)
    fig.suptitle("Bagaimana penjelasan dibuka?", y=0.985)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.93, bottom=0.18, hspace=0.45, wspace=0.4)
    return fig


def build(root: Path) -> list[dict]:
    """Write candidate panels through the shared geometry-checked exporter."""
    from .gender_lm_candidate_common import save_panel

    _style()
    manifest = json.loads((BUNDLE / "manifest.json").read_text(encoding="utf-8"))
    source_manifest = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != STATUS or manifest["primary_correctness_filter_applied"]:
        raise ValueError("Unexpected source status or correctness selection")
    if Path(manifest["source_run"]).name != SOURCE.name:
        raise ValueError("Frozen source run has changed")
    if manifest["hidden_reasoning_used"]:
        raise ValueError("Hidden reasoning is outside this presentation")
    tables = {name: pd.read_csv(BUNDLE / name) for name in BUNDLE_TABLES}
    sources = {name: pd.read_csv(SOURCE / name) for name in SOURCE_TABLES}
    coverage = prepare_coverage(tables["response_counts_model_persona.csv"])
    numeric = prepare_direct_numeric(tables["direct_gender_paired_numeric.csv"])
    if set(numeric.outcome_id) != set(PRIMARY) or len(numeric) != 384:
        raise ValueError("Expected all 16 primary indicators, six models and four scenarios")
    descriptive = sources["condition_descriptives.csv"].drop_duplicates(["model_id", "persona", "outcome_id"]).copy()
    descriptive = descriptive.loc[descriptive.outcome_id.isin(PRIMARY)]
    raw_columns = ["model_id", "persona", "text_available", "word_count"] + [m + suffix for m in MARKERS for suffix in ["__count", "__density"]]
    raw = pd.read_csv(SOURCE / "response_records.csv", usecols=raw_columns)
    counts = aggregate_marker_counts(raw, MARKERS)
    del raw
    profile = descriptive.merge(counts.drop(columns="n_available"), on=["model_id", "persona", "outcome_id"], how="left", validate="one_to_one")
    profile = profile.merge(sources["marker_registry.csv"][["marker_id", "theoretical_source", "validation_status"]],
                            left_on="outcome_id", right_on="marker_id", how="left", validate="many_to_one")
    profile["measurement_unit"] = np.where(profile.outcome_id.isin(MARKERS), "per_100_words", "native_numeric_unit")
    records = []

    def add(stem, title, question, caption, fig, table, section="main"):
        record = save_panel(root, "gsm8k_" + stem, title, question, caption, fig, table, section=section)
        record["source_status"] = STATUS
        records.append(record)
        return record

    totals = coverage[["target", "n_records", "n_text_available", "n_missing_text", "n_technical_only", "n_source_gap"]].sum()
    coverage_record = add("01_coverage", "Ketersediaan teks", "Berapa respons yang dapat diukur pada tiap persona?",
        f"Setiap sel menampilkan jumlah teks tersedia dan persentase terhadap target 1.319 soal. "
        f"Dari {int(totals.target):,} slot, {int(totals.n_records):,} rekaman dan {int(totals.n_text_available):,} teks tersedia. "
        f"Terdapat {int(totals.n_missing_text)} missing text, {int(totals.n_technical_only)} technical-only, dan {int(totals.n_source_gap)} source gap. "
        f"Seluruh respons tersedia masuk analisis, tanpa filter jawaban benar. Status: {STATUS}.",
        _coverage_figure(coverage), coverage)
    coverage_record["note"] = "N = netral; E = eksplisit; I1/I2/I3 = tiga narasi implisit; M = Man; W = Woman. Status kandidat belum menggantikan audit precision/recall manual."
    add("02_verbosity", "Panjang penjelasan", "Apakah persona mengubah panjang penjelasan pada model yang sama?",
        "Titik adalah median dan garis adalah IQR, bukan confidence interval. Skala jumlah kata berbeda antarpanel agar variasi dalam model terbaca. "
        "CSV memuat n_available untuk seluruh 54 kombinasi model dan persona; teks tidak tersedia tidak diberi panjang nol.",
        _verbosity_figure(descriptive), descriptive.loc[descriptive.outcome_id.eq("word_count")])
    add("03_absolute_profile", "Profil absolut penanda", "Seberapa umum 13 penanda utama pada skenario netral?",
        "Persentase respons dengan minimal satu kemunculan menjelaskan kelangkaan penanda sebelum effect size dibaca. "
        "CSV memuat seluruh 16 indikator × sembilan persona × enam model, termasuk jumlah kemunculan mentah, denominator, "
        "rata-rata densitas per respons per 100 kata, dan densitas gabungan per 100 kata. Tiga indikator panjang/leksikal memakai satuan asalnya. "
        "Sumber kamus dan status validasi tetap mengikuti marker_registry.csv; tidak ada kamus baru.",
        _profile_figure(profile), profile)
    for page, outcomes in enumerate([PRIMARY[:8], PRIMARY[8:]], start=1):
        selected = numeric.loc[numeric.outcome_id.isin(outcomes)].copy()
        record = add(f"0{3 + page}_direct_effects_page{page}", f"Efek gender langsung — halaman {page}",
            "Apakah arah perbedaan Man–Woman konsisten pada E, I1, I2, dan I3?",
            "Nilai positif berarti indikator lebih tinggi pada Man; negatif pada Woman. Setiap model memiliki posisi vertikal sendiri, "
            "tanpa pergeseran x atau median gabungan. Angka di bawah kelompok titik adalah rentang n_nonzero dan n_pairs enam model. "
            "Titik kosong menandai n_nonzero < 20 sebagai petunjuk deskriptif, bukan ambang uji. "
            "Tabel dukungan menyajikan denominator dan p Holm tiap titik. CI delta median dalam sumber tidak berlaku untuk rank-biserial.",
            rank_page(selected, outcomes, f"Perbedaan gender langsung ({page}/2)"), selected)
        support_path = root / "tables" / f"gsm8k_direct_support_page{page}.csv"
        if support_path.exists():
            raise FileExistsError(support_path)
        support = selected[["model_id", "outcome_id", "panel_scenario", "rank_biserial", "n_nonzero", "n_pairs", "p_holm", "sparse_support"]].copy()
        support["model"] = support.model_id.map(MODEL_LABELS)
        support["indicator"] = support.outcome_id.map(OUTCOME_LABELS)
        support.to_csv(support_path, index=False)
        record["support_csv"] = support_path.relative_to(root).as_posix()
        record["note"] = ("Contoh kelangkaan: DeepSeek, E, rujukan orang pertama mempunyai rank-biserial −1 dengan "
                          "n_nonzero = 1 dari n_pairs = 1.298 dan p Holm = 1. Nilai ekstrem ini tidak membuktikan efek gender besar.")
    selected_lexical = select_lexical_examples(tables["lexical_contrast_token_weights.csv"])
    lexical_record = add("06_actual_expressions", "Ungkapan aktual", "Token apa yang menyertai perubahan dari N ke E?",
        "Lima token per identitas dipilih menurut besar absolut delta prevalensi rata-rata setara enam model pada E − N. "
        "Pilihan ini deskriptif, bukan inferensi berdasarkan token terpilih. CSV mempertahankan top 5 untuk seluruh 14 kombinasi "
        "identitas dan kontras, termasuk I1/I2/I3 terhadap N dan E, beserta denominator dan kesepakatan arah antarmodel. "
        "Token soal dapat ikut terbawa; perbedaan kata tidak otomatis berarti stereotip gender.",
        _lexical_figure(selected_lexical), selected_lexical)
    pronouns = tables["gender_pronoun_class_prevalence.csv"]
    add("07_pronoun_profile", "Profil pronomina", "Apakah penggunaan pronomina berubah menurut persona?",
        "Prevalensi dihitung dari seluruh teks terlihat yang tersedia. Bentuk titik membedakan kelas pronomina; warna membedakan model. "
        "Pronomina dapat merujuk tokoh dalam soal GSM8K, sehingga tidak langsung mengukur cara model menyapa pengguna. "
        "Jumlah respons yang memuat kelas dan denominator dipertahankan dalam CSV.",
        _pronoun_profile_figure(pronouns), pronouns)
    direct_pronouns = tables["paired_gender_pronoun_direct.csv"].copy()
    direct_pronouns["panel_scenario"] = direct_pronouns.left_persona.map(scenario)
    direct_pronouns["plot_value"] = 100 * direct_pronouns.prevalence_delta
    direct_pronouns["plot_ci_low"] = 100 * direct_pronouns.ci_low
    direct_pronouns["plot_ci_high"] = 100 * direct_pronouns.ci_high
    add("08_pronoun_direct", "Perbandingan pronomina langsung", "Apakah pronomina berbeda antara Man dan Woman pada soal yang sama?",
        "Delta prevalensi Man − Woman dalam percentage points dan CI bootstrap berpasangan 95% berasal langsung dari sumber. "
        "CSV memuat n_pairs, discordant, p Holm dan kedua prevalensi; uji sumber adalah McNemar exact. "
        "Perbandingan E/I1/I2/I3 tetap terpisah; interpretasi referen tokoh soal tetap diperlukan.",
        _pronoun_direct_figure(direct_pronouns), direct_pronouns)
    opening = tables["opening_composition.csv"].drop_duplicates(["model_id", "persona", "opening_category"])
    opening_record = add("09_opening_detail", "Komposisi pembuka", "Apakah pola pembuka yang dominan berubah antarpersona?",
        "Komposisi mencakup semua teks tersedia per persona. Kategori dengan jumlah nol di semua kelompok tidak diberi warna, "
        "tetapi tetap ada pada CSV. Skenario N yang tersimpan pada dua identitas di sumber dihitung sekali. "
        "Uji multinomial berpasangan sumber disertakan sebagai tabel tambahan; persentase komposisi bukan hitungan independen antarkategori.",
        _opening_figure(opening), opening, section="detail")

    supplemental = [
        ("numeric_neutral_implicit", "Seluruh efek numerik terhadap N dan E", sources["paired_numeric.csv"]),
        ("presence_neutral_implicit", "Seluruh perbandingan presence terhadap N dan E", sources["paired_binary.csv"]),
        ("direct_presence", "Seluruh perbandingan presence Man − Woman", tables["direct_gender_paired_binary.csv"]),
        ("direct_availability", "Uji ketersediaan dan technical-only Man − Woman", tables["direct_gender_availability.csv"]),
        ("direct_sensitivity", "Sensitivitas both-correct untuk kontras langsung", tables["direct_gender_correctness_comparison.csv"]),
        ("sensitivity_neutral_implicit", "Sensitivitas both-correct terhadap N dan E", tables["correctness_sensitivity_comparison.csv"]),
        ("opening_tests", "Uji pembuka berpasangan terhadap N dan E", tables["paired_opening_multinomial.csv"]),
        ("opening_direct_tests", "Uji pembuka berpasangan Man − Woman", tables["direct_gender_opening_multinomial.csv"]),
        ("lexical_full", "Seluruh token dan kontras leksikal", tables["lexical_contrast_token_weights.csv"]),
        ("gender_terms_full", "Istilah gender: hitungan mentah dan denominator", tables["gendered_term_model_persona.csv"]),
        ("marker_registry", "Definisi dan sumber kamus yang dibekukan", sources["marker_registry.csv"]),
        ("manual_audit_status", "Status audit manual sumber", tables["manual_audit_status.csv"]),
        ("surface_audit", "Audit pemisahan teks terlihat", tables["visible_surface_audit.csv"]),
    ]
    supplemental_links = []
    for stem, title, table in supplemental:
        path = root / "tables" / f"gsm8k_{stem}.csv"
        if path.exists():
            raise FileExistsError(path)
        table.to_csv(path, index=False)
        supplemental_links.append({"title": title, "csv": path.relative_to(root).as_posix()})
    coverage_record["supplemental_tables"] = supplemental_links
    coverage_record["source_pipeline_version"] = source_manifest.get("pipeline_version", manifest["source_pipeline_version"])
    lexical_record["note"] = "Pilihan top 5 tidak menjadi keluarga uji baru dan tidak menggantikan seluruh tabel token sumber."
    opening_record["note"] = "Detail pembuka tetap bagian paket kandidat, meskipun ditempatkan setelah alur utama."
    return records
