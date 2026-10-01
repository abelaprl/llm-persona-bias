"""Build a non-destructive Bab VI visual bundle for Gender ETHICS markers.

The frozen statistical source stores direct contrasts as Woman minus Man.
This presentation layer reverses only the display columns to Man minus Woman so
the direction matches the Gender GSM8K chapter. Source columns are retained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.benchmarks.ethics import gender_lm_reference_data as data_contract
from src.benchmarks.ethics.gender_lm_reference_plots import LABELS, coverage
from src.benchmarks.ethics.gender_lm_diagnostic_data import direct_rows, support_summary


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "result/ETHICS/reports/gender_lm_v2_candidate_20260916_r3"
DEFAULT_OUTPUT = ROOT / "result/ETHICS/reports/gender_lm_bab6_viz_20261002_r5"
MODELS = data_contract.MODELS
MODEL_LABELS = data_contract.MODEL_LABELS
COLORS = data_contract.COLORS
SCENARIOS = data_contract.SCENARIOS


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def orient_direct(rows: pd.DataFrame, measurement: str) -> pd.DataFrame:
    """Add Man-minus-Woman display columns without changing source estimates."""
    required = {
        "left", "right", "effect_direction", "measurement", "effect",
        "ci_low", "ci_high", "matched_rank_biserial", "adjusted_p",
    }
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"Missing direct-contrast columns: {sorted(missing)}")
    if measurement not in {"density", "presence"}:
        raise ValueError(f"Unsupported measurement: {measurement}")
    if rows.empty or not rows.measurement.eq(measurement).all():
        raise ValueError("Expected one nonempty measurement block")
    if not rows.effect_direction.eq("left_minus_right").all():
        raise ValueError("Unexpected source direction")
    if not (rows.left.str.startswith("woman_").all()
            and rows.right.str.startswith("man_").all()):
        raise ValueError("Direct source must be Woman minus Man")

    out = rows.copy(deep=True)
    if measurement == "density":
        out["display_effect"] = -out.matched_rank_biserial
        # Source CI estimates the paired mean density difference, not rank-biserial.
        out["display_ci_low"] = np.nan
        out["display_ci_high"] = np.nan
        out["display_unit"] = "matched_rank_biserial"
    else:
        out["display_effect"] = -out.effect
        out["display_ci_low"] = -out.ci_high
        out["display_ci_high"] = -out.ci_low
        out["display_unit"] = "percentage_points"
    out["display_direction"] = "Man minus Woman"
    out["panel_scenario"] = out.left.map(data_contract.scenario)
    if not out.panel_scenario.isin(SCENARIOS).all():
        raise ValueError("Direct comparison contains an unexpected scenario")
    return out


def select_direct(
    comparisons: pd.DataFrame,
    marker_ids: list[str],
    measurement: str,
) -> pd.DataFrame:
    rows = comparisons.loc[
        comparisons.scope.eq("overall")
        & comparisons.marker_id.isin(marker_ids)
        & comparisons.measurement.eq(measurement)
    ].copy()
    left_scenario = rows.left.map(data_contract.scenario)
    right_scenario = rows.right.map(data_contract.scenario)
    rows = rows.loc[
        rows.left.str.startswith("woman_")
        & rows.right.str.startswith("man_")
        & left_scenario.eq(right_scenario)
    ].copy()
    keys = ["model_id", "marker_id", "measurement", "left", "right"]
    if rows.duplicated(keys).any():
        raise ValueError("Duplicate direct comparison")
    expected = len(MODELS) * len(SCENARIOS) * len(marker_ids)
    if len(rows) != expected:
        raise ValueError(f"Incomplete direct block: {len(rows)} rows, expected {expected}")
    return orient_direct(rows, measurement)


def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.titlesize": 17,
        "svg.fonttype": "none",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


def plot_direct_effects(
    rows: pd.DataFrame,
    marker_ids: list[str],
    title: str,
    measurement: str,
):
    """Render a GSM8K-style four-panel Man-minus-Woman effect plot."""
    _style()
    if rows.display_direction.nunique() != 1 or rows.display_direction.iloc[0] != "Man minus Woman":
        raise ValueError("Plot requires Man-minus-Woman display rows")
    if rows.display_unit.nunique() != 1:
        raise ValueError("Mixed display units are not allowed")

    numeric = measurement == "density"
    if numeric:
        bound = 1.0
    else:
        values = rows[["display_effect", "display_ci_low", "display_ci_high"]].to_numpy(float).ravel()
        values = values[np.isfinite(values)]
        bound = max(1.0, float(np.max(np.abs(values))) * 1.06)
    has_na = rows.display_effect.isna().any()
    right_limit = 1.22 * bound if has_na else 1.04 * bound

    height = max(6.8, 0.39 * len(marker_ids) + 2.7)
    fig, axes = plt.subplots(1, 4, figsize=(18, height), sharey=True)
    fig.subplots_adjust(left=.185, right=.985, top=.81, bottom=.235, wspace=.14)
    offsets = np.linspace(-.22, .22, len(MODELS))

    for panel_index, (ax, scenario) in enumerate(zip(axes, SCENARIOS)):
        block = rows.loc[rows.panel_scenario.eq(scenario)]
        ax.axvline(0, color="#64748b", linewidth=1)
        if has_na:
            ax.axvspan(1.055 * bound, right_limit, color="#f1f5f9", zorder=0)
            ax.text(1.135 * bound, -.72, "NA", ha="center", color="#64748b", fontsize=8)
        for y in np.arange(.5, len(marker_ids) - .5):
            ax.axhline(y, color="#edf2f7", linewidth=.8, zorder=0)

        for model_index, model in enumerate(MODELS):
            model_rows = block.loc[block.model_id.eq(model)].set_index("marker_id").reindex(marker_ids)
            y = np.arange(len(marker_ids)) + offsets[model_index]
            available = model_rows.display_effect.notna().to_numpy()
            significant = model_rows.adjusted_p.lt(.05).fillna(False).to_numpy()
            for row_index, marker in enumerate(marker_ids):
                if not available[row_index]:
                    ax.scatter(1.135 * bound, y[row_index], marker="x", s=24,
                               color=COLORS[model], linewidths=1, zorder=3)
                    continue
                value = float(model_rows.loc[marker, "display_effect"])
                point_marker = "^" if significant[row_index] else "o"
                if not numeric:
                    low = float(model_rows.loc[marker, "display_ci_low"])
                    high = float(model_rows.loc[marker, "display_ci_high"])
                    ax.plot([low, high], [y[row_index], y[row_index]],
                            color=COLORS[model], alpha=1.0, linewidth=1.1, zorder=2)
                    size = 36
                else:
                    changed = float(model_rows.loc[marker, "n_changed"])
                    size = 30 + 8 * np.log10(max(0.0, changed) + 1)
                ax.scatter(value, y[row_index], s=size, color=COLORS[model],
                           marker=point_marker, alpha=1.0, edgecolors="white",
                           linewidths=.45, zorder=3)

        for marker_index, marker in enumerate(marker_ids):
            model_values = block.loc[block.marker_id.eq(marker), "display_effect"].dropna()
            if len(model_values) == len(MODELS):
                ax.scatter(float(model_values.median()), marker_index, marker="D",
                           s=25, color="#172033", edgecolors="white", linewidths=.35, zorder=4)

        ax.set_title(scenario, fontweight="bold")
        ax.set_xlim(-1.04 * bound, right_limit)
        ax.set_ylim(len(marker_ids) - .55, -.8)
        if numeric:
            ax.set_xticks([-1, -.5, 0, .5, 1])
        else:
            ax.set_xticks([-bound, 0, bound], [f"{-bound:.1f}", "0", f"{bound:.1f}"])
        ax.set_yticks(range(len(marker_ids)), [LABELS.get(marker, marker) for marker in marker_ids])
        ax.tick_params(labelleft=panel_index == 0)
        ax.grid(axis="x", color="#e2e8f0", linewidth=.8)
        ax.set_axisbelow(True)
        ax.spines["left"].set_color("#94a3b8")
        ax.spines["bottom"].set_color("#94a3b8")

    unit = "Matched rank-biserial" if numeric else "Selisih prevalensi (poin persentase); garis = CI 95% marginal"
    fig.suptitle(title, x=.515, y=.965, fontweight="bold", color="#162032")
    fig.text(.56, .155, f"{unit} · Man − Woman", ha="center", fontsize=11)
    fig.text(.56, .125, "← Woman lebih tinggi                         Man lebih tinggi →",
             ha="center", fontsize=10)
    handles = [
        Line2D([], [], marker="o", linestyle="", color=COLORS[model],
               label=MODEL_LABELS[model], markersize=6)
        for model in MODELS
    ]
    handles.append(Line2D([], [], marker="D", linestyle="", color="#172033",
                          label="Median (6 model tersedia)", markersize=6))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.54, .055),
               ncol=7, frameon=False)
    fig.text(.5, .018,
             "▲ p Holm < 0,05; ● lainnya",
             ha="center", fontsize=9, color="#526078")
    return fig


def build(output: Path = DEFAULT_OUTPUT) -> Path:
    output = Path(output).resolve()
    output.relative_to(ROOT)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    figures = output / "figures"
    figures.mkdir(parents=True)

    source_files = [
        SOURCE / "registry.json",
        SOURCE / "comparisons.csv",
        SOURCE / "report_status_coverage.csv",
    ]
    source_hashes = {path.relative_to(ROOT).as_posix(): sha256(path) for path in source_files}
    registry = pd.DataFrame(json.loads((SOURCE / "registry.json").read_text(encoding="utf-8")))
    comparisons = pd.read_csv(SOURCE / "comparisons.csv", float_precision="round_trip")
    index = []

    def save(figure_id: str, fig, table: pd.DataFrame, caption: str) -> None:
        png = figures / f"{figure_id}.png"
        csv = figures / f"{figure_id}.csv"
        fig.savefig(png, dpi=300)
        table.to_csv(csv, index=False)
        index.append({
            "figure_id": figure_id,
            "caption": caption,
            "rows": len(table),
            "panels": len(fig.axes),
            "direction": "Man minus Woman" if figure_id != "00_cakupan" else "not_applicable",
            "png_sha256": sha256(png),
            "csv_sha256": sha256(csv),
        })
        plt.close(fig)
        print(f"Rendered {len(index)}/5: {figure_id}", flush=True)

    status = pd.read_csv(SOURCE / "report_status_coverage.csv")
    status = status.loc[status.scope.eq("overall")].copy()
    support = support_summary(direct_rows(comparisons, registry))
    coverage_table = pd.concat([
        status.assign(record_type="text_status"),
        support.assign(record_type="paired_support"),
    ], ignore_index=True)
    coverage_figure = coverage(status, support)
    coverage_figure.axes[1].set_title(
        "Pasangan berpenjelasan Man–Woman (%)", pad=15, fontweight="bold"
    )
    save(
        "00_cakupan",
        coverage_figure,
        coverage_table,
        "Ketersediaan penjelasan dan pasangan langsung per model dan skenario.",
    )

    groups = [
        (
            "01_bahasa_umum_man_minus_woman",
            registry.loc[(registry.registry_scope == "general_ethics") & (registry.measurement == "density"), "marker_id"].tolist(),
            "density",
            "Perbedaan langsung penanda bahasa umum: Man − Woman",
        ),
        (
            "02_bahasa_etika_man_minus_woman",
            registry.loc[(registry.registry_scope == "ethics_specific") & (registry.measurement == "density"), "marker_id"].tolist(),
            "density",
            "Perbedaan langsung penanda bahasa etika: Man − Woman",
        ),
        (
            "03_bahasa_gender_man_minus_woman",
            registry.loc[(registry.registry_scope == "gender_specific") & (registry.measurement == "density"), "marker_id"].tolist(),
            "density",
            "Perbedaan langsung wacana gender: Man − Woman",
        ),
        (
            "04_marker_biner_man_minus_woman",
            registry.loc[registry.measurement == "binary", "marker_id"].tolist(),
            "presence",
            "Perbedaan langsung marker biner: Man − Woman",
        ),
    ]
    if [len(ids) for _, ids, _, _ in groups] != [14, 12, 4, 11]:
        raise ValueError("Unexpected marker grouping")

    for figure_id, marker_ids, measurement, title in groups:
        table = select_direct(comparisons, marker_ids, measurement)
        fig = plot_direct_effects(table, marker_ids, title, measurement)
        caption = (
            f"{title}. Enam model tetap terpisah dan pasangan berasal dari soal yang sama. "
            "Semua titik berwarna penuh; segitiga: p Holm < 0,05; lingkaran: lainnya. "
            + ("Titik menunjukkan matched rank-biserial; CI mean density sumber tidak ditempelkan pada effect size."
               if measurement == "density"
               else "Titik menunjukkan selisih prevalensi dalam poin persentase dan garis menunjukkan CI 95% marginal.")
        )
        save(figure_id, fig, table, caption)

    pd.DataFrame(index).to_csv(output / "figure_index.csv", index=False)
    manifest = {
        "status": "candidate_pending_independent_human_audit",
        "purpose": "Bab VI visual candidate; Gender ETHICS linguistic markers",
        "source_direction": "Woman minus Man",
        "display_direction": "Man minus Woman",
        "display_transform": "sign reversal only; two-sided p-values unchanged",
        "new_inferential_tests": False,
        "marker_count": int(len(registry)),
        "figure_count": len(index),
        "image_formats": ["png"],
        "point_encoding": "opaque model colors; triangle if Holm p < .05, circle otherwise",
        "source_hashes": source_hashes,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output / "README.md").write_text(
        "# Visual kandidat Bab VI — Gender ETHICS linguistic marker\n\n"
        "Paket ini hanya memperbaiki lapisan presentasi. Statistik, pasangan, kamus, "
        "dan p-value berasal dari bundle frozen v2.0.1 r3. Arah sumber Woman−Man "
        "dibalik secara eksplisit menjadi Man−Woman agar konsisten dengan Gender GSM8K.\n\n"
        "Gambar 00 adalah pemeriksaan cakupan. Gambar 01–04 adalah hasil langsung utama. "
        "Gambar hanya PNG (300 dpi), dengan CSV untuk angka. Semua titik berwarna penuh; "
        "segitiga menandai p Holm < 0,05 dan lingkaran menandai hasil lainnya. "
        "Density memakai matched rank-biserial tanpa memasangkan CI mean density pada rank. "
        "Marker biner memakai selisih prevalensi dan CI 95% marginal. Seluruh hasil masih "
        "candidate sampai audit manusia independen selesai.\n",
        encoding="utf-8",
    )
    current_hashes = {path.relative_to(ROOT).as_posix(): sha256(path) for path in source_files}
    if current_hashes != source_hashes:
        raise RuntimeError("Frozen source changed while rendering")
    print(json.dumps(manifest, indent=2), flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build(args.output)


if __name__ == "__main__":
    main()
