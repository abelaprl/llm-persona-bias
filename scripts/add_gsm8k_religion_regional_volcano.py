"""Add vs-neutral numeric volcano plots to the GSM8K religion/regional bundles.

The script only renders existing paired statistics. It does not read raw model
responses, rerun marker extraction, recompute tests, or overwrite prior bundles.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURE_FILENAME = "14_numeric_effect_volcano_vs_neutral.png"
POINT_TABLE_FILENAME = "volcano_numeric_vs_neutral_points.csv"
CONTRASTS = (
    "explicit_vs_neutral",
    "implicit_1_vs_neutral",
    "implicit_2_vs_neutral",
    "implicit_3_vs_neutral",
)
CONTRAST_LABELS = {
    "explicit_vs_neutral": "E − N",
    "implicit_1_vs_neutral": "I1 − N",
    "implicit_2_vs_neutral": "I2 − N",
    "implicit_3_vs_neutral": "I3 − N",
}

FAMILY_COLORS = {
    "text_length": "#264653",
    "lexical_diversity": "#2A9D8F",
    "reasoning_organization": "#E9C46A",
    "epistemic_stance": "#E76F51",
    "logical_connective": "#6A4C93",
    "self_reference": "#1982C4",
    "engagement": "#8AC926",
    "persona_accommodation": "#7C3AED",
    "persona_meta_discourse": "#0F766E",
    "persona_context_carryover": "#A16207",
    "religious_language_authority": "#9D174D",
    "religious_language_devotional": "#BE123C",
    "religious_language_formula": "#C2410C",
    "religious_language_general": "#0369A1",
    "religious_language_identity": "#4338CA",
    "religious_language_practice": "#047857",
    "cultural_language_formula": "#C2410C",
    "cultural_language_general": "#0369A1",
    "cultural_language_identity": "#4338CA",
    "cultural_language_practice": "#047857",
}
FAMILY_LABELS = {
    "text_length": "Panjang teks",
    "lexical_diversity": "Keragaman leksikal",
    "reasoning_organization": "Organisasi penalaran",
    "epistemic_stance": "Sikap epistemik",
    "logical_connective": "Konektif logis",
    "self_reference": "Rujukan diri",
    "engagement": "Engagement",
    "persona_accommodation": "Akomodasi persona",
    "persona_meta_discourse": "Meta-discourse persona",
    "persona_context_carryover": "Context carryover",
    "religious_language_authority": "Bahasa agama: otoritas",
    "religious_language_devotional": "Bahasa agama: devosional",
    "religious_language_formula": "Bahasa agama: formula",
    "religious_language_general": "Bahasa agama: umum",
    "religious_language_identity": "Bahasa agama: identitas",
    "religious_language_practice": "Bahasa agama: praktik",
    "cultural_language_formula": "Bahasa budaya: formula",
    "cultural_language_general": "Bahasa budaya: umum",
    "cultural_language_identity": "Bahasa budaya: identitas",
    "cultural_language_practice": "Bahasa budaya: praktik",
}


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    display_name: str
    source_bundle: Path
    output_bundle: Path
    source_notebook: Path
    output_notebook: Path
    bundle_version: str
    identities: tuple[str, ...]
    identity_labels: dict[str, str]
    expected_outcomes: int


SPECS = (
    DatasetSpec(
        dataset_id="religion",
        display_name="agama",
        source_bundle=REPO_ROOT
        / "result/GMS8K/reports/Religion_Persona/linguistic_markers/visual/"
        "religion_gsm8k_bab4_visual_v1_3_0_20260818",
        output_bundle=REPO_ROOT
        / "result/GMS8K/reports/Religion_Persona/linguistic_markers/visual/"
        "religion_gsm8k_bab4_visual_v1_3_1_volcano_20260825",
        source_notebook=REPO_ROOT
        / "notebooks/religion/linguistic_marker_religion_gsm8k_bab4_visual_v1.ipynb",
        output_notebook=REPO_ROOT
        / "notebooks/religion/"
        "linguistic_marker_religion_gsm8k_bab4_visual_v1_3_1_volcano.ipynb",
        bundle_version="religion-gsm8k-bab4-visual-v1.3.1-volcano",
        identities=("islam", "christian", "jewish", "atheist"),
        identity_labels={
            "islam": "Islam",
            "christian": "Christian",
            "jewish": "Jewish",
            "atheist": "Atheist",
        },
        expected_outcomes=27,
    ),
    DatasetSpec(
        dataset_id="regional",
        display_name="regional-kultural",
        source_bundle=REPO_ROOT
        / "result/GMS8K/reports/Regional_Cultural_Persona/linguistic_markers/visual/"
        "regional_gsm8k_bab4_visual_v1_1_0_20260818",
        output_bundle=REPO_ROOT
        / "result/GMS8K/reports/Regional_Cultural_Persona/linguistic_markers/visual/"
        "regional_gsm8k_bab4_visual_v1_1_1_volcano_20260825",
        source_notebook=REPO_ROOT
        / "notebooks/regional cultural/"
        "linguistic_marker_regional_cultural_gsm8k_bab4_visual_v1.ipynb",
        output_notebook=REPO_ROOT
        / "notebooks/regional cultural/"
        "linguistic_marker_regional_cultural_gsm8k_bab4_visual_v1_1_1_volcano.ipynb",
        bundle_version="regional-gsm8k-bab4-visual-v1.1.1-volcano",
        identities=(
            "north_american",
            "east_asian",
            "south_asian",
            "sub_saharan_african",
        ),
        identity_labels={
            "north_american": "Amerika Utara",
            "east_asian": "Asia Timur",
            "south_asian": "Asia Selatan",
            "sub_saharan_african": "Afrika Sub-Sahara",
        },
        expected_outcomes=30,
    ),
)


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 13.5,
            "axes.titlesize": 15.5,
            "axes.labelsize": 16.5,
            "xtick.labelsize": 13.5,
            "ytick.labelsize": 13.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.titlesize": 21,
            "legend.frameon": False,
            "legend.fontsize": 11.5,
        }
    )


def _load_points(spec: DatasetSpec, source_run: Path) -> pd.DataFrame:
    source_path = source_run / "paired_numeric.csv"
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    numeric = pd.read_csv(source_path)
    required = {
        "model_id",
        "identity",
        "outcome_id",
        "outcome_family",
        "analysis_role",
        "contrast_id",
        "n_nonzero",
        "rank_biserial",
        "p_holm",
    }
    missing = sorted(required.difference(numeric.columns))
    if missing:
        raise ValueError(f"{source_path} is missing required columns: {missing}")

    points = numeric[
        numeric["analysis_role"].eq("primary")
        & numeric["identity"].isin(spec.identities)
        & numeric["contrast_id"].isin(CONTRASTS)
    ].copy()
    for column in ("n_nonzero", "rank_biserial", "p_holm"):
        points[column] = pd.to_numeric(points[column], errors="coerce")
    invalid = points[["n_nonzero", "rank_biserial", "p_holm"]].isna().any(axis=1)
    if invalid.any():
        raise ValueError(f"Found {int(invalid.sum())} invalid numeric volcano rows")

    models = tuple(sorted(points["model_id"].unique()))
    outcomes = tuple(sorted(points["outcome_id"].unique()))
    expected_rows = (
        len(spec.identities) * len(CONTRASTS) * len(models) * spec.expected_outcomes
    )
    if len(outcomes) != spec.expected_outcomes or len(points) != expected_rows:
        raise ValueError(
            f"Unexpected {spec.dataset_id} volcano grid: rows={len(points)} "
            f"(expected {expected_rows}), outcomes={len(outcomes)} "
            f"(expected {spec.expected_outcomes}), models={len(models)}"
        )
    duplicate_keys = ["model_id", "identity", "outcome_id", "contrast_id"]
    if points.duplicated(duplicate_keys).any():
        raise ValueError(f"Duplicate volcano points for keys {duplicate_keys}")
    return points.sort_values(duplicate_keys).reset_index(drop=True)


def _volcano_panel(ax: plt.Axes, data: pd.DataFrame, title: str) -> None:
    y_cap = 50.0
    p_values = data["p_holm"].clip(lower=np.nextafter(0.0, 1.0))
    panel = data.assign(y=(-np.log10(p_values)).clip(upper=y_cap))
    for family, family_data in panel.groupby("outcome_family", sort=True):
        point_size = 15 + 8 * np.log10(family_data["n_nonzero"].clip(lower=0) + 1)
        significant = family_data["p_holm"].lt(0.05).to_numpy()
        ax.scatter(
            family_data["rank_biserial"],
            family_data["y"],
            s=point_size,
            alpha=np.where(significant, 0.92, 0.34),
            color=FAMILY_COLORS.get(family, "#64748B"),
            edgecolors="white",
            linewidths=0.45,
        )
    ax.axvline(0, color="#334155", linewidth=0.85)
    ax.axhline(-math.log10(0.05), color="#94A3B8", linewidth=0.85, linestyle="--")
    ax.set_xlim(-1.02, 1.02)
    ax.set_ylim(-0.2, y_cap + 1)
    ax.set_title(title, fontweight="bold")
    ax.grid(color="#E2E8F0", linewidth=0.5, alpha=0.75)


def _render_volcano(spec: DatasetSpec, points: pd.DataFrame, image_path: Path) -> None:
    _style()
    fig, axes = plt.subplots(
        len(spec.identities),
        len(CONTRASTS),
        figsize=(24, 20.5),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for row, identity in enumerate(spec.identities):
        for column, contrast in enumerate(CONTRASTS):
            selected = points[
                points["identity"].eq(identity) & points["contrast_id"].eq(contrast)
            ]
            _volcano_panel(
                axes[row, column],
                selected,
                f"{spec.identity_labels[identity]} · {CONTRAST_LABELS[contrast]}",
            )

    families = sorted(points["outcome_family"].unique())
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=FAMILY_COLORS.get(family, "#64748B"),
            label=FAMILY_LABELS.get(family, family.replace("_", " ").title()),
            markersize=9,
        )
        for family in families
    ]
    fig.supylabel("−log₁₀(p Holm), dipotong pada 50", x=0.006)
    fig.supxlabel(
        "Rank-biserial (arah: kondisi persona − Neutral)",
        y=0.093,
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.5, 0.008),
        columnspacing=1.1,
        handletextpad=0.45,
    )
    fig.suptitle(
        f"GSM8K {spec.display_name.title()}: effect map marker numerik terhadap Neutral\n"
        "Efek besar berada jauh dari nol; ukuran titik mengikuti jumlah pasangan nonzero",
        y=0.997,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.025, 0.145, 0.995, 0.945), h_pad=1.65, w_pad=1.05)
    fig.savefig(image_path, dpi=235, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _append_figure_index(bundle_dir: Path, spec: DatasetSpec) -> None:
    index_path = bundle_dir / "figure_index.csv"
    index = pd.read_csv(index_path)
    index = index[index["figure"].ne(FIGURE_FILENAME)].copy()
    index.loc[len(index)] = {
        "figure": FIGURE_FILENAME,
        "description": (
            f"Volcano marker numerik {spec.display_name} vs Neutral: "
            "rank-biserial × −log10(p Holm), ukuran titik = n_nonzero."
        ),
    }
    index.to_csv(index_path, index=False)


def _update_manifest(bundle_dir: Path, spec: DatasetSpec, source_bundle: Path) -> None:
    path = bundle_dir / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    figures = [name for name in manifest.get("generated_figures", []) if name != FIGURE_FILENAME]
    tables = [name for name in manifest.get("generated_tables", []) if name != POINT_TABLE_FILENAME]
    manifest["bundle_version"] = spec.bundle_version
    manifest["generated_figures"] = [*figures, FIGURE_FILENAME]
    manifest["generated_tables"] = [*tables, POINT_TABLE_FILENAME]
    manifest["volcano_augmentation"] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_visual_bundle": str(source_bundle.resolve()),
        "statistics_source": "source_run/paired_numeric.csv",
        "analysis_role": "primary",
        "contrasts": list(CONTRASTS),
        "x": "rank_biserial",
        "y": "-log10(p_holm)",
        "point_size": "n_nonzero",
        "statistics_recomputed": False,
        "llm_inference_rerun": False,
        "renderer": str(Path(__file__).resolve()),
    }
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _replace_strings(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, str):
        for old, new in replacements.items():
            value = value.replace(old, new)
        return value
    if isinstance(value, list):
        return [_replace_strings(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: _replace_strings(item, replacements) for key, item in value.items()}
    return value


def _new_markdown_cell(source: str) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def _new_image_cell(image_path: Path) -> dict[str, Any]:
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "outputs": [
            {
                "data": {
                    "image/png": encoded,
                    "text/plain": ["<IPython.core.display.Image object>"],
                },
                "metadata": {},
                "output_type": "display_data",
            }
        ],
        "source": [f'show_figure("{FIGURE_FILENAME}", width=1900)\n'],
    }


def _write_notebook(spec: DatasetSpec, image_path: Path) -> None:
    notebook = json.loads(spec.source_notebook.read_text(encoding="utf-8"))
    relative_notebook = spec.output_notebook.relative_to(REPO_ROOT)
    notebook_expression = " / ".join(
        ["REPO_ROOT", *(repr(part) for part in relative_notebook.parts)]
    )
    notebook = _replace_strings(
        notebook,
        {
            spec.source_bundle.name: spec.output_bundle.name,
            spec.source_notebook.name: spec.output_notebook.name,
        },
    )
    notebook = _replace_strings(
        notebook,
        {
            f"Path.cwd() / 'notebooks' / '{spec.output_notebook.name}'": notebook_expression,
            f"REPO_ROOT / 'notebooks' / '{spec.output_notebook.name}'": notebook_expression,
            str(spec.source_notebook.parent / "notebooks" / spec.output_notebook.name): str(
                spec.output_notebook
            ),
            str(REPO_ROOT / "notebooks" / spec.output_notebook.name): str(
                spec.output_notebook
            ),
        },
    )
    cells = notebook.get("cells")
    if not isinstance(cells, list):
        raise ValueError(f"Notebook has no cell list: {spec.source_notebook}")

    insertion_index = next(
        (
            index
            for index, cell in enumerate(cells)
            if cell.get("cell_type") == "markdown"
            and "".join(cell.get("source", [])).lstrip().startswith("## 8.")
        ),
        None,
    )
    if insertion_index is None:
        raise ValueError(f"Could not find section 8 in {spec.source_notebook}")

    significant = pd.read_csv(spec.output_bundle / POINT_TABLE_FILENAME)["p_holm"].lt(0.05).sum()
    total = len(pd.read_csv(spec.output_bundle / POINT_TABLE_FILENAME, usecols=["p_holm"]))
    explanation = (
        "## 7A. Volcano marker numerik terhadap Neutral\n\n"
        "Visual ini mengikuti definisi volcano pada analisis gender: sumbu X adalah "
        "*rank-biserial*, sumbu Y adalah $-\\log_{10}(p_{Holm})$, dan ukuran titik "
        "mengikuti `n_nonzero`. Setiap baris mempertahankan identitas secara terpisah; "
        "setiap kolom mempertahankan E/I1/I2/I3 sebagai contrast terpisah. Tidak ada "
        "pooling identitas atau proxy implicit.\n\n"
        f"Tabel yang divisualisasikan memuat **{total:,}** titik primary vs Neutral; "
        f"**{int(significant):,}** memiliki `p_holm < 0.05`. Angka berasal langsung "
        "dari `paired_numeric.csv`; tidak ada inferensi LLM atau penghitungan ulang uji.\n"
    )
    cells[insertion_index:insertion_index] = [
        _new_markdown_cell(explanation),
        _new_image_cell(image_path),
    ]
    spec.output_notebook.parent.mkdir(parents=True, exist_ok=True)
    spec.output_notebook.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _repair_existing_notebook_path(spec: DatasetSpec) -> None:
    """Repair the inherited final-cell notebook path in an existing output."""

    notebook = json.loads(spec.output_notebook.read_text(encoding="utf-8"))
    relative_notebook = spec.output_notebook.relative_to(REPO_ROOT)
    notebook_expression = " / ".join(
        ["REPO_ROOT", *(repr(part) for part in relative_notebook.parts)]
    )
    notebook = _replace_strings(
        notebook,
        {
            f"Path.cwd() / 'notebooks' / '{spec.output_notebook.name}'": notebook_expression,
            f"REPO_ROOT / 'notebooks' / '{spec.output_notebook.name}'": notebook_expression,
            str(spec.source_notebook.parent / "notebooks" / spec.output_notebook.name): str(
                spec.output_notebook
            ),
            str(REPO_ROOT / "notebooks" / spec.output_notebook.name): str(
                spec.output_notebook
            ),
        },
    )
    spec.output_notebook.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def build(spec: DatasetSpec) -> dict[str, Any]:
    for required in (spec.source_bundle, spec.source_notebook):
        if not required.exists():
            raise FileNotFoundError(required)
    if spec.output_bundle.exists():
        raise FileExistsError(f"Refusing to overwrite existing bundle: {spec.output_bundle}")
    if spec.output_notebook.exists():
        raise FileExistsError(f"Refusing to overwrite existing notebook: {spec.output_notebook}")

    shutil.copytree(spec.source_bundle, spec.output_bundle)
    manifest = json.loads((spec.output_bundle / "manifest.json").read_text(encoding="utf-8"))
    source_run = Path(manifest["source_run"])
    points = _load_points(spec, source_run)
    point_path = spec.output_bundle / POINT_TABLE_FILENAME
    points.to_csv(point_path, index=False)
    image_path = spec.output_bundle / "images" / FIGURE_FILENAME
    _render_volcano(spec, points, image_path)
    _append_figure_index(spec.output_bundle, spec)
    _update_manifest(spec.output_bundle, spec, spec.source_bundle)
    _write_notebook(spec, image_path)

    return {
        "dataset": spec.dataset_id,
        "bundle": str(spec.output_bundle),
        "notebook": str(spec.output_notebook),
        "figure": str(image_path),
        "point_table": str(point_path),
        "points": len(points),
        "significant_holm": int(points["p_holm"].lt(0.05).sum()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("all", "religion", "regional"),
        default="all",
        help="Dataset bundle to augment (default: all).",
    )
    parser.add_argument(
        "--repair-notebook-paths",
        action="store_true",
        help="Only repair inherited notebook-location text in existing outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = SPECS if args.dataset == "all" else tuple(
        spec for spec in SPECS if spec.dataset_id == args.dataset
    )
    if args.repair_notebook_paths:
        for spec in selected:
            _repair_existing_notebook_path(spec)
        print(json.dumps([str(spec.output_notebook) for spec in selected], indent=2))
        return
    summaries = [build(spec) for spec in selected]
    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
