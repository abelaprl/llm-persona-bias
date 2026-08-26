"""Render thesis-readable GSM8K religion and regional marker figures.

This renderer is visual-only. It consumes the existing source-aligned tables,
does not recompute inference, and writes to new versioned output directories.
The denser four-column figures are split or rearranged so their typography
remains legible after scaling to the thesis text block.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
RELIGION_SOURCE = (
    REPO_ROOT
    / "result"
    / "GMS8K"
    / "reports"
    / "Religion_Persona"
    / "linguistic_markers"
    / "visual"
    / "religion_gsm8k_bab4_visual_v1_3_1_volcano_20260825"
    / "volcano_numeric_vs_neutral_points.csv"
)
REGIONAL_SOURCE = (
    REPO_ROOT
    / "result"
    / "GMS8K"
    / "reports"
    / "Regional_Cultural_Persona"
    / "linguistic_markers"
    / "visual"
    / "regional_gsm8k_bab4_visual_v1_1_1_volcano_20260825"
    / "volcano_numeric_vs_neutral_points.csv"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "result"
    / "GMS8K"
    / "reports"
    / "thesis_marker_readability_v3_20260826"
)

IDENTITY_LABELS = {
    "islam": "Islam",
    "christian": "Kristen",
    "jewish": "Yahudi",
    "atheist": "Ateis",
    "north_american": "Amerika Utara",
    "east_asian": "Asia Timur",
    "south_asian": "Asia Selatan",
    "sub_saharan_african": "Afrika Sub-Sahara",
}
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
}
FAMILY_LABELS = {
    "text_length": "Panjang respons",
    "lexical_diversity": "Variasi kata",
    "reasoning_organization": "Susunan penyelesaian",
    "epistemic_stance": "Kepastian dan keraguan",
    "logical_connective": "Kata penghubung",
    "self_reference": "Kata ganti diri",
    "engagement": "Sapaan kepada pengguna",
}

def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--religion-source", type=Path, default=RELIGION_SOURCE)
    parser.add_argument("--regional-source", type=Path, default=REGIONAL_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_figure(fig: plt.Figure, stem: Path) -> list[Path]:
    outputs = [stem.with_suffix(".png"), stem.with_suffix(".pdf")]
    fig.savefig(outputs[0], dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(outputs[1], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return outputs


def _religion_legend() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=color,
            markeredgecolor="white",
            markersize=12,
            label=FAMILY_LABELS[family],
        )
        for family, color in FAMILY_COLORS.items()
    ]


def _render_volcano_group(
    points: pd.DataFrame,
    identities: tuple[str, str],
    output_dir: Path,
) -> list[Path]:
    y_cap = 50.0
    selected = points[
        points["analysis_role"].eq("primary")
        & points["identity"].isin(identities)
        & points["contrast_id"].isin(CONTRASTS)
        & points["outcome_family"].isin(FAMILY_COLORS)
    ].copy()
    selected["plot_y"] = (
        -np.log10(selected["p_holm"].clip(lower=np.nextafter(0.0, 1.0)))
    ).clip(upper=y_cap)

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(18, 13.5),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for row, identity in enumerate(identities):
        for column, contrast in enumerate(CONTRASTS):
            ax = axes[row, column]
            panel = selected[
                selected["identity"].eq(identity)
                & selected["contrast_id"].eq(contrast)
            ]
            for family, family_data in panel.groupby("outcome_family"):
                point_size = 42 + 22 * np.log10(
                    family_data["n_nonzero"].clip(lower=0) + 1
                )
                ax.scatter(
                    family_data["rank_biserial"],
                    family_data["plot_y"],
                    s=point_size,
                    color=FAMILY_COLORS[family],
                    alpha=0.90,
                    edgecolors="white",
                    linewidths=0.9,
                )
            ax.axvline(0, color="#475569", linewidth=1.2)
            ax.axhline(
                -math.log10(0.05),
                color="#94A3B8",
                linestyle="--",
                linewidth=1.2,
            )
            ax.set_xlim(-1.02, 1.02)
            ax.set_ylim(-0.2, y_cap + 1)
            ax.set_xticks((-1.0, -0.5, 0.0, 0.5, 1.0))
            ax.set_yticks((0, 10, 20, 30, 40, 50))
            ax.tick_params(axis="both", labelsize=22, width=1.1, length=5)
            ax.set_title(
                f"{IDENTITY_LABELS[identity]}\n{CONTRAST_LABELS[contrast]}",
                fontsize=23,
                fontweight="bold",
                pad=12,
                linespacing=1.05,
            )
            ax.grid(color="#E2E8F0", linewidth=0.8, alpha=0.80)
            ax.spines[["top", "right"]].set_visible(False)

    identity_text = " dan ".join(IDENTITY_LABELS[item] for item in identities)
    fig.suptitle(
        "Ukuran efek penanda linguistik umum terhadap skenario netral\n"
        f"Persona {identity_text}",
        y=0.992,
        fontsize=28,
        fontweight="bold",
    )
    fig.supylabel(
        "−log₁₀(p Holm), dibatasi pada 50",
        x=0.012,
        fontsize=25,
        fontweight="semibold",
    )
    fig.supxlabel(
        "Rank-biserial (skenario persona − N)",
        y=0.158,
        fontsize=25,
        fontweight="semibold",
    )
    fig.legend(
        handles=_religion_legend(),
        title="Kelompok penanda",
        title_fontsize=23,
        fontsize=21,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=4,
        columnspacing=1.0,
        handletextpad=0.4,
        frameon=False,
    )
    fig.text(
        0.5,
        0.122,
        "Ukuran titik = jumlah pasangan dengan selisih nonnol; "
        "garis putus-putus = batas p Holm 0,05.",
        ha="center",
        fontsize=19,
        color="#334155",
    )
    fig.tight_layout(rect=(0.035, 0.19, 0.995, 0.91), h_pad=1.8, w_pad=1.0)
    stem = output_dir / (
        "15_general_numeric_effect_volcano_vs_neutral_"
        + "_".join(identities)
    )
    return _save_figure(fig, stem)


def main() -> int:
    args = _parser().parse_args()
    religion_source = args.religion_source.resolve()
    regional_source = args.regional_source.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty; choose a new versioned path: {output_root}"
        )

    religion_dir = output_root / "religion" / "images"
    regional_dir = output_root / "regional" / "images"
    religion_dir.mkdir(parents=True, exist_ok=True)
    regional_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.linewidth": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    religion = pd.read_csv(religion_source)
    regional = pd.read_csv(regional_source)

    generated: list[Path] = []
    generated.extend(
        _render_volcano_group(
            religion,
            ("islam", "christian"),
            religion_dir,
        )
    )
    generated.extend(
        _render_volcano_group(
            religion,
            ("jewish", "atheist"),
            religion_dir,
        )
    )
    generated.extend(
        _render_volcano_group(
            regional,
            ("north_american", "east_asian"),
            regional_dir,
        )
    )
    generated.extend(
        _render_volcano_group(
            regional,
            ("south_asian", "sub_saharan_african"),
            regional_dir,
        )
    )

    manifest = {
        "bundle_version": "gsm8k-thesis-marker-readability-v3.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "visual_only": True,
        "statistics_recomputed": False,
        "hidden_reasoning_used": False,
        "religion_source": {
            "path": str(religion_source),
            "sha256": _sha256(religion_source),
            "rows": int(len(religion)),
        },
        "regional_source": {
            "path": str(regional_source),
            "sha256": _sha256(regional_source),
            "rows": int(len(regional)),
        },
        "generated_files": [str(path.relative_to(output_root)) for path in generated],
        "layout_changes": {
            "religion": "four-by-four grid split into two two-by-four figures",
            "regional": "four-by-four volcano grid split into two two-by-four figures",
            "formats": ["png_300dpi", "pdf_vector"],
        },
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Generated {len(generated)} files in {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
