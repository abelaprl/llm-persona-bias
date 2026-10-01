"""Rerender the ETHICS Gender direct-comparison bundle with GSM8K-scale type.

This is a visual-only revision. Exact v1.2 tables are copied byte-for-byte,
statistics are not recomputed, and all figures are redrawn in a new directory.
Version 1.3.1 also aligns visible terminology with the thesis by using
``skenario`` instead of ``kondisi`` for persona settings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
import pandas as pd
import seaborn as sns

from src.benchmarks.ethics import build_gender_ethics_direct_thesis_visuals_v1_2 as base


BUNDLE_VERSION = "gender-ethics-direct-thesis-visuals-v1.3.1"
REFERENCE_NOTEBOOK = "notebooks/gender/linguistic_marker_gender_gsm8k_direct_gender_v3.ipynb"
SAVE_SVG = True

TABLE_FILES = (
    "gender_term_response_hits.csv.gz",
    "gender_term_surface_reconstruction_validation.csv",
    "gender_term_prevalence_by_model_identity_condition_subset.csv",
    "gender_term_direct_man_woman_response_only.csv",
    "general_marker_direct_man_woman_descriptive.csv",
    "gender_marker_direct_man_woman_by_subset_descriptive.csv",
    "response_outcomes_direct_man_woman_by_subset.csv",
    "opening_direct_man_woman_descriptive.csv",
)

FIGURE_SIZES = {
    "surface": (22.0, 14.0),
    "term_absolute": (21.0, 15.0),
    "term_direct": (23.5, 17.0),
    "general_marker": (24.0, 19.0),
    "gender_marker": (24.0, 16.0),
    "commit": (22.0, 14.0),
    "outcome": (22.0, 15.0),
    "opening": (16.0, 9.5),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reuse-figures-from", type=Path)
    parser.add_argument("--reuse-through-number", type=int, default=0)
    parser.add_argument("--start-number", type=int, default=1)
    parser.add_argument("--end-number", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--png-only", action="store_true")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _visual_class(filename: str) -> str:
    number = int(filename[:2])
    if number <= 3:
        return "surface"
    if number == 4:
        return "term_absolute"
    if number <= 8:
        return "term_direct"
    if number <= 12:
        return "general_marker"
    if number <= 16:
        return "gender_marker"
    if number == 17:
        return "commit"
    if number <= 19:
        return "outcome"
    return "opening"


def _style() -> None:
    """Use the same typography scale as the Gender GSM8K v3 renderer."""

    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 14.5,
            "axes.titlesize": 18,
            "axes.labelsize": 17.5,
            "xtick.labelsize": 15.5,
            "ytick.labelsize": 15.5,
            "figure.titlesize": 20,
            "legend.fontsize": 17,
            "legend.markerscale": 1.25,
            "legend.handlelength": 2.1,
            "legend.handletextpad": 0.7,
            "legend.columnspacing": 1.25,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _apply_readable_type(fig: plt.Figure, filename: str) -> None:
    """Normalize type that was locally overridden by the v1.2 plot functions."""

    visual_class = _visual_class(filename)
    fig.set_size_inches(*FIGURE_SIZES[visual_class], forward=True)

    # The v1.2 plotting functions use an older visible label. Keep the CSV
    # schema column ``condition`` unchanged, but align figure prose with the
    # thesis terminology for experimental persona settings.
    exact_text_replacements = {
        "Akurasi pada pasangan committed dan coverage-nya": (
            "Akurasi pada pasangan committed dan cakupan pasangannya"
        ),
    }
    substring_replacements = (
        ("Kondisi", "Skenario"),
        ("paired coverage", "cakupan pasangan committed"),
        (" — coverage", " — cakupan"),
    )
    for artist in fig.findobj(match=Text):
        original = artist.get_text()
        revised = exact_text_replacements.get(original, original)
        for old, new in substring_replacements:
            revised = revised.replace(old, new)
        if revised != original:
            artist.set_text(revised)

    if fig._suptitle is not None:
        fig._suptitle.set_fontsize(20)
        fig._suptitle.set_fontweight("bold")

    for text in fig.texts:
        if text is not fig._suptitle:
            text.set_fontsize(max(text.get_fontsize(), 16.5))

    for ax in fig.axes:
        ax.title.set_fontsize(18)
        ax.title.set_fontweight("bold")
        ax.xaxis.label.set_fontsize(17.5)
        ax.yaxis.label.set_fontsize(17.5)
        for label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
            label.set_fontsize(15.0)

        # Heatmap cell annotations need to remain smaller than axis labels,
        # but must survive reduction to notebook/thesis page width.
        for annotation in ax.texts:
            annotation.set_fontsize(max(annotation.get_fontsize(), 12.5))
            annotation.set_fontweight("semibold")

        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize(17)

    for legend in fig.legends:
        for text in legend.get_texts():
            text.set_fontsize(17)


def _save_readable(fig: plt.Figure, image_dir: Path, filename: str) -> Path:
    _apply_readable_type(fig, filename)
    png = image_dir / filename
    svg = png.with_suffix(".svg")
    fig.savefig(png, dpi=240, bbox_inches="tight", facecolor="white")
    if SAVE_SVG:
        fig.savefig(svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png


def _copy_exact_tables(source_bundle: Path, output_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for filename in TABLE_FILES:
        source = source_bundle / filename
        if not source.is_file():
            raise FileNotFoundError(source)
        target = output_dir / filename
        shutil.copy2(source, target)
        source_hash = _sha256(source)
        target_hash = _sha256(target)
        if source_hash != target_hash:
            raise ValueError(f"Copied table differs from source: {filename}")
        hashes[filename] = target_hash
    return hashes


def _copy_reused_figures(
    *,
    source_bundle: Path,
    reuse_bundle: Path,
    output_dir: Path,
    through_number: int,
) -> dict[str, object]:
    """Reuse already-readable figures that contain no revised visible term."""

    reuse_bundle = reuse_bundle.resolve()
    reuse_manifest_path = reuse_bundle / "manifest.json"
    reuse_index_path = reuse_bundle / "figure_index.csv"
    if not reuse_manifest_path.is_file() or not reuse_index_path.is_file():
        raise FileNotFoundError(f"Incomplete reuse bundle: {reuse_bundle}")
    reuse_manifest = json.loads(reuse_manifest_path.read_text(encoding="utf-8"))
    if reuse_manifest.get("bundle_version") != "gender-ethics-direct-thesis-visuals-v1.3.0":
        raise ValueError("Reuse source must be the completed v1.3.0 visual bundle.")
    if reuse_manifest.get("visual_revision_only") is not True:
        raise ValueError("Reuse source is not marked as a visual-only revision.")
    for filename in TABLE_FILES:
        if _sha256(reuse_bundle / filename) != _sha256(source_bundle / filename):
            raise ValueError(f"Reuse source table differs from v1.2: {filename}")

    reuse_index = pd.read_csv(reuse_index_path)
    copied_hashes: dict[str, str] = {}
    image_dir = output_dir / "images"
    for filename in reuse_index["figure"]:
        if int(filename[:2]) > through_number:
            continue
        source_png = reuse_bundle / "images" / filename
        target_png = image_dir / filename
        if not source_png.is_file():
            raise FileNotFoundError(source_png)
        if target_png.exists():
            raise FileExistsError(target_png)
        shutil.copy2(source_png, target_png)
        copied_hashes[filename] = _sha256(target_png)

    provenance: dict[str, object] = {
        "reuse_bundle": str(reuse_bundle),
        "reuse_manifest_sha256": _sha256(reuse_manifest_path),
        "reuse_through_number": through_number,
        "reused_png_sha256": copied_hashes,
        "reason": (
            "Figures 01-17 contain no visible 'Kondisi' label; they are reused "
            "byte-for-byte from v1.3.0. Figures 18-20 are rerendered with "
            "thesis-aligned 'Skenario' terminology."
        ),
    }
    (output_dir / "reused_figure_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return provenance


def rerender(
    source_bundle: Path,
    output_dir: Path,
    *,
    reuse_figures_from: Path | None = None,
    reuse_through_number: int = 0,
    start_number: int = 1,
    end_number: int = 20,
    resume: bool = False,
    png_only: bool = False,
) -> Path:
    global SAVE_SVG
    SAVE_SVG = not png_only
    source_bundle = source_bundle.resolve()
    output_dir = output_dir.resolve()
    source_manifest_path = source_bundle / "manifest.json"
    if not source_manifest_path.is_file():
        raise FileNotFoundError(source_manifest_path)
    if not 1 <= start_number <= end_number <= 20:
        raise ValueError("Figure range must satisfy 1 <= start <= end <= 20.")
    if not 0 <= reuse_through_number <= 20:
        raise ValueError("Reuse range must satisfy 0 <= reuse-through-number <= 20.")
    if reuse_through_number and reuse_figures_from is None:
        raise ValueError("--reuse-through-number requires --reuse-figures-from.")
    if reuse_figures_from is not None and reuse_through_number == 0:
        raise ValueError("--reuse-figures-from requires a positive reuse range.")
    if output_dir.exists() and not resume:
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    if output_dir.exists() and (output_dir / "manifest.json").exists():
        raise FileExistsError(f"Completed bundle will not be modified: {output_dir}")

    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("bundle_version") != "gender-ethics-direct-thesis-visuals-v1.2.0":
        raise ValueError("Source must be the validated ETHICS Gender v1.2 bundle.")
    required_validations = (
        "rows_match_source",
        "visible_rows_match_source",
        "term_reconstruction_all_match",
        "direct_general_row_count",
        "direct_gender_row_count",
        "direct_term_row_count",
        "response_outcome_row_count",
        "all_figures_exist",
        "all_svg_figures_exist",
        "marker_inference_disabled",
    )
    if not all(source_manifest["validations"].get(key) is True for key in required_validations):
        raise ValueError("Source bundle did not pass all required v1.2 validations.")
    if source_manifest.get("hidden_reasoning_used") is not False:
        raise ValueError("Source bundle unexpectedly used hidden reasoning.")

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=False)
        image_dir = output_dir / "images"
        image_dir.mkdir()
        table_hashes = _copy_exact_tables(source_bundle, output_dir)
        reuse_provenance = (
            _copy_reused_figures(
                source_bundle=source_bundle,
                reuse_bundle=reuse_figures_from,
                output_dir=output_dir,
                through_number=reuse_through_number,
            )
            if reuse_figures_from is not None
            else None
        )
    else:
        image_dir = output_dir / "images"
        if not image_dir.is_dir():
            raise FileNotFoundError(image_dir)
        table_hashes = {}
        for filename in TABLE_FILES:
            target = output_dir / filename
            if not target.is_file() or _sha256(target) != _sha256(source_bundle / filename):
                raise ValueError(f"Resume table differs from source: {filename}")
            table_hashes[filename] = _sha256(target)
        reuse_provenance_path = output_dir / "reused_figure_provenance.json"
        reuse_provenance = (
            json.loads(reuse_provenance_path.read_text(encoding="utf-8"))
            if reuse_provenance_path.is_file()
            else None
        )

    term_prevalence = pd.read_csv(output_dir / "gender_term_prevalence_by_model_identity_condition_subset.csv")
    direct_terms = pd.read_csv(output_dir / "gender_term_direct_man_woman_response_only.csv")
    direct_general = pd.read_csv(output_dir / "general_marker_direct_man_woman_descriptive.csv")
    direct_gender = pd.read_csv(output_dir / "gender_marker_direct_man_woman_by_subset_descriptive.csv")
    response_outcomes = pd.read_csv(output_dir / "response_outcomes_direct_man_woman_by_subset.csv")
    opening = pd.read_csv(output_dir / "opening_direct_man_woman_descriptive.csv")

    _style()
    base._save = _save_readable
    selected_terms = list(source_manifest["selected_gender_terms"])
    direct_term_selected = direct_terms[
        direct_terms["term"].isin(selected_terms) & ~direct_terms["subset"].eq("all")
    ]
    term_limit = max(1.0, float(direct_term_selected["prevalence_delta_pp"].abs().max()))
    gender_limit = max(1.0, float(direct_gender["prevalence_delta_pp"].abs().max()) * 1.08)
    source_figure_index = pd.read_csv(source_bundle / "figure_index.csv")
    descriptions = source_figure_index.set_index("figure")["description"].to_dict()
    figures: list[tuple[str, str]] = []

    def selected(number: int) -> bool:
        return start_number <= number <= end_number

    def assert_new_target(filename: str) -> None:
        png = image_dir / filename
        svg = png.with_suffix(".svg")
        if png.exists() or (SAVE_SVG and svg.exists()):
            raise FileExistsError(f"Refusing to overwrite rendered figure: {png}")

    def add_expected(filename: str, fallback_description: str) -> None:
        figures.append((filename, descriptions.get(filename, fallback_description)))

    for number, outcome in enumerate(base.SURFACE_OUTCOMES, start=1):
        filename = f"{number:02d}_direct_verbosity_{outcome}.png"
        if selected(number):
            assert_new_target(filename)
            base._plot_surface_heatmap(direct_general, image_dir, outcome, number)
        add_expected(filename, f"Direct Man-Woman {outcome} by model, subset, and condition.")
    filename = "04_gender_words_absolute_man_woman.png"
    if selected(4):
        assert_new_target(filename)
        base._plot_term_absolute(term_prevalence, selected_terms, image_dir)
    add_expected(filename, "Absolute response-only gender-term prevalence for Man and Woman.")
    for number, condition in enumerate(base.CONDITIONS, start=5):
        filename = f"{number:02d}_direct_gender_words_{condition}.png"
        if selected(number):
            assert_new_target(filename)
            base._plot_term_direct_heatmap(
                direct_terms, selected_terms, image_dir, condition, number, term_limit
            )
        add_expected(filename, f"Direct Man-Woman response-only gender terms for {condition}.")
    for number, condition in enumerate(base.CONDITIONS, start=9):
        filename = f"{number:02d}_direct_general_markers_{condition}.png"
        if selected(number):
            assert_new_target(filename)
            base._plot_marker_forest(
                direct_general[direct_general["outcome_family"].eq("general_marker")],
                image_dir,
                condition=condition,
                outcomes=base.GENERAL_MARKERS,
                labels=base.GENERAL_LABELS,
                value="rank_biserial",
                xlabel="Paired rank-biserial Man − Woman (deskriptif)",
                title=f"Marker linguistik umum: Man − Woman pada {base.CONDITION_LABELS[condition]}",
                filename=filename,
                xlimit=1.03,
            )
        add_expected(filename, f"Direct general-marker paired effect for {condition} by subset.")
    for number, condition in enumerate(base.CONDITIONS, start=13):
        filename = f"{number:02d}_direct_gender_markers_{condition}.png"
        if selected(number):
            assert_new_target(filename)
            base._plot_marker_forest(
                direct_gender,
                image_dir,
                condition=condition,
                outcomes=base.GENDER_MARKERS,
                labels=base.GENDER_MARKER_LABELS,
                value="prevalence_delta_pp",
                xlabel="Man − Woman: Δ prevalensi (poin persentase; deskriptif)",
                title=f"Marker gender: Man − Woman pada {base.CONDITION_LABELS[condition]}",
                filename=filename,
                xlimit=gender_limit,
            )
        add_expected(filename, f"Direct gender-marker prevalence for {condition} by subset.")
    final_plots = (
        (17, "17_direct_commit_rate_man_woman.png", lambda: base._plot_commit_delta(response_outcomes, image_dir), "Direct Man-Woman commit-rate difference by subset."),
        (18, "18_direct_accuracy_committed_with_coverage.png", lambda: base._plot_accuracy_coverage(response_outcomes, image_dir), "Committed-pair accuracy with paired coverage."),
        (19, "19_direct_answer_distribution_with_coverage.png", lambda: base._plot_distribution_coverage(response_outcomes, image_dir), "Answer distribution with commit-rate coverage."),
        (20, "20_direct_opening_composition_man_woman.png", lambda: base._plot_opening(opening, image_dir), "Direct opening-composition difference."),
    )
    for number, filename, plotter, description in final_plots:
        if selected(number):
            assert_new_target(filename)
            plotter()
        add_expected(filename, description)

    missing_png = [name for name, _ in figures if not (image_dir / name).is_file()]
    if missing_png:
        print(
            f"[PARTIAL] Rendered figures {start_number:02d}-{end_number:02d}; "
            f"{len(missing_png)} PNG still missing. Resume without replacing existing figures."
        )
        return output_dir

    figure_index = pd.DataFrame(figures, columns=["figure", "description"])
    figure_index["png_size_bytes"] = figure_index["figure"].map(
        lambda name: (image_dir / name).stat().st_size
    )
    figure_index["svg_size_bytes"] = figure_index["figure"].map(
        lambda name: (
            (image_dir / name).with_suffix(".svg").stat().st_size
            if (image_dir / name).with_suffix(".svg").is_file()
            else pd.NA
        )
    )
    figure_index.to_csv(output_dir / "figure_index.csv", index=False, encoding="utf-8")

    table_hashes_match = all(
        _sha256(source_bundle / filename) == _sha256(output_dir / filename)
        for filename in TABLE_FILES
    )
    validations = dict(source_manifest["validations"])
    validations.update(
        {
            "all_figures_exist": all((image_dir / name).is_file() for name in figure_index["figure"]),
            "all_svg_figures_exist": all(
                (image_dir / name).with_suffix(".svg").is_file() for name in figure_index["figure"]
            ),
            "exact_tables_byte_identical_to_v1_2": table_hashes_match,
            "statistics_recomputed": False,
            "hidden_reasoning_used": False,
            "implicit_proxies_aggregated": False,
            "reused_figures_hash_match": (
                all(
                    _sha256(image_dir / name) == expected
                    for name, expected in reuse_provenance["reused_png_sha256"].items()
                )
                if reuse_provenance is not None
                else True
            ),
        }
    )
    if not table_hashes_match or not validations["all_figures_exist"]:
        raise ValueError(f"v1.3 validation failed: {validations}")

    manifest = dict(source_manifest)
    manifest.update(
        {
            "bundle_version": BUNDLE_VERSION,
            "parent_bundle": str(source_bundle),
            "parent_manifest_sha256": _sha256(source_manifest_path),
            "visual_revision_only": True,
            "statistics_recomputed": False,
            "png_notebook_primary": True,
            "svg_export_complete": validations["all_svg_figures_exist"],
            "typography_reference": REFERENCE_NOTEBOOK,
            "visible_persona_setting_term": "skenario",
            "implicit_proxies_aggregated": False,
            "reuse_provenance": reuse_provenance,
            "typography": {
                "font_size": 14.5,
                "axes_title_size": 18,
                "axes_label_size": 17.5,
                "tick_label_size": 15.0,
                "figure_title_size": 20,
                "legend_size": 17,
                "heatmap_annotation_min_size": 12.5,
            },
            "exact_table_sha256": table_hashes,
            "generated_tables": [*TABLE_FILES, "figure_index.csv"],
            "generated_figures": figure_index["figure"].tolist(),
            "generated_svg_figures": [
                Path(name).with_suffix(".svg").name
                for name in figure_index["figure"]
                if (image_dir / name).with_suffix(".svg").is_file()
            ],
            "validations": validations,
        }
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[DONE] Readable visual bundle: {output_dir}")
    return output_dir


def main() -> int:
    args = _parser().parse_args()
    rerender(
        args.source_bundle,
        args.output_dir,
        reuse_figures_from=args.reuse_figures_from,
        reuse_through_number=args.reuse_through_number,
        start_number=args.start_number,
        end_number=args.end_number,
        resume=args.resume,
        png_only=args.png_only,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
