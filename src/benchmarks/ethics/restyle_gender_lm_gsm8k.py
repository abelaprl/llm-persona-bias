"""Render the frozen ETHICS figure tables with the GSM8K notebook style."""
from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.benchmarks.ethics.gender_lm_v2 import reporting as report

BUNDLE = ROOT / "result/benchmarks/ethics/reports/gender_lm_v2_candidate_20260916_r3"
REFERENCE = ROOT / "src/tools/build_gender_lm_visual_revision_v2.py"
STYLE_VERSION = "gender-ethics-gsm8k-presentation-v1"

LABELS = {
    "organization_explicit_step": "Langkah eksplisit", "organization_bullet": "Bullet",
    "organization_section_header": "Section header", "stance_hedge": "Hedge",
    "stance_booster": "Booster", "stance_assumption": "Asumsi eksplisit",
    "stance_self_correction": "Self-correction", "stance_contextual_hedge": "Hedge kontekstual",
    "stance_distancing": "Distancing", "connective_causal": "Konektif kausal",
    "connective_contrastive": "Konektif kontras", "connective_conditional": "Konektif kondisional",
    "connective_additive": "Konektif aditif", "connective_concessive": "Konektif konsesif",
    "self_reference_first_person": "Rujukan orang pertama", "self_reference_inclusive_we": "Inclusive we",
    "engagement_reader_directive": "Direktif pembaca", "elaboration_code_gloss": "Code gloss",
    "ai_identity_statement": "Identitas AI", "response_stance_committed": "Stance committed",
    "response_stance_conditional": "Stance kondisional", "response_stance_refusal": "Refusal",
    "affect_empathy_validation": "Empati / validasi", "framing_deontological": "Framing deontologis",
    "framing_consequentialist": "Framing konsekuensialis", "framing_virtue_ethics": "Framing virtue ethics",
    "framing_religious": "Framing keagamaan", "framing_cultural": "Framing budaya",
    "moral_evaluation_positive": "Evaluasi moral positif", "moral_evaluation_negative": "Evaluasi moral negatif",
    "gender_persona_direct_address": "Sapaan persona gender",
    "gender_persona_acknowledgment": "Pengakuan persona gender",
    "gender_second_person_identity_reference": "Identitas orang kedua",
    "gender_hypothetical_identity_reference": "Identitas hipotetis",
    "gender_first_person_identity_reference": "Identitas orang pertama",
    "gender_perspective_adoption": "Adopsi perspektif gender",
    "gender_identity_relevance_denial": "Penolakan relevansi gender",
    "gender_identity_neutralization": "Netralisasi identitas gender",
    "gender_identity_discourse": "Wacana identitas gender",
    "gender_role_framing": "Framing peran gender",
    "gendered_normative_evaluation": "Evaluasi normatif gender",
}
FAMILIES = {
    "reasoning_organization": "Organisasi", "epistemic_stance": "Sikap epistemik",
    "logical_connective": "Konektif logis", "self_reference": "Rujukan diri",
    "engagement": "Engagement", "textual_elaboration": "Elaborasi",
    "model_identity": "Identitas model", "response_stance": "Sikap respons",
    "affective_language": "Bahasa afektif", "ethical_content_framing": "Framing etika",
    "moral_evaluation_language": "Evaluasi moral", "gender_persona_uptake": "Respons terhadap persona gender",
    "gender_conditional_identity_reference": "Rujukan gender kondisional",
    "gender_identity_discourse": "Wacana identitas gender", "gender_role_discourse": "Peran gender",
    "gendered_normative_language": "Bahasa normatif gender",
}


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_reference_style():
    """Reuse only the visual constants/function, without importing legacy statistics."""
    tree = ast.parse(REFERENCE.read_text(encoding="utf-8"))
    selected = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_plot_style":
            selected.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in
                                                 {"MODEL_LABELS", "IDENTITY_COLORS"} for t in node.targets):
            selected.append(node)
    namespace = {"plt": plt}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(REFERENCE), "exec"), namespace)
    namespace["_plot_style"]()
    plt.rcParams["svg.fonttype"] = "none"
    return namespace["MODEL_LABELS"], namespace["IDENTITY_COLORS"]


def readable(text, model_labels, wrap=None):
    text = str(text)
    statuses = {"substantive_explanation": "Penjelasan tersedia", "verdict_only": "Verdict saja",
                "technical_only": "Teknis saja", "unknown_surface": "Status tidak diketahui"}
    for source, target in sorted({**model_labels, **LABELS, **FAMILIES, **statuses}.items(), key=lambda x: -len(x[0])):
        text = text.replace(source, target)
    return textwrap.fill(text, wrap, break_long_words=False) if wrap else text


def scenario(persona):
    if persona == "neutral":
        return "N"
    return "E" if "explicit" in persona else "I" + persona.rsplit("_", 1)[-1]


def forest_from_table(data, kind, models, colors):
    """Put scenarios in columns and model-specific paired estimates in rows."""
    data = data.copy()
    direct = kind == "forest_woman_man"
    interaction = kind == "forest_interaction"
    data["display_scenario"] = [scenario(left) if direct or kind == "forest_neutral"
                                else scenario(left) + " − " + scenario(right)
                                for left, right in zip(data.left, data.right)]
    data["display_gender"] = "direct" if direct or interaction else data.left.str.split("_").str[0]
    scenarios = list(dict.fromkeys(data.display_scenario))
    order = {key: i for i, key in enumerate(("E", "I1", "I2", "I3", "I1 − E", "I2 − E", "I3 − E",
                                            "I1 − I2", "I1 − I3", "I2 − I3"))}
    scenarios.sort(key=lambda key: order.get(key, 99))
    ids = list(dict.fromkeys(data.marker_id))
    genders = [gender for gender in ("man", "woman", "direct") if gender in set(data.display_gender)]
    row_keys = [(model, gender) for model in models for gender in genders]
    width = max(20, 4.1 * len(scenarios) + 3.5)
    height = (4.7 if len(genders) == 2 else 3.15) * len(ids) + 2.0
    fig, axes = plt.subplots(len(ids), len(scenarios), figsize=(width, height), squeeze=False)
    plotted = 0
    unavailable = 0
    for row_index, marker_id in enumerate(ids):
        marker_data = data.loc[data.marker_id.eq(marker_id)]
        finite = marker_data[["effect", "ci_low", "ci_high"]].to_numpy(float)
        bound = max(float(np.nanmax(np.abs(finite))) * 1.12, 0.001) if np.isfinite(finite).any() else 1
        for column, condition in enumerate(scenarios):
            ax = axes[row_index, column]
            panel = marker_data.loc[marker_data.display_scenario.eq(condition)]
            assert not panel.duplicated(["model_id", "display_gender"]).any()
            for point in panel.itertuples():
                y = len(row_keys) - 1 - row_keys.index((point.model_id, point.display_gender))
                if pd.isna(point.effect):
                    ax.text(0, y, "NA", ha="center", va="center", fontsize=14)
                    unavailable += 1
                    continue
                color = colors.get(point.display_gender, "#264653")
                alpha = .95 if pd.notna(point.adjusted_p) and point.adjusted_p < .05 else .55
                if pd.notna(point.ci_low) and pd.notna(point.ci_high):
                    ax.plot([point.ci_low, point.ci_high], [y, y], color=color, alpha=alpha, linewidth=1.45)
                    ax.vlines([point.ci_low, point.ci_high], y - .12, y + .12, color=color, alpha=alpha, linewidth=1.1)
                ax.plot(point.effect, y, marker="o", linestyle="", markersize=7.5, color=color,
                        markerfacecolor="none" if point.n_changed < 10 else color, alpha=alpha)
                plotted += 1
            ax.axvline(0, color="#334155", linewidth=.8)
            ax.set_xlim(-bound, bound)
            ax.set_ylim(-.65, len(row_keys) - .35)
            ax.grid(axis="x", color="#E2E8F0", linewidth=.6)
            ax.locator_params(axis="x", nbins=4)
            if row_index == 0:
                ax.set_title(condition, fontweight="bold")
            if column == 0:
                ax.set_yticks(range(len(row_keys) - 1, -1, -1),
                             [models[model] + (" · " + gender.title() if gender != "direct" else "")
                              for model, gender in row_keys], fontsize=14.5)
                ax.set_ylabel(textwrap.fill(LABELS[marker_id], 18), fontweight="bold", fontsize=16)
            else:
                ax.set_yticks([])
    if len(genders) == 2:
        fig.legend([Line2D([], [], marker="o", color=colors[g]) for g in genders],
                   [g.title() for g in genders], loc="lower center", bbox_to_anchor=(.5, .62 / height), ncol=2)
    units = {"count": "Selisih rerata jumlah marker", "density": "Selisih rerata marker per 100 kata",
             "presence": "Selisih prevalensi (poin persentase)"}
    xlabel = units[data.measurement.iloc[0]] + " · CI berpasangan nominal 95%"
    if direct:
        xlabel = "Woman − Man · " + xlabel
    if interaction:
        xlabel = "Perubahan selisih Woman − Man · " + xlabel
    fig.supxlabel(xlabel, y=(1.15 if len(genders) == 2 else .78) / height, fontsize=17.5)
    fig.subplots_adjust(left=3.7 / width, right=1 - .3 / width, top=1 - 1.05 / height,
                        bottom=(2.15 if len(genders) == 2 else 1.8) / height, hspace=.32, wspace=.14)
    assert plotted + unavailable == len(data)
    return fig, {"plotted_points": plotted, "unavailable_points": unavailable, "source_rows": len(data)}


class StyledFigures:
    def __init__(self, output, selected=None):
        self.root = output
        self.folder = output / "figures"
        self.folder.mkdir()
        self.original = pd.read_csv(BUNDLE / "figure_index.csv").fillna("").set_index("figure_id")
        self.rows = []
        self.audits = []
        self.selected = selected
        self.models, self.colors = load_reference_style()
        self.expected_hashes = {str(REFERENCE.relative_to(ROOT)): digest(REFERENCE)}

    def save(self, fig, data, name, title, scope, kind, measurement="", estimand="", denominator=""):
        stem = report._slug(name)
        if self.selected and kind not in self.selected:
            plt.close(fig)
            return
        original = self.original.loc[stem]
        source = BUNDLE / original.source_csv
        assert digest(source) == original.source_sha256
        frozen = pd.read_csv(source)
        pd.testing.assert_frame_equal(data.reset_index(drop=True), frozen, check_dtype=False,
                                      check_exact=False, rtol=1e-11, atol=1e-12)
        point_audit = {}
        if kind.startswith("forest_"):
            plt.close(fig)
            fig, point_audit = forest_from_table(frozen, kind, self.models, self.colors)
        else:
            width, height = fig.get_size_inches()
            scale = max(1, 21 / width)
            fig.set_size_inches(width * scale, max(height * scale, 7))
            for ax in fig.axes:
                ax.set_title(readable(ax.get_title(), self.models, 38), fontsize=18, fontweight="bold")
                for text in ax.get_xticklabels() + ax.get_yticklabels():
                    text.set_fontsize(15.5)
                ax.xaxis.label.set_fontsize(17.5)
                ax.yaxis.label.set_fontsize(17.5)
                for text in ax.texts:
                    text.set_fontsize(13.5 if kind == "marker_atlas" else 14.5)
                if kind != "audit":
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                if kind == "neutral_triplet_map":
                    ax.locator_params(axis="both", nbins=4)
                    units = {"count": "selisih rerata jumlah marker",
                             "density": "selisih rerata marker / 100 kata",
                             "presence": "selisih prevalensi (pp)"}
                    ax.set_xlabel("Man − Neutral\n" + units[measurement], fontsize=17.5,
                                  color=self.colors["man"])
                # Existing plots use fixed tick locators for named marker rows.
                yticks = ax.get_yticklabels()
                if any(t.get_text() in LABELS for t in yticks):
                    ax.set_yticks(ax.get_yticks(), [readable(t.get_text(), self.models, 18) for t in yticks], fontsize=14.5)
            for text in fig.findobj(Text):
                text.set_text(readable(text.get_text(), self.models))
            for legend in fig.legends:
                for text in legend.get_texts():
                    text.set_fontsize(15.5)
                    text.set_text(readable(text.get_text(), self.models, 32))
                legend.get_title().set_fontsize(17)
            if kind == "neutral_triplet_map":
                fig.legends[0].set_bbox_to_anchor((.5, .045))
                fig.legends[1].set_bbox_to_anchor((.5, .205))
            bottom = .31 if kind == "neutral_triplet_map" else .17 if fig.legends else .1
            fig.tight_layout(rect=(.01, bottom, .99, .91), h_pad=2.4, w_pad=2.3)
        fig.suptitle(readable(title, self.models, 110), fontsize=20, fontweight="bold", y=1 - .18 / fig.get_figheight())
        note = "Candidate · keseluruhan ETHICS · NA ≠ 0 · satu respons = satu observasi · denominator tersedia pada tabel sumber."
        if kind == "response_length":
            note = "Candidate · kotak: IQR; garis hitam: median; titik: rerata; whisker: persentil 5–95 (bukan CI). NA ≠ 0."
        elif kind == "condition_profile":
            note = "Candidate · pergeseran terhadap Neutral · CI nominal 95% · titik kosong: n_changed < 10 · NA ≠ 0."
        elif kind == "neutral_triplet_map":
            note = "Candidate · common triplet pada soal yang sama · dekat diagonal bukan bukti kesetaraan · warna: marker; bentuk: skenario."
        if kind.startswith("forest_"):
            note = "Candidate · pekat: p Holm < 0,05; pudar: lainnya · titik kosong: n_changed < 10 · CI nominal 95%."
        fig.text(.02, .12 / fig.get_figheight(), note, fontsize=13.5, va="bottom")
        png = self.root / original.png_path
        svg = self.root / original.svg_path
        local_csv = self.root / original.source_csv
        shutil.copy2(source, local_csv)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        outside = []
        for text in fig.findobj(Text):
            if not text.get_visible() or not text.get_text():
                continue
            box = text.get_window_extent(renderer)
            if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.width + 1 or box.y1 > fig.bbox.height + 1:
                outside.append(text.get_text())
        fig.savefig(png, dpi=230, bbox_inches="tight", pad_inches=.18, facecolor="white")
        fig.savefig(svg, bbox_inches="tight", pad_inches=.18, facecolor="white")
        plt.close(fig)
        row = original.to_dict()
        row.update(figure_id=stem, png_sha256=digest(png), svg_sha256=digest(svg), style_version=STYLE_VERSION,
                   original_png_sha256=original.png_sha256)
        self.rows.append(row)
        self.audits.append(dict(figure_id=stem, figure_kind=kind, source_csv_identical=digest(local_csv) == original.source_sha256,
                               untrimmed_canvas_outside_text=outside, tight_bbox_export=True, **point_audit))
        self.expected_hashes[str(source.relative_to(ROOT))] = original.source_sha256
        print(f"Rendered {len(self.rows)}/88: {kind} / {stem}", flush=True)
        gc.collect()


def merge_parts(output, parts, replace_kinds=()):
    original = pd.read_csv(BUNDLE / "figure_index.csv").set_index("figure_id")
    indices, audits, hashes = [], [], {}
    manifest = None
    for part in parts:
        manifest = json.loads((part / "presentation_manifest.json").read_text(encoding="utf-8"))
        assert manifest["style_version"] == STYLE_VERSION
        hashes.update(manifest["source_hashes"])
        index = pd.read_csv(part / "figure_index.csv")
        if replace_kinds and part != parts[-1]:
            index = index.loc[~index.figure_kind.isin(replace_kinds)]
        for row in index.itertuples():
            expected = original.loc[row.figure_id]
            assert row.source_sha256 == expected.source_sha256
            for name, expected_hash in ((row.png_path, row.png_sha256), (row.svg_path, row.svg_sha256),
                                        (row.source_csv, row.source_sha256)):
                source, target = part / name, output / name
                assert digest(source) == expected_hash
                if target.exists():
                    raise FileExistsError(target)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        indices.append(index)
        selected_ids = set(index.figure_id)
        audits.extend(row for row in json.loads((part / "visual_checks.json").read_text(encoding="utf-8"))
                      if row["figure_id"] in selected_ids)
    combined = pd.concat(indices, ignore_index=True)
    assert combined.figure_id.is_unique and set(combined.figure_id) == set(original.index)
    combined = combined.set_index("figure_id").loc[original.index].reset_index()
    assert all(digest(ROOT / name) == expected_hash for name, expected_hash in hashes.items())
    combined.to_csv(output / "figure_index.csv", index=False)
    (output / "visual_checks.json").write_text(json.dumps(audits, indent=2), encoding="utf-8")
    manifest.update(figures=len(combined), source_hashes=hashes, selected_kinds=None,
                    source_parts=[str(path.relative_to(ROOT)) for path in parts])
    (output / "presentation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Merged and verified {len(combined)} figures; all source CSVs are byte-identical", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--kinds", nargs="+", help="Render selected figure kinds for visual inspection.")
    parser.add_argument("--merge-parts", nargs="+", type=Path)
    parser.add_argument("--replace-kinds", nargs="+", default=[])
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    if args.merge_parts:
        merge_parts(output, [path.resolve() for path in args.merge_parts], args.replace_kinds)
        return
    book = StyledFigures(output, set(args.kinds) if args.kinds else None)
    report.COLORS.update(book.colors)
    registry = json.loads((BUNDLE / "registry.json").read_text(encoding="utf-8"))
    markers = report._registry(registry, pd.DataFrame())
    assert set(LABELS) == {m["marker_id"] for m in markers}
    frames = {}
    for name in ("report_status_coverage", "response_length_summary", "marker_summary", "comparisons",
                 "interaction_comparisons", "neutral_triplet_comparisons", "audit/marker_audit_summary"):
        path = BUNDLE / (name + ".csv")
        book.expected_hashes[str(path.relative_to(ROOT))] = digest(path)
        frames[name] = pd.read_csv(path)
    models = sorted(frames["marker_summary"].model_id.unique())
    enabled = lambda kind: not args.kinds or kind in args.kinds
    if enabled("coverage"):
        report._coverage_figure(book, frames["report_status_coverage"], models)
    if enabled("response_length"):
        report._response_length_figure(book, frames["response_length_summary"], models)
    if enabled("marker_atlas"):
        report._atlas(book, frames["marker_summary"], markers, models)
    combined = pd.concat([frames["comparisons"], frames["interaction_comparisons"]], ignore_index=True)
    display = report._registered_measurements(combined, markers)
    if not args.kinds or any(kind.startswith("forest_") for kind in args.kinds):
        report._forest(book, display, markers, models)
    if enabled("condition_profile"):
        report._condition_profiles(book, display, markers, models)
    if enabled("neutral_triplet_map"):
        report._triplet_maps(book, report._registered_measurements(frames["neutral_triplet_comparisons"], markers), markers, models)
    if enabled("audit"):
        report._audit_figure(book, frames["audit/marker_audit_summary"], markers)
    index = pd.DataFrame(book.rows)
    expected = book.original if not args.kinds else book.original.loc[book.original.figure_kind.isin(args.kinds)]
    assert set(index.figure_id) == set(expected.index)
    assert all(digest(ROOT / name) == expected_hash for name, expected_hash in book.expected_hashes.items())
    index.to_csv(output / "figure_index.csv", index=False)
    (output / "visual_checks.json").write_text(json.dumps(book.audits, indent=2), encoding="utf-8")
    manifest = dict(status="candidate", style_version=STYLE_VERSION, figures=len(index), source_bundle=str(BUNDLE.relative_to(ROOT)),
                    style_reference=str(REFERENCE.relative_to(ROOT)), source_hashes=book.expected_hashes,
                    all_source_csvs_identical=True, source_files_unchanged=True, statistics_recomputed=False,
                    font_family="DejaVu Sans", base_font_pt=14.5, figure_title_pt=20, panel_title_pt=18,
                    axis_label_pt=17.5, tick_label_pt=15.5, dpi=230, selected_kinds=args.kinds)
    (output / "presentation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in manifest.items() if key != "source_hashes"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
