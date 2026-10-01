"""Build a new, source-bound diagnostic presentation without recomputing tests."""
import argparse
import gc
import hashlib
import html
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd
from src.benchmarks.ethics import gender_lm_diagnostic_data as data_tools
from src.benchmarks.ethics import gender_lm_diagnostic_plots as plots
from src.benchmarks.ethics.restyle_gender_lm_gsm8k import BUNDLE, LABELS, report


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


class DiagnosticWriter:
    def __init__(self, output):
        self.root = Path(output)
        self.root.mkdir(parents=True, exist_ok=False)
        (self.root / "figures").mkdir()
        (self.root / "diagnostic_tables").mkdir()
        self.rows = []
        self.audits = []

    def save(self, fig, data, name, kind, title, marker_ids, audit):
        stem = "diagnostic_" + name
        png, svg = "figures/" + stem + ".png", "figures/" + stem + ".svg"
        csv = "diagnostic_tables/" + stem + ".csv"
        for path in [png, svg, csv]:
            if (self.root / path).exists():
                raise FileExistsError(self.root / path)
        data.to_csv(self.root / csv, index=False)
        readback = pd.read_csv(self.root / csv, float_precision="round_trip")
        pd.testing.assert_frame_equal(data.reset_index(drop=True), readback,
                                      check_dtype=False, check_exact=True)
        fig.savefig(self.root / png, dpi=230, bbox_inches="tight", pad_inches=.20, facecolor="white")
        fig.savefig(self.root / svg, bbox_inches="tight", pad_inches=.20, facecolor="white")
        plt.close(fig)
        self.rows.append(dict(figure_id=stem, figure_kind=kind, title=title,
                              marker_ids=";".join(marker_ids), png_path=png, svg_path=svg, source_csv=csv,
                              png_sha256=digest(self.root/png), svg_sha256=digest(self.root/svg),
                              source_sha256=digest(self.root/csv), audit_status="candidate"))
        self.audits.append(dict(figure_id=stem, csv_roundtrip_exact=True, **audit))
        print(f"Rendered {len(self.rows)}: {stem}", flush=True)
        gc.collect()


def build(output, markers=None):
    writer = DiagnosticWriter(output)
    names = ["registry.json", "response_markers.csv.gz", "marker_spans_reviewed.csv.gz", "comparisons.csv"]
    source_hashes = {name: digest(BUNDLE/name) for name in names}
    registry_list = json.loads((BUNDLE / "registry.json").read_text(encoding="utf-8"))
    registry = pd.DataFrame(registry_list)
    ids = registry.marker_id.tolist() if markers is None else markers
    if not set(ids).issubset(set(registry.marker_id)):
        raise ValueError("Unknown requested marker")
    print("Reading frozen metadata, spans and contrasts; no raw/held-out text is read", flush=True)
    responses = pd.read_csv(BUNDLE/"response_markers.csv.gz", usecols=["response_id", "model_id", "persona", "text_available"])
    spans = pd.read_csv(BUNDLE/"marker_spans_reviewed.csv.gz", usecols=["response_id", "marker_id", "reference_type", "flags"])
    comparisons = pd.read_csv(BUNDLE/"comparisons.csv", float_precision="round_trip")
    assert set(responses.model_id) == set(plots.MODEL_LABELS)
    primary = data_tools.direct_rows(comparisons, registry)
    all_direct = data_tools.direct_rows(comparisons, registry, primary_only=False)
    gender_ids = registry.loc[registry.registry_scope.eq("gender_specific"), "marker_id"].tolist()
    reference = data_tools.reference_summary(responses, spans, set(gender_ids))
    fig, audit = plots.reference_figure(reference)
    writer.save(fig, reference, "gender_reference_context", "gender_reference_context",
                "Konteks rujukan gender per respons", gender_ids, audit)
    support = data_tools.support_summary(primary)
    fig, audit = plots.support_overview(support)
    writer.save(fig, support, "pair_support_overview", "pair_support_overview",
                "Dukungan pasangan dan marker low-support", registry.marker_id.tolist(), audit)
    overall = primary.loc[primary.scope.eq("overall")]
    for family, page, page_ids in report._marker_pages(report._registry(registry_list, pd.DataFrame())):
        selected = [marker for marker in page_ids if marker in ids]
        if not selected:
            continue
        part = overall.loc[overall.marker_id.isin(selected)]
        fig, audit = plots.support_detail(part, selected)
        writer.save(fig, part, f"support_markers_{page:02}", "marker_pair_support",
                    f"Dukungan marker · kelompok {page}", selected, audit)
    for marker in ids:
        part = primary.loc[primary.marker_id.eq(marker)]
        assert len(part) == 6*4*5
        fig, audit = plots.subset_figure(part)
        writer.save(fig, part, "subset_"+marker, "subset_sensitivity",
                    "Sensitivitas subset · " + LABELS[marker], [marker], audit)
        if registry.set_index("marker_id").loc[marker, "measurement"] == "density":
            part = all_direct.loc[all_direct.marker_id.eq(marker) & all_direct.scope.eq("overall") &
                                  all_direct.measurement.isin(["count", "density"])]
            assert len(part) == 6*4*2
            fig, audit = plots.normalization_figure(part)
            writer.save(fig, part, "normalization_"+marker, "normalization_sensitivity",
                        "Sensitivitas normalisasi · " + LABELS[marker], [marker], audit)
    index = pd.DataFrame(writer.rows)
    index.to_csv(writer.root / "diagnostic_figure_index.csv", index=False)
    assert all(digest(BUNDLE/name) == expected for name, expected in source_hashes.items())
    manifest = dict(status="candidate", complete_registry=set(ids)==set(registry.marker_id),
                    figure_count=len(index), source_bundle=BUNDLE.relative_to(ROOT).as_posix(),
                    source_hashes=source_hashes, markers=ids, statistics_recomputed=False,
                    reference_unit="one_response_per_indicator", reference_indicators_overlap=True,
                    binary_normalization_pages=False, old_figures_modified=False,
                    kind_counts=index.figure_kind.value_counts().to_dict(),
                    plotted_points=sum(row.get("numeric_points", 0) for row in writer.audits),
                    na_points=sum(row.get("na_points", 0) for row in writer.audits),
                    source_files_unchanged=True, font_family="DejaVu Sans", dpi=230)
    (writer.root / "diagnostic_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (writer.root / "diagnostic_visual_checks.json").write_text(json.dumps(writer.audits, indent=2), encoding="utf-8")
    sections = []
    for row in index.itertuples():
        sections.append(f'<section id="{row.figure_id}"><h2>{html.escape(row.title)}</h2>'
                        f'<p>Candidate · <a href="{row.png_path}">PNG penuh</a> · '
                        f'<a href="{row.svg_path}">SVG</a> · <a href="{row.source_csv}">CSV sumber</a></p>'
                        f'<img loading="lazy" src="{row.png_path}" alt="{html.escape(row.title)}"></section>')
    contents = ''.join(f'<li><a href="#{row.figure_id}">{html.escape(row.title)}</a></li>' for row in index.itertuples())
    page = ('<!doctype html><html lang="id"><meta charset="utf-8"><title>Diagnostik Gender ETHICS</title>'
            '<style>body{font:17px system-ui;margin:2rem;color:#172554}img{max-width:100%;height:auto}'
            'section{margin:4rem 0;border-top:1px solid #ccc}a{color:#2563eb}</style>'
            '<h1>Pelengkap visual Gender ETHICS</h1><p>Seluruh hasil tetap candidate. '
            'CI nominal 95%; p Holm dari r3. Unknown bukan bukti ketiadaan rujukan. '
            'Respons tanpa penjelasan tidak dianggap nol marker. Klik PNG untuk membaca ukuran penuh.</p>'
            '<details><summary>Daftar seluruh halaman</summary><ol>'+contents+'</ol></details>'+''.join(sections)+'</html>')
    (writer.root / "diagnostic_gallery.html").write_text(page, encoding="utf-8")
    print(json.dumps({k:v for k,v in manifest.items() if k not in {"source_hashes", "markers"}}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--markers", nargs="+")
    args = parser.parse_args()
    build(args.output_dir, args.markers)
