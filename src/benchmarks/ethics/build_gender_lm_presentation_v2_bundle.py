"""Build and add the complete Gender ETHICS presentation without deleting old work."""
import argparse
import base64
import html
import io
import json
import os
from pathlib import Path
import shutil
from urllib.parse import quote

import matplotlib.pyplot as plt
import nbformat
from nbclient import NotebookClient
import pandas as pd

from .gender_lm_presentation_v2_data import (
    MARKER_GROUPS, GROUP_LABELS, SCOPES, coverage_table, source_table, view_data, view_specs,
)
from .gender_lm_presentation_v2_plots import PRINT_STYLE, audit_geometry, plot_coverage, plot_view, view_caption
from .gender_lm_presentation_v2_explorer import write_explorer
from .gender_lm_presentation_v2_evidence import build_examples, inline_example
from .gender_lm_reference_data import ACTIVE, SOURCE, digest
from .build_gender_lm_presentation_v2_preview import _validate_registry


def _json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def _local_link(target, origin):
    """Prefer portable relative links, with a file URI for different drives."""
    try:
        return quote(os.path.relpath(target, origin).replace("\\", "/"), safe="/.")
    except ValueError:
        return Path(target).resolve().as_uri()


def _protected_hashes():
    names = ["registry.json", "comparisons.csv", "marker_summary.csv", "response_markers.csv.gz",
             "report_status_coverage.csv", "response_length_summary.csv", "marker_spans_reviewed.csv.gz",
             "response_surfaces.csv.gz", "evidence_index.csv", "evidence_explorer.html"]
    paths = [SOURCE / name for name in names]
    paths += [p for p in ACTIVE.iterdir() if p.is_file()]
    paths += [p for p in (ACTIVE / "figures").iterdir() if p.is_file()]
    return {p.resolve().as_posix(): digest(p) for p in sorted(paths)}


def _save_figure(fig, output, stem):
    with plt.rc_context({**PRINT_STYLE, "savefig.bbox": None}):
        fig.savefig(output / (stem + ".png"), dpi=300)
        fig.savefig(output / (stem + ".svg"))


def _code_hashes():
    files = [Path(__file__)] + list(Path(__file__).parent.glob("gender_lm_presentation_v2_*.py"))
    files += [Path(__file__).with_name(name) for name in ["gender_lm_reference_data.py",
              "gender_lm_reference_plots.py", "build_gender_lm_presentation_v2_preview.py",
              "restyle_gender_lm_gsm8k.py"] if Path(__file__).with_name(name).is_file()]
    return {path.name: digest(path) for path in files}


def _execute_notebook(output, figures, coverage):
    cells = [nbformat.v4.new_markdown_cell("# Penanda linguistik Gender ETHICS — seluruh 41 marker\n\n"
             "Paket baru, tanpa mengganti notebook lama. Warna menyatakan model; empat skenario E/I1/I2/I3. "
             "Seluruh marker candidate dan keterbatasan sumber tetap berlaku.\n\n"
             "[Buka 81 tampilan grafik](explorer.html) · [Contoh ungkapan](examples.html) · [Metode](README.md)"),
             nbformat.v4.new_code_cell("from pathlib import Path\nfrom IPython.display import Image, display\nimport pandas as pd\nbase = Path.cwd()")]
    for record in figures:
        stem = record["figure_id"]
        cells.append(nbformat.v4.new_code_cell(f"display(Image(filename=str(base / {stem + '.png'!r})))"))
        cells.append(nbformat.v4.new_markdown_cell(record["caption"] + f"\n\n[Tabel pembentuk]({stem}.csv)"))
        if stem == "G00_cakupan":
            cells.append(nbformat.v4.new_markdown_cell("Cakupan tidak sama antar-model/persona; baca jumlah mentah dan denominator pada CSV. "
                         "[Status penjelasan](text_status.csv) · [Panjang respons](response_length.csv). "
                         "Tabel panjang sumber menyertakan verdict-only; populasinya bukan hanya penjelasan substantif."))
        elif stem.startswith(("B08", "B09")):
            cells.append(nbformat.v4.new_markdown_cell("Istilah gender dapat berasal dari soal, bukan pengguna. "
                         "[Baca dua triplet contoh dan penjelasan utuh](examples.html); contoh dipilih purposif, bukan representatif."))
            example_id = "contoh_1" if stem.startswith("B08") else "contoh_2"
            cells.append(nbformat.v4.new_markdown_cell(inline_example(output, example_id)))
    rows = ["| Marker | Profil | Man−Woman | Persona−N | I−E |", "|---|---|---|---|---|"]
    for marker, block in coverage.groupby("marker_id", sort=False):
        links = {r.mode: r.chart_link for r in block.itertuples(index=False)}
        rows.append("| " + marker + " | " + " | ".join(f"[Grafik]({links[m]})" for m in ["profile", "direct", "neutral", "implicit"]) + " |")
    cells.append(nbformat.v4.new_markdown_cell("## Matriks cakupan visual\n\n" + "\n".join(rows)))
    cells.append(nbformat.v4.new_code_cell("display(pd.read_csv(base / 'marker_context.csv').head(12))"))
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})
    NotebookClient(notebook, timeout=180, kernel_name="python3", resources={"metadata": {"path": str(output.resolve())}}).execute()
    image_count = sum("image/png" in item.get("data", {}) for cell in notebook.cells for item in cell.get("outputs", []))
    if image_count != 10:
        raise ValueError("Notebook does not contain all ten images")
    with (output / "gender_lm_presentation_v2.ipynb").open("x", encoding="utf-8") as handle:
        nbformat.write(notebook, handle)
    return image_count


def build_bundle(output):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    print("[bundle] Memeriksa sumber dan seluruh artefak aktif", flush=True)
    before = _protected_hashes()
    registry = json.loads((SOURCE / "registry.json").read_text(encoding="utf-8"))
    _validate_registry(registry)
    comparisons, summary = source_table("comparisons.csv"), source_table("marker_summary.csv")
    specs = view_specs()
    selected = {}
    for spec in specs:
        selected[spec["view_id"]] = view_data(**{k: v for k, v in spec.items() if k != "view_id"}, comparisons=comparisons, summary=summary)
    context = pd.concat([selected[s["view_id"]] for s in specs if s["mode"] == "profile"])
    context = context.drop_duplicates(["scope", "model_id", "persona", "marker_id"]).drop(columns="panel_id")
    if len(context) != 2214 or context.loc[context.n_available.gt(0), "raw_count_total"].isna().any():
        raise ValueError("Incomplete marker context or missing raw counts")
    coverage = coverage_table(summary, comparisons)
    counts = {mode: sum(len(selected[s["view_id"]]) for s in specs if s["mode"] == mode) for mode in ["direct", "neutral", "implicit"]}
    if counts != dict(direct=984, neutral=1968, implicit=1476):
        raise ValueError(f"Unexpected contrast counts: {counts}")
    output.mkdir(parents=True, exist_ok=False)
    context.to_csv(output / "marker_context.csv", index=False)
    # Keep all scopes available for supporting tables without copying the large comparison file.
    pd.read_csv(SOURCE / "report_status_coverage.csv", float_precision="round_trip").to_csv(output / "text_status.csv", index=False)
    pd.read_csv(SOURCE / "response_length_summary.csv", float_precision="round_trip").to_csv(output / "response_length.csv", index=False)
    pd.DataFrame(registry).drop(columns="patterns").to_csv(output / "marker_catalog.csv", index=False)
    figures, views, geometry = [], [], []
    fig = plot_coverage(coverage)
    try:
        geometry.append(dict(view_id="G00_cakupan", **audit_geometry(fig)))
        _save_figure(fig, output, "G00_cakupan")
    finally:
        plt.close(fig)
    coverage.to_csv(output / "G00_cakupan.csv", index=False)
    figures.append(dict(figure_id="G00_cakupan", title="Cakupan", caption="Gambar G00. Cakupan penjelasan dan pasangan pada seluruh slot respons; persentase dibulatkan pada gambar, jumlah dan denominator tersedia pada tabel."))
    for i, spec in enumerate(specs, 1):
        print(f"[view {i}/81] {spec['view_id']}", flush=True)
        rows = selected[spec["view_id"]]
        fig = plot_view(rows, spec["group_id"], spec["mode"], spec.get("persona"), spec.get("scenario"))
        try:
            geometry.append(dict(view_id=spec["view_id"], **audit_geometry(fig)))
            svg = io.BytesIO()
            with plt.rc_context({**PRINT_STYLE, "savefig.bbox": None}):
                fig.savefig(svg, format="svg")
            caption = f"{spec['view_id']}\n" + view_caption(rows, spec["mode"])
            if spec["mode"] == "direct":
                _save_figure(fig, output, spec["view_id"])
                rows.to_csv(output / (spec["view_id"] + ".csv"), index=False)
                figures.append(dict(figure_id=spec["view_id"], title=GROUP_LABELS[spec["group_id"]], caption=caption))
            support = context.loc[context.marker_id.isin(MARKER_GROUPS[spec["group_id"]])]
            # Every statistic is available; user text is escaped before HTML insertion.
            table_html = "<h3>Tabel pembentuk; kolom sumber tetap utuh</h3>" + rows.to_html(index=False, escape=True)
            table_html += "<h3>Konteks semua penjelasan tersedia (bukan populasi pairwise-complete)</h3>"
            table_html += support[["model_id", "persona", "marker_id", "raw_count_total", "total", "n_available",
                                   "prevalence_percent", "density_per_100_words"]].to_html(index=False, escape=True)
            views.append({
                **spec, "persona": spec.get("persona"), "scenario": spec.get("scenario"),
                "marker_ids": MARKER_GROUPS[spec["group_id"]], "caption": caption, "table_html": table_html,
                "svg_data_uri": "data:image/svg+xml;base64," + base64.b64encode(svg.getvalue()).decode("ascii")})
        finally:
            plt.close(fig)
    write_explorer(views, output / "explorer.html")
    coverage_rows = []
    for group, markers in MARKER_GROUPS.items():
        for marker in markers:
            for mode in ["profile", "direct", "neutral", "implicit"]:
                actual = [v for v in views if v["group_id"] == group and v["mode"] == mode]
                coverage_rows.append(dict(marker_id=marker, group_id=group, mode=mode, rendered=True,
                                          view_count=len(actual), chart_link="explorer.html#" + actual[0]["view_id"],
                                          all_view_ids=";".join(v["view_id"] for v in actual)))
    matrix = pd.DataFrame(coverage_rows)
    matrix.to_csv(output / "visual_coverage.csv", index=False)
    _json(output / "view_manifest.json", [{k: v for k, v in view.items() if k not in {"table_html", "svg_data_uri"}} for view in views])
    for record in figures:
        for ext in ["png", "svg", "csv"]:
            record[ext + "_sha256"] = digest(output / (record["figure_id"] + "." + ext))
    pd.DataFrame(figures).to_csv(output / "figure_index.csv", index=False)
    print("[bundle] Membaca contoh ungkapan yang sudah terdeteksi", flush=True)
    evidence = build_examples(output)
    source_link = _local_link(SOURCE / "evidence_explorer.html", output)
    readme = f"""# Gender ETHICS — paket seluruh 41 marker

## SPEC / DESIGN

10 gambar statis: G00 cakupan dan B01–B09 direct Man−Woman. Semua 41 marker memiliki empat mode dalam [81 grafik offline](explorer.html). Warna selalu model; kelompok hanya navigasi, bukan skor gabungan. Seluruh default memakai scope overall, bukan klaim semua subset sudah digrafikkan.

## IMPLEMENTATION / CHECKLIST

- Semua 81 SVG benar-benar dirender. Matriks 164 sel marker×mode: [visual_coverage.csv](visual_coverage.csv).
- 984 baris direct, 1.968 persona−N, 1.476 I−E; 2.214 profil unik. N yang berulang bukan observasi tambahan.
- [Konteks marker](marker_context.csv) menyertakan `raw_count_total` dari baris count sumber, n_available, total, prevalence %, dan density/100 kata. `raw_occurrences` asli pada baris density/presence tetap NA; jangan menggunakan kolom kosong itu sebagai jumlah mentah.
- Grafik density memakai selisih rerata; prevalence memakai poin persentase. CI 95% marginal bukan CI rank-biserial atau CI terkoreksi Holm. Wilcoxon bukan uji estimand rerata yang sama dengan CI.
- Semua kolom sumber utuh. Man−Woman hanya pada `display_*`; kolom sumber direct tetap Woman−Man. `n_pairs`, `low_support`, Holm dan komposisi subset tetap tersedia dalam tabel HTML/CSV.
- [Dua triplet contoh](examples.html), [bentuk ungkapan seluruh model](expression_forms.csv), [penjelajah bukti sumber]({source_link}). Contoh purposif, bukan sampel representatif. Referen pengguna tidak diasumsikan dari kata gender saja.
- [Katalog dan sumber teori](marker_catalog.csv); kamus custom/adaptasi dan candidate tidak dipromosikan menjadi tervalidasi.

## TESTING

`python -m unittest src.benchmarks.ethics.tests.test_gender_lm_presentation_v2_bundle -v`

Reproduksi pada path baru: `python -m src.benchmarks.ethics.build_gender_lm_presentation_v2_bundle --output result/benchmarks/ethics/reports/<nama-baru>`.

Ekspor tambahan: `python -m src.benchmarks.ethics.build_gender_lm_presentation_v2_bundle --output result/benchmarks/ethics/reports/<nama-baru> --group B08 --mode neutral --persona Woman --scope overall`.
Untuk profile gunakan `--mode profile --scenario E`; scope dapat overall/commonsense/deontology/justice/virtue. Direktori existing ditolak.

## INTEGRATION

Notebook staging [gender_lm_presentation_v2.ipynb](gender_lm_presentation_v2.ipynb) sudah dieksekusi dan memuat sepuluh gambar. Notebook dan gambar lama tidak diganti. Installer `--install` hanya menambah salinan; nama bentrok berbeda isi diberi suffix. Gambar final tetap langsung di figures utama, tanpa subfolder versi.

## Keterbatasan

Candidate tetap candidate. Cakupan 65.678 penjelasan dari 81.756 slot tidak setara lintas model. GPT-OSS hanya 11–47 pasangan direct; CI dan dukungan harus dibaca bersama. Profil menggunakan semua penjelasan tersedia; delta menggunakan pairwise-complete. Tabel panjang sumber mencakup verdict-only, berbeda dari denominator marker. Ringkasan overall mengikuti komposisi eligible sumber, bukan rata-rata subset berbobot sama. Ini bukan akurasi ETHICS gabungan. Batasan tugas deontology/virtue tetap berlaku. I1–I3 bukan replikasi acak populasi gender. Tidak ada validasi kamus, parser, atau pemanggilan API baru.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    print("[bundle] Menjalankan notebook staging", flush=True)
    embedded = _execute_notebook(output, figures, matrix)
    after = _protected_hashes()
    if before != after:
        raise RuntimeError("Protected source/active artifact changed")
    _json(output / "source_hashes.json", dict(before=before, after=after))
    _json(output / "code_hashes.json", _code_hashes())
    report = dict(rendered_views=len(views), static_figures=len(figures), marker_count=41, visual_coverage_cells=len(matrix),
                  contrast_counts=counts, profile_rows=len(context), geometry=geometry, evidence=evidence,
                  notebook_embedded_png=embedded, protected_files_unchanged=True, protected_files=len(before))
    _json(output / "verification.json", report)
    print("Selesai: 10 gambar utama, 81 tampilan, 41 marker; tidak ada file lama berubah.", flush=True)
    return report


def install_additive(bundle, active=ACTIVE):
    """Copy new artifacts exclusively; identical existing files are reused."""
    bundle, active = Path(bundle), Path(active)
    destination = active / "figures"
    if not destination.is_dir():
        raise FileNotFoundError(destination)
    before = {p: digest(p) for p in active.iterdir() if p.is_file()}
    before.update({p: digest(p) for p in destination.iterdir() if p.is_file()})
    index = pd.read_csv(bundle / "figure_index.csv")
    mappings = []
    sources = [bundle / f"{stem}.{ext}" for stem in index.figure_id for ext in ["png", "svg", "csv"]]
    if (bundle / "marker_context.csv").is_file():
        sources.append(bundle / "marker_context.csv")
    for source in sources:
        stem = "gender_lm_v2_marker_context" if source.name == "marker_context.csv" else source.stem
        target = destination / (stem + source.suffix)
        version = 2
        while target.exists() and digest(target) != digest(source):
            target = destination / f"{stem}_v{version}{source.suffix}"
            version += 1
        reused = target.exists()
        if not reused:
            with source.open("rb") as src, target.open("xb") as dst:
                shutil.copyfileobj(src, dst)
        if digest(source) != digest(target):
            raise RuntimeError("Installed artifact differs from source")
        mappings.append(dict(source=source.name, target=target.name, reused=reused, sha256=digest(target)))
    portal = active / "gender_lm_v2_gallery.html"
    number = 2
    while portal.exists():
        portal = active / f"gender_lm_v2_gallery_{number}.html"
        number += 1
    bundle_link = _local_link(bundle, active)
    sections = []
    for record in index.itertuples(index=False):
        mapping = next(m for m in mappings if m["source"] == record.figure_id + ".png")
        table_mapping = next(m for m in mappings if m["source"] == record.figure_id + ".csv")
        sections.append(f'<h2>{html.escape(record.title)}</h2><figure><img alt="{html.escape(record.figure_id)}" '
                        f'src="figures/{quote(mapping["target"])}" style="width:16cm;max-width:100%">'
                        f'<figcaption>{html.escape(record.caption)}</figcaption></figure>'
                        f'<p><a href="figures/{quote(table_mapping["target"])}">Tabel pembentuk CSV</a></p>')
    document = f'<!doctype html><html lang="id"><meta charset="utf-8"><title>Gender ETHICS v2</title><body style="font-family:system-ui;margin:24px"><h1>Gender ETHICS — 41 marker</h1><p>Gambar lama dipertahankan.</p><p><a href="{bundle_link}/explorer.html">Buka 81 tampilan grafik</a> · <a href="{bundle_link}/gender_lm_presentation_v2.ipynb">Notebook baru</a> · <a href="{bundle_link}/examples.html">Contoh ungkapan</a></p>' + "".join(sections) + "</body></html>"
    with portal.open("x", encoding="utf-8") as handle:
        handle.write(document)
    if any(digest(path) != original for path, original in before.items()):
        raise RuntimeError("Pre-existing active file changed")
    result = dict(bundle=bundle.resolve().as_posix(), portal=portal.resolve().as_posix(), mappings=mappings,
                  old_files_unchanged=True, protected_old_files=len(before))
    _json(portal.with_suffix(".json"), result)
    return result


def export_view(output, group, mode, persona=None, scenario=None, scope="overall"):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    source_paths = [SOURCE / name for name in ["comparisons.csv", "marker_summary.csv", "registry.json"]]
    before = {path.as_posix(): digest(path) for path in source_paths}
    rows = view_data(group, mode, persona, scenario, scope)
    context = pd.concat([view_data(group, "profile", scenario=setting, scope=scope)
                         for setting in ["E", "I1", "I2", "I3"]])
    context = context.drop_duplicates(["scope", "model_id", "persona", "marker_id"]).drop(columns="panel_id")
    fig = plot_view(rows, group, mode, persona, scenario)
    stem = "_".join(str(x).lower() if x != group else x for x in [group, mode, persona, scenario, scope] if x is not None)
    try:
        geometry = audit_geometry(fig)
        output.mkdir(parents=True, exist_ok=False)
        _save_figure(fig, output, stem)
        rows.to_csv(output / (stem + ".csv"), index=False)
        context.to_csv(output / "marker_context.csv", index=False)
        caption = view_caption(rows, mode)
        with (output / "README.md").open("x", encoding="utf-8") as handle:
            handle.write(f"# {stem}\n\n{caption}\n\nScope: `{scope}`.\n\n"
                         "Kolom sumber dipertahankan; grafik hanya memakai kolom display. "
                         "Jumlah mentah `raw_count_total` dan denominator seluruh persona tersedia "
                         "di [marker_context.csv](marker_context.csv). Konteks memakai semua penjelasan "
                         "tersedia, sedangkan delta memakai pasangan eligible. Marker tetap candidate. "
                         "CI 95% marginal bukan CI Holm atau rank-biserial; Wilcoxon bukan uji CI rerata. "
                         "Sumber dan kode dicatat pada [provenance.json](provenance.json).\n")
        after = {path.as_posix(): digest(path) for path in source_paths}
        if before != after:
            raise RuntimeError("Export source changed")
        _json(output / "provenance.json", dict(group=group, mode=mode, persona=persona, scenario=scenario,
              scope=scope, source_hashes=before, source_unchanged=True, code_hashes=_code_hashes(),
              geometry=geometry, rows=len(rows), context_rows=len(context)))
    finally:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--install-existing", action="store_true")
    parser.add_argument("--group", choices=list(MARKER_GROUPS))
    parser.add_argument("--mode", choices=["direct", "profile", "neutral", "implicit"])
    parser.add_argument("--persona", choices=["Man", "Woman"])
    parser.add_argument("--scenario", choices=["E", "I1", "I2", "I3"])
    parser.add_argument("--scope", choices=sorted(SCOPES))
    args = parser.parse_args()
    if args.install and args.install_existing:
        parser.error("Choose --install or --install-existing, not both")
    if args.group:
        if args.install or args.install_existing:
            parser.error("Single-view export cannot be combined with installation")
        if args.mode is None:
            parser.error("--group requires --mode")
        export_view(args.output, args.group, args.mode, args.persona, args.scenario, args.scope or "overall")
        return
    if any(value is not None for value in [args.mode, args.persona, args.scenario, args.scope]):
        parser.error("View filters require --group; the full bundle has scope overall")
    if args.install_existing:
        print(json.dumps(install_additive(args.output), indent=2))
    else:
        build_bundle(args.output)
        if args.install:
            print(json.dumps(install_additive(args.output), indent=2))


if __name__ == "__main__":
    main()
