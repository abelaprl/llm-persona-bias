"""Audit new figure artifacts and prepare executed overall-only notebooks."""
from __future__ import annotations

import argparse
import base64
import html
import json
import os
from pathlib import Path
import sys
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.benchmarks.ethics import complete_lm_notebook_presentation as prior
from src.benchmarks.ethics.present_ethics_lm_overall import sha256, write_json, GENDER_MARKERS
import nbformat
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

KEYS = ["population", "model_id", "comparison", "left_persona", "right_persona", "outcome_id", "scale"]
VALUES = ["estimate", "ci_low", "ci_high", "signed_effect", "effect_size", "n_pairs", "p_raw", "p_holm"]


def audit_group(bundle, group):
    output = bundle / group
    destination = output / "audit"
    destination.mkdir()
    index = pd.read_csv(output / "figure_index.csv").fillna("")
    comparisons = pd.read_csv(output / "all_comparisons.csv")
    manifest = json.loads((output / "source_manifest.json").read_text(encoding="utf-8"))
    assert set(comparisons.subset) == {"ethics_all"}
    assert not comparisons.duplicated(KEYS).any()
    for path, expected in manifest["source_hashes"].items():
        assert sha256(Path(path)) == expected, path
    assert sha256(output / "notebook_before.ipynb") == manifest["notebook_before_sha256"]
    records = []
    for row in index.to_dict("records"):
        data = pd.read_csv(output / row["table"])
        assert len(data) == row["rows"]
        assert row["clipped_text"] == 0
        if row["section"] in ("core", "specific_direct"):
            checked = data.merge(comparisons, on=KEYS, how="left", suffixes=("_figure", "_source"), validate="one_to_one", indicator=True)
            assert checked._merge.eq("both").all()
            for col in VALUES:
                np.testing.assert_allclose(checked[col + "_figure"], checked[col + "_source"], rtol=0, atol=0, equal_nan=True)
        if row["section"] == "sensitivity":
            source = pd.read_csv(output / "sensitivity.csv")
            expected = source.loc[source.tier_all.eq("core_confirmatory")].reset_index(drop=True)
            pd.testing.assert_frame_equal(data, expected)
        if row["section"] == "coverage":
            assert data.total.sum() == manifest["source_responses"]
            assert (data.committed + data.abstain + data.missing).eq(data.total).all()
            np.testing.assert_allclose(data.commit_rate, data.committed / data.total)
        if row["section"] == "specific":
            expected = pd.read_csv(output / "specific_descriptives.csv")
            expected = expected.loc[expected.model_id.eq(data.model_id.iloc[0])].reset_index(drop=True)
            pd.testing.assert_frame_equal(data, expected)
        if row["section"] == "opening":
            np.testing.assert_allclose(data.groupby(["model_id", "persona"]).proportion.sum(), 1)
        with Image.open(output / row["figure"]) as picture:
            picture.verify()
        with Image.open(output / row["figure"]) as picture:
            width, height = picture.size
        assert width >= 1500 and height >= 600
        assert "<svg" in (output / row["svg"]).read_text(encoding="utf-8")
        records.append({**row, "png_sha256": sha256(output / row["figure"]), "svg_sha256": sha256(output / row["svg"]),
                        "table_sha256": sha256(output / row["table"]), "width": width, "height": height,
                        "numeric_audit": "pass", "visual_audit": "pending"})
    pd.DataFrame(records).to_csv(destination / "numeric_audit.csv", index=False)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 20)
    for start in range(0, len(index), 6):
        sheet = Image.new("RGB", (2000, 2280), "#e5e5e5")
        draw = ImageDraw.Draw(sheet)
        for j, row in enumerate(index.iloc[start:start+6].itertuples()):
            x, y = (j % 2) * 1000, (j // 2) * 760
            with Image.open(output / row.figure) as figure:
                figure = figure.convert("RGB")
                figure.thumbnail((990, 715))
                sheet.paste(figure, (x + (1000-figure.width)//2, y + 35))
            draw.text((x+10, y+8), f"{start+j+1:02d} {Path(row.figure).stem}", font=font, fill="black")
        sheet.save(destination / f"contact_{start//6+1:02d}.png")
    write_json(destination / "numeric_validation.json", {"valid": True, "figures": len(index), "source_hashes_verified": len(manifest["source_hashes"]),
               "statistical_figure_values_match": True, "source_unchanged": True, "backup_unchanged": True})
    print(f"[{group}] Numeric audit passed; {len(index)} figures ready for contact-sheet review", flush=True)


def make_notebook(bundle, group, review_note):
    output = bundle / group
    for name in ["audit/visual_audit.csv", "paired_examples.json", "notebook_after.ipynb", "notebook_validation.json"]:
        assert not (output / name).exists(), f"Use a new output bundle; existing artifact: {name}"
    manifest = json.loads((output / "source_manifest.json").read_text(encoding="utf-8"))
    audit = pd.read_csv(output / "audit/numeric_audit.csv").fillna("")
    assert audit.numeric_audit.eq("pass").all()
    for row in audit.itertuples():
        assert sha256(output / row.figure) == row.png_sha256
        assert sha256(output / row.svg) == row.svg_sha256
        assert sha256(output / row.table) == row.table_sha256
    audit["visual_audit"] = "pass_contact_sheet_and_selected_full_size"
    audit["reviewer"] = "Codex: presentation audit only"
    audit["review_note"] = review_note
    audit.to_csv(output / "audit/visual_audit.csv", index=False)
    title, target_path, _ = prior.CONFIG[group]
    target = ROOT / target_path
    rel = lambda p: quote(os.path.relpath(p, target.parent).replace("\\", "/"), safe="/:_")
    dictionary = GENDER_MARKERS.parent / "marker_registry.csv" if group == "gender" else Path(manifest["source_bundle"]) / "outcome_dictionary.csv"
    d = pd.read_csv(output / "all_comparisons.csv")
    coverage = pd.read_csv(output / "figure_tables/coverage.csv")
    verbosity = pd.read_csv(output / "figure_tables/verbosity.csv")
    short = verbosity.loc[verbosity.model_id.eq("openai_gpt-oss-20b")]
    one_word = int(short['median'].eq(1).sum())
    example_source = ROOT / "result/ETHICS/reports/presentation_completion_20260913_r5" / group / "paired_examples.json"
    assert not (output / "paired_examples.json").exists()
    (output / "paired_examples.json").write_bytes(example_source.read_bytes())
    total, available = int(coverage.total.sum()), int(coverage.available.sum())
    markdown = nbformat.v4.new_markdown_cell
    code = nbformat.v4.new_code_cell
    cells = [markdown(f"""# {title} — keseluruhan ETHICS

**Candidate · penyajian gabungan, 13 September 2026.** Semua hasil utama menggunakan keseluruhan ETHICS. Model dan skenario persona tetap terpisah.

Sumber mencakup **{total:,} respons**, dengan **{available:,} teks tersedia** ({available/total:.2%}). Satu respons berbobot sama; hasil mengikuti komposisi korpus yang tersedia. Pasangan memakai model + subset asal + ID soal. P-value dan effect size dihitung pada pasangan respons gabungan, bukan dirata-ratakan dari tabel hasil subset.

Analisis utama memakai seluruh teks penjelasan tersedia, termasuk *abstain*, tanpa hidden reasoning. *Missing* dan ukuran yang tidak tersedia tetap NA. Perbandingan *both committed* hanya sensitivitas. Density dihitung per 100 kata per respons; count mentah disertakan pada tabel.

**Catatan panjang teks:** GPT-OSS mempunyai median **1 kata pada {one_word} dari {len(short)} persona** di domain ini ([tabel panjang respons]({rel(output/'figure_tables/verbosity.csv')})). Flag teks tersedia mengikuti sumber beku dan tidak menjamin adanya penjelasan substantif. Marker rendah pada respons pendek tidak dapat langsung diartikan sebagai gaya bahasa yang sama atau tidak adanya bias.

**Cara membaca:** E = explicit; I1/I2/I3 = tiga skenario implicit; N = neutral. Biru berarti sisi pertama lebih tinggi, merah sisi pembanding lebih tinggi, putih efek nol, abu-abu data tidak tersedia. Bintang berarti p Holm < 0,05 dalam keluarga uji sumber. Tanpa bintang bukan bukti kedua kondisi setara.

CI numerik adalah CI median selisih berpasangan, **bukan CI rank-biserial**. CI presence adalah CI selisih prevalensi. Bootstrap sumber: {manifest['bootstrap_resamples']:,} replikasi. Keluarga Holm dipertahankan dari engine domain; jumlah hasil signifikan antardomain tidak dijadikan peringkat bias.

[Penjelajah semua marker]({rel(output/'explorer.html')}) · [Tabel lengkap dan sensitivitas]({rel(output/'all_comparisons.csv')}) · [Kamus, definisi dan sumber marker]({rel(dictionary)}) · [Notebook sebelumnya sebagai pendukung]({rel(output/'notebook_before.ipynb')})

Kamus bersumber dari registry beku; adaptasi/aturan operasional buatan sendiri tidak dianggap instrumen psikometrik tervalidasi. Penggabungan ini untuk linguistic marker; tidak ada akurasi jawaban gabungan.
"""), code(f'''from pathlib import Path
import hashlib, html, json, os
import pandas as pd
from IPython.display import display, Image, Markdown, HTML
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src/benchmarks/ethics").is_dir())
BUNDLE = ROOT / {str(output.relative_to(ROOT)).replace(chr(92), '/')!r}
figures = pd.read_csv(BUNDLE / "figure_index.csv").fillna("")
audit = pd.read_csv(BUNDLE / "audit/visual_audit.csv").fillna("")
assert figures.aggregation.eq("ethics_all").all()
for row in audit.itertuples():
    for artifact, expected in [(row.figure, row.png_sha256), (row.svg, row.svg_sha256), (row.table, row.table_sha256)]:
        assert hashlib.sha256((BUNDLE / artifact).read_bytes()).hexdigest() == expected
comparisons = pd.read_csv(BUNDLE / "all_comparisons.csv")
assert set(comparisons.subset) == {{"ethics_all"}}

def show_figures(section, comparison=None):
    selected = figures.loc[figures.section.eq(section)]
    if comparison is not None:
        selected = selected.loc[selected.comparison.eq(comparison)]
    for row in selected.itertuples():
        display(Image(filename=str(BUNDLE / row.figure), width=1200))
        display(Markdown(row.caption))
        links = []
        for filename, label in [(row.svg, "SVG"), (row.figure, "PNG resolusi penuh"), (row.table, "Tabel angka")]:
            relative = os.path.relpath(BUNDLE / filename).replace("\\\\", "/")
            links.append('<a href="' + html.escape(relative, quote=True) + '">' + label + '</a>')
        display(HTML(" · ".join(links)))

def show_table(filename, caption):
    table = pd.read_csv(BUNDLE / filename)
    display(HTML("<details><summary>" + html.escape(caption) + "</summary>" + table.to_html(index=False, float_format=lambda v: f"{{v:.5g}}") + "</details>"))
display(Markdown(f"**{{len(figures)}} gambar**; seluruh artefak gambar/tabel cocok dengan hash audit."))
''')]
    sections = [
        ("1. Cakupan teks dan panjang respons", "Coverage teks adalah dukungan linguistic marker, bukan commit rate jawaban. Tabel melaporkan keduanya bersama total, abstain, dan missing.", "show_figures('coverage')\nshow_table('figure_tables/coverage.csv', 'Total, coverage teks, committed, abstain dan missing per model/persona')\nshow_figures('verbosity')"),
        ("2. Perbandingan langsung antaridentitas", "Bandingkan identitas pada skenario yang sama. Semua marker inti dan enam model ditampilkan, termasuk hasil tidak signifikan.", "show_figures('core', 'direct')"),
        ("3. Setiap identitas terhadap neutral", "Setiap panel menunjukkan persona pada skenario terkait dikurangi baseline N yang sama. Baseline tidak diperlakukan sebagai observasi baru ketika muncul pada beberapa kontras.", "show_figures('core', 'vs_neutral')"),
        ("4. Implicit terhadap explicit dalam identitas yang sama", "I1−E, I2−E, dan I3−E menunjukkan perubahan bahasa antarskenario dalam identitas yang sama. Hasil menggambarkan respons atas prompt yang dikirim; desain ini bukan identifikasi kausal sifat demografis.", "show_figures('core', 'vs_explicit')"),
        ("5. Marker khusus identitas", "Uji langsung ditampilkan lebih dahulu. Panel density memakai rank-biserial; panel presence memakai selisih prevalensi (poin persentase), dengan colorbar terpisah. Grafik prevalensi berikutnya bersifat deskriptif. Marker eksploratori lainnya tetap tersedia di penjelajah.", "show_figures('specific_direct')\nshow_figures('specific')\nshow_table('specific_descriptives.csv', 'Dukungan respons, count mentah, density, dan prevalensi seluruh marker khusus')"),
        ("6. Pembuka respons", "Komposisi pembuka dibaca sebagai outcome multinomial. Uji Stuart–Maxwell dan p Holm dilaporkan pada tabel terpisah, bukan sebagai sejumlah uji biner independen.", "show_figures('opening')\nshow_table('opening.csv', 'Uji pembuka respons berpasangan')"),
        ("7. Sensitivitas terhadap committed", "Analisis utama tetap seluruh teks. Pembatasan pada kedua jawaban committed dapat menyeleksi respons. Hanya kontras yang tersedia pada kedua populasi dibandingkan; nilai yang tidak tersedia tidak diisi.", "show_figures('sensitivity')"),
    ]
    for heading, text, source in sections:
        cells.extend([markdown(f"## {heading}\n\n{text}"), code(source)])
    cells.extend([markdown("""## 8. Contoh respons untuk membantu membaca marker

Satu pasangan berikut diambil dari contoh sumber terdahulu: pasangan pertama menurut urutan model/soal/persona pada pemilihan deterministik, tanpa memilih berdasarkan p-value atau GT. Teks ditampilkan utuh. Ini ilustrasi, bukan sampel audit representatif dan bukan penilaian benar–salah.
"""), code('''examples = json.loads((BUNDLE / 'paired_examples.json').read_text(encoding='utf-8'))
pair = examples[:2]
assert len(pair) == 2
assert pair[0]['model'] == pair[1]['model'] and pair[0]['question_id'] == pair[1]['question_id'] and pair[0]['subset'] == pair[1]['subset']
for item in pair:
    label = item['model'] + ' · ' + item['persona'] + ' · ' + item['question_id']
    display(HTML('<details><summary>' + html.escape(label) + '</summary><p style="white-space:pre-wrap">' + html.escape(item['lexical_text']) + '</p><p>Sumber: ' + html.escape(item['source_file']) + '</p></details>'))
'''), markdown(f"""## 9. Tabel lengkap, provenance, dan batas klaim

Penjelajah memuat **{int(d.population.eq('all_visible').sum()):,} baris uji utama**, dengan filter model, skenario, identitas/perbandingan, marker dan skala. Semua effect size, CI, dukungan pasangan, p mentah, p Holm, hasil tidak signifikan serta tidak dapat diuji tetap tersedia. Kolom `source_file` dan `source_row` menunjuk baris sumber nol-based.

[Buka penjelajah offline]({rel(output/'explorer.html')}) · [CSV semua uji]({rel(output/'all_comparisons.csv')}) · [CSV sensitivitas dan retensi pasangan]({rel(output/'sensitivity.csv')}) · [Audit gambar per file]({rel(output/'audit/visual_audit.csv')}) · [Manifest sumber]({rel(output/'source_manifest.json')})

Tabel [komposisi korpus]({rel(output/'subset_weights.csv')}) dan [asal pasangan per outcome]({rel(output/'pair_composition.csv')}) tersedia sebagai provenance bobot gabungan. Keduanya bukan penyajian hasil terpisah per subset.

**Batas interpretasi:** efek gabungan bergantung komposisi korpus dan dapat menutupi arah yang berbeda pada bagian korpus. Konstruk prompt deontology/virtue tetap membatasi klaim tentang moralitas; marker mengukur bahasa penjelasan yang dihasilkan. Banyak outcome diuji dengan keluarga Holm sumber, bukan satu koreksi global atas semua outcome. Sign test pada kasus yang dipilih engine masih perlu diselaraskan dengan Bab IV.

**Audit penyajian oleh Codex:** seluruh gambar ditinjau melalui contact sheet, dengan pembesaran gambar terpilih, pemeriksaan clipping otomatis, kecocokan angka tabel, dan pemeriksaan integritas sumber. Audit ini tidak mengisi label audit semantik manusia. Status ilmiah tetap **candidate** sampai validasi marker manusia dan gerbang metode selesai.
"""), code("show_table('audit/visual_audit.csv', 'Daftar audit seluruh gambar')\nshow_table('summary.csv', 'Jumlah uji signifikan, tidak signifikan, dan tidak dapat diuji — bukan peringkat bias')")])
    nb = nbformat.v4.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}, "ethics_aggregation": "overall", "result_status": "candidate"})
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output
    from traitlets.config import Config
    config = Config(); config.HistoryManager.hist_file = ":memory:"
    shell = InteractiveShell.instance(config=config)
    previous_cwd = Path.cwd()
    executed = 0
    try:
        os.chdir(target.parent)
        shell.reset(new_session=False)
        for cell in nb.cells:
            if cell.cell_type != "code":
                continue
            executed += 1
            with capture_output() as captured:
                result = shell.run_cell(cell.source, store_history=False)
            assert result.success, captured.stdout + captured.stderr
            cell.execution_count = executed
            cell.outputs = []
            if captured.stdout:
                cell.outputs.append(nbformat.v4.new_output("stream", name="stdout", text=captured.stdout))
            if captured.stderr:
                cell.outputs.append(nbformat.v4.new_output("stream", name="stderr", text=captured.stderr))
            for rich in captured.outputs:
                cell.outputs.append(nbformat.v4.new_output("display_data", data=rich.data, metadata=rich.metadata))
    finally:
        os.chdir(previous_cwd)
        InteractiveShell.clear_instance()
    pngs = [base64.b64decode(o.data['image/png']) for c in nb.cells for o in c.get('outputs', []) if 'image/png' in o.get('data', {})]
    assert len(pngs) == len(audit)
    assert {__import__('hashlib').sha256(p).hexdigest() for p in pngs} == set(audit.png_sha256)
    assert not any(o.output_type == "error" for c in nb.cells for o in c.get("outputs", []))
    nbformat.validate(nb)
    assert not (output / "notebook_after.ipynb").exists()
    nbformat.write(nb, output / "notebook_after.ipynb")
    assert sha256(target) == manifest["notebook_before_sha256"]
    result = {"notebook": str(target_path), "notebook_sha256": sha256(output / "notebook_after.ipynb"), "code_cells_executed": executed,
              "errors": 0, "images": len(pngs), "main_aggregation": "ethics_all", "old_notebook_preserved": True,
              "source_responses": total, "available_text": available, "visual_review_note": review_note,
              "example_source": str(example_source), "example_sha256": sha256(example_source)}
    write_json(output / "notebook_validation.json", result)
    print(f"[{group}] Executed {executed} code cells, {len(pngs)} verified images, zero errors", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--action", required=True, choices=["audit", "finalize", "install"])
    parser.add_argument("--review-note")
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    assert bundle.is_relative_to(ROOT / "result/ETHICS/reports")
    if args.action == "install":
        prior.install_prepared(bundle)
    elif args.action == "audit":
        for group in prior.CONFIG:
            audit_group(bundle, group)
    else:
        assert args.review_note, "Provide the actual completed visual review scope"
        results = {group: make_notebook(bundle, group, args.review_note) for group in prior.CONFIG}
        write_json(bundle / "validation.json", {"valid": True, **results})


if __name__ == "__main__":
    main()
