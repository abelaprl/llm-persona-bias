"""Add source-bound diagnostic pages to the existing notebook, preserving base figures."""
import hashlib
import json
from pathlib import Path
import shutil

import nbformat
import pandas as pd


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def selected_bundle(root, notebook_dir):
    config = notebook_dir / "diagnostic_source.json"
    if not config.exists():
        return None
    bundle = (root / json.loads(config.read_text(encoding="utf-8"))["diagnostic_bundle"]).resolve()
    if not bundle.is_relative_to((root / "result/ETHICS/reports").resolve()):
        raise ValueError("Diagnostic bundle must remain within ETHICS reports")
    manifest = json.loads((bundle / "diagnostic_manifest.json").read_text(encoding="utf-8"))
    if not manifest["complete_registry"] or not manifest["source_files_unchanged"]:
        raise ValueError("Only complete verified diagnostic bundles can be installed")
    return bundle


def install_assets(bundle, notebook_dir):
    index = pd.read_csv(bundle / "diagnostic_figure_index.csv")
    names = ["diagnostic_figure_index.csv", "diagnostic_manifest.json", "diagnostic_visual_checks.json", "diagnostic_gallery.html"]
    names += [getattr(row, field) for row in index.itertuples() for field in ("png_path", "svg_path", "source_csv")]
    copies = []
    for name in names:
        source, target = (bundle / name).resolve(), (notebook_dir / name).resolve()
        if not source.is_relative_to(bundle.resolve()) or not target.is_relative_to(notebook_dir.resolve()):
            raise ValueError("Diagnostic artifact path escapes bundle or notebook directory")
        expected = digest(source)
        if target.exists() and digest(target) != expected:
            raise FileExistsError(target)
        copies.append((source, target, expected))
    for row in index.itertuples():
        for field, hash_field in [("png_path", "png_sha256"), ("svg_path", "svg_sha256"), ("source_csv", "source_sha256")]:
            assert digest(bundle / getattr(row, field)) == getattr(row, hash_field)
    for source, target, expected in copies:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
        assert digest(target) == expected
    print(f"Verified {len(index)} diagnostic PNG/SVG/CSV sets and gallery", flush=True)


def diagnostic_cells(root, notebook_dir):
    bundle = selected_bundle(root, notebook_dir)
    if bundle is None:
        return [], 0
    index = pd.read_csv(bundle / "diagnostic_figure_index.csv")
    default_ids = index.marker_ids.eq("stance_hedge") & index.figure_kind.isin(["subset_sensitivity", "normalization_sensitivity"])
    count = 2 + int(index.figure_kind.eq("marker_pair_support").sum()) + int(default_ids.sum())
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    cells = [md("""## 9. Konteks rujukan gender — tambahan diagnostik

Halaman tambahan memakai metadata/span dan statistik beku r3. **Tidak ada
aturan marker, p-value, atau CI yang dihitung ulang.** Kolom konteks dihitung
per respons, bukan per span: satu respons dapat memiliki rujukan pengguna
eksplisit sekaligus kutipan atau rujukan unknown. Karena itu persentase antar-
kolom tidak harus berjumlah 100%. Denominator adalah penjelasan tersedia;
respons tanpa penjelasan tetap tercatat pada coverage, bukan nol marker.

`explicit_user` adalah label otomatis kandidat, bukan coreference tervalidasi.
Overlap istilah dengan skenario adalah flag terpisah, bukan bukti bahwa
gender pengguna diadopsi atau bahwa tokoh tertentu dirujuk. Seluruh halaman
tambahan tersedia pada galeri dan indeks; hanya detail marker pilihan yang
ditanam pada notebook untuk menjaga keterbacaan.
"""), code('''DIAGNOSTIC_CONFIG = json.loads((NOTEBOOK_DIR / "diagnostic_source.json").read_text(encoding="utf-8"))
DIAGNOSTIC_BUNDLE = ROOT / DIAGNOSTIC_CONFIG["diagnostic_bundle"]
diagnostics = pd.read_csv(NOTEBOOK_DIR / "diagnostic_figure_index.csv").fillna("")
diagnostic_manifest = json.loads((NOTEBOOK_DIR / "diagnostic_manifest.json").read_text(encoding="utf-8"))
assert diagnostic_manifest["complete_registry"] and not diagnostic_manifest["statistics_recomputed"]
assert diagnostics.figure_id.is_unique and diagnostics.audit_status.eq("candidate").all()
pd.testing.assert_frame_equal(diagnostics, pd.read_csv(DIAGNOSTIC_BUNDLE / "diagnostic_figure_index.csv").fillna(""))
for row in diagnostics.itertuples():
    for name, expected in [(row.png_path, row.png_sha256), (row.svg_path, row.svg_sha256), (row.source_csv, row.source_sha256)]:
        assert hashlib.sha256((NOTEBOOK_DIR / name).read_bytes()).hexdigest() == expected, name
for name, expected in diagnostic_manifest["source_hashes"].items():
    assert hashlib.sha256((BUNDLE / name).read_bytes()).hexdigest() == expected, name
diagnostic_shown_ids = []

def diagnostic_link(name, label):
    assert (NOTEBOOK_DIR / name).is_file(), name
    return '<a href="' + html.escape(name, quote=True) + '">' + html.escape(label) + '</a>'

def show_diagnostic(row):
    display(Markdown("### " + row.title))
    display(Image(filename=str(NOTEBOOK_DIR / row.png_path), width=1800))
    display(HTML(" · ".join([diagnostic_link(row.png_path, "PNG penuh"), diagnostic_link(row.svg_path, "SVG"),
                             diagnostic_link(row.source_csv, "CSV sumber") ])))
    diagnostic_shown_ids.append(row.figure_id)

display(HTML(diagnostic_link("diagnostic_gallery.html", "Galeri semua halaman tambahan") + " · " +
             diagnostic_link("diagnostic_figure_index.csv", "Indeks dan hash")))
for row in diagnostics.loc[diagnostics.figure_kind.eq("gender_reference_context")].itertuples():
    show_diagnostic(row)
'''), md("""## 10. Dukungan pasangan — bedakan N dan pasangan yang berbeda

Ringkasan kiri memakai pasangan penjelasan lengkap / seluruh soal untuk
Woman−Man pada kondisi sepadan. Ringkasan kanan menghitung marker dengan
`n_changed < 10` pada ukuran utama registry: density atau presence.
Hitungan marker tidak menambah jumlah soal atau observasi independen.

Pada halaman detail, isi sel adalah `n_changed / n_pairs`. Bingkai merah
menandai low-support; bukan berarti efek pasti tidak ada. Pada forest
sensitivitas, urutan angka dibalik secara eksplisit sebagai
`n_pairs / n_changed` agar jumlah sampel dibaca terlebih dahulu.
"""), code('''for row in diagnostics.loc[diagnostics.figure_kind.isin(["pair_support_overview", "marker_pair_support"])].itertuples():
    show_diagnostic(row)
'''), md("""## 11. Sensitivitas subset dan normalisasi — pilih marker

Ubah `DIAGNOSTIC_MARKER` lalu jalankan sel di bawah. Seluruh 41 marker
mempunyai halaman overall + empat subset, dengan semua model dan E/I1/I2/I3
tetap terpisah. CI nominal 95% bukan CI simultan; kepekatan titik mengikuti
p Holm tersimpan, dan titik kosong berarti `n_changed < 10`. NA tidak
diganti dengan nol. Overall mencakup subset tersebut, bukan sampel independen.

Untuk 30 marker dengan ukuran utama density, halaman kedua membandingkan
count dan density pada sumbu terpisah. Unit berbeda, sehingga besar angka
antarkolom tidak boleh disamakan. Perubahan setelah normalisasi bukan bukti
kausal bahwa verbosity menjelaskan seluruh perbedaan. Sebelas marker binary
tetap memakai presence dan tidak diberi halaman count–density tambahan.

Hedge dipilih sebagai contoh navigasi, bukan karena hasil signifikansinya.
Tabel tautan di bawah menyediakan semua marker tanpa filter signifikansi.
"""), code('''DIAGNOSTIC_MARKER = "stance_hedge"
assert DIAGNOSTIC_MARKER in set(registry.marker_id), DIAGNOSTIC_MARKER
chosen = diagnostics.loc[diagnostics.marker_ids.eq(DIAGNOSTIC_MARKER) &
                        diagnostics.figure_kind.isin(["subset_sensitivity", "normalization_sensitivity"])]
assert not chosen.empty
for row in chosen.itertuples():
    show_diagnostic(row)
if not chosen.figure_kind.eq("normalization_sensitivity").any():
    display(Markdown("Marker binary: presence adalah ukuran utama; tidak ada halaman count–density tambahan."))
navigation = []
for row in diagnostics.loc[diagnostics.figure_kind.isin(["subset_sensitivity", "normalization_sensitivity"])].itertuples():
    navigation.append("<tr><td>" + html.escape(row.title) + "</td><td>" +
                      " · ".join([diagnostic_link(row.png_path, "PNG"), diagnostic_link(row.svg_path, "SVG"),
                                  diagnostic_link(row.source_csv, "CSV")]) + "</td></tr>")
display(HTML("<table><tr><th>Semua halaman marker</th><th>Berkas</th></tr>" + "".join(navigation) + "</table>"))
assert set(diagnostics.loc[diagnostics.figure_kind.eq("subset_sensitivity"), "marker_ids"]) == set(registry.marker_id)
assert set(diagnostics.loc[diagnostics.figure_kind.eq("normalization_sensitivity"), "marker_ids"]) == set(registry.loc[registry.measurement.eq("density"), "marker_id"])
print("Halaman diagnostik tersedia:", len(diagnostics), "; audit manusia tetap belum selesai.")
''')]
    return cells, count
