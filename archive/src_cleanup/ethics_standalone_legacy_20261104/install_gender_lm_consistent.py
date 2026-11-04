"""Stage and install presentation-only thesis/notebook changes with backups."""
from __future__ import annotations
import argparse
from datetime import datetime
import difflib
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.benchmarks.ethics.gender_lm_terminology import display_text

THESIS = Path("D:/ProposalTA-18222008")
ETHICS = ROOT / "result/benchmarks/ethics/reports/gender_lm_print_20261002_r8"
GSM = ROOT / "result/benchmarks/ethics/reports/gender_gsm8k_lm_consistent_20261002_r4"
STAGE = ROOT / "result/benchmarks/ethics/reports/gender_lm_integration_20261002_r3"
NOTEBOOK = ROOT / "notebooks/gender/linguistic marker/linguistic_marker_gender_gsm8k_direct_gender_v3.ipynb"
GSM_PREFIX = "image/BabVI/gsm8k/gender/consistent_20261002"
ETHICS_PREFIX = "image/BabVI/ethics/gender/linguistic_marker_20261002"


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def figure(prefix, name, caption, label):
    return ("\\begin{figure}[H]\n    \\centering\n"
            f"    \\includegraphics[width=\\textwidth]{{{prefix}/{name}.png}}\n"
            f"    \\caption{{{caption}}}\n    \\label{{{label}}}\n\\end{{figure}}\n")


def ethics_main():
    first = figure(ETHICS_PREFIX, "N02_bahasa_etika_vs_netral",
                   r"Ukuran efek penanda linguistik etika terhadap skenario netral (\textit{candidate})",
                   "fig:ethics-gender-lm-netral")
    second = figure(ETHICS_PREFIX, "02_bahasa_etika_man_minus_woman",
                    r"Perbedaan penanda linguistik etika antara persona laki-laki dan perempuan (\textit{candidate})",
                    "fig:ethics-gender-lm-langsung")
    return r"""% Presentation-only integration; frozen statistics, no audit promotion.
Penanda linguistik dihitung pada teks penjelasan yang tersedia, termasuk
penjelasan pada respons \textit{abstain}; teks yang tidak tersedia tidak
dianggap memiliki frekuensi penanda nol. Ketersediaan penjelasan dan pasangan
respons disajikan pada Lampiran~\ref{fig:ethics-gender-lm-cakupan}.
Gambar~\ref{fig:ethics-gender-lm-netral} membandingkan densitas penanda linguistik
etika pada E, I1, I2, dan I3 terhadap N. Satu titik mewakili satu penanda pada
satu model; warna membedakan kelompok penanda dan ukuran titik mengikuti
jumlah pasangan dengan selisih bukan nol.

""" + first + r"""
Pada keluaran \textit{candidate}, ukuran efek tersedia untuk 544 dari 576
kombinasi penanda, model, persona, dan skenario; 133 perbandingan memiliki
$p_{\mathrm{Holm}}<0{,}05$. Sumbu horizontal menunjukkan \textit{rank-biserial}
dengan arah persona dikurangi netral, sedangkan sumbu vertikal menunjukkan
$-\log_{10}(p_{\mathrm{Holm}})$ dan dibatasi pada 50. Garis putus-putus menandai
$p_{\mathrm{Holm}}=0{,}05$. Signifikansi dan besarnya efek tetap dibaca secara
terpisah; hasil ini belum menjadi kesimpulan akhir sebelum audit manusia selesai.

Gambar~\ref{fig:ethics-gender-lm-langsung} membandingkan persona laki-laki dan
perempuan secara langsung pada soal yang sama. Nilai positif berarti densitas
penanda lebih tinggi pada persona laki-laki, sedangkan nilai negatif berarti
lebih tinggi pada persona perempuan. Warna menunjukkan model, segitiga menandai
$p_{\mathrm{Holm}}<0{,}05$, dan lingkaran menunjukkan hasil lainnya.

""" + second + r"""
Dari 288 kombinasi penanda, model, dan skenario, 263 memiliki ukuran efek yang
tersedia dan 15 perbandingan memiliki $p_{\mathrm{Holm}}<0{,}05$. Rinciannya
adalah dua pada E, tujuh pada I1, dua pada I2, dan empat pada I3. Berlian
menunjukkan median ketika keenam estimasi model tersedia; tanda silang pada
kolom NA menunjukkan estimasi yang tidak tersedia. Grafik penanda linguistik
umum, wacana gender, dan penanda biner disajikan pada
Lampiran~\ref{sec:lampiran-ethics-gender-lm}. Penanda biner memakai selisih
prevalensi dalam poin persentase, bukan \textit{rank-biserial}.
"""


def ethics_appendix():
    entries = [
        ("00_cakupan", "Ketersediaan penjelasan dan pasangan analisis; L: laki-laki, P: perempuan", "cakupan"),
        ("02_panjang_penjelasan", "Panjang penjelasan menurut skenario; titik: median, garis: rentang antarkuartil", "panjang"),
        ("N01_bahasa_umum_vs_netral", "Ukuran efek penanda linguistik umum terhadap skenario netral", "umum-netral"),
        ("N03_wacana_gender_vs_netral", "Ukuran efek penanda linguistik wacana gender terhadap skenario netral", "wacana-netral"),
        ("01_bahasa_umum_man_minus_woman", "Perbedaan penanda linguistik umum antara persona laki-laki dan perempuan", "umum-langsung"),
        ("03_bahasa_gender_man_minus_woman", "Perbedaan penanda linguistik wacana gender antara persona laki-laki dan perempuan", "wacana-langsung"),
        ("04_marker_biner_man_minus_woman", "Selisih prevalensi penanda biner antara persona laki-laki dan perempuan; garis: interval kepercayaan 95\\% marginal", "biner-langsung"),
    ]
    text = r"""\section{Penanda Linguistik ETHICS Domain Gender}
\label{sec:lampiran-ethics-gender-lm}

Gambar berikut melengkapi penanda linguistik ETHICS pada Bab VI. Seluruh
hasil masih berstatus \textit{candidate} dan belum menggantikan audit manusia
independen. Koreksi Holm mengikuti keluarga pengujian pada hasil sumber,
bukan dihitung ulang berdasarkan pembagian gambar. Pada grafik langsung,
segitiga menunjukkan $p_{\mathrm{Holm}}<0{,}05$ dan lingkaran menunjukkan
hasil lainnya; warna membedakan model.

"""
    for name, caption, label in entries:
        text += "\\clearpage\n" + figure(ETHICS_PREFIX, name, caption + r" (\textit{candidate})", "fig:ethics-gender-lm-"+label) + "\n"
    return text


def prepare():
    import nbformat
    import pandas as pd
    from nbclient import NotebookClient
    if STAGE.exists():
        raise FileExistsError(STAGE)
    STAGE.mkdir(parents=True)
    files = []

    def stage_text(target, content):
        staged = STAGE / target.name
        staged.write_text(content, encoding="utf-8")
        files.append(dict(target=str(target), staged=str(staged), before=sha(target) if target.exists() else None,
                          after=sha(staged)))
        if target.exists():
            diff = difflib.unified_diff(target.read_text(encoding="utf-8").splitlines(True), content.splitlines(True),
                                        fromfile=str(target), tofile=str(target))
            (STAGE/(target.name+".diff")).write_text("".join(diff), encoding="utf-8")

    gsm_index = pd.read_csv(GSM/"figure_index.csv")
    main_path = THESIS/"Bab VI - Evaluasi.tex"
    appendix_path = THESIS/"Lampiran-D.tex"
    for target in (main_path, appendix_path):
        text = target.read_text(encoding="utf-8")
        pattern = r"\\includegraphics\s*\[[^\]]*\]\s*\{image/BabVI/gsm8k/gender/([^}]+)\}"
        used = []
        def replace(match):
            name = match.group(1)
            if name not in set(gsm_index.parent_figure):
                return match.group(0)
            used.append(name)
            return rf"\includegraphics[width=\textwidth]{{{GSM_PREFIX}/{name}}}"
        text = re.sub(pattern, replace, text)
        if target == main_path:
            text = text.replace("notebook v1.4.1\n\\textit{candidate}", "notebook v1.5\n\\textit{candidate}")
            anchor = "independen selesai.\n\n\\subsubsection{Ringkasan Domain Gender}"
            assert text.count(anchor) == 1
            text = text.replace(anchor, "independen selesai.\n\n\\input{Bab VI - ETHICS Gender Penanda Linguistik.tex}\n\n\\subsubsection{Ringkasan Domain Gender}")
        else:
            text = text.replace("\\caption{Jumlah respons yang dapat dibandingkan antara persona laki-laki dan perempuan}",
                                "\\caption{Selisih ketersediaan teks respons antara persona laki-laki dan perempuan}")
            text = text.replace("\\caption{Selisih kategori pembuka respons antara persona laki-laki dan perempuan}",
                                "\\caption{Perbedaan komposisi pembuka respons antara persona laki-laki dan perempuan}")
            for original, scopes, base_label in [
                ("05_proxy_presence_forest_structure_logic.png", ["Penghubung sebab-akibat", r"LaTeX \textit{display}"], "fig:lampiran-marker-struktur-gender"),
                ("16_direct_gender_availability_forest.png", ["Hanya konten teknis"], "fig:lampiran-jumlah-respons-langsung-gender"),
            ]:
                pattern_fig = r"(\\begin\{figure\}\[H\](?:(?!\\end\{figure\}).)*?" + re.escape(original) + r"(?:(?!\\end\{figure\}).)*?\\end\{figure\})"
                match = re.search(pattern_fig, text, re.S)
                assert match
                extra = ""
                for page, scope in enumerate(scopes, 2):
                    new_name = original.removesuffix(".png")+f"_p{page:02d}"
                    extra += "\n\\clearpage\n" + figure(GSM_PREFIX, new_name, f"{scope}: lanjutan perbandingan penanda linguistik gender", base_label+f"-p{page}")
                text = text[:match.end()] + extra + text[match.end():]
            anchor = r"\section{Hasil Tambahan ETHICS Domain Gender}"
            assert text.count(anchor) == 1
            text = text.replace(anchor, "\\input{Lampiran-D-ETHICS-Penanda-Linguistik.tex}\n\n"+anchor)
            direct_end = r"\label{fig:lampiran-marker-gender-laki-perempuan}"+"\n\\end{figure}"
            text = text.replace(direct_end, direct_end+r"""

Pada grafik langsung, warna membedakan model; segitiga menunjukkan
$p_{\mathrm{Holm}}<0{,}05$, sedangkan lingkaran menunjukkan hasil lainnya.
Ukuran titik mengikuti jumlah pasangan dengan selisih bukan nol dan berlian
menunjukkan median enam model. Arah ukuran efek tetap laki-laki dikurangi perempuan.
""")
        assert len(used) == (2 if target == main_path else 5), used
        stage_text(target, text)
    stage_text(THESIS/"Bab VI - ETHICS Gender Penanda Linguistik.tex", ethics_main())
    stage_text(THESIS/"Lampiran-D-ETHICS-Penanda-Linguistik.tex", ethics_appendix())
    for bundle, subdir, prefix, index in ((GSM,"images",GSM_PREFIX,gsm_index),
                                        (ETHICS,"figures",ETHICS_PREFIX,pd.read_csv(ETHICS/"figure_index.csv"))):
        for row in index.itertuples():
            name = row.figure if subdir == "images" else row.figure_id+".png"
            source = bundle/subdir/name
            target = THESIS/prefix/name
            assert not target.exists(), target
            files.append(dict(target=str(target), staged=str(source), before=None, after=sha(source)))
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    source = notebook.cells[1].source
    start, end = source.index("candidates = []"), source.index("FIG_DIR =")
    source = source[:start] + f'BUNDLE_DIR = REPO_ROOT / {GSM.relative_to(ROOT).as_posix()!r}\nmanifest = json.loads((BUNDLE_DIR / "manifest.json").read_text(encoding="utf-8"))\n' + source[end:]
    start, end = source.index("def show_figure("), source.index("display(Markdown(")
    source = source[:start] + '''def show_figure(name: str, width: int = 1000):
    index = pd.read_csv(BUNDLE_DIR / "figure_index.csv")
    pages = index.loc[index.parent_figure.eq(name)].sort_values("page")
    if pages.empty:
        raise KeyError(name)
    for row in pages.itertuples():
        display(Image(filename=str(FIG_DIR / row.figure), width=width))

''' + source[end:]
    notebook.cells[1].source = source
    for cell in notebook.cells:
        if cell.cell_type == "markdown":
            cell.source = display_text(cell.source)
        if cell.cell_type == "code":
            cell.outputs, cell.execution_count = [], None
    notebook.cells[0].source += "\n\n**Pembaruan tampilan 2 Oktober 2026:** istilah diseragamkan dengan ETHICS; warna titik penuh, segitiga pada perbandingan numerik langsung menandai p Holm < 0,05. Statistik dan status audit tidak diubah. Gambar pendukung padat ditampilkan dalam beberapa halaman."
    print("Executing updated GSM8K notebook...", flush=True)
    NotebookClient(notebook, timeout=240, kernel_name="python3", resources={"metadata":{"path":str(ROOT)}}).execute()
    nbformat.validate(notebook)
    count = sum("image/png" in o.get("data",{}) for c in notebook.cells if c.cell_type=="code" for o in c.outputs)
    assert count == len(gsm_index)-1, (count, len(gsm_index))  # Historical word cloud is not displayed.
    stage_text(NOTEBOOK, nbformat.writes(notebook))
    (STAGE/"install_plan.json").write_text(json.dumps(files, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Prepared {len(files)} exact targets; {count} notebook PNG outputs; review .diff files in {STAGE}", flush=True)


def install():
    files = json.loads((STAGE/"install_plan.json").read_text(encoding="utf-8"))
    for record in files:
        target, staged = Path(record["target"]).resolve(), Path(record["staged"]).resolve()
        assert target.is_relative_to(THESIS.resolve()) or target == NOTEBOOK.resolve()
        assert staged.is_relative_to(ROOT/"result/benchmarks/ethics/reports")
        assert sha(staged) == record["after"]
        assert (sha(target) if target.exists() else None) == record["before"], f"Changed since staging: {target}"
    backup = THESIS/"_arsip"/("gender_lm_consistent_"+datetime.now().strftime("%Y%m%d_%H%M%S"))
    backup.mkdir(parents=True)
    for record in files:
        if record["before"] is not None:
            target = Path(record["target"])
            destination = backup/target.name
            shutil.copy2(target, destination)
            assert sha(destination) == record["before"]
    for record in files:
        target = Path(record["target"])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(record["staged"], target)
        assert sha(target) == record["after"]
    (backup/"install_plan.json").write_text(json.dumps(files, indent=2), encoding="utf-8")
    print(f"Installed {len(files)} checked targets. Backup: {backup}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    install() if args.install else prepare()
