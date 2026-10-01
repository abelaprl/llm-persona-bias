"""Build a fresh, reference-led presentation from frozen Gender ETHICS results."""
import argparse
import html
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.benchmarks.ethics import gender_lm_reference_data as d
from src.benchmarks.ethics import gender_lm_reference_plots as p
from src.benchmarks.ethics.gender_lm_diagnostic_data import direct_rows, support_summary

SOURCE_NAMES = ["registry.json", "marker_summary.csv", "comparisons.csv", "response_length_summary.csv",
                "report_status_coverage.csv", "response_markers.csv.gz", "availability_tests.csv"]


def create_notebook(bundle, index):
    nb = nbformat.v4.new_notebook()
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    nb.cells = [md("""# Penanda linguistik Gender — ETHICS

Alur presentasi mengikuti Religion/Regional, dengan bentuk panel dan warna model dari Gender GSM8K.
**Candidate, belum tervalidasi melalui audit manusia independen.** Sumber frozen v2.0.1 r3; 81.756 respons,
enam model, sembilan persona, 41 marker. Empat subset digabung hanya untuk bahasa, bukan akurasi.

Satu respons = satu observasi. Pengukuran memakai penjelasan final, bukan hidden reasoning, dan mencakup
*abstain* berpenjelasan. Missing, teknis, dan verdict-only tidak dianggap penjelasan dengan nol marker.
E = explicit, I1/I2/I3 = tiga skenario implicit, N = skenario netral.

Gambar utama memakai format landscape. Titik warna mengidentifikasi **model**, bukan persona.
Persona dan skenario dibedakan melalui panel. PNG untuk notebook; SVG untuk pembesaran/cetak.
Jangan memperkecil panel lebar ke 16 cm tanpa memeriksa keterbacaan.""")]
    setup = f'''from pathlib import Path
import hashlib, json, os, sys
import pandas as pd
from IPython.display import Image, Markdown, HTML, display

ROOT = next(x for x in [Path.cwd(), *Path.cwd().parents] if (x / 'src/benchmarks/ethics').is_dir())
sys.path.insert(0, str(ROOT))
BUNDLE = ROOT / {bundle.relative_to(d.ROOT).as_posix()!r}
SOURCE = ROOT / {d.SOURCE.relative_to(d.ROOT).as_posix()!r}
ACTIVE = ROOT / {d.ACTIVE.relative_to(d.ROOT).as_posix()!r}
DOCUMENT = ACTIVE if Path.cwd() != BUNDLE else BUNDLE
FIGURES = DOCUMENT / 'figures'
index = pd.read_csv(BUNDLE / 'figure_index.csv')
manifest = json.loads((BUNDLE / 'manifest.json').read_text(encoding='utf-8'))
for relative, expected in manifest['source_hashes'].items():
    with (ROOT / relative).open('rb') as handle:
        actual = hashlib.file_digest(handle, 'sha256').hexdigest()
    assert actual == expected, 'Sumber frozen berubah: ' + relative

def show(figure_id):
    row = index.loc[index.figure_id.eq(figure_id)].iloc[0]
    path = FIGURES / (figure_id + '.png')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row.png_sha256
    display(Image(filename=str(path), width=1700))
    display(Markdown(row.caption))
    links = [f'<a href="figures/{{figure_id}}.{{ext}}">{{ext.upper()}}</a>' for ext in ['png','svg','csv']]
    display(HTML(' · '.join(links)))

def table_link(path, label):
    return '<a href="' + os.path.relpath(path, DOCUMENT).replace(chr(92), '/') + '">' + label + '</a>'

display(HTML('<style>.jp-Notebook {{ max-width: none; }} .output_png img {{ max-width: 100%; }}</style>'))
'''
    nb.cells.append(code(setup))
    sections = [
        ("1. Coverage dan panjang penjelasan", ["01_cakupan", "02_panjang_penjelasan"],
         "Coverage dibaca sebelum membandingkan frekuensi. Median/IQR panjang mengikuti sumber frozen: verdict-only bernilai nol kata, sedangkan denominator marker hanya penjelasan tersedia. Dua denominator ini tidak boleh disamakan."),
        ("2. Profil marker: apa yang muncul?", ["03_profil_umum_explicit", "04_profil_etika_explicit", "05_profil_gender_semua_skenario"],
         "Profil umum dan etika memakai E sebagai pengantar, seperti profil explicit pada Religion. Profil gender menampilkan seluruh N/E/I1/I2/I3. Semua profil memakai prevalence (% respons dengan ≥1 hit); jumlah mentah dan density per 100 kata ikut disimpan di CSV. Berlian hitam hanya median deskriptif enam model, bukan penggabungan observasi atau hasil uji. Kategori tidak saling eksklusif."),
        ("3. Perbandingan langsung Woman − Man", ["06_efek_langsung_umum", "07_efek_langsung_domain", "08_efek_langsung_biner"],
         "Bentuk empat panel mengikuti chart konsistensi Gender GSM8K, tetapi arah tetap **Woman−Man**, sesuai sumber ETHICS (GSM8K memakai arah sebaliknya). Nilai positif berarti lebih tinggi pada Woman. Seluruh 41 marker tercakup: 30 density dan 11 biner. Density memakai matched rank-biserial; CI mean sumber tetap di CSV dan tidak ditempel pada rank. Marker biner memakai delta prevalence pp dengan CI 95% marginal. Holm mengikuti keluarga pengujian sumber; bukan koreksi global untuk semua grafik. Pudar bukan berarti effect nol. Ukuran titik rank menunjukkan pasangan berubah; n_pairs dan n_changed ada di CSV."),
        ("4. Persona dibanding N dan implicit dibanding explicit", ["09_gender_terhadap_netral", "10_gender_implicit_vs_explicit"],
         "Sebelas marker gender ditampilkan dalam unit prevalence yang sama. Perbandingan dilakukan berpasangan pada soal yang sama, per model. I1/I2/I3 tidak dilebur. Efek marker umum/etika dan hasil per subset tetap tersedia pada penjelajah di bawah; tidak dipilih menurut signifikansi."),
        ("5. Variasi kategori dan konteks rujukan", ["11_keragaman_kategori", "12_konteks_rujukan"],
         "ECDF menghitung jumlah kategori gender berbeda per respons (0–11), bukan kekayaan bahasa menyeluruh. Pengulangan hit tidak menambah kategori. Nilai dipengaruhi kamus dan panjang respons; tabel strata panjang disediakan. Referen/konteks adalah flag otomatis kandidat; kategori dapat overlap. Istilah yang juga ada dalam soal bukan bukti coreference atau asal kausal bahasa. Tidak ada uji statistik baru pada dua grafik deskriptif ini."),
    ]
    for title, ids, description in sections:
        nb.cells.append(md("## " + title + "\n\n" + description))
        nb.cells.append(code("\n".join(f"show({fid!r})" for fid in ids)))
    nb.cells += [md("""## 6. Penjelajah lengkap: seluruh 41 marker dan kontras

Ubah `marker_id`, `scope`, dan `measurement` untuk membaca profil/kontras lengkap. Sumber tetap menyediakan
jumlah mentah, density/100 kata, prevalence, effect size, CI, p Holm, dan dukungan pasangan.
Tabel default hanya membatasi tampilan, bukan analisis. Kamus mencantumkan sumber dan adaptasi buatan sendiri.
Untuk gambar tambahan, fungsi `lihat_efek()` merender pilihan di notebook tanpa menambah file di figures."""),
        code("""summary = pd.read_csv(SOURCE / 'marker_summary.csv', float_precision='round_trip')
comparisons = pd.read_csv(SOURCE / 'comparisons.csv', float_precision='round_trip')
registry = pd.DataFrame(json.loads((SOURCE / 'registry.json').read_text(encoding='utf-8')))
display(registry[['marker_id','registry_scope','measurement','theoretical_source','custom_adaptation']])

marker_id = 'stance_hedge'
scope = 'overall'
measurement = 'density'
display(summary.loc[summary.marker_id.eq(marker_id) & summary.scope.eq(scope)])
chosen = comparisons.loc[comparisons.marker_id.eq(marker_id) & comparisons.scope.eq(scope) & comparisons.measurement.eq(measurement)]
display(chosen[['model_id','left','right','n_pairs','n_changed','effect','measurement_unit','ci_low','ci_high','matched_rank_biserial','adjusted_p','family_id']].head(48))
display(Markdown(f'{len(chosen)} baris cocok; tampilan dibatasi 48 baris. Akses variabel `chosen` untuk semuanya.'))

def lihat_efek(marker_ids, comparison='neutral', measurement='density'):
    from src.benchmarks.ethics.gender_lm_reference_data import select_contrasts
    from src.benchmarks.ethics.gender_lm_reference_plots import effects
    import matplotlib.pyplot as plt
    selected = select_contrasts(comparisons, marker_ids, comparison, measurement)
    fig = effects(selected, marker_ids, 'Penjelajah kontras terpilih', comparison, measurement)
    display(fig)
    plt.close(fig)
    return selected
"""), md("## 7. Respons utuh, denominator, dan batas kesimpulan\n\nContoh pada evidence explorer berasal dari paket frozen; bukan sampel representatif atau validasi manusia. Cacat prompt deontology/virtue tidak diperbaiki oleh analisis bahasa. Tidak ada akurasi gabungan ETHICS. Marker kamus tidak mencakup seluruh ekspresi gender; kamus dan referen belum menjadi human gold."),
        code("""links = [table_link(SOURCE / 'evidence_explorer.html', 'Respons utuh dan bukti marker'),
         table_link(SOURCE / 'marker_spans_reviewed.csv.gz', 'Occurrence dengan provenance'),
         table_link(SOURCE / 'availability_tests.csv', 'Uji availability: effect size dan Holm'),
         table_link(BUNDLE / 'tables/diversity_by_length.csv', 'Keragaman menurut rentang panjang'),
         table_link(BUNDLE / 'tables/diversity_per_response.csv.gz', 'Keragaman per respons'),
         table_link(SOURCE / 'report.md', 'Metode, audit, dan keterbatasan sumber')]
display(HTML('<br>'.join(links)))
availability = pd.read_csv(SOURCE / 'availability_tests.csv')
display(availability.loc[availability.scope.eq('overall'), ['model_id','outcome','n_responses','cramers_v','adjusted_p','availability_rate_range_pp']])
display(pd.read_csv(BUNDLE / 'tables/diversity_by_length.csv').head(20))
"""), md("## Reproduksi\n\nJalankan builder dengan nama output baru. Menjalankan notebook hanya memuat hasil, tanpa API dan tanpa scoring ulang.\n\n```powershell\n.\\.venv\\Scripts\\python.exe -X utf8 src/benchmarks/ethics/build_gender_lm_reference_presentation.py --run-id reproduction_01\n```")]
    nb.metadata = {"kernelspec":{"display_name":"Python 3", "language":"python", "name":"python3"}, "language_info":{"name":"python"}}
    nbformat.write(nb, bundle / d.NOTEBOOK)


def build(output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    output.relative_to(d.ROOT)
    output.mkdir(parents=True)
    (output / "figures").mkdir()
    (output / "tables").mkdir()
    inputs = [d.SOURCE / name for name in SOURCE_NAMES]
    reference_path = d.DIAGNOSTICS / "diagnostic_tables/diagnostic_gender_reference_context.csv"
    inputs.append(reference_path)
    before = {x.relative_to(d.ROOT).as_posix():d.digest(x) for x in inputs}
    registry = pd.DataFrame(json.loads((d.SOURCE / "registry.json").read_text(encoding="utf-8")))
    registry.to_csv(output / "tables/marker_catalog.csv", index=False)
    summary = pd.read_csv(d.SOURCE / "marker_summary.csv", float_precision="round_trip")
    comparisons = pd.read_csv(d.SOURCE / "comparisons.csv", float_precision="round_trip")
    profile = d.profiles(summary)
    group_ids = {s: registry.loc[registry.registry_scope.eq(s), "marker_id"].tolist() for s in registry.registry_scope.unique()}
    gender_ids = group_ids["gender_specific"]
    density_ids = registry.loc[registry.measurement.eq("density"), "marker_id"].tolist()
    general_density = [m for m in density_ids if m in group_ids["general_ethics"]]
    domain_density = [m for m in density_ids if m not in general_density]
    binary_ids = registry.loc[registry.measurement.eq("binary"), "marker_id"].tolist()
    assert [len(general_density), len(domain_density), len(binary_ids), len(gender_ids)] == [14,16,11,11]
    assert set(density_ids + binary_ids) == set(registry.marker_id)
    index = []

    def save(fid, fig, table, caption):
        for ext in ["png","svg"]:
            fig.savefig(output / "figures" / (fid + "." + ext), dpi=180)
        table.to_csv(output / "figures" / (fid + ".csv"), index=False)
        index.append(dict(figure_id=fid, caption=caption, rows=len(table), panels=len(fig.axes),
                          width_inches=fig.get_figwidth(), height_inches=fig.get_figheight(),
                          png_sha256=d.digest(output / "figures" / (fid + ".png")),
                          svg_sha256=d.digest(output / "figures" / (fid + ".svg")),
                          csv_sha256=d.digest(output / "figures" / (fid + ".csv"))))
        plt.close(fig)
        print(f"Rendered {len(index)}/12: {fid}", flush=True)

    status = pd.read_csv(d.SOURCE / "report_status_coverage.csv")
    status = status.loc[status.scope.eq("overall")].copy()
    support = support_summary(direct_rows(comparisons, registry))
    support.to_csv(output / "tables/paired_support.csv", index=False)
    save("01_cakupan",p.coverage(status,support),pd.concat([status.assign(record_type="text_status"), support.assign(record_type="paired_support")],ignore_index=True),
         "**Gambar 1.** Ketersediaan penjelasan dan pasangan langsung. Denominator per sel adalah 1.514; status lengkap dan jumlah pasangan tersedia pada CSV. Banyak respons awal tidak menjamin banyak pasangan berpenjelasan.")
    length = pd.read_csv(d.SOURCE / "response_length_summary.csv", float_precision="round_trip")
    length = length.loc[length.scope.eq("overall")].copy()
    save("02_panjang_penjelasan",p.verbosity(length),length,
         "**Gambar 2.** Median kata dan IQR per model pada N/E/I1/I2/I3. Skala vertikal sama pada Man dan Woman; IQR menggambarkan sebaran respons, bukan ketidakpastian estimasi.")
    for fid, scope, title in [("03_profil_umum_explicit","general_ethics","Profil penanda bahasa umum pada explicit"),
                              ("04_profil_etika_explicit","ethics_specific","Profil penanda khusus etika pada explicit"),
                              ("05_profil_gender_semua_skenario","gender_specific","Cara persona gender muncul dalam penjelasan")]:
        ids = group_ids[scope]
        settings = None if scope=="gender_specific" else ["E"]
        table = profile.loc[profile.marker_id.isin(ids)].copy()
        if settings: table = table.loc[table.persona.str.contains("explicit")].copy()
        save(fid,p.profile(table,ids,title,settings),table,
             f"**Gambar {len(index)+1}.** {title}. Titik warna = model; berlian = median enam model. Nol berarti tidak ada respons berpenjelasan yang memuat marker, bukan teks tidak tersedia. Jumlah mentah dan density/100 kata ikut tersedia pada CSV.")
    jobs = [("06_efek_langsung_umum",general_density,"direct","density","Perbedaan langsung bahasa umum: Woman − Man"),
            ("07_efek_langsung_domain",domain_density,"direct","density","Perbedaan langsung bahasa etika dan gender: Woman − Man"),
            ("08_efek_langsung_biner",binary_ids,"direct","presence","Perbedaan langsung marker biner: Woman − Man"),
            ("09_gender_terhadap_netral",gender_ids,"neutral","presence","Perubahan marker gender terhadap skenario netral"),
            ("10_gender_implicit_vs_explicit",gender_ids,"implicit","presence","Perubahan marker gender: implicit dibanding explicit")]
    for fid,ids,comparison,measurement,title in jobs:
        table = d.select_contrasts(comparisons,ids,comparison,measurement)
        save(fid,p.effects(table,ids,title,comparison,measurement),table,
             f"**Gambar {len(index)+1}.** {title}. Perkiraan tetap terpisah per model, berpasangan pada soal yang sama. Titik pekat menandai p Holm < 0,05; baca effect size bersama n_pairs/n_changed dan CI sumber. Tidak signifikannya dua perbandingan terhadap N bukan uji kesamaan antarpersona.")
    columns = ["response_id","model_id","persona","subset","question_id","text_available","text_status","word_count"] + [m+"__count" for m in gender_ids]
    responses = pd.read_csv(d.SOURCE / "response_markers.csv.gz", usecols=columns)
    diversity = d.category_diversity(responses,gender_ids)
    diversity["raw_occurrences"] = diversity[[m+"__count" for m in gender_ids]].sum(axis=1,min_count=1).where(diversity.text_available)
    diversity["density_per_100_words"] = (100*diversity.raw_occurrences/diversity.word_count.replace(0,np.nan)).where(diversity.text_available)
    diversity.to_csv(output / "tables/diversity_per_response.csv.gz",index=False,compression="gzip")
    diversity["length_bin"] = pd.cut(diversity.word_count,[0,49,99,199,399,np.inf],labels=["1–49","50–99","100–199","200–399","400+"],include_lowest=False)
    strata = diversity.loc[diversity.text_available].groupby(["model_id","persona","length_bin"],observed=True).agg(
        n_available=("response_id","size"),mean_categories=("category_count","mean"),raw_occurrences=("raw_occurrences","sum"),
        mean_density_per_100_words=("density_per_100_words","mean"),mean_words=("word_count","mean")).reset_index()
    strata.to_csv(output / "tables/diversity_by_length.csv",index=False)
    ecdf=[]
    for (model,identity),g in diversity.groupby(["model_id","persona"]):
        values=g.category_count.dropna()
        for threshold in range(12):
            n=int(values.le(threshold).sum())
            ecdf.append(dict(model_id=model,persona=identity,threshold=threshold,n_le_threshold=n,n_available=len(values),total=len(g),
                             percent=100*n/len(values) if len(values) else np.nan, raw_occurrences=g.raw_occurrences.sum(min_count=1),
                             mean_density_per_100_words=g.density_per_100_words.mean()))
    save("11_keragaman_kategori",p.diversity(diversity),pd.DataFrame(ecdf),
         "**Gambar 11.** Distribusi kumulatif jumlah kategori gender berbeda. Kurva lebih ke kanan berarti lebih banyak kategori terdeteksi. Ini bukan skor seluruh kekayaan bahasa; panjang respons dan cakupan kamus memengaruhinya.")
    reference = pd.read_csv(reference_path,float_precision="round_trip").rename(columns={"indicator":"marker_id","percent":"value"})
    reference_ids = ["any_gender","explicit_user","quoted_or_hypothetical","unknown","scenario_overlap"]
    save("12_konteks_rujukan",p.profile(reference,reference_ids,"Konteks penyebutan gender: rujukan pengguna atau skenario?"),reference,
         "**Gambar 12.** Lima indikator referen/konteks otomatis. Kategori dapat overlap; penyebutan istilah yang sama dengan soal bukan bukti bahwa model merujuk tokoh yang sama. Semua flag masih kandidat, bukan coreference tervalidasi.")
    after = {x.relative_to(d.ROOT).as_posix():d.digest(x) for x in inputs}
    assert before == after
    index = pd.DataFrame(index)
    index.to_csv(output / "figure_index.csv",index=False)
    manifest = dict(status="candidate_pending_independent_human_audit", source_hashes=before,
                    figure_count=len(index),marker_count=len(registry),response_count=len(responses),source_unchanged=True,
                    direction="Woman minus Man",new_inferential_tests=False,
                    references=["Religion ETHICS v3 r4 print r2", "Regional ETHICS v3 presentation r2", "Gender GSM8K direct v3"])
    (output / "manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    create_notebook(output,index)
    gallery = '<!doctype html><meta charset="utf-8"><title>Gender ETHICS — penanda linguistik</title><style>body{font:16px sans-serif;margin:2rem}img{width:100%}section{margin-bottom:4rem}</style><h1>Penanda linguistik Gender ETHICS</h1><p>Candidate. Seluruh statistik berasal dari sumber frozen. Lihat notebook untuk metode dan tabel.</p>'
    for row in index.itertuples():
        gallery += f'<section><img src="figures/{row.figure_id}.png"><p>{html.escape(row.caption)}</p><a href="figures/{row.figure_id}.csv">CSV</a> · <a href="figures/{row.figure_id}.svg">SVG</a></section>'
    (output / "gallery.html").write_text(gallery,encoding="utf-8")
    print(json.dumps(manifest,indent=2),flush=True)
    return output


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--run-id",required=True)
    args=parser.parse_args()
    if not args.run_id.replace("_","").replace("-","").isalnum():
        raise ValueError("Use a simple run ID")
    build(d.ROOT / "result/ETHICS/reports" / ("gender_lm_reference_presentation_"+args.run_id))


if __name__ == "__main__":
    main()
