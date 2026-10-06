# Gender ETHICS — paket seluruh 41 marker

## SPEC / DESIGN

10 gambar statis: G00 cakupan dan B01–B09 direct Man−Woman. Semua 41 marker memiliki empat mode dalam [81 grafik offline](explorer.html). Warna selalu model; kelompok hanya navigasi, bukan skor gabungan. Seluruh default memakai scope overall, bukan klaim semua subset sudah digrafikkan.

## IMPLEMENTATION / CHECKLIST

- Semua 81 SVG benar-benar dirender. Matriks 164 sel marker×mode: [visual_coverage.csv](visual_coverage.csv).
- 984 baris direct, 1.968 persona−N, 1.476 I−E; 2.214 profil unik. N yang berulang bukan observasi tambahan.
- [Konteks marker](marker_context.csv) menyertakan `raw_count_total` dari baris count sumber, n_available, total, prevalence %, dan density/100 kata. `raw_occurrences` asli pada baris density/presence tetap NA; jangan menggunakan kolom kosong itu sebagai jumlah mentah.
- Grafik density memakai selisih rerata; prevalence memakai poin persentase. CI 95% marginal bukan CI rank-biserial atau CI terkoreksi Holm. Wilcoxon bukan uji estimand rerata yang sama dengan CI.
- Semua kolom sumber utuh. Man−Woman hanya pada `display_*`; kolom sumber direct tetap Woman−Man. `n_pairs`, `low_support`, Holm dan komposisi subset tetap tersedia dalam tabel HTML/CSV.
- [Dua triplet contoh](examples.html), [bentuk ungkapan seluruh model](expression_forms.csv), [penjelajah bukti sumber](../gender_lm_v2_candidate_20260916_r3/evidence_explorer.html). Contoh purposif, bukan sampel representatif. Referen pengguna tidak diasumsikan dari kata gender saja.
- [Katalog dan sumber teori](marker_catalog.csv); kamus custom/adaptasi dan candidate tidak dipromosikan menjadi tervalidasi.

## TESTING

`python -m unittest src.MAIN_PARSING.ETHICS.tests.test_gender_lm_presentation_v2_bundle -v`

Reproduksi pada path baru: `python -m src.MAIN_PARSING.ETHICS.build_gender_lm_presentation_v2_bundle --output result/ETHICS/reports/<nama-baru>`.

Ekspor tambahan: `python -m src.MAIN_PARSING.ETHICS.build_gender_lm_presentation_v2_bundle --output result/ETHICS/reports/<nama-baru> --group B08 --mode neutral --persona Woman --scope overall`.
Untuk profile gunakan `--mode profile --scenario E`; scope dapat overall/commonsense/deontology/justice/virtue. Direktori existing ditolak.

## INTEGRATION

Notebook staging [gender_lm_presentation_v2.ipynb](gender_lm_presentation_v2.ipynb) sudah dieksekusi dan memuat sepuluh gambar. Notebook dan gambar lama tidak diganti. Installer `--install` hanya menambah salinan; nama bentrok berbeda isi diberi suffix. Gambar final tetap langsung di figures utama, tanpa subfolder versi.

## Keterbatasan

Candidate tetap candidate. Cakupan 65.678 penjelasan dari 81.756 slot tidak setara lintas model. GPT-OSS hanya 11–47 pasangan direct; CI dan dukungan harus dibaca bersama. Profil menggunakan semua penjelasan tersedia; delta menggunakan pairwise-complete. Tabel panjang sumber mencakup verdict-only, berbeda dari denominator marker. Ringkasan overall mengikuti komposisi eligible sumber, bukan rata-rata subset berbobot sama. Ini bukan akurasi ETHICS gabungan. Batasan tugas deontology/virtue tetap berlaku. I1–I3 bukan replikasi acak populasi gender. Tidak ada validasi kamus, parser, atau pemanggilan API baru.
