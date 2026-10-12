# Hasil revisi gambar ETHICS regional — 9 Oktober 2026

## SPEC

Pengguna menyetujui pembuatan paket PNG/PDF dan proof A4 sebelum integrasi
tesis. Implementasi mengikuti
[rencana](RENCANA_REVISI_GAMBAR_ETHICS_REGIONAL_BAB6_14CM_20261009.md).
Seluruh enam model, 17 skenario, dan empat subset pada tabel beku dipakai;
tidak melakukan sampling, eksperimen/API, perubahan parser, atau uji ulang.
`.tex`, folder gambar aktif, notebook aktif, dan pointer `LATEST_RUN.json`
tidak diubah.

Run akhir:
`D:/TugasAkhir/result/analyses/accuracy/runs/regional_ethics_bab6_14cm_20261009_072105_170416/`.

- [Panduan dan indeks PNG](../result/analyses/accuracy/runs/regional_ethics_bab6_14cm_20261009_072105_170416/MULAI_DI_SINI.md)
- [Proof A4 lengkap](../result/analyses/accuracy/runs/regional_ethics_bab6_14cm_20261009_072105_170416/proof/regional_14cm.pdf)
- [Caption](../result/analyses/accuracy/runs/regional_ethics_bab6_14cm_20261009_072105_170416/CAPTIONS.md)
- [Mapping integrasi](../result/analyses/accuracy/runs/regional_ethics_bab6_14cm_20261009_072105_170416/mapping_integrasi.csv)

## DESIGN PLAN

Amerika Utara–Asia Timur (AU/AT) dan Asia Selatan–Afrika Sub-Sahara (AS/ASS)
dipisah. ID sumber tetap `north_american`, `east_asian`, `south_asian`, dan
`sub_saharan_african`. Deskriptif memakai delapan kolom persona + N;
efek persona–N memakai delapan kolom. N berulang hanya untuk tampilan.

Semua gambar baru tepat 14 cm, PDF vektor, PNG 300 dpi, dan teks minimal
8 pt. Heatmap dua subset ditambah tinggi menjadi 16,764 cm setelah inspeksi
awal agar dua baris angka/bintang lebih lega. Gambar alasan setinggi
19,939 cm; jumlah respons ditulis di bawah label skenario dan dijelaskan
sebagai n. Skala alasan 0–100% sama pada seluruh model.

Penyebut tetap dibedakan: akurasi committed; coverage terhadap total;
keberhasilan efektif terhadap nonmissing; stance Wrong terhadap committed;
alasan terhadap abstain dan terhadap nonmissing. Pooling I menggunakan
hitungan, bukan rerata persentase. Seluruh efek 10–13 memakai risk difference
dalam poin persentase, keluarga Holm tetap 16 per model/subset/outcome,
tanpa menambahkan syarat Q Cochran.

## IMPLEMENTATION

Kode baru, tidak memanggil ekstraktor atau fungsi pengujian statistik:

- `src/benchmarks/ethics/render_regional_ethics_14cm.py`
- `src/benchmarks/ethics/validate_regional_ethics_14cm.py`

Paket memuat 29 gambar: 28 PNG baru beserta PDF vektornya, dan satu PNG token
historis identik. Pembagiannya:

| Isi | Jumlah | Halaman proof |
|---|---:|---|
| Distribusi seluruh respons | 2 | 1–2 |
| Alasan terhadap abstain | 4 | 3–6 |
| Akurasi committed + coverage | 2 | 7–8 |
| Efek akurasi persona–N | 2 | 9–10 |
| Alasan terhadap nonmissing | 4 | 11–14 |
| Keberhasilan efektif + coverage | 2 | 15–16 |
| Stance Deontology/Virtue + coverage | 2 | 17–18 |
| Efek coverage, empat subset | 4 | 19–22 |
| Efek keberhasilan efektif | 2 | 23–24 |
| Efek stance | 2 | 25–26 |
| Token historis, tidak dirender ulang | 1 | 27 |
| Distribusi committed terpisah | 2 | 28–29 |

Sepuluh gambar utama berada di `bab6/`. Sisanya di `pelengkap/` dan
`pelengkap_committed/`. Istilah pelengkap **bukan izin untuk menghapus atau
memindahkan gambar lama dari Bab VI**. Kesepuluh referensi gambar lama
tetap tercakup dalam mapping. Gambar 04/alasan terhadap abstain adalah
tambahan, bukan pengganti diam-diam gambar 05/nonmissing.

`figure_index.csv` mencatat ukuran, font, subset, penyebut, caption, dan
halaman proof. `tables/` memuat ringkasan dengan coverage, tabel pasangan
lengkap, alasan sumber, nilai setiap sel, hitungan distribusi, dan audit
geometri teks. Semua output ditulis pada run baru; iterasi sebelum run
akhir tidak digunakan untuk pemasangan dan tidak dihapus.

## CHECKLIST

- [x] Sumber aktif dan 10 referensi `.tex` diinventarisasi.
- [x] Seluruh gambar prioritas dan pelengkap yang direncanakan tersedia.
- [x] Nilai sel, penyebut, arah, n pasangan, p Holm, dan keluarga dicocokkan.
- [x] Semua sumber yang dilindungi hash tetap identik.
- [x] PDF vektor 14 cm, PNG 300 dpi, dan font minimal 8 pt untuk gambar baru.
- [x] Proof 29 halaman beserta caption lengkap diperiksa secara digital.
- [ ] Cetak fisik: belum dilakukan.
- [ ] Integrasi `.tex`: sengaja belum dilakukan, menunggu review pengguna.

## TESTING

Sumber: tiga tabel `ethics_regional_parity_condition_summary.csv`,
`ethics_regional_parity_paired_effects_vs_neutral.csv`, dan
`ethics_regional_parity_abstain_reasons.csv` di folder notebook aktif,
ditambah tabel kategori untuk verifikasi pooling.

| Pemeriksaan | Hasil |
|---|---|
| Total respons unik | 154.428 |
| Status | 112.035 committed; 40.570 abstain; 1.823 missing |
| Ringkasan per model/skenario/subset | 408 baris |
| Sel akurasi unik | 204; 216 sel tampilan karena 12 N diulang |
| Efek persona–N | 960, seluruhnya dicocokkan satu-ke-satu |
| Signifikan Holm, akurasi / coverage / keberhasilan / stance | 9 / 178 / 68 / 42, tidak berubah |
| Keluarga Holm | 16 untuk setiap model/subset/outcome |
| Sumber terlindungi | 78 berkas, tanpa perubahan hash |
| PDF baru | 28 vektor, lebar MediaBox 14 cm |
| Caption utuh pada halaman yang sama | 29/29 |
| Kesalahan atau overfull hbox/vbox pada proof | 0 |

Validator independen lulus terhadap run akhir. Memeriksa ulang nilai plot
dengan tabel beku, keunikan sel, seluruh hash sumber/output, dimensi PDF,
ketiadaan raster di PDF gambar baru, DPI PNG, n/persentase, dan teks caption
utuh hasil ekstraksi PDF. Uji helper penyebut nol menghasilkan NA, bukan 0%.
Tidak menghitung ulang McNemar atau Holm.

Inspeksi digital mencakup semua 29 halaman. Setelah tinggi heatmap diubah,
16 halaman yang berubah dibuka ulang; 13 halaman lain identik byte-for-byte
dengan PNG proof yang telah diperiksa. Catatan tahap inspeksi ada di
`visual_review.json`. `checks.json` adalah hasil otomatis sebelum inspeksi
dan dibiarkan utuh, bukan ditimpa untuk mengganti statusnya.

Reproduksi dari `D:/TugasAkhir`:

```powershell
.\.venv\Scripts\python.exe src\benchmarks\ethics\render_regional_ethics_14cm.py
.\.venv\Scripts\python.exe src\benchmarks\ethics\validate_regional_ethics_14cm.py result\analyses\accuracy\runs\regional_ethics_bab6_14cm_20261009_072105_170416
```

Perintah pertama selalu membuat run baru; untuk memvalidasi regenerasi,
ganti argumen perintah kedua dengan path run yang dicetak. Membutuhkan
XeLaTeX, pdfinfo, pdftoppm, dan pdftotext yang tersedia pada PATH. Versi
dependensi Python tersimpan pada `manifest.json`.

## INTEGRATION

Paket siap ditinjau pengguna sebelum disalin ke repo TA. `.tex` regional
tetap berhash SHA256
`01003eb359d52488817c3c79697abf964e6080f47e866c7128205529f78039a8`.
Tidak menjalankan kompilasi tesis atau menyunting bab lain pada pekerjaan ini.

Saat pemasangan nanti, gambar baru jangan diperkecil di bawah 14 cm.
Gunakan caption pendek pada proof, dan pindahkan rincian metode yang masih
diperlukan ke prosa. Sesuaikan penjelasan jika memakai distribusi total
sebagai pengganti distribusi committed; keduanya tidak mempunyai penyebut
yang sama.

### Batasan yang tersisa

Grafik token historis pada halaman 27 masih bertulisan kecil pada lebar
14 cm. Aset itu sengaja dipertahankan identik sesuai rencana; jangan
mengklaim ia memenuhi standar font gambar baru. Versi final token yang lebih
terbaca memerlukan pekerjaan tersendiri. Grafik alasan tetap paling padat,
walau memenuhi pemeriksaan geometri dan font 8 pt; cek cetak fisik tetap
diperlukan. Keseragaman visual dan preservasi angka ini bukan validasi ulang
parser, representativitas persona, atau pairing historis.
