# Linguistic marker ETHICS Gender v2

Pipeline **candidate** untuk seluruh respons sumber, dengan 41 marker dan perbandingan neutral, woman, man pada explicit, implicit 1, 2, dan 3. Angka deteksi belum menjadi bukti precision/recall 99%. Tidak ada penilaian benar/salah atau pembacaan ground truth dalam modul ini.

## Menjalankan dari akar repositori

Pakai nama output baru yang belum ada:

```powershell
.venv/Scripts/python.exe -m src.benchmarks.ethics.gender_lm_v2.run --output-dir result/benchmarks/ethics/reports/gender_lm_v2_candidate_run_baru --mode candidate --workers 4 --bootstrap-resamples 5000 --permutations 10000
```

Sumber default dikunci dalam `source_config.json`: 81.756 respons, enam model, sembilan persona, empat subset. Satu perintah menjalankan ekstraksi, paket audit, statistik, dan laporan. Untuk eksekusi bertahap, gunakan `--stage extract`, lalu `--stage analyze`, lalu `--stage report` pada direktori yang sama. Tahap yang sudah menghasilkan artefak tidak boleh diulang dengan menimpa file. Percobaan yang gagal dipertahankan; gunakan direktori baru untuk eksekusi ulang lengkap.

```powershell
.venv/Scripts/python.exe -m unittest discover -s src/benchmarks/ethics/gender_lm_v2/tests -v
```

Dependensi memakai lingkungan repositori: pandas, numpy, scipy, matplotlib. Pengujian menggunakan unittest, tanpa instalasi pytest. Multiprocessing dan direktori temporer Windows dapat memerlukan izin sandbox tambahan; proses ini seluruhnya lokal, tanpa API berbayar.

## Artefak utama

| Artefak | Isi |
|---|---|
| `report.html` | Atlas keseluruhan dataset, per model, seluruh persona; forest dan perbandingan implisit |
| `evidence_explorer.html` | Respons penuh dengan filter dan highlight span; semua varian soal dapat dibandingkan |
| `response_markers.csv.gz` | Satu respons per observasi, count, presence, density per 100 kata |
| `response_surfaces.csv.gz` | Raw, visible, explanation, structure; hidden reasoning dimask pada surface analisis |
| `marker_spans_reviewed.csv.gz` | Span, rule ID, offset, status candidate, dan flag rujukan yang dipakai analisis |
| `comparisons.csv` | 12 contrast utama dan 12 sekunder, tiga measurement, overall dan tiap subset |
| `neutral_triplet_comparisons.csv` | Woman-neutral, man-neutral, dan woman-man pada soal dengan triplet lengkap yang sama |
| `interaction_comparisons.csv` | Tiga difference-of-differences Woman-Man antarjenis implisit |
| `availability_tests.csv` | Uji permutasi dalam blok soal, effect size, koreksi Holm |
| `baseline_delta.csv` | Perubahan marker dan denominator terhadap baseline, per model/persona/subset |
| `audit/` | Worksheet dua penilai, teks penuh, hit per aturan, split soal, evaluator dan status audit |
| `figure_index.csv` | PNG, SVG, CSV sumber, satuan, denominator, dan hash gambar |
| `*_manifest.json` | Parameter, hitungan, hash sumber/artefak, status dan keterbatasan |

Visual default mencakup semua subset dengan model tetap terpisah. Tabel tetap menyediakan tiap subset dan ketiga measurement. API `build_report(..., render_subset_figures=True, full_detail=True)` dapat merender tambahan subset dan ketiga measurement untuk semua forest pada direktori laporan baru.

## Kontrak interpretasi

- Detektor memakai teks saja; metadata model/persona tidak memilih aturan. Scenario hanya untuk flag overlap, bukan pemberian label jawaban atau klaim otomatis tentang rujukan gender.
- Missing, verdict-only, technical-only, unknown tetap dihitung; marker mereka NA. Semua penjelasan tersedia dianalisis, termasuk refusal, tanpa filter committed atau akurasi.
- Overall berarti komposisi soal yang tersedia di corpus, bukan rerata empat subset berbobot sama. Tidak ada angka akurasi ETHICS agregat.
- Direct contrast selalu Woman-Man. Positif berarti marker lebih tinggi pada woman. Narasi persona implisit juga berbeda; hasil bukan identifikasi sifat intrinsik atau efek gender murni.
- CI 95% adalah interval nominal selisih rerata berpasangan; matched rank-biserial dicatat terpisah. Count, density, dan presence ikut satu family Holm secara konservatif. P-value tidak menguji kualitas detektor.
- `baseline_delta.csv` memuat jumlah (`old_sum`, `new_sum`) per measurement; jumlah density bukan rerata density. Perubahan surface dan aturan masih tercampur: tabel ini bukan estimasi error semantik atau ablation kausal.

## Audit dan target 99%

Worksheet yang belum diisi tidak dihitung sebagai anotasi manusia. Audit discovery agen dan fixture regresi hanya bukti pengembangan. Penilai independen menandai semua span reference pada surface sesuai registry; `raw_text` hanya untuk memeriksa transformasi, bukan memasukkan hidden reasoning ke analisis marker. Evaluasi recall tidak memakai rasio respons yang memiliki hit.

Split menggunakan cluster soal sehingga model dan persona dari soal yang sama tidak tersebar antar-split. Soal yang sudah dibaca sebelumnya dikecualikan dari held-out. Precision mengambil hit development yang tidak dipakai discovery; recall mengambil respons non-held-out secara independen dari hit. Held-out terkunci pada pemilihan worksheet pengembangan, tetapi penjelajah seluruh corpus tetap menampilkan teksnya; ini bukan penyembunyian data secara fisik. Penilai held-out perlu prosedur akses terpisah. Aturan semantic memerlukan minimum 100 hit audit, explicit 50, atau semua hit jika lebih sedikit. Underfilled quota tetap tercatat, bukan dianggap lulus.

Promosi operasional per aturan membutuhkan review manusia ganda lengkap, adjudikasi dan nol false positive. Itu belum membuktikan 99%: evaluator memerlukan reference independen dan batas bawah CI per marker memenuhi target. Marker langka bisa tetap inconclusive. Klaim simultan seluruh marker memerlukan penanganan tambahan; CI individual bukan sertifikat simultan.

`--mode validated` memeriksa gate audit dan menolak bukti yang belum lengkap. Rilis saat ini tetap menghasilkan artefak candidate secara konservatif; publikasi hasil accepted memerlukan rilis baru yang mengikat reference audit, registry, kode instrumen, dan validasi parser. Tidak ada promosi otomatis dalam eksekusi ini.

Tujuh kategori tambahan yang diusulkan dalam SPEC belum dipromosikan menjadi marker baru. Prioritas rilis ini adalah memperbaiki gap nyata dan menyediakan evaluasi seluruh 41 marker lama. Penambahan berikutnya memerlukan scan, pembacaan respons penuh, pengecekan duplikasi konstruk, dan audit per aturan; menambah nama kategori saja tidak membuktikan cakupan lebih tinggi.

## Koreksi wrapper murni v2.0.1

Pembacaan ulang menemukan 12 respons berupa `The answer is 0.` atau wrapper bold sejenis yang keliru masuk penjelasan pada r2. Fullmatch wrapper diperbaiki, dengan fixture respons nyata. R3 menghitung ulang semua surface dari teks asli, memeriksa hash sumber dan instrumen, lalu mengekstrak ulang respons yang berubah. Evidence respons yang identik dipertahankan dari artefak reviewed r2; manifest mencatat keterbatasan snapshot instrumen historis r2. Statistik, paket audit, dan visual dihitung ulang. Runner utama di atas tetap dapat menjalankan semuanya dari sumber mentah tanpa reuse.

Modul `revision --parent-output <r2> --output-dir <direktori-baru>` tersedia khusus koreksi substantive menjadi verdict-only; perubahan detector/registry atau transisi status lain ditolak. Daftar `changed_responses.csv` membuat setiap koreksi dapat diperiksa. Prefix inline pada respons yang masih berisi teks alami tidak diperluas pada koreksi ini; pengaruhnya terhadap denominator masih memerlukan audit.
