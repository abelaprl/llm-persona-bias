# Arsip data ETHICS

Folder ini dibuat pada 2026-11-03 dan diperbarui pada 2026-11-04 untuk
memisahkan dataset kanonis dari raw source, fixture, duplikat, dan dataset
eksperimen lama.

## Kriteria pemindahan

Hanya `data/ethics/ETHICS_test_4subsets_10percent_from200.json` yang digunakan
sebagai dataset kanonis pipeline utama. File tersebut berisi 1.514 soal:
388 commonsense, 359 deontology, 270 justice, dan 497 virtue. Kecocokannya
dengan output V9 telah diverifikasi berdasarkan 1.514 ID unik tanpa ID hilang
atau tambahan.

File lain dipindahkan ke sini jika hanya berfungsi sebagai raw source, fixture,
duplikat, pecahan dataset, atau dataset eksperimen lama. Pemindahan tidak
mengubah isi file dan tidak menghapus data. Untuk memulihkan file, pindahkan
kembali ke `data/ethics/` dengan nama yang sama.

Beberapa runner lama masih mempunyai default yang menunjuk `2each`, `5each`,
`800`, atau `ethics_sample_20`. Runner tersebut bukan pipeline utama 1.514 soal
dan perlu diberi `--samples_file data/ethics/ETHICS_test_4subsets_10percent_from200.json`
jika hendak dijalankan terhadap dataset kanonis.

## Manifest

| File | Baris | Ukuran (byte) | Alasan diarsipkan |
|---|---:|---:|---|
| `ETHICS_4subsets_100q.json` | 400 | 214.176 | Sampel turunan tanpa referensi aktif |
| `ETHICS_4subsets_50q.json` | 200 | 89.899 | Sampel turunan tanpa referensi aktif |
| `ETHICS_commonsense_test.json` | 3.964 | 4.067.281 | Raw test; bukan input langsung pipeline utama |
| `ETHICS_commonsense_test_10percent_from200.json` | 388 | 808.410 | Pecahan pembentuk dataset gabungan; tidak dibaca langsung |
| `ETHICS_commonsense_test_200.json` | 200 | 427.745 | Sampel eksperimen lama tanpa referensi aktif |
| `ETHICS_commonsense_train.json` | 13.910 | 15.162.583 | Split train tidak dipakai evaluasi atau builder aktif |
| `ETHICS_commonsense_validation.json` | 3.885 | 3.338.137 | Raw validation; hanya sumber fixture lama |
| `ETHICS_deontology_test_10percent_from200.json` | 359 | 168.971 | Pecahan pembentuk dataset gabungan; tidak dibaca langsung |
| `ETHICS_deontology_test_200.json` | 200 | 94.073 | Sampel eksperimen lama tanpa referensi aktif |
| `ETHICS_justice_test_10percent_from200.json` | 270 | 128.080 | Pecahan pembentuk dataset gabungan; tidak dibaca langsung |
| `ETHICS_justice_test_200.json` | 200 | 95.266 | Sampel eksperimen lama tanpa referensi aktif |
| `ethics_sample_20.json` | 20 | 19.215 | Fixture runner test lama |
| `ETHICS_test_4subsets_2each.json` | 8 | 6.096 | Fixture prompt-variant lama |
| `ETHICS_test_4subsets_5each.json` | 20 | 23.530 | Fixture eksplorasi/V8 lama |
| `ETHICS_test_4subsets_800.json` | 1.514 | 1.320.999 | Duplikat byte-identik dataset kanonis; nama `800` tidak sesuai isi |
| `ETHICS_test_5subsets_1000.json` | 1.000 | 810.303 | Dataset eksperimen lima subset tanpa referensi aktif |
| `ETHICS_utilitarianism_test_200.json` | 200 | 106.214 | Subset tidak digunakan pipeline ETHICS empat subset aktif |
| `ETHICS_virtue_test_10percent_from200.json` | 497 | 215.547 | Pecahan pembentuk dataset gabungan; tidak dibaca langsung |
| `ETHICS_virtue_test_200.json` | 200 | 86.986 | Sampel eksperimen lama tanpa referensi aktif |

Total: 19 file, 27.183.511 byte.
