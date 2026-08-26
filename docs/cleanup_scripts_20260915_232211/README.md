# Cleanup scripts — 15 September 2026

## SPEC

Cakupan: lima file berubah, dua file untracked, dan 13 penghapusan staged
di `scripts/`. Tujuan: membuang kode yang tidak dipakai pada alur aktif,
mempertahankan reproduksi analisis, dan memperjelas daftar penghapusan lama.
Tidak ada eksperimen atau rerun API; tidak ada pemrosesan ulang dataset mentah.

## DESIGN PLAN

Scan referensi membaca 681 file kode, dokumentasi, dan sumber sel notebook.
Builder v3 masih mengimpor v2 dan analisis leksikon. Builder Religion/Regional
dirujuk dokumen progres; generator volcano dijelaskan dalam catatan visualisasi.
Perender keterbacaan membaca tabel titik yang dihasilkan generator tersebut.
Ketujuh file dipertahankan.

Scan tidak menemukan pemanggil `_render_regional_identity` di luar definisinya.
Fungsi itu juga tidak dipanggil `main()` pada perender keterbacaan saat ini.
`_model_legend` dan lima konstanta hanya mendukung cabang visual lama itu.
Kode tersebut dipilih untuk dihapus; ketiadaan referensi saja tidak dipakai
untuk menghapus file builder atau bukti penelitian.

## IMPLEMENTATION

File yang dirapikan: `scripts/render_gsm8k_marker_readable_v1.py`.

- Hapus `_render_regional_identity` dan `_model_legend`.
- Hapus `MODEL_LABELS`, `MODEL_ORDER`, `MODEL_COLORS`, `GENERAL_OUTCOMES`,
  dan `OUTCOME_LABELS` yang hanya dipakai oleh dua fungsi tersebut.
- Jumlah baris turun dari 536 menjadi 343: berkurang 193 baris.
- Struktur sintaks seluruh bagian yang dipertahankan identik sebelum/sesudah;
  `main()` tetap menghasilkan empat visual volcano dalam format PNG/PDF.
- Tambahkan [panduan scripts](../../scripts/README.md).

Salinan byte asli tersedia di [source_before_cleanup.zip](source_before_cleanup.zip).
SHA256 dan status awal dicatat dalam [audit.json](audit.json). ZIP disertakan
secara khusus dalam commit cleanup sebagai salinan pemulihan source untracked.

## Tiga belas penghapusan yang sudah staged

Seluruh path berikut berada di `scripts/` dan sudah tidak ada sebelum cleanup ini.
Isi diperiksa dari versi `HEAD`; perubahan staging dipertahankan.

| File | Isi/peran versi terakhir |
|---|---|
| `all_mismatches_compact.csv` | Ekspor ringkas kasus mismatch |
| `all_mismatches_full_responses.json` | Ekspor respons lengkap untuk review mismatch |
| `analyze_apply_report.json` | Output analisis apply report |
| `gsm8k_regional_scan.txt` | Ringkasan scan kegagalan regional |
| `mismatch_100_full_responses.json` | Sampel respons mismatch |
| `mismatch_examples_full_responses.json` | Contoh review mismatch |
| `commit_ethics_day2.sh` | Otomasi commit batch data tertentu |
| `commit_ethics_day3.sh` | Otomasi commit dua batch data tertentu |
| `scan_empty_predictions_arcee.py` | Scan prediction kosong Arcee |
| `scan_moonshot_kimi_k2_errors.py` | Scan error Moonshot Kimi |
| `tmp_scan_qwen3_8b_403.py` | Scan sementara error 403 Qwen |
| `tmp_rebuild_results_all_models_csv.py` | Rebuild sementara CSV seluruh model |
| `stratified_sample_squadv2.py` | Utilitas stratified sampling SQuAD v2 |

Empat artefak yang dihapus masih disebut utilitas historis:
`all_mismatches_compact.csv`, `all_mismatches_full_responses.json`,
`analyze_apply_report.json`, dan `mismatch_examples_full_responses.json`.
Pemanggil terdapat di `src/analysis/dump_mismatches_to_csv.py`,
`dump_all_wrong_full_responses.py`, `export_mismatches_review_examples.py`,
dan `analyze_apply_report.py`. Ini dicatat sebagai dependensi historis;
cleanup tidak menyimpulkan seluruh 13 file sudah terbukti tidak berguna.

## CHECKLIST

- [x] Tujuh file analisis/reproduksi dipertahankan.
- [x] Dua fungsi tidak terpakai dan lima konstanta pendukung dihapus.
- [x] Struktur sintaks kode yang dipertahankan identik.
- [x] Daftar 13 penghapusan sebelumnya didokumentasikan.
- [x] Verifikasi render dan integritas akhir lulus; lihat [verification.json](verification.json).

## TESTING

Verifikasi membandingkan kode aktif sebelum/sesudah, menjalankan perender
yang dirapikan dari tabel titik tersimpan ke folder sementara baru, memeriksa
PNG/PDF yang dihasilkan, dan membandingkan hash tabel input. Tidak ada hasil
lama ditimpa. Lima builder lain tidak dijalankan ulang dalam cleanup ini.

Hasil: empat PNG terbaca dan memiliki resolusi 300 DPI, serta empat PDF
memiliki header dan penutup yang valid. Hash kedua tabel input tetap sama.
Sebanyak 69 file lain pada snapshot `scripts/` identik byte demi byte,
staging tidak berubah, dan seluruh 13 tautan dokumentasi tersedia.

## INTEGRATION

Gunakan panduan scripts untuk membedakan builder, kode pendukung, dan
perender. Tidak ada file tambahan dihapus; tidak dilakukan stage, commit,
atau push pada pekerjaan ini. Mekanisme impor dan rumus statistik lima
builder yang sedang berubah tidak dimodifikasi.

## Batas pemeriksaan

Scan bersifat statis; pemanggil di luar repositori tidak dapat dipastikan.
Satu notebook arsip lebih dari 20 MB dilewati:
`notebooks/archive/20260910_superseded_notebooks/religion/linguistic_marker_religion_ethics_bab4_visual_v1_1.ipynb`.
Folder `src/MAIN_PARSING/ETHICS/tests/pooled_test_93c4zanf/` tidak terbaca karena
izin akses. Nama `render_gsm8k_marker_readable_v1.py` dipertahankan untuk
kompatibilitas path meskipun manifest di dalamnya menyebut versi bundle v3.
