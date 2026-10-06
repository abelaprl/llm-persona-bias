# FAILED RUN — jangan gunakan sebagai hasil

Run ini berhenti pada fase 9/10 (`Automated validation`) sebelum notebook dan
manifest final dibuat. Tidak ada file lama yang dihapus atau ditimpa.

Kontrak row count lulus:

- `scoring_summary`: 408;
- `descriptives`: 22.848;
- `within_pairs`: 56.784;
- `direct_pairs`: 50.400;
- `omnibus`: 5.376;
- `opening_pairs`: 1.536.

Penyebab gagal adalah asumsi validator yang terlalu sempit: validator awal
menganggap semua baris `n_pairs=0` hanya dapat berasal dari GPT-OSS `ttr_300`.
Audit tabel aktual menemukan 212 zero-pair rows: 27 pada populasi `all_visible`
karena surface `ttr_300` GPT-OSS tidak tersedia, sedangkan sisanya berada pada
populasi sensitivitas `both_committed`/`both_correct` ketika tidak ada pasangan
respons yang lolos filter untuk keseluruhan kontras. Pemeriksaan raw-count awal
juga keliru karena menuntut raw count pada baris tanpa pasangan.

Run pengganti harus menggunakan validator yang:

1. mengklasifikasikan setiap zero-pair row berdasarkan alasan availability;
2. menolak zero parsial yang tidak dikenal;
3. menuntut raw count hanya ketika `n_pairs > 0`;
4. tetap menyimpan baris kosong dengan statistik inferensial `NA` tanpa imputasi.

Status: `failed_validation_not_for_analysis`.
