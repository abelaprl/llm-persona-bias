# Pemeriksaan kandidat visual bersama

Tanggal: 21 September 2026. Cakupan: penyajian Gender GSM8K dan ETHICS; bukan validasi ulang kamus atau inferensi sumber.

- [verification.json](verification.json): 7 pasangan template cocok; 26 PNG, 26 SVG, dan 26 CSV gambar. Notebook GSM8K memuat 9 PNG dan ETHICS 17 PNG, tanpa keluaran error.
- [marker_coverage.csv](marker_coverage.csv): seluruh 16 indikator numerik GSM8K dan 41 marker ETHICS tercakup. Ketujuh marker pada ringkasan utama tidak menggantikan detail lengkap.
- [gsm8k_all_effects.csv](gsm8k_all_effects.csv) mempertahankan 384 baris dan seluruh kolom sumber; [ethics_all_effects.csv](ethics_all_effects.csv) mempertahankan 984 baris. Pemeriksaan independen memakai toleransi numerik 1e-12. Pembalikan tanda/batas CI hanya terdapat pada kolom tampilan ETHICS.
- Masing-masing tabel U06 berisi 72 baris; seluruh density cocok dengan `100 * raw_count / total_words`. Masing-masing U07 berisi 3 cuplikan yang cocok persis dengan substring teks sumber dan offset tersimpan.
- Seluruh 168 referensi lokal pada galeri, notebook, dan README ditemukan. Sebanyak 116 hash sumber/artefak sebelumnya tetap cocok; 11 hash kode juga diperiksa ulang sesudah revisi.
- Review independen menemukan satu overlap judul–subtitle U01. Posisi subtitle diperbaiki identik pada kedua benchmark; review ulang mengukur jarak 9,32 pt dan tidak menemukan blocker tersisa. Kedua PNG/SVG dan gambar notebook telah diperbarui; byte PNG pertama pada setiap notebook cocok dengan U01 terbaru.
- Sebanyak 14 uji khusus kandidat lulus sesudah perbaikan (20,877 detik), termasuk uji jarak judul minimal 4 pt, pemetaan statistik, bentuk bersama, dan provenance contoh.
- Suite regresi menyeluruh lulus: 304 uji dalam 471,552 detik. Proses ini dimulai sebelum penambahan uji overlap U01; karena itu, 14 uji khusus di atas dijalankan ulang setelah perbaikan. Warning SciPy untuk fixture konstan dan fallback event-loop Windows tidak menghasilkan kegagalan.

Perintah uji khusus:

```powershell
python -X utf8 -B -m unittest src.MAIN_PARSING.ETHICS.tests.test_gender_lm_unified src.MAIN_PARSING.ETHICS.tests.test_gender_lm_unified_build src.MAIN_PARSING.ETHICS.tests.test_gender_lm_unified_evidence -q
```

Perintah suite regresi:

```powershell
python -X utf8 -B -m unittest discover -s src/MAIN_PARSING/ETHICS/tests -p 'test_*.py' -q
```

Keterbatasan tetap berlaku: bentuk bersama tidak menyamakan estimand, kamus, atau populasi teks antardataset. Contoh kalimat bersifat purposif dan instrumen tetap berstatus candidate. Hasil ini belum mengganti notebook utama.
