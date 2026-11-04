# Regional — template visual bersama

[Buka galeri](index.html). [Notebook GSM8K](GSM8K_unified.ipynb) · [Notebook ETHICS](ETHICS_unified.ipynb).

**Peringatan: U04/U05 GSM8K ditahan karena masalah pairing soal.** Lihat [audit sumber](PAIRING_AUDIT.md). Kandidat visual ini bukan validasi ulang statistik.

## SPEC dan DESIGN

Bentuk bersama tidak menyamakan kamus, populasi, atau estimand antardataset. Hasil tetap candidate, bukan instrumen yang sudah tervalidasi. Seluruh teks tersedia pada sumber dipertahankan termasuk abstain, tanpa filter jawaban benar. Missing bukan nol. CI marginal bukan keputusan Holm; IQR bukan CI. Contoh kalimat bersifat purposif.

Perubahan hanya penyajian statistik yang sudah tersedia. Tidak ada API, sampling statistik, ekstraksi ulang, perubahan kamus, maupun pengujian inferensial baru. U01–U07 memakai satu fungsi plot per jenis analisis pada kedua benchmark. Semua skenario tersedia; I1–I3 dan katalog lengkap diletakkan di detail agar cerita utama tetap terbaca.

Profil dan katalog mencakup seluruh marker sumber. Tabel kontras hanya memuat estimand yang diadaptasi ke template; statistik lain, termasuk kontras binary GSM8K, tetap tersedia pada paket sumber yang ditautkan dari galeri.

## IMPLEMENTATION

PNG 240 dpi, SVG, dan CSV tersedia. Lebar 16 cm; DejaVu Sans minimal 9 pt. Warna serta posisi model tetap. E/I1/I2/I3 bukan urutan waktu atau dosis. Kontras utama adalah persona − N, bukan perbandingan langsung dua identitas. U06 menghitung 100 × hit / total kata; bukan rerata density respons. Kutipan U07 berupa substring nyata; pemilihan contoh tidak mewakili prevalensi.

### GSM8K

Seluruh respons sumber; available berarti visible/explanation text tersedia.

Median kata dan IQR; missing tidak diubah menjadi nol kata.

Prevalensi respons dengan marker (%); seluruh catalog candidate dipertahankan.

Kontras GSM8K ditahan: ditemukan ID sama dengan isi soal berbeda. Template ditampilkan tanpa nilai. Kolom source_* dan statistik lama dipertahankan sebagai arsip, bukan hasil yang layak ditafsirkan. Audit dan perhitungan ulang belum dilakukan.

Kontras GSM8K ditahan: ditemukan ID sama dengan isi soal berbeda. Template ditampilkan tanpa nilai. Kolom source_* dan statistik lama dipertahankan sebagai arsip, bukan hasil yang layak ditafsirkan. Audit dan perhitungan ulang belum dilakukan.

GSM8K memakai label term/group beku (bukan selalu frasa literal tunggal); ETHICS memakai accepted matched_text literal case-sensitive. Enam teratas per identitas, tie alfabetis; placeholder NA berarti tidak ada term teramati tambahan, bukan respons missing. Inventory kandidat kedua benchmark tidak identik.

Triplet purposif N/E/I1: model dan soal sama; teks literal, tanpa ground truth.

### ETHICS

Seluruh respons sumber; available berarti visible/explanation text tersedia.

Median kata dan IQR; missing tidak diubah menjadi nol kata.

Prevalensi respons dengan marker (%); seluruh catalog candidate dipertahankan.

Paired rank-biserial persona minus Neutral; tanpa CI karena CI sumber milik median delta.

Median paired difference per 100 kata dengan CI bootstrap 95% marginal dan Holm sumber.

GSM8K memakai label term/group beku (bukan selalu frasa literal tunggal); ETHICS memakai accepted matched_text literal case-sensitive. Enam teratas per identitas, tie alfabetis; placeholder NA berarti tidak ada term teramati tambahan, bukan respons missing. Inventory kandidat kedua benchmark tidak identik.

Triplet purposif N/E/I1: model dan soal sama; teks literal, tanpa ground truth.

## CHECKLIST dan TESTING

- Seluruh marker sumber ditelusuri melalui marker_coverage.csv dan katalog.
- Geometri pasangan diperiksa di layout_checks.json.
- Hasil eksekusi notebook, jumlah gambar dan hash tersedia di verification.json.
- Seluruh artefak sebelum tugas ini tetap utuh; sumber dicek sebelum/sesudah.

## INTEGRATION

Ini kandidat terpisah, belum mengganti notebook utama. Reproduksi dari D:\TugasAkhir ke **folder baru**:

```powershell
python -X utf8 -B -m src.MAIN_PARSING.ETHICS.build_identity_lm_unified --dimension regional --output "notebooks/regional/linguistic marker/candidate/unified_rerun"
```
