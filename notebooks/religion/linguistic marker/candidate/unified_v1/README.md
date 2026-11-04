# Religion — template visual bersama

[Buka galeri](index.html). [Notebook GSM8K](GSM8K_unified.ipynb) · [Notebook ETHICS](ETHICS_unified.ipynb).

**Peringatan: U04/U05 GSM8K ditahan karena masalah pairing soal.** Lihat [audit sumber](PAIRING_AUDIT.md). Kandidat visual ini bukan validasi ulang statistik.

## SPEC dan DESIGN

Bentuk bersama tidak menyamakan kamus, populasi, atau estimand antardataset. Hasil tetap candidate, bukan instrumen yang sudah tervalidasi. Seluruh teks tersedia pada sumber dipertahankan termasuk abstain, tanpa filter jawaban benar. Missing bukan nol. CI marginal bukan keputusan Holm; IQR bukan CI. Contoh kalimat bersifat purposif.

Perubahan hanya penyajian statistik yang sudah tersedia. Tidak ada API, sampling statistik, ekstraksi ulang, perubahan kamus, maupun pengujian inferensial baru. U01–U07 memakai satu fungsi plot per jenis analisis pada kedua benchmark. Semua skenario tersedia; I1–I3 dan katalog lengkap diletakkan di detail agar cerita utama tetap terbaca.

Profil dan katalog mencakup seluruh marker sumber. Tabel kontras hanya memuat estimand yang diadaptasi ke template; statistik lain, termasuk kontras binary GSM8K, tetap tersedia pada paket sumber yang ditautkan dari galeri.

## IMPLEMENTATION

PNG 240 dpi, SVG, dan CSV tersedia. Lebar 16 cm; DejaVu Sans minimal 9 pt. Warna serta posisi model tetap. E/I1/I2/I3 bukan urutan waktu atau dosis. Kontras utama adalah persona − N, bukan perbandingan langsung dua identitas. U06 menghitung 100 × hit / total kata; bukan rerata density respons. Kutipan U07 berupa substring nyata; pemilihan contoh tidak mewakili prevalensi.

### GSM8K

Denominator adalah seluruh slot sumber per model-persona; available hanya teks visible yang dapat diukur. Missing/technical/source gap tidak dijadikan nol marker.

Titik adalah median jumlah kata dan rentang adalah IQR Q1–Q3, bukan confidence interval. Populasi mengikuti surface beku GSM8K; status candidate v1.3; audit Codex bukan validasi manusia independen.

Estimand deskriptif adalah persentase respons teks tersedia dengan ≥1 hit. Raw occurrence, jumlah teks, dan total kata dipertahankan; semua marker catalog disertakan.

Kontras GSM8K ditahan: ditemukan ID sama dengan isi soal berbeda. Template ditampilkan tanpa nilai. Kolom source_* dan statistik lama dipertahankan sebagai arsip, bukan hasil yang layak ditafsirkan. Audit dan perhitungan ulang belum dilakukan.

Kontras GSM8K ditahan: ditemukan ID sama dengan isi soal berbeda. Template ditampilkan tanpa nilai. Kolom source_* dan statistik lama dipertahankan sebagai arsip, bukan hasil yang layak ditafsirkan. Audit dan perhitungan ulang belum dilakukan.

Enam ungkapan religion-specific per identitas dipilih deterministik dari pooled raw hit pada E; nilai per model = 100×raw_count/total_words. Pemilihan deskriptif, bukan keluarga inferensi baru.

Triplet purposif N/E/I1 memakai soal dan model yang sama, tanpa GT. Excerpt adalah substring eksak ≤230 karakter; bukan contoh representatif. GSM8K memilih triplet dengan jumlah hit tujuh core marker tertinggi; full_text berasal dari visible_text.

### ETHICS

Denominator adalah seluruh slot sumber per model-persona; available hanya teks visible yang dapat diukur. Missing/technical/source gap tidak dijadikan nol marker.

Titik adalah median jumlah kata dan rentang adalah IQR Q1–Q3, bukan confidence interval. Populasi mengikuti surface beku ETHICS; status candidate v3; dua review manusia independen belum selesai.

Estimand deskriptif adalah persentase respons teks tersedia dengan ≥1 hit. Raw occurrence, jumlah teks, dan total kata dipertahankan; semua marker catalog disertakan.

ETHICS: selisih prevalensi berpasangan persona−neutral dalam percentage points pada populasi text_all; CI hanya jika ci_status=available. Status candidate v3; dua review manusia independen belum selesai.

ETHICS: paired mean density difference persona−neutral (occurrence/100 kata), populasi text_all; interval marginal bootstrap hanya ditampilkan saat ci_status=available. Status candidate v3; dua review manusia independen belum selesai; p Holm tetap terpisah dari CI.

Enam ungkapan religion-specific per identitas dipilih deterministik dari pooled raw hit pada E; nilai per model = 100×raw_count/total_words. Pemilihan deskriptif, bukan keluarga inferensi baru.

Triplet purposif N/E/I1 memakai soal dan model yang sama, tanpa GT. Excerpt adalah substring eksak ≤230 karakter; bukan contoh representatif. ETHICS memakai card_04 beku: raw final_output sebagai konteks dapat memuat baris Answer. Marker sumber diukur pada linguistic_text surface lexical hasil normalisasi, bukan pada raw final_output atau cuplikan ini.

## CHECKLIST dan TESTING

- Seluruh marker sumber ditelusuri melalui marker_coverage.csv dan katalog.
- Geometri pasangan diperiksa di layout_checks.json.
- Hasil eksekusi notebook, jumlah gambar dan hash tersedia di verification.json.
- Seluruh artefak sebelum tugas ini tetap utuh; sumber dicek sebelum/sesudah.

## INTEGRATION

Ini kandidat terpisah, belum mengganti notebook utama. Reproduksi dari D:\TugasAkhir ke **folder baru**:

```powershell
python -X utf8 -B -m src.MAIN_PARSING.ETHICS.build_identity_lm_unified --dimension religion --output "notebooks/religion/linguistic marker/candidate/unified_rerun"
```
