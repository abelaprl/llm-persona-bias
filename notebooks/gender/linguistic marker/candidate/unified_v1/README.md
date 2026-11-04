# Gender — satu aturan visual untuk dua benchmark

[Buka galeri berpasangan](index.html). [GSM8K](GSM8K_unified.ipynb) · [ETHICS](ETHICS_unified.ipynb).

## SPEC dan DESIGN

Perubahan hanya penyajian. Seluruh respons yang tersedia pada sumber tetap menjadi populasi; tidak ada sampling baru untuk statistik, API call, ekstraksi, kamus, atau pengujian inferensial baru. Tujuh jenis gambar utama memakai fungsi yang sama, bukan implementasi berbeda per benchmark. Galeri memasangkan hasilnya agar kesamaan bentuk dapat diperiksa langsung.

[Kontrak pemilihan chart](visual_contract.csv): coverage mengambil heatmap GSM8K; panjang respons mengambil satu panel ETHICS; ringkasan menggabungkan cakupan lintas marker GSM8K dan pemisahan titik model serta panel 2×2 ETHICS; detail memakai CI yang sesuai; kata/ungkapan disertai konteks.

## IMPLEMENTATION dan batas statistik

- Font DejaVu Sans minimal 9 pt; lebar gambar 16 cm. PNG 240 dpi dan SVG tersedia.
- U03/U04/U05 memakai tujuh marker bersama yang ditetapkan sebelum pemilihan berdasarkan data. Semua 16 indikator numerik GSM8K dan 41 marker ETHICS tetap ada pada detail dan marker_coverage.csv.
- U04: titik tanpa CI untuk keduanya; GSM8K rank-biserial, ETHICS delta rerata density. Susunan sama tidak berarti angkanya sebanding. CI ditampilkan pada U05.
- U05: GSM8K median selisih density, ETHICS rerata selisih density. CI milik estimand masing-masing; tidak memakai CI median untuk rank, tidak mengubah median menjadi rerata.
- CI marginal tidak sama dengan keputusan Holm. Titik kosong menunjukkan <20 pasangan berubah, bukan signifikansi; semua pasangan, p-value dan effect size tetap ada di CSV.
- U02 ETHICS mempertahankan verdict-only sebagai 0 kata. Denominator panjang respons berbeda dari denominator penjelasan substantif pada marker; missing tidak menjadi nol.
- U06 jumlah hit dibagi total kata ×100, dengan raw count, total_words dan n_available disimpan. Ini pooled density kata, bukan mean density respons. Pemilihan istilah deskriptif, bukan uji baru.
- Kamus/surface berbeda: GSM8K kata pada visible_text, ETHICS frasa candidate pada explanation_text. Referen dapat berasal dari soal. Semua instrumen tetap candidate, belum selesai audit manual.
- U07 triplet purposif: model/soal sama di dalam benchmark, bukan soal sama lintas benchmark. Teks penuh, lokasi source dan offset cuplikan disimpan; contoh bukan sampel representatif.

[Candidate sebelumnya beserta opening, pronomina, perbandingan netral/implisit, kamus dan bukti](../index.html) tetap utuh. Tidak ada berkas lama dihapus.

## CHECKLIST dan TESTING

- [x] Fungsi plot bersama dan geometri ketujuh pasangan dicocokkan.
- [x] Seluruh indikator sumber tetap tercakup; raw count/denominator tersedia.
- [x] PNG/SVG/CSV dibuat pada folder baru; angka inferensial sumber tidak dihitung ulang.
- [x] Kedua notebook dieksekusi; hasil tercatat di verification.json.
- [x] Sumber dan candidate sebelumnya diperiksa dengan hash sebelum/sesudah.

## INTEGRATION dan reproduksi

Gunakan galeri untuk membandingkan jenis visualisasi terlebih dahulu. Belum mengganti notebook utama. Dari D:\TugasAkhir, gunakan **path baru** untuk membangun ulang:

```powershell
python -m src.MAIN_PARSING.ETHICS.build_gender_lm_unified --output "notebooks/gender/linguistic marker/candidate/unified_rerun"
```
