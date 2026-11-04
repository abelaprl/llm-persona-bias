# Revisi ETHICS gender untuk Bab VI

[Buka daftar PNG/PDF dan caption](../../../result/benchmarks/ethics/reports/gender_ethics_bab6_revision_20261008_004712_065174/MULAI_DI_SINI.md).

Folder gambar:
`D:\TugasAkhir\result\benchmarks\ethics\reports\gender_ethics_bab6_revision_20261008_004712_065174\bab6`

Enam jenis gambar menjadi sembilan PNG/PDF:

1. `00`: distribusi; definisi Wrong dan gabungan I.
2. `01`: akurasi Commonsense + Justice pada committed, tanpa Neutral Avg,
   disertai coverage.
3. `03`: delta dalam poin persentase; panel I−E dan Man I−Woman I.
4. `06a`, `06b`, `06c`: persona−N, I−E, Man−Woman; skala h bersama −0,6…+0,6.
   `*` = Holm; `†` tambahan = Q untuk antarskenario; Man−Woman tidak memakai Q.
5. `08`: keputusan Wrong (% dari committed), Deontology/Virtue terpisah.
6. `04a`, `04b`: alasan abstain, masing-masing tiga model; skala 0–100% dan
   n abstain per kolom, dengan I1–I3 terpisah.

PDF individual memakai sel dan teks vektor. Proof A4 di folder `proof/`
merupakan pratinjau raster dengan lebar gambar 14 cm. Cetak 100% untuk
pemeriksaan fisik. Caption di `CAPTIONS.md`; coverage dan angka di `tables/`.

Kode langsung di [notebook revisi](ETHICS_Gender_BabVI_Revisi_20261008.ipynb).
Run All membuat direktori baru. Tidak perlu menjalankan notebook lama.

Naskah `.tex`, PNG aktif lama, hasil parsing, dan nilai-p tidak diubah.
Paket ini memvalidasi presentasi, bukan mengaudit ulang kebenaran statistik.
