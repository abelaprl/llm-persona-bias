# ETHICS religion — revisi lebar 14 cm

[Buka hasil PNG/PDF dan caption](../../../result/analyses/accuracy/runs/religion_ethics_bab6_14cm_20261008_230544_472975/MULAI_DI_SINI.md).

Folder:
`D:\TugasAkhir\result\analyses\accuracy\runs\religion_ethics_bab6_14cm_20261008_230544_472975`

Isi paket:

- `bab6/06_accuracy_*`: dua heatmap akurasi Ateis–Kristen dan Islam–Yahudi;
  masing-masing Commonsense/Justice, delapan skenario + N. Baris bawah c
  menunjukkan commit rate. Tabel lengkap mendampingi akurasi.
- `bab6/11_effect_persona_vs_N_*`: dua heatmap efek, masing-masing delapan
  kolom. Tetap risk difference (poin persentase) pada pasangan sama-sama
  committed; `*` Holm <0,05 dari keluarga 16, bukan delapan. Tidak memakai
  prasyarat Q atau mengubah metrik menjadi Cohen's h.
- `bab6/00_total_response_distribution_*`: dua gambar empat status dengan
  penyebut seluruh respons; I menggabungkan hitungan I1–I3.
- `bab6/04_abstain_*`: empat gambar alasan abstain (dua kelompok agama ×
  dua kelompok tiga model), skala 0–100% dan n per skenario.
- `pelengkap_committed/`: dua grafik Wrong/Acceptable berpenyebut committed.
  Jangan disamakan dengan distribusi seluruh respons.

N diulang identik sebagai referensi, bukan tambahan data. PDF individual
vektor, tepat 14 cm. Proof A4 di `proof/` untuk pemeriksaan cetak 100%;
preview digital sudah diperiksa, cetak fisik belum. Caption di `CAPTIONS.md`;
coverage, denominator, n pasangan dan pemeriksaan tersedia di `tables/`.

Kode langsung di [notebook revisi](ETHICS_Religion_BabVI_14cm_20261008.ipynb).
Run All membuat direktori baru. Notebook lama, angka sumber, nilai-p,
keluarga Holm, dan `.tex` tidak diubah. Grafik di luar permintaan, termasuk
I−E/antaragama dan stance Deontology/Virtue, tidak diganti oleh paket ini.
