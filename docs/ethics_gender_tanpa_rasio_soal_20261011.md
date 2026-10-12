# Penghapusan anotasi rasio soal ETHICS gender

## SPEC

Hapus anotasi jumlah pasangan berbentuk angka/angka dan k/n atau n/k beserta
keterangannya pada gambar aktif ETHICS gender. Pertahankan angka ukuran efek,
persentase coverage, skala, label pengukuran, CI, simbol signifikansi, dan TP/ND.
Cakupan seluruh gambar yang terkait, tanpa sampling atau penghitungan statistik ulang.

## DESIGN PLAN

Gunakan renderer notebook dan tabel tersimpan. Bersihkan objek teks sebelum
ekspor PNG/PDF dan pasang filter yang sama di notebook agar perubahan bertahan
saat Run All. Simpan hasil dan backup di direktori baru; hasil historis tetap utuh.

## IMPLEMENTATION

Filter: `src/benchmarks/ethics/gender_figure_count_labels.py`.
Renderer: `src/benchmarks/ethics/remove_gender_ethics_plot_status_label.py`
menerima filter opsional, tanpa mengubah perilaku default.
Ekspor PDF memakai `src/benchmarks/ethics/clean_gender_count_labels_pdf.py`:
objek teks rasio dihapus secara transparan dari PDF vektor, lalu PNG diekspor
kembali pada dimensi aslinya. Pemeriksaan membandingkan seluruh path grafik,
multiset angka desimal, serta jumlah tanda signifikansi sebelum/sesudah.
Versi PNG tanpa PDF tetap dirender dari tabel dan sumber notebook.

## CHECKLIST

- [x] Identifikasi rasio pada sel heatmap, kolom samping forest plot, dan footer.
- [x] Tentukan perubahan hanya pada objek teks gambar.
- [x] Render ulang, validasi, dan pasang 490 PNG serta 375 PDF terdampak.
- [x] Cadangkan seluruh 865 gambar aktif sebelum pemasangan.
- [x] Perbarui hash pada indeks dan keluaran PNG notebook yang cocok.

## TESTING

Verifikasi tidak ada anotasi rasio tersisa, hash tabel tetap sama, dan lihat
contoh heatmap, forest plot, serta grafik coverage setelah render.
Seluruh 490 PNG lolos pemeriksaan integritas. Seluruh 375 PDF lolos
pemeriksaan ketiadaan rasio, kesamaan path grafik, angka desimal, serta
jumlah bintang. Lima sel presentasi notebook yang diedit lolos kompilasi.
Pemeriksaan visual mencakup coverage, forest plot, dan heatmap termasuk
grafik dengan jarak rapat antara footer dan skala warna.

## INTEGRATION

Notebook dan folder gambar aktif diperbarui setelah verifikasi. Backup gambar
dan notebook disimpan dalam direktori run baru untuk pemulihan.

Run hasil:
`result/analyses/linguistic_markers/runs/gender_ethics_without_counts_20261011_1056/`.
Manifest validasi ada di `manifest.json`; backup di `backup_active_before_install/`.
Folder aktif: `notebooks/gender/linguistic marker/ETHICS/figures/`.
Hasil historis di `result/` tidak ditimpa. Perubahan hanya presentasi;
parsing, statistik, persona, dan runner tidak dihitung ulang atau diubah.

Perintah render ulang dari sumber tersimpan:
`python -X utf8 -m src.benchmarks.ethics.remove_gender_figure_count_labels`.
Pemasangan: tambahkan `--install <direktori_run>` setelah pemeriksaan.
Ekspor PDF menggunakan PyMuPDF 1.28.2 yang dipasang lokal pada
`.codex_tmp/gender_ethics_counts_20261011/pdf_tools/`; Run All notebook
memakai filter Matplotlib dan tidak memerlukan pustaka PDF tersebut.
