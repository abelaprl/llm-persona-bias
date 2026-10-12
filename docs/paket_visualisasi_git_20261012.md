# Paket kode visualisasi — 12 Oktober 2026

## SPEC

Simpan skrip visualisasi yang belum dilacak beserta dokumentasi pendamping
melalui satu commit kecil pada `ethics-accuracy`. Cakupan pekerjaan adalah
kode dan dokumentasi; tidak menjalankan eksperimen atau mengambil sampel
dataset. Hasil penelitian dan notebook aktif tetap berada pada versi lokalnya.

## DESIGN PLAN

Cadangkan indeks Git serta patch staging lama sebelum menyusun ulang staging.
Masukkan hanya daftar file yang diperiksa, lalu cocokkan daftar tersebut dengan
isi commit. Verifikasi ukuran setiap blob dan kesamaan commit lokal dengan
branch remote setelah push.

## IMPLEMENTATION

Paket berisi 12 file Python pada `src/benchmarks/ethics/`:

- Filter anotasi: `gender_figure_count_labels.py`, `identity_figure_count_labels.py`.
- Penghapusan rasio: `clean_gender_count_labels_pdf.py`,
  `remove_gender_figure_count_labels.py`, `remove_identity_figure_count_labels.py`.
- Renderer label: `remove_gender_ethics_plot_status_label.py`,
  `remove_regional_plot_status_label.py`, `remove_religion_plot_status_label.py`.
- Presentasi regional: `render_regional_ethics_14cm.py`,
  `render_regional_gsm8k_without_axes.py`, `validate_regional_ethics_14cm.py`.
- Tes filter: `test_identity_figure_count_labels.py`.

Import pada dua renderer regional disesuaikan agar mendukung pemanggilan
sebagai modul maupun skrip langsung. Aturan pengukuran tidak diubah.

Dokumen pendamping:

- [Rasio gender](ethics_gender_tanpa_rasio_soal_20261011.md).
- [Rasio regional dan religion](ethics_regional_religion_tanpa_rasio_soal_20261011.md).
- [Rencana regional 14 cm](RENCANA_REVISI_GAMBAR_ETHICS_REGIONAL_BAB6_14CM_20261009.md).
- [Hasil regional 14 cm](HASIL_REVISI_GAMBAR_ETHICS_REGIONAL_BAB6_14CM_20261009.md).

## CHECKLIST

- [x] Batasi paket pada 12 skrip dan 5 dokumen, termasuk catatan ini.
- [x] Periksa sintaks dan import seluruh 12 modul.
- [x] Periksa antarmuka bantuan tiga renderer dalam dua cara pemanggilan.
- [x] Jalankan lima tes filter yang tersedia.
- [ ] Render ulang seluruh gambar: di luar cakupan commit ini.

## TESTING

Perintah tes yang dijalankan, dengan hasil lima tes lulus:

```powershell
python -B -X utf8 -m unittest src.benchmarks.ethics.test_identity_figure_count_labels -v
```

Seluruh 12 modul lolos pemeriksaan AST dan import. Enam pemeriksaan `--help`
lulus untuk `remove_regional_plot_status_label`,
`remove_religion_plot_status_label`, dan `remove_identity_figure_count_labels`,
masing-masing melalui `python -m src.benchmarks.ethics.<nama>` dan
`python src/benchmarks/ethics/<nama>.py`.

Pemeriksaan ini memvalidasi pemuatan kode, antarmuka, dan perilaku filter yang
dites. Angka hasil render dalam empat dokumen pendamping adalah catatan
pekerjaan sebelumnya; seluruh gambar tidak divalidasi ulang pada pekerjaan Git ini.

## INTEGRATION

Skrip membutuhkan NumPy, pandas, Matplotlib, dan Pillow. Pembersih PDF juga
membutuhkan PyMuPDF; implementasi yang disimpan masih mencari instalasi lokal
di `.codex_tmp/gender_ethics_counts_20261011/pdf_tools/` sebelum lokasi paket
Python biasa. Renderer regional 14 cm memerlukan sumber tesis pada
`D:/ProposalTA-18222008` serta XeLaTeX dan utilitas Poppler.

Skrip ini merekam workflow presentasi tertentu: path run, snapshot notebook,
dan jumlah gambar yang diharapkan masih spesifik. Beberapa snapshot berada
di `.codex_tmp/` dan tidak disertakan dalam Git. Tabel dan notebook masukan
juga tidak ditambahkan oleh paket ini. Karena itu, clone saja belum cukup
untuk mereproduksi seluruh render; siapkan masukan yang disebut setiap skrip.

Opsi `--install` mengganti gambar aktif setelah pemeriksaan dan backup.
Tidak ada rendering, pemasangan gambar, atau kompilasi tesis yang dijalankan
untuk membuat commit ini. `verify_rebuild_ta_lm_no_ratios.py` tetap lokal
karena menangani publikasi `TA.pdf` pada repositori tesis terpisah.
