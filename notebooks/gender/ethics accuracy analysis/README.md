# ETHICS Gender Accuracy Analysis

**Yang dipakai:** [ETHICS_Gender_Accuracy_Analysis_Lengkap.ipynb](ETHICS_Gender_Accuracy_Analysis_Lengkap.ipynb).
`figures/` berisi **20 gambar utama dan pendukung** (PNG/SVG), `per_subset/` berisi 8 gambar detail.
Gambar tak terpakai dan bundle lama berada di `archive/20260910_cleanup/`.
Data di `tables/` tetap diperlukan. [Daftar gambar aktif](active_figures.csv).
[Panduan semua notebook ETHICS](../../ETHICS_MULAI_DI_SINI.md).

Notebook yang sama menyajikan chart gabungan empat subset sebagai bagian utama,
diikuti rincian per subset. Grafik utama memakai format terpadu di bawah;
diagram diagnostik pendukung mempertahankan pembesaran teks nonjudul 2 pt.

## Format visual yang diselaraskan

Grafik akurasi memakai format Regional/Religion: 3 desimal, anotasi 20 pt,
sumbu Y 0–1, serta palet GSM8K Regional (empat warna pertama dan Neutral).
Heatmap utama memakai 3 panel vertikal: Man (7), Woman (7), dan Man–Woman (4).
File p-value `09` lama kini menampilkan Exact McNemar; Holm berada di
`09_ethics_gender_combined_holm_pvalue_heatmap`. Nilai dan cakupan tidak berubah.
Reproduksi: `python src/MAIN_PARSING/ETHICS/harmonize_regional_religion_ethics_visuals.py`.

## Cakupan parent

- Tiga chart utama `00_ethics_combined_four_subsets_*`: distribusi jawaban,
  delta terhadap netral, dan delta implicit–explicit; seluruhnya memakai empat subset.
- Distribusi menampilkan wrong, acceptable, abstain, dan missing sebagai proporsi
  seluruh respons per model dan kategori persona.
- Delta memakai proporsi wrong pada respons committed; positif berarti lebih
  banyak jawaban wrong. Delta bersifat deskriptif, tidak disertai p-value gabungan.
- Hitungan semua subset dijumlahkan; subset dengan lebih banyak respons berbobot
  lebih besar. Tiga varian implicit dipool berdasarkan hitungan, bukan rata-rata persentase.
Dua metrik utama dihitung pada keempat subset: akurasi penuh (n_correct/total) dan proporsi wrong pada respons committed (n_answer_wrong/committed). Grafik dua metrik tersedia pada 00_ethics_<domain>_two_metrics_all_subsets_heatmap.
- Status `committed`, `abstain`, dan `missing`: dipool dari seluruh empat subset.
- Penyebab `abstain`: chart khusus ETHICS dengan denominator `% dari abstain`.
- Judul visual parent memakai `ETHICS Gender`, tanpa kata `overall`.

## Artefak utama

- `ETHICS_Gender_Accuracy_Analysis_Lengkap.ipynb`
- `figures/` — 16 visual yang ditampilkan di notebook dan 9 visual terdahulu
  sebagai pelengkap; seluruhnya PNG/SVG dengan teks non-judul +2 pt.
- `tables/` — tabel akurasi, status, alasan abstain, dan paired effects.
- `bundle_manifest.json` — provenance serta scope metrik.
- `output_index.csv` — hash artefak lokal.
- `tables/ethics_combined_four_subsets.csv` — 30 baris model × kategori persona,
  dengan hitungan wrong/acceptable, coverage, abstain, dan missing.
- `tables/font_size_audit.csv` — ukuran font sebelum/sesudah per elemen teks.

Visual terdahulu yang memakai akurasi gabungan Commonsense/Justice tetap tersedia
sebagai pelengkap historis. Visual tersebut tidak mewakili gabungan empat subset
dan tidak ditampilkan sebagai hasil utama notebook.

## Rincian per subset

Rincian keempat subset tampil setelah chart gabungan di notebook dan tersedia di:

```text
per_subset/commonsense/
per_subset/deontology/
per_subset/justice/
per_subset/virtue/
```

Setiap child berisi tabel `condition_summary.csv`, `abstain_reason.csv`,
`paired_effects.csv`, serta visual subset.

## Bangun ulang

```powershell
python src/MAIN_PARSING/ETHICS/render_gender_combined_existing.py
python src/MAIN_PARSING/ETHICS/update_gender_notebook_existing.py
```

Bundle lama disimpan di `per_subset/archive_v1/` dan tidak dihapus.
