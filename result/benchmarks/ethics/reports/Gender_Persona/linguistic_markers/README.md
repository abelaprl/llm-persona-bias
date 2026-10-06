# ETHICS — Linguistic Marker Results

Indeks navigasi hasil analisis linguistic marker ETHICS untuk eksperimen
`Gender_Persona`. File ini hanya mendokumentasikan lokasi dan status artefak;
tidak mengubah isi hasil analisis.

## Bundle aktif

| Tujuan | Path | Status |
|---|---|---|
| Analisis overall seluruh ETHICS | `analysis/gender_ethics_linguistic_markers_overall_candidate_r2_20260905/` | **ACTIVE**; agregasi lintas subset, bukan penggabungan accuracy |
| Analisis detail per subset/dataset | `analysis/gender_ethics_linguistic_markers_v1_4_candidate_r4_20260826/` | **ACTIVE**; dipakai untuk detail dan lampiran |
| Source extraction response-level | `../linguistic_markers_gender_candidate_20260821_v1_6_0/` | **ACTIVE INPUT**; sumber marker level respons |
| Visual detail per subset | `visual/gender_ethics_direct_thesis_visuals_v1_3_1_r4_20260825/` | **ACTIVE** untuk visual detail yang sudah ada |

## Notebook aktif

- Overall: `D:/TugasAkhir/notebooks/linguistic_marker_gender_ethics_overall_v1_0_candidate.ipynb`
- Overall executed: `D:/TugasAkhir/notebooks/linguistic_marker_gender_ethics_overall_v1_0_candidate_executed.ipynb`
- Detail per subset: `D:/TugasAkhir/notebooks/gender/linguistic_marker_gender_ethics_v1_4_1_candidate.ipynb`

Notebook overall menghasilkan enam visual utama di bundle `overall...r2`,
sedangkan notebook detail tetap dipertahankan untuk analisis per subset dan
lampiran.

## Superseded / archive

Artefak versi lama dipindahkan ke folder `analysis/archive/` atau
`visual/archive/` hanya jika tidak memiliki referensi aktif yang terdeteksi.
Tidak ada artefak yang dihapus atau ditimpa.

Bundle berikut sengaja tetap berada di lokasi lama karena masih direferensikan
oleh dokumentasi atau notebook historis:

- `analysis/gender_ethics_linguistic_markers_v1_4_candidate_20260825/`
- `analysis/gender_ethics_linguistic_markers_notebook_v1_4_1_candidate_20260826/`
- `visual/gender_ethics_bab4_visual_supplement_v1_0_0_20260825/`
- `visual/gender_ethics_direct_thesis_visuals_v1_2_0_r4_20260825/`
- `visual/gender_ethics_direct_thesis_visuals_v1_3_0_r3_20260825/`

Status `candidate` tetap dipertahankan. Bundle overall `r1` adalah versi
superseded karena masalah duplicate direct Man–Woman rows; gunakan `r2`.

## Batas folder

`D:/TugasAkhir/result/linguistic_markers/` adalah hasil linguistic marker
GSM8K dan tidak dicampur dengan bundle ETHICS ini.

Semua path di atas bersifat relatif terhadap folder indeks ini kecuali path
notebook yang ditulis sebagai path absolut untuk menghindari ambiguitas.
