# GSM8K Religion Accuracy Analysis

Visual dan notebook ini mempertahankan bentuk Figure 05, 06, dan 26 yang dirujuk
oleh `D:/ProposalTA-18222008/Bab VI - Evaluasi.tex` pada baris 750, 775, dan 842.
Renderer membaca tabel read-only dari bundle v5 pada folder `tables`. Figure 05/06
menggunakan `religion_explicit_implicit_and_deltas.csv`; Figure 26 menggunakan
`religion_paired_effects_significance.csv`.

`neutral` berasal dari `neutral_persona_accuracy_six_models.csv` karena baseline
tersebut disimpan terpisah dari 16 kondisi persona agama. Rata-rata I1–I3 adalah
agregasi deskriptif, bukan uji statistik baru. Uji inferensial tetap membaca hasil
v5: Cochran's Q, Exact McNemar, Cohen's h, dan koreksi Holm dalam family 3/4/24.

Standardisasi mengikuti notebook Gender/Regional: judul utama 23 pt bold, judul panel
heatmap 19 pt bold, angka batang 17 pt normal, angka delta 16 pt normal di luar batang,
dan label/legend normal. Keluarga warna identitas stabil: Atheist biru, Christian
merah, Islam ungu, dan Jewish oranye. Figure 26 tidak memuat footnote di dalam gambar.
`render_manifest.json` menyimpan hash figure dan seluruh tabel yang dibaca notebook.
Tidak ada API call, inference, parsing ulang, rescoring, atau perubahan output lama.
