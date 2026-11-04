# Analisis performa persona agama — ETHICS
Penyelarasan 6 Oktober 2026 langsung dalam dua notebook yang ada. Buka notebook lengkap; notebook Heatmap_06_13 mereproduksi tujuh heatmap.
PNG aktif berada di figures/. Run terbaru dan lokasi arsip: LATEST_RUN.json; hash tabel/gambar: presentation_manifest.json dan output_index.csv.

Akurasi hanya per subset Commonsense/Justice, dibaca bersama commit_rate. Deontology/Virtue memakai distribusi wrong/acceptable.
Ringkasan stance/status gabungan bersifat deskriptif berbobot respons, bukan akurasi gabungan.
Empat grafik gabungan kecocokan label lama ada di _arsip, bukan hasil utama. Semua tabel lama dipertahankan.
Detail per subset kini juga tersedia langsung di figures/ (01_<subset>_performance dan 02_<subset>_status). Folder per_subset lama tidak dipakai notebook aktif.

N: netral; E: eksplisit; I1-I3: implisit. Rerata I1-I3 hanya deskriptif. Warna identitas E/I sama dengan GSM8K; warna status memiliki arti tersendiri.
* selalu p Holm <0,05, termasuk pada gambar p mentah. Nilai p/efek, keputusan dan keluarga sumber tidak diubah.
Heatmap 10-13: persona-N, risk difference, keluarga 16/model/subset/outcome. N pasangan dan discordant tersedia pada ethics_religion_parity_paired_effects_vs_neutral.csv.
Heatmap 17: 52 kontras/model untuk masing-masing Commonsense/Justice, outcome ketepatan seluruh pasangan, non-committed dianggap tidak benar pada tabel sumber; keluarga 7/24.
Cohen's h dihitung sumber sebagai kiri-kanan; jangan menyamakan dengan conditional accuracy pada pasangan committed.
NA tidak diganti nol. Normalisasi warna efek simetris per gambar; bandingkan angka, bukan intensitas lintas gambar.
PNG token pendukung 14-16 dipertahankan dari paket sebelumnya, bukan dihitung ulang pada pekerjaan ini; tradeoff 16 mempunyai scope akurasi Commonsense+Justice dan scope token empat subset, bukan akurasi moral gabungan empat subset.

Tidak ada API, parser/scoring/uji ulang atau perubahan Bab VI. Keseragaman tampilan tidak membuktikan validitas seluruh pipeline.
