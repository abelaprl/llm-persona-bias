# Linguistic marker religion — GSM8K
Tanggal: 2026-10-05. Status: **kandidat; audit manusia independen belum selesai**.

## Lokasi
Notebook aktif: linguistic_marker_religion_gsm8k_bab4_visual_v1_3_1_volcano.ipynb. PNG aktif: figures/. Sumber lengkap keluaran ini: D:\TugasAkhir\result\linguistic_markers\religion_gsm8k_raw_rebuild_20261005_084234_591790.
Tidak ada HTML/SVG baru. Hasil lama disimpan di _arsip/, bukan dihapus.
Run All membaca seluruh korpus yang tersedia, membuat direktori hasil baru, lalu memasang gambar setelah pemeriksaan lolos.

## Yang berubah
- Pairing memakai isi soal yang cocok dengan dataset dan terverifikasi muncul dalam pesan pengguna terakhir. Nomor file saja tidak cukup.
- GSM8K memakai seluruh 1.319 soal test; ETHICS memakai seluruh 1.514 soal dalam eksperimen empat subset, bukan seluruh benchmark ETHICS global.
- API final kosong tidak dipulihkan dari reasoning. Semua respons tetap diinventarisasi, tanpa filter committed/correct.
- Kamus dibekukan: 30 marker; definisi/regex tidak ditambah. Setiap sumber, definisi, dan pengecualian ada di registry.json.
- Densitas dan prevalensi dipisahkan. Jumlah mentah, densitas per 100 kata, dan kehadiran disimpan per respons.
- Seluruh 52 kontras/model tersedia: 16 persona–N, 12 I–E, 24 antaridentitas. Arah selalu **kiri dikurangi kanan**, tertulis di judul.
- Warna tidak dipudarkan berdasarkan p-value. Heatmap mencantumkan k/n, arsiran dukungan rendah, dan tanda signifikansi.
- Keluarga Holm yang sama mencakup seluruh 2,340 hipotesis terencana/model dalam paket benchmark ini, termasuk subset dan gabungan untuk ETHICS. Hipotesis tidak teruji tetap dihitung dalam ukuran keluarga, p-nya tetap NA.

## Membaca gambar
- **ND**: tidak ada pasangan penjelasan yang dapat dibandingkan. Bukan efek nol.
- **TP**: pasangan ada tetapi semua selisih nol; n tetap terlihat. Rank-biserial tidak terdefinisi dan tidak diubah menjadi nol. Untuk prevalensi, selisih aktual nol tetap tersedia di CSV.
- **k**: pasangan dengan selisih nonnol; **n**: pasangan penjelasan yang dibandingkan.
- Arsiran/† atau segitiga: k<20, hanya aturan penandaan dukungan rendah, bukan batas universal validitas.
- Bintang atau tepi hitam: p Holm<0,05.
- Rank-biserial dihitung dari peringkat selisih densitas per respons. Nilai ±1 tidak berarti perubahan luas; selalu baca k/n.
- Prevalensi: selisih persentase respons dengan marker, dalam poin persentase; uji McNemar eksak.
- Garis CI pada karakteristik adalah CI **selisih rata-rata**, bukan CI rank-biserial. Bootstrap 2.000 kali pada pasangan; ETHICS gabungan distratifikasi subset dengan bobot pasangan tersedia. CI disembunyikan bila n<2, k<6, degenerat, atau strata berisi <2 pasangan.
- Profil semua marker bersifat deskriptif. Tidak menyiratkan signifikansi; semua efek inferensial tetap ada di CSV.

## Kebijakan statistik dan perbandingan dengan paket lama
Ini bukan sekadar perubahan tata letak. Pasangan soal dan kanal teks diperbaiki.
Uji numerik kini diselaraskan dengan paket gender: signed-rank dua sisi, exact sign permutation pada k≤20 (mendukung ties), aproksimasi normal pada k>20.
Uji tanda disimpan sebagai sensitivitas dalam p_sign_sensitivity dan p_sign_holm.
Paket GSM8K lama memilih sign/Wilcoxon secara adaptif dan keluarga Holm lebih kecil; paket ETHICS lama memakai randomisasi selisih rata-rata.
Jadi p-value dan banyaknya tanda signifikan lama–baru **tidak boleh dibandingkan seolah hanya data yang berubah**.
Ukuran mean_delta, median_delta, n_pairs, n_nonzero, dan rank-biserial tersedia untuk penelaahan.
Tidak dibuat klaim bahwa ambang k atau satu p-value menjamin kualitas pengukuran.

## Keterbatasan yang tetap berlaku
1. Validasi manusia independen kamus/parsing belum selesai; gambar bukan hasil final siap tesis.
2. I–E membandingkan paket persona dan konteks percakapan; implicit dapat memiliki warmup assistant panjang. Bukan efek identitas murni. Lihat conversation_context.csv.
3. Kesimpulan hanya untuk penjelasan tersedia; cakupan rendah bukan bukti model tidak memiliki suatu ciri bahasa.
4. ETHICS gabungan berbobot pasangan yang tersedia, bukan empat subset berbobot sama. Baca pair_support.csv dan pooled_subset_direction_check.csv; utamakan analisis subset.
5. Definisi tokenisasi/kamus antarbenchmark tidak sepenuhnya sama. Titik temu adalah 14 marker bahasa umum dan rancangan pelaporannya; marker spesifik tugas tidak dipaksakan identik.
6. Deontology pada eksperimen tidak memasukkan excuse; virtue menggunakan instruksi moral umum. Tidak ada perubahan scoring atau klaim akurasi gabungan.
7. Rujukan teori pada registry bukan bukti semua regex tervalidasi; kamus korpus/custom tetap kandidat.

## Tabel penting
- source_audit.csv: provenance tiap JSON, kanal, status pairing, dan konteks percakapan.
- pairing_audit.csv: perbandingan pairing nomor lama vs isi soal.
- response_records.csv.gz: seluruh pengukuran dan final text; tidak menyimpan reasoning tersembunyi.
- coverage.csv, pair_support.csv: denominator respons dan pasangan.
- outcome_registry.csv, paired_statistics.csv.gz: outcome, efek, dukungan, p mentah/Holm, metode, status CI.
- figure_index.csv: indeks semua PNG.
- manifest.json: hash sumber dan keterbatasan.
