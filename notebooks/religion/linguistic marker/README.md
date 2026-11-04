# Linguistic marker religion

Pembaruan: 2026-10-05. Paket tetap **kandidat**, bukan hasil akhir tervalidasi untuk tesis.

## Notebook aktif

- [GSM8K](GSM8K/linguistic_marker_religion_gsm8k_bab4_visual_v1_3_1_volcano.ipynb)
- [ETHICS](ETHICS/linguistic_marker_religion_ethics_v3_candidate_20260916_r4_overall.ipynb)

Masing-masing notebook berisi pengolahan sumber, pemeriksaan pasangan soal,
statistik, dan kode plotting. Jalankan **Run All** dari lingkungan Python proyek.
Tidak ada skrip implementasi baru atau pemanggilan API berbayar.

PNG hasil eksekusi yang sudah lolos pemeriksaan dipasang di `GSM8K/figures/`
dan `ETHICS/figures/`. `LATEST_RUN.json` dan `figure_index.csv` di masing-masing
folder menunjukkan keluaran yang benar-benar sudah selesai, bukan sekadar
notebook yang telah diedit.

ETHICS menyediakan hasil per subset dan ringkasan gabungan. Efek gabungan berbobot
pasangan tersedia, bukan empat subset berbobot sama. Baca dukungan pasangan
sebelum menafsirkan ukuran efek.

Notebook/gambar lama dipertahankan di `_arsip/` atau lokasi historisnya.
Notebook GSM8K lama di folder induk `notebooks/religion/` adalah artefak historis;
gunakan tautan aktif di atas untuk penghitungan baru. Paket `candidate/unified_v1`
tidak diubah oleh pembaruan ini. LaTeX Bab IV/VI tidak disentuh.

Kamus marker tetap kandidat. Perbaikan teknis pairing, kanal teks, statistik,
dan tampilan tidak menggantikan audit manusia independen atas validitas kamus.

## Catatan audit tambahan

- Pemeriksaan 50 sampel acak (seed 20261005) dari 241 respons GSM8K lama yang
  hanya berisi bilangan mengonfirmasi bahwa seluruh 50 teks final memang hanya
  bilangan. Pemeriksaan agen ini bukan audit manusia independen.
- Uji seleksi kanal pada 12 respons nyata (enam model, dua benchmark) memberi
  keluaran identik ketika metadata GT/prediction dihapus atau diganti.
- Uji numerik tambahan: 50 kasus signed-rank berties dibandingkan dengan
  enumerasi tanda, dua kasus dibandingkan dengan permutasi SciPy, serta
  pemeriksaan pembalikan arah dan Holm dengan hipotesis tidak tersedia.
- Dataset `data/ETHICS_test_4subsets_10percent_from200.json` memiliki 359 baris
  Deontology tetapi 338 teks skenario unik: 39 baris berada dalam 18 kelompok
  teks berulang (maksimal tiga pengulangan). Eksperimen tidak menyertakan
  excuse, sehingga ini juga merupakan keterbatasan unit prompt. Notebook tetap
  memakai satu respons per observasi dan tidak menghapus baris diam-diam;
  sebelum klaim inferensial final, ketergantungan/pengulangan prompt tersebut
  perlu diperiksa dengan analisis sensitivitas tingkat skenario.
- Audit seluruh JSON ETHICS menemukan 180 respons DeepSeek yang isi soalnya
  tidak cocok dengan ID pada dataset kanonis: lima ID awal Deontology dan
  Virtue pada seluruh 17 skenario, serta lima ID awal Commonsense dan Justice
  pada netral. Pencocokan isi menghasilkan 17 baris ambigu dan 326 baris yang
  bertabrakan pada kunci soal (termasuk pasangan baris kanonisnya).
  **343 baris ditahan dari pairing, bukan dihapus dari inventaris.** Kebijakan
  saat ini konservatif: semua anggota kunci yang bertabrakan ditahan; tidak
  memilih salah satunya berdasarkan jawaban atau hasil statistik. Restorasi
  baris membutuhkan audit provenance percobaan, bukan menebak file terbaik.
  Bukti: `result/analyses/linguistic_markers/runs/religion_ethics_raw_rebuild_20261005_084413_121886/source_audit.csv`.
- Pada seluruh sumber GSM8K terdapat tiga JSON tidak valid dan 12 baris yang
  bertabrakan pada kunci soal. Semua tetap dicatat; tidak diisi dengan tebakan.
  Bukti: `result/analyses/linguistic_markers/runs/religion_gsm8k_raw_rebuild_20261005_084234_591790/source_audit.csv`.
