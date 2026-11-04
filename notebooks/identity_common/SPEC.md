# Penyamaan pengukuran identitas, 15 Oktober 2026

## SPEC
Permintaan: pronomina dan penyebutan gender, serta sapaan/sudut pandang agama,
ketidakberkaitan agama dengan soal, dan doa/berkat diukur pada GSM8K dan ETHICS.
Gunakan seluruh respons enam model, seluruh persona dan N/E/I1–I3; ETHICS
mencakup empat subset serta gabungan. Tidak ada API, scoring atau perubahan soal.

## DESIGN PLAN
- Input: empat snapshot final-only yang dipakai notebook aktif, dengan hash sumber.
- Gender: dua kelompok pronomina dan lima ukuran identitas yang sudah ada pada
  ETHICS (direct address, acknowledgment, second-person identity reference,
  first-person identity reference, perspective adoption). Ukuran tetap terpisah.
- Agama: direct address, perspective adoption, moral authority, secular humanist,
  pluralist, identity irrelevance dan blessing invocation. Gunakan aturan ETHICS
  yang sama pada kedua tugas; simpan frekuensi dan kemunculan secara terpisah.
- Aturan tidak diperluas dari hasil p-value. Seluruh keluaran baru memiliki folder
  tersendiri. Hasil lama tetap utuh. Analisis bersama ini merupakan keluarga Holm
  tersendiri yang ditetapkan sebelum pengujian: gender 7 x 18, agama 11 x 52,
  dikalikan lima cakupan pada ETHICS, per model. Angka lama tidak diganti diam-diam.
- Pronomina: teks terlihat; identitas: teks penjelasan. Respons tanpa teks bernilai
  NA. Respons label saja tetap masuk penyebut pronomina; abstain tidak dikeluarkan.
- Pairing memakai subset + ID kanonis + hash isi soal dan pair_eligible, bukan ID saja.
- Binary: Exact McNemar, delta pp, Cohen h, discordances, bootstrap CI delta.
  Density: Wilcoxon, matched rank-biserial, delta/100 kata, bootstrap CI delta.
  Bootstrap 2000, seed tetap. Holm mencakup seluruh hipotesis keluarga termasuk
  perbandingan tanpa perubahan (p=1); pasangan kosong dicatat sebagai NA.

## IMPLEMENTATION
Modul CLI dan notebook bersama, ditautkan dari empat notebook aktif. Snapshot
registry, hash kode/input, fitur per respons, profil, hasil pasangan, audit dan
ringkasan tersedia bersama hasil baru. Integrasi naskah setelah pemeriksaan.

## CHECKLIST / TESTING / INTEGRATION
Periksa pemisahan reasoning, angka/label saja, rujukan tokoh vs pengguna, deduplikasi
pasangan, hasil nol vs NA, Holm, kesetaraan aturan lintas tugas, dan sampel respons
nyata. Hit transfer aturan diperiksa dengan konteks respons. Naskah dan tabel
keluarga Holm harus menyatakan cakupan yang betul. Compile dan periksa PDF akhir.
