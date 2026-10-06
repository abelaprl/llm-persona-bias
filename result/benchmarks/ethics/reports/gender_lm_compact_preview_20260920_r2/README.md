# Pratinjau ringkas penanda linguistik Gender ETHICS

14 figur utama · 41 penanda · 6 model · 9 persona · candidate, belum validasi manusia. Notebook aktif dan 168 grafik lama tidak diubah.

Unit observasi: satu respons. Teks final tersedia, termasuk abstain; bukan reasoning channel. Profil memakai prevalensi (%) agar satuannya sama. Inferensi tetap memakai densitas (kejadian/100 kata) atau presence sesuai registry. Jumlah mentah tersedia dalam tabel profil. Respons missing/teknis bukan nol; verdict-only mempunyai penjelasan sepanjang nol kata.

N = neutral; M/W = Man/Woman; E = eksplisit; I1–I3 = skenario implisit. L31/L33 = Llama 3.1/3.3; MIN = Ministral; DSR1 = DeepSeek R1; GPT = GPT-OSS; QWN = Qwen3 Next.

Warna efek biru = negatif, merah muda = positif. Semua efek, termasuk nonsignifikan, ditampilkan; * berarti p Holm < 0,05, bukan CI yang dikoreksi. CI nominal, n_pairs, n_changed, dan p lengkap ada di tabel serta lampiran. CI untuk selisih rerata tidak identik dengan estimand uji Wilcoxon. Tidak ada penghitungan ulang uji atau pooling p-value.

Gambar 10–13 hanya peta arah efek, bukan uji robustness baru. Efek kecil dekat nol dapat berpindah tanda; lihat estimasi dan CI sebelum menyimpulkan. Perbandingan neutral dan E–I serta forest plot lengkap tetap tersedia di lampiran.

## Reproduksi

Dari D:/TugasAkhir:

```powershell
python -X utf8 -B -m src.MAIN_PARSING.ETHICS.build_gender_lm_compact_preview --output result/ETHICS/reports/NAMA_BARU
```

## Batas dan pemetaan

SPEC: peringkasan presentasi, tanpa mengubah ekstraksi/statistik.

DESIGN: 2 cakupan/dukungan, 3 profil, 3 efek gender, 1 konteks, 2 subset, 2 normalisasi, 1 interaksi. IMPLEMENTATION: seluruh keluaran di folder baru; hasil lama ditautkan, bukan dihapus.

CHECKLIST/TESTING: unit test untuk tanda, NA, nol, satuan, dan penolakan overwrite; round-trip nilai CSV; 14 figur 16 cm/300 dpi; hash sumber dan notebook aktif sebelum/sesudah. Rincian mesin di verification.json. INTEGRATION: pratinjau saja, belum dipasang ke notebook aktif.

Pemadatan mengorbankan tampilan CI pada halaman utama, bukan menyembunyikan CI dari hasil. Lihat tabel/forest plot sebelum menginterpretasikan efek. Kamus dan sumber teoritis tetap mengikuti registry.json; penanda adaptasi/buatan sendiri tetap berstatus candidate. Ringkasan overall di sini adalah penanda linguistik, bukan akurasi ETHICS gabungan.
