# Catatan validasi setelah build

**Jangan gunakan tabel sensitivitas both_correct pada run r4 sebagai hasil akhir.**

Pemeriksaan pascabuild menemukan `analysis_is_correct` bertipe desimal 1.0/0.0 tidak dikenali oleh helper nullable_bool v2. Akibatnya perhitungan ulang conditional untuk populasi both_correct menghasilkan pasangan kosong. Ini bug konversi pelaporan; ekstraksi 154.428 respons dan perubahan dua idiom tetap cocok dengan replay yang diverifikasi.

Perbaikan mempertahankan nullable boolean dan telah diuji dengan CSV aktual: pada pemeriksaan within conditional both_correct, 144 baris kembali memiliki 5–206 pasangan, tanpa pasangan kosong, serta effect/p_raw identik dengan baseline. Tidak ada output r4 yang ditimpa. Laporan pengganti dibangun pada `../regional_lm_v3_20260916_r5/`; gunakan setelah manifest-nya menyatakan complete dan pemeriksaan akhir selesai.
