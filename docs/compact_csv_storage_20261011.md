# Penyimpanan CSV ringkas dan perbaikan push Git

Pengerjaan 11 Oktober 2026. Sebanyak **39 arsip CSV unik pada 50 path** yang ukurannya
minimal 20 MiB dan memperkenalkan blob baru dalam commit lokal diubah dari
gzip ke Zstandard untuk penyimpanan Git. Semua CSV asli tetap berada di
lokasi lokal semula. Tidak ada baris, kolom, angka, atau teks yang dihapus.

| Ukuran agregat 50 file fisik | Sebelum | Sesudah |
|---|---:|---:|
| Byte arsip | 3,182,561,182 | 2,108,110,220 |
| GiB arsip | 2.964 | 1.963 |

Pengurangan ukuran arsip: **33.76%**. Tabel menghitung
file fisik; deduplikasi objek Git dapat menghemat ruang tambahan.
Arsip Zstandard terbesar: **81.07 MiB**.

SPEC: mengurangi ukuran penyimpanan dan memperbaiki riwayat lokal yang belum
terkirim, dengan mempertahankan seluruh isi CSV dan tanggal commit.

DESIGN: kompresi Zstandard level 9, window 128 MiB, pencarian pengulangan jarak
jauh, dan checksum frame. SHA-256 atas byte CSV yang didekompresi menjadi uji
kesetaraan isi. File baru memakai akhiran `.csv.zst` di folder yang sama.

IMPLEMENTATION: [`compact_csv_archives.py`](../src/tools/compact_csv_archives.py)
melakukan kompresi, verifikasi, dan pemulihan. Pemetaan path, hash gzip asal,
hash CSV, dan hash arsip baru tercatat dalam
[`compact_csv_manifest_20261011.json`](compact_csv_manifest_20261011.json).
Gzip asli dilepas dari pelacakan Git dan diabaikan melalui `.gitignore`, tetapi
tidak dihapus dari disk. Tidak ada API model yang dijalankan.

CHECKLIST:

- [x] 50 arsip lolos verifikasi hash isi CSV setelah dekompresi.
- [x] Gzip asli tidak berubah selama kompresi.
- [x] Semua blob baru dalam riwayat pengganti berada di bawah 100 MiB.
- [x] Pohon akhir riwayat data hanya berbeda pada 50 penggantian format arsip.
- [x] Tanggal author dan committer dari commit asli dipertahankan.
- [x] Batas atas ukuran paket tiap bagian berada di bawah 1 GiB.
- [ ] Push ke remote: belum dilakukan oleh pekerjaan ini.

TESTING: sembilan tes otomatis lulus untuk kesetaraan byte (termasuk BOM,
CRLF, kolom multiline, nol di depan angka, dan desimal panjang), pemulihan,
penolakan penimpaan, deteksi arsip rusak, pembatasan path, dry run push,
penolakan remote yang berubah, pelanjutan push yang sudah sebagian terkirim,
penolakan publikasi hasil dari sumber gzip yang terpotong, serta penggunaan
`pack.useSparse=false` saat push agar rename tidak mengirim ulang objek lama.
Jalankan ulang dari akar repo:

```powershell
python src/tools/test_compact_csv_archives.py
python src/tools/compact_csv_archives.py verify --manifest docs/compact_csv_manifest_20261011.json
```

INTEGRATION: skrip lokal tetap dapat memakai gzip asli. Pada clone baru,
pulihkan seluruh path gzip sebelum menjalankan analisis lama:

```powershell
python src/tools/compact_csv_archives.py restore --manifest docs/compact_csv_manifest_20261011.json
```

Kompresi dan pemulihan memerlukan Python 3.14 dengan modul `compression.zstd`.
Pemulihan melewati file yang sudah ada dengan isi identik dan menolak menimpa
file dengan isi berbeda. **Hash isi CSV tetap identik, tetapi hash container
gzip hasil pemulihan dapat berbeda** karena header dan proses kompresi.
Hash gzip asal tetap disimpan untuk penelusuran artefak lokal asli. Verifikasi
reproduksi lintas format menggunakan `csv_sha256`, bukan hash container gzip.

Keterbatasan sumber: ditemukan **1 blob gzip lama rusak pada 1 path**.
Blob tersebut sudah ada dalam commit asal dengan isi yang sama, dan ukurannya
di bawah 100 MiB. File itu tetap disimpan dalam format gzip aslinya di Git
dan di disk; hasil dekompresi parsial tidak dipakai sebagai pengganti:

- `result/benchmarks/ethics/reports/regional_lm_v3_20260916_r3/audit_response_index.csv.gz`: `Compressed file ended before the end-of-stream marker was reached`.

Pekerjaan ini tidak memperbaiki isi arsip yang terpotong dan tidak menjalankan
ulang analisis atau API. Daftar lengkap kegagalan terdapat pada `failures`
dan `invalid_source_paths` dalam manifest kompresi.

Riwayat data semula memiliki 9 commit lokal. Riwayat pengganti memuat
44 bagian data, lalu satu commit kecil untuk alat dan dokumentasi.

| Tanggal WIB | Commit data lama | Bagian data baru |
|---|---:|---:|
| 2026-10-05 | 1 | 10 |
| 2026-10-06 | 3 | 29 |
| 2026-10-07 | 4 | 4 |
| 2026-10-08 | 1 | 1 |

Batas perencanaan adalah 256 MiB byte blob baru yang belum pernah ditambahkan
dalam rangkaian ini. Sebanyak 3 paket Git diuji aktual dengan
`pack-objects --thin --window=0 --compression=1 --threads=2` tanpa mengirimnya.
Paket aktual terbesar yang diuji: **248.18 MiB**.
Untuk seluruh bagian, batas atas dihitung dari blob baru per bagian ditambah
seluruh objek non-blob baru dalam rangkaian, serta overhead kompresi zlib,
representasi penyimpanan objek, dan header seluruh objek. Penghitungan sengaja
memasukkan objek tree/commit dan overhead dari bagian lain agar konservatif.
Batas atas terbesar: **289.41 MiB**.
Kolom `pack_bytes` adalah batas atas, sedangkan `trial_pack_bytes` hanya ada
pada paket yang benar-benar dibentuk. Ukuran transfer aktual dapat berbeda
menurut negosiasi objek dan konfigurasi Git.
Skrip push menetapkan `pack.useSparse=false` agar batas atas berbasis objek baru
berlaku juga pada commit yang memindahkan folder. Mode sparse bawaan Git dapat
mengikutsertakan objek lama pada rename dan menghasilkan paket lebih besar
daripada himpunan objek baru. Perilaku ini dijelaskan dalam
[dokumentasi Git pack-objects](https://git-scm.com/docs/git-pack-objects#Documentation/git-pack-objects.txt---no-sparse).
Rincian hash commit, tanggal, dan ukuran paket tersedia dalam
[`push_batches_20261011.json`](push_batches_20261011.json).

Lihat rencana push (tidak mengirim apa pun):

```powershell
python src/tools/push_pending_in_batches.py
```

Kirim bertahap jika sudah siap:

```powershell
python src/tools/push_pending_in_batches.py --execute
```

Skrip melanjutkan dari commit terakhir yang sudah ada di remote dan tidak
memakai force push. Kirim hanya branch `ethics-accuracy`; backup lokal
`backup/pre-csv-compaction-20261011` mempertahankan sejarah lama beserta
blob gzip besarnya. Perintah push biasa yang sekaligus mengirim seluruh
rangkaian masih dapat melampaui batas ukuran satu push.

Dasar format dan batas layanan:
[Python Zstandard](https://docs.python.org/3.14/library/compression.zstd.html),
[batas file GitHub](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github),
[batas push GitHub](https://docs.github.com/en/get-started/using-git/troubleshooting-the-2-gb-push-limit).
