# Penghapusan anotasi rasio soal ETHICS regional dan religion

## SPEC

Hapus rasio jumlah soal/pasangan (`angka/angka`, `k=… / n=…`) dan
keterangan pembilang/penyebut dari seluruh gambar aktif ETHICS regional dan
religion. Persentase, nilai efek, CI, label sumbu, simbol signifikansi, dan
status ND/TP/F tetap dipertahankan. Tidak menjalankan eksperimen baru atau
mengambil sampel dataset; gunakan seluruh hasil tersimpan yang mendasari gambar.

## DESIGN PLAN

Render hanya bagian presentasi coverage dan heatmap dari notebook, menggunakan
tabel statistik tersimpan. Simpan hasil pada direktori run baru di `result/`.
Bandingkan gambar awal dengan rekonstruksinya serta perbedaan piksel di luar
area teks yang diubah. Cadangkan gambar dan notebook sebelum pemasangan.
Perubahan sumber notebook terbatas pada teks rasio agar Run All konsisten.

## IMPLEMENTATION

Filter teks: `src/benchmarks/ethics/identity_figure_count_labels.py`.
Renderer dan pemasang: `src/benchmarks/ethics/remove_identity_figure_count_labels.py`.
Sumber notebook memanggil filter sebelum menyimpan PNG; jalur reuse regional
tidak boleh mengambil PNG coverage/heatmap lama yang belum dibersihkan.

Indeks aktif mencatat 439 PNG regional (395 terdampak) dan 406 PNG religion
(365 terdampak). Sumber angka: `figure_index.csv` pada masing-masing direktori
aktif. Tidak ada PDF aktif dalam kedua direktori gambar tersebut.

Run staging:

- `result/analyses/linguistic_markers/runs/regional_ethics_without_count_labels_20261011_115327_234017/`
- `result/analyses/linguistic_markers/runs/religion_ethics_without_count_labels_20261011_115335_313264/`

Salinan notebook sebelum perubahan ada di `original.ipynb` pada masing-masing
run. Konfigurasi mencatat hash seluruh PNG aktif dan tabel masukan. Audit
per-scope mencatat perubahan teks dan jumlah piksel berubah di luar area teks.

## CHECKLIST

- [x] Identifikasi bentuk anotasi dari sumber notebook dan gambar aktif.
- [x] Batasi perubahan pada presentasi, tanpa parsing atau statistik ulang.
- [x] Render dan validasi semua 760 gambar terdampak.
- [x] Perbarui sumber notebook dan pasang gambar dengan backup.

## TESTING

Periksa persamaan hash tabel, ketiadaan teks rasio pada hasil render, dimensi
PNG, dan perubahan piksel hanya dalam batas teks. Kompilasi sel notebook
yang diubah dan tinjau contoh gambar dari kedua dimensi.

Lima unit test filter telah lolos:
`python -X utf8 -m unittest src.benchmarks.ethics.test_identity_figure_count_labels -v`.
Seluruh sel kode notebook berhasil dikompilasi. Perubahan sumber hanya sel 7
regional dan sel 9 religion (indeks mulai nol). Pemeriksaan awal rekonstruksi
PNG cakupan dan heatmap kedua dimensi cocok persis dengan gambar aktif.
Render berikutnya membandingkan PNG akhir langsung dengan gambar aktif:
setiap piksel di luar batas teks yang diedit wajib identik. Multiset angka
desimal dan jumlah simbol signifikansi dalam teks juga wajib identik.

## INTEGRATION

Target aktif: `notebooks/{regional,religion}/linguistic marker/ETHICS/`.
Folder arsip dan seluruh hasil historis `result/` tidak ditimpa.

Pemasangan selesai: 395 PNG regional diganti dan 44 PNG tetap identik;
365 PNG religion diganti dan 41 PNG tetap identik. Seluruh gambar yang diganti
memiliki backup dengan hash yang cocok dengan versi awal. Manifest tiap run
ada di `manifest.json`; rincian masukan ada di `config.json`. Tidak ada keluaran
PNG tertanam pada kedua notebook yang perlu diperbarui.

Pemeriksaan render seluruh 760 PNG menunjukkan nol perubahan piksel di luar
area teks yang diedit dan nol anotasi rasio tersisa pada objek teks gambar.
Hash tabel masukan tetap identik. Notebook hanya berubah pada sel presentasi;
selain sel 7 regional dan sel 9 religion, isi sel lain tetap sama.

Perintah untuk run baru (tanpa menjalankan parsing/statistik):

```powershell
$plotDimension = 'regional' # Change to 'religion' for the other dimension.
$plotRun = python -X utf8 src/benchmarks/ethics/remove_identity_figure_count_labels.py --prepare $plotDimension
if ($LASTEXITCODE -ne 0) { throw 'Preparation failed' }
foreach ($plotScope in @('commonsense', 'deontology', 'justice', 'virtue', 'gabungan')) {
    python -X utf8 src/benchmarks/ethics/remove_identity_figure_count_labels.py --run $plotRun --scope $plotScope
    if ($LASTEXITCODE -ne 0) { throw "Rendering failed: $plotScope" }
}
python -X utf8 src/benchmarks/ethics/remove_identity_figure_count_labels.py --install $plotRun
```

Pemasangan memerlukan izin tulis folder notebook. Backup berada di
`backup_active_before_install/`, sedangkan `original.ipynb` menyimpan notebook
sebelum penambahan filter. Filter notebook bekerja otomatis saat gambar dirender
ulang; tidak ada perubahan pada data, persona, aturan ekstraksi, atau runner API.
