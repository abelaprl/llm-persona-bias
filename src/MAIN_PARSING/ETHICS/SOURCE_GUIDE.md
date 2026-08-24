# Panduan kode ETHICS dan analisis terkait

## Mulai dari kebutuhan

| Kebutuhan | Entry point atau panduan |
|---|---|
| Parsing jawaban ETHICS | [README parsing](README.md): `run_parse.py`, `build_csv.py`, dan `checks.py` |
| Statistik dan notebook marker Gender | [run_gender_ethics_lm_workspace.py](run_gender_ethics_lm_workspace.py); [petunjuk reproduksi](../../../notebooks/gender/linguistic%20marker/README_ETHICS.md) |
| Builder marker Religion | [build_religion_ethics_linguistic_markers_v2.py](build_religion_ethics_linguistic_markers_v2.py); [catatan eksekusi](../../../docs/linguistic_marker_religion_ethics_v2_0_2_execution_2026-08-27.md) |
| Builder marker Regional/Cultural | [build_regional_cultural_ethics_linguistic_markers_v2.py](build_regional_cultural_ethics_linguistic_markers_v2.py); [catatan eksekusi](../../../docs/regional_cultural_ethics_linguistic_markers_v2_execution.md) |
| Presentasi marker lintas kelompok persona | [present_ethics_lm_overall.py](present_ethics_lm_overall.py); [catatan presentasi](../../../docs/ethics_lm_overall_presentation_20260913.md) |
| Notebook akurasi ETHICS yang dipakai | [ETHICS_MULAI_DI_SINI.md](../../../notebooks/ETHICS_MULAI_DI_SINI.md) |
| Builder dan perender akurasi ETHICS/GSM8K | [src/analysis](../../analysis); [panduan visualisasi ETHICS](../../../docs/ethics_accuracy_visualization_handoff_2026-09-07.md) |
| Library ekstraksi marker dan statistik | [Linguistic Marker Builder](../Linguistic%20Marker%20Builder/README.md) |

Daftar ini menunjukkan lokasi kode, bukan pengesahan metodologi atau hasil.
Ikuti batas interpretasi, status candidate, dan aturan keluaran baru pada
dokumentasi masing-masing pipeline.

## Kode yang tetap diperlukan

- `build_*`, `create_*`, dan `render_*` dapat diperlukan untuk mereproduksi
  tabel, gambar, dan notebook, walaupun sebelumnya hanya dijalankan sekali.
- `audit_*`, `validate_*`, `finalize_*`, `tests/`, dan `templates/` menyimpan
  pemeriksaan, bukti, atau komponen reproduksi.
- Versi lama dapat menjadi dependensi versi baru. Contohnya,
  `build_gender_ethics_linguistic_markers_v1_5.py` masih merujuk builder v1.4.
- Folder `Linguistic Marker Builder` dipakai melalui loader kompatibilitas
  pada `run_gender_ethics_lm_workspace.py`. Mengganti namanya memerlukan
  pemeriksaan impor dan seluruh pemanggilnya.
- Status Git `??` hanya menyatakan path belum dilacak. Tidak ditemukannya
  referensi nama juga belum membuktikan bahwa sebuah skrip tidak diperlukan:
  entry point dapat dijalankan langsung dari terminal.

## Cleanup 15 September 2026

Empat skrip pemindahan arsip bertanggal tetap telah dihapus dari folder ini
setelah seluruh 130 hash file tujuan pada tiga catatan arsip diverifikasi.
Kode analisis dan path impor dipertahankan.

[Laporan cleanup](../../../docs/cleanup_src_20260915_223200/README.md) memuat
daftar penghapusan, batas pemeriksaan, inventaris seluruh 139 file untracked
pada awal cleanup, dan ZIP pemulihan untuk empat skrip tersebut.
