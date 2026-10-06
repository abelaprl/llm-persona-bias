# INCOMPLETE RUN — phase 9 passed, notebook write failed

Run ini lulus fase 9/10 (`Automated validation`) tetapi berhenti pada fase
10/10 sebelum source notebook dapat ditulis ke `notebooks/regional`.

Error:

`PermissionError: [Errno 13] Permission denied` saat membuka
`linguistic_marker_regional_cultural_ethics_thesis_v2_0_1_candidate_20260828.ipynb`.

Dataset, tabel, figure, provenance, dan `automated_validation.json` sudah
terbentuk, tetapi folder ini **tidak boleh dipakai sebagai bundle lengkap**
karena tidak memiliki executed notebook dan manifest final.

Penyelesaian dilakukan secara additive ke folder baru dengan prosedur:

1. salin seluruh artifact phase-9 kecuali manifest/penanda kegagalan;
2. verifikasi source-tree digest sama sebelum dan sesudah salin;
3. buat dan eksekusi notebook pada destination baru;
4. hitung hash seluruh file destination;
5. jalankan validator read-back eksternal.

Status: `incomplete_phase_10_not_for_final_analysis`.
