# Run tidak sah

Seluruh 154.428 respons telah dipindai dan kontrak coverage lolos, tetapi run
berhenti ketika `pattern_coverage_summary.json` ditulis karena dua nilai boolean
NumPy belum dikonversi menjadi boolean Python. File tabel yang sempat tertulis
tidak dipakai sebagai bundle final.

Perbaikan tipe dijalankan pada direktori write-once v2.0.2 tanpa menimpa file
run parsial ini.
