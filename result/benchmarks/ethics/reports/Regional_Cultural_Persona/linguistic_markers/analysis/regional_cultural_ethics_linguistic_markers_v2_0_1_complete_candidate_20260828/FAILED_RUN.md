# FAILED COMPLETION — Windows path-length limit

Completion berhenti pada langkah 2/5 saat menyalin provenance ke destination
yang namanya terlalu panjang. `shutil.copytree` menerima `WinError 3` untuk
tiga file dengan nama panjang di bawah subfolder provenance.

Tidak ada notebook atau final manifest yang dibuat. Folder ini parsial dan
tidak boleh digunakan. Penyelesaian berikutnya memakai nama destination lebih
pendek, tanpa menghapus folder ini.

Status: `failed_copy_windows_path_length_not_for_analysis`.
