# ETHICS Gender Linguistic Markers v1.4

Status hasil: **candidate**.

Bundle ini menghitung 21 outcome numerik pada tiga keluarga perbandingan dengan
empat subset ETHICS tetap terpisah. Analisis utama memakai seluruh visible text;
analisis sensitivity memakai pasangan yang kedua responsnya committed.

Pemeriksaan otomatis selesai: **True**.
Audit precision manusia: **0/1689**.
Audit recall manusia: **0/162**.

Karena audit manusia belum selesai, p-value, CI, effect size, tabel, dan gambar
di bundle ini adalah **candidate statistical results**, bukan klaim final.

## Petunjuk audit manusia

1. Isi `human_precision_audit.csv`: `reviewer`, `is_true_positive`,
   `audit_note`, lalu set `human_review_complete=true` per baris.
2. Isi `human_recall_audit.csv`: `reviewer`, `manual_marker_ids`,
   `missed_span_notes`, `audit_note`, lalu set `human_review_complete=true`.
3. Jangan memakai ground truth saat menilai precision/recall marker.
4. Setelah seluruh baris selesai, jalankan ulang validator/manifes; jangan
   mengubah flag audit secara manual tanpa mencocokkan isi lembar.
