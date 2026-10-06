# ETHICS Gender Linguistic Markers v1.4

Status hasil: **candidate**.

Bundle ini menghitung 21 outcome numerik pada tiga keluarga perbandingan dengan
empat subset ETHICS tetap terpisah. Analisis utama memakai seluruh visible text;
analisis sensitivity memakai pasangan yang kedua responsnya committed.

Pemeriksaan otomatis selesai: **True**.
Audit precision manusia: **0/3378 judgment**.
Audit recall manusia: **0/324 judgment**.

Karena audit manusia belum selesai, p-value, CI, effect size, tabel, dan gambar
di bundle ini adalah **candidate statistical results**, bukan klaim final.

## Petunjuk audit manusia

1. Reviewer 1 dan reviewer 2 mengisi file masing-masing secara independen.
2. Precision: isi `reviewer_id`, `is_true_positive`, dan `audit_note`; baca
   `visible_response_full`, bukan hanya potongan `match_context`.
3. Recall: isi `reviewer_id`, `manual_marker_ids`, `missed_span_notes`, dan
   `audit_note` setelah membaca seluruh `visible_text`.
4. Set `human_review_complete=true` hanya setelah satu item selesai dinilai.
5. Jangan memakai ground truth atau melihat lembar reviewer lain selama audit.
6. Status final memerlukan inter-rater agreement dan adjudikasi disagreement;
   jangan mengubah flag manifest secara manual.
