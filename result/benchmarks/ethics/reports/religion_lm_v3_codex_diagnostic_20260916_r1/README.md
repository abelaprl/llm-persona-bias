# Audit diagnostik AI Religion ETHICS v3

Baca [laporan audit](../../../../docs/religion_lm_v3_codex_audit_20260916.md).

60 respons acak development dibaca utuh; 1.204 hit diperiksa, 21 ditandai false positive oleh Codex, dan 31 kemunculan terlewat didokumentasikan. Dua respons terarah menambah dua contoh false positive. Hasil bukan gold manusia dan tidak mempromosikan aturan.

- `finding_spans.csv`: bukti temuan dengan offset, hash, response key, marker/rule dan alasan.
- `reviewed_final_sources.json`: 62 final sumber utuh, tanpa GT/thinking.
- `reviews.json`: catatan reviewer untuk seluruh 60 respons acak.
- `occurrence_reviews.csv`: 1.204 keputusan hit AI; no_issue_identified bukan gold true positive.
- `rule_review_summary.csv`: seluruh 129 aturan, termasuk aturan yang belum teramati.
- `summary.json`, `manifest.json`, `verification.json`: hitungan, desain sampel dan integritas artefak.
- `authored_findings.json`: keputusan reviewer sebelum pengikatan otomatis ke span raw.
- `sample_full.json`, `targeted_full.json`: sampel dan provenance deteksi.
