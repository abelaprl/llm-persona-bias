# GSM8K Regional Cultural rerun manifests

- `pending/` kosong: tidak ada model yang masih wajib di-rerun.
- `closed_audit/qwen_unresolved_after_retries_63.csv` menyimpan 63 target Qwen
  terakhir. Setelah refresh, 6 masih dapat diparse secara aman dan 57 tetap
  uncommitted; semua persona Qwen sudah di atas 99% coverage.
- `history/qwen_initial_1039.csv` menyimpan manifest awal untuk refresh seluruh
  1.039 target setelah recovery selesai.
- `history/qwen_reasoning1024_round_108.csv` menyimpan 108 target round
  sebelumnya; 45 berhasil dan 63 masih gagal.
- `closed_audit/` berisi DeepSeek, Llama 3.1, Llama 3.3, Ministral, dan
  GPT-OSS. Technical rerun kedua Llama sudah selesai; jangan dijalankan lagi.
- Llama 3.3 masih mempunyai genuine model non-answer, tetapi tidak lagi
  memiliki provider/API failure. Non-answer itu dipertahankan sebagai hasil
  model dan tidak diimputasi.
- OLMo tidak termasuk scope.
- Semua manifest memakai raw group `regional_cultural_persona`,
  `temperature=0`, dan `max_tokens=4096`.
- Perintah lengkap: `docs/run_gsm8k_regional_cultural_rerun.md`.
