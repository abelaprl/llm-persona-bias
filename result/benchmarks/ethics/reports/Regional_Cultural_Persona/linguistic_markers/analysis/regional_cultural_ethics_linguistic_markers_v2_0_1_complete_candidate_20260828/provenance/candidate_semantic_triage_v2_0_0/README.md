# AI semantic triage kandidat Regional/Cultural v2.0.0

Artefak ini menambahkan catatan audit tanpa mengubah registry v1.0.16.

- 100/100 respons kandidat yang tidak overlap dengan marker regional dibaca;
- 51 respons utuh untuk kandidat `culture_custom_explicit` dibaca secara
  terstratifikasi per model (10 per model, kecuali seluruh 1 hit GPT-OSS);
- hasil AI untuk `custom` adalah 51 TP, 0 FP, tetapi **belum dipromosikan**;
- dua reviewer manusia independen tetap wajib.

Label TP hanya berarti frasa custom/customs benar-benar ada dan digunakan
sebagai rujukan kebiasaan. Audit ini tidak menilai kebenaran faktual klaim
budaya model dan tidak melihat ground truth ETHICS.
