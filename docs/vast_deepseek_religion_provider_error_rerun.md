# DeepSeek GSM8K Religion provider-error rerun via Vast.ai

Run ini hanya menargetkan 552 baris yang pada report kanonik memenuhi
`technical_failure=true` dan `error_class=provider_error`. Report lama tidak
ditimpa. Manifest tidak memuat ground truth.

## Perintah lokal

Bangun ulang manifest dari report kanonik:

```powershell
python scripts\build_gsm8k_religion_coverage_manifest.py `
  --model deepseek_deepseek-r1-distill-qwen-32b `
  --results result\GMS8K\reports\Religion_Persona\deepseek_deepseek-r1-distill-qwen-32b\results.csv `
  --output result\rerun_manifests\gsm8k_deepseek\deepseek_religion_provider_errors.csv `
  --only-provider-errors
```

Validasi tanpa API call atau perubahan raw JSON:

```powershell
python scripts\rerun_deepseek_32b_vast.py
```

Setelah vLLM siap dan tunnel lokal `127.0.0.1:8000` mengarah ke port internal
vLLM, jalankan smoke test lalu resume semua target:

```powershell
python -u scripts\rerun_deepseek_32b_vast.py `
  --execute --limit 3 --concurrency 1

python -u scripts\rerun_deepseek_32b_vast.py `
  --execute --concurrency 16
```

Wrapper mengunci model alias
`deepseek/deepseek-r1-distill-qwen-32b`, temperature `0.0`, dan
`max_tokens=4096`. Respons hanya mengganti raw JSON apabila sukses, memiliki
konten final, tidak membawa error top-level maupun `choices[0].error`, dan
`finish_reason=stop`. Setiap sumber dibackup sebelum request; attempt gagal
ditulis terpisah.

Re-parse ke report baru:

```powershell
python -u -m src.MAIN_PARSING.GSM8K.Religion_Persona.run `
  --model deepseek_deepseek-r1-distill-qwen-32b `
  --reparse_results result\GMS8K\reports\Religion_Persona\deepseek_deepseek-r1-distill-qwen-32b\results.csv `
  --refresh_manifest result\rerun_manifests\gsm8k_deepseek\deepseek_religion_provider_errors.csv `
  --out_root result\GMS8K\reports_vast_rerun_20260814
```

## Hasil 14 Agustus 2026

- 552/552 raw JSON valid, berisi konten final, dan selesai dengan `stop`.
- Report baru berisi 21.104 kunci unik, tepat 1.319 per persona.
- `technical_failure=0`, `provider_error=0`, `missing=0`.
- 21.003 committed; commit rate 99,5214%.
- Akurasi seluruh unit 93,0487%; akurasi pada committed 93,4962%.
- Tepat 552 baris berubah dan semuanya berada di manifest; 20.552 baris lain
  identik dengan report lama.
- Seluruh 552 target memiliki backup; tidak ada failed-attempt.
- Instance Vast `47656080` sudah di-destroy dan daftar instance kembali kosong.

Receipt lengkap beserta hash SHA-256 ada di
`result/GMS8K/reports_vast_rerun_20260814/Religion_Persona/deepseek_deepseek-r1-distill-qwen-32b/rerun_validation.json`.
