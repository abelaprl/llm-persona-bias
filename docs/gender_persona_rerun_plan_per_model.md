# Rencana Rerun ETHICS v9 Gender Persona per Model

Tanggal audit: 2026-07-16  
Sumber: `result/ETHICS/v9_pilot/Gender Persona` dan `result/reports/v9_pilot/gender`

## Ringkasan keputusan

Ada 4.174 baris berstatus `missing`, tetapi terdiri dari dua kelompok:

| Kelompok | Jumlah | Tindakan |
|---|---:|---|
| Tidak memiliki content maupun reasoning | 3.424 | Rerun wajib |
| Memiliki reasoning, tetapi final content kosong | 750 | Rerun untuk memperoleh final answer |
| **Total tanpa final answer valid** | **4.174** | **Rerun jika report mensyaratkan final answer** |

Yang disebut “kosong” di dokumen ini adalah output assistant kosong. File JSON tetap ada dan biasanya menyimpan metadata atau error API.

## Selective rerun yang digunakan

Runner sekarang memiliki `--rerun_manifest`. Gunakan bersama `--json_only` agar:

- hanya pasangan `persona_folder + question_id` dalam manifest yang masuk loop;
- hanya persona yang muncul di manifest yang mendapat warmup;
- JSON target boleh diganti walaupun sudah ada;
- ID lain sama sekali tidak diproses;
- tidak ada CSV/report yang ditulis.

Tanpa `--rerun_manifest`, runner tetap memiliki perilaku lama:

- `--skip_existing`: semua JSON yang sudah ada dilewati, termasuk JSON gagal;
- tanpa `--skip_existing`: semua sampel pada persona terpilih dijalankan ulang dan file valid ikut ditimpa.

Jadi runner perlu mendapat mode selektif terlebih dahulu, misalnya:

```text
--rerun_manifest <csv-or-json>
```

atau:

```text
--rerun_failed_only
--rerun_reason no_output,reasoning_without_final
```

Manifest memilih pasangan tepat:

```text
model_slug + persona_folder + question_id/file
```

dan hanya menimpa JSON yang tercantum dalam manifest. Tidak perlu memindahkan JSON target agar `--skip_existing` bekerja; manifest adalah override eksplisit untuk target tersebut.

## Model order

| Model | Model order | Model slug |
|---|---:|---|
| GPT OSS 20B | 31 | `openai_gpt-oss-20b` |
| Qwen3 Next 80B Thinking | 32 | `qwen_qwen3-next-80b-a3b-thinking` |
| Llama 3.3 70B Instruct | 35 | `meta-llama_llama-3.3-70b-instruct` |
| Llama 3.1 8B Instruct | 40 | `meta-llama_llama-3.1-8b-instruct` |
| Ministral 8B | 41 | `mistralai_ministral-8b-2512` |
| DeepSeek R1 Distill Qwen 32B | 43 | `deepseek_deepseek-r1-distill-qwen-32b` |

## Rerun wajib: 3.424 no-output

### Matriks model × persona

| Persona | DeepSeek | Llama 3.1 | Llama 3.3 | Ministral | GPT OSS | Qwen3 | Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| `man_explicit_1` | 2 | 5 | 9 | 10 | 6 | 9 | 41 |
| `man_implicit_1` | 1 | 199 | 192 | 231 | 6 | 241 | 870 |
| `man_implicit_2` | 4 | 9 | 205 | 147 | 5 | 10 | 380 |
| `man_implicit_3` | 0 | 184 | 224 | 242 | 5 | 173 | 828 |
| `woman_explicit_1` | 2 | 6 | 11 | 8 | 6 | 14 | 47 |
| `woman_implicit_1` | 0 | 157 | 147 | 154 | 5 | 145 | 608 |
| `woman_implicit_2` | 1 | 8 | 14 | 12 | 5 | 166 | 206 |
| `woman_implicit_3` | 3 | 5 | 231 | 196 | 5 | 4 | 444 |
| **Total** | **13** | **573** | **1.033** | **1.000** | **43** | **762** | **3.424** |

### Penyebab no-output

| Penyebab | Jumlah |
|---|---:|
| HTTP 403 `Key limit exceeded` | 3.090 |
| SSL unexpected EOF | 232 |
| DNS gagal resolve OpenRouter | 81 |
| Embedded provider error 502/504 tetapi `success=true` | 12 |
| Successful response dengan content dan reasoning `null` | 6 |
| Endpoint 404 | 1 |
| Connection aborted/disconnected | 2 |
| **Total** | **3.424** |

## Rerun reasoning-only: 750

Kelompok ini tidak benar-benar kosong. Reasoning tersedia, tetapi `message.content=null`, sehingga tidak ada final answer resmi.

| Model | `length` | `content_filter` | `error` | Total |
|---|---:|---:|---:|---:|
| DeepSeek R1 Distill | 97 | 0 | 9 | 106 |
| GPT OSS 20B | 450 | 15 | 20 | 485 |
| Qwen3 Thinking | 140 | 18 | 1 | 159 |
| **Total** | **687** | **33** | **30** | **750** |

Sinyal di tail reasoning:

| Sinyal | Jumlah |
|---|---:|
| `Answer: 0/1` | 91 |
| Leading digit | 1 |
| Semantic conclusion | 4 |
| Tidak ada verdict eksplisit di tail | 654 |

Sebanyak 96 dapat disimpan sebagai `candidate_answer_from_reasoning`, tetapi jangan dianggap final answer tanpa keputusan metodologis. Untuk konsistensi report, seluruh 750 sebaiknya di-rerun.

## Rencana per model

### 1. DeepSeek R1 Distill Qwen 32B — order 43

- No-output wajib rerun: **13**
- Reasoning-only: **106**
- Total tanpa final answer: **119**

No-output per persona:

| Persona | Jumlah |
|---|---:|
| `man_explicit_1` | 2 |
| `man_implicit_1` | 1 |
| `man_implicit_2` | 4 |
| `man_implicit_3` | 0 |
| `woman_explicit_1` | 2 |
| `woman_implicit_1` | 0 |
| `woman_implicit_2` | 1 |
| `woman_implicit_3` | 3 |

Catatan rerun: 97 reasoning-only berhenti karena `length`. Jangan sekadar mengulang konfigurasi yang sama. Pertimbangkan final-answer-first atau menaikkan token budget secara terkontrol.

### 2. Llama 3.1 8B Instruct — order 40

- No-output wajib rerun: **573**
- Reasoning-only: **0**
- Total tanpa final answer: **573**

| Persona | Jumlah |
|---|---:|
| `man_explicit_1` | 5 |
| `man_implicit_1` | 199 |
| `man_implicit_2` | 9 |
| `man_implicit_3` | 184 |
| `woman_explicit_1` | 6 |
| `woman_implicit_1` | 157 |
| `woman_implicit_2` | 8 |
| `woman_implicit_3` | 5 |

Mayoritas adalah kegagalan API, bukan refusal model.

### 3. Llama 3.3 70B Instruct — order 35

- No-output wajib rerun: **1.033**
- Reasoning-only: **0**
- Total tanpa final answer: **1.033**

| Persona | Jumlah |
|---|---:|
| `man_explicit_1` | 9 |
| `man_implicit_1` | 192 |
| `man_implicit_2` | 205 |
| `man_implicit_3` | 224 |
| `woman_explicit_1` | 11 |
| `woman_implicit_1` | 147 |
| `woman_implicit_2` | 14 |
| `woman_implicit_3` | 231 |

Termasuk 6 completion dengan `finish_reason=stop` tetapi content dan reasoning `null`.

### 4. Ministral 8B — order 41

- No-output wajib rerun: **1.000**
- Reasoning-only: **0**
- Total tanpa final answer: **1.000**

| Persona | Jumlah |
|---|---:|
| `man_explicit_1` | 10 |
| `man_implicit_1` | 231 |
| `man_implicit_2` | 147 |
| `man_implicit_3` | 242 |
| `woman_explicit_1` | 8 |
| `woman_implicit_1` | 154 |
| `woman_implicit_2` | 12 |
| `woman_implicit_3` | 196 |

Seluruh missing model ini tidak memiliki content/reasoning dan didominasi API failure.

### 5. GPT OSS 20B — order 31

- No-output wajib rerun: **43**
- Reasoning-only: **485**
- Total tanpa final answer: **528**

| Persona | No-output |
|---|---:|
| `man_explicit_1` | 6 |
| `man_implicit_1` | 6 |
| `man_implicit_2` | 5 |
| `man_implicit_3` | 5 |
| `woman_explicit_1` | 6 |
| `woman_implicit_1` | 5 |
| `woman_implicit_2` | 5 |
| `woman_implicit_3` | 5 |

Reasoning-only: 450 `length`, 15 `content_filter`, dan 20 `error`. Model ini memerlukan strategi final-answer-first atau token budget yang berbeda.

### 6. Qwen3 Next 80B Thinking — order 32

- No-output wajib rerun: **762**
- Reasoning-only: **159**
- Total tanpa final answer: **921**

| Persona | No-output |
|---|---:|
| `man_explicit_1` | 9 |
| `man_implicit_1` | 241 |
| `man_implicit_2` | 10 |
| `man_implicit_3` | 173 |
| `woman_explicit_1` | 14 |
| `woman_implicit_1` | 145 |
| `woman_implicit_2` | 166 |
| `woman_implicit_3` | 4 |

Reasoning-only: 140 `length`, 18 `content_filter`, dan 1 `error`.

## Urutan rerun yang disarankan

1. Perbaiki API key/limit terlebih dahulu; jangan mengulang 3.090 kasus dengan key yang masih mencapai limit.
2. Tambahkan selective rerun/manifest ke runner.
3. Buat backup atau output staging; jangan langsung menimpa JSON lama.
4. Rerun 3.424 no-output terlebih dahulu.
5. Validasi bahwa setiap hasil staging memiliki final content non-empty.
6. Rerun 750 reasoning-only dengan konfigurasi yang mencegah reasoning menghabiskan token.
7. Untuk 33 `content_filter`, simpan status khusus jika percobaan tetap difilter.
8. Promosikan hasil staging ke lokasi final hanya jika `success=true` dan final content non-empty.
9. Rebuild CSV/report per model setelah semua batch tervalidasi.
10. Bandingkan row count, duplicate ID, missing rate, dan coverage persona sebelum/sesudah.

## Bentuk manifest yang disarankan

```csv
model_order,model_slug,persona_folder,file,question_id,rerun_reason,old_finish_reason
40,meta-llama_llama-3.1-8b-instruct,man_implicit_1,ethics_commonsense_0144.json,ethics_commonsense_0144,api_failure,
43,deepseek_deepseek-r1-distill-qwen-32b,man_explicit_1,ethics_deontology_0124.json,ethics_deontology_0124,reasoning_without_final,length
```

Manifest aktual sebaiknya dibuat langsung dari JSON sumber agar alasan rerun tidak hanya bergantung pada CSV.

## Command runner setelah mode manifest diimplementasikan

Contoh interface yang direkomendasikan (belum tersedia pada runner saat dokumen ini dibuat):

```powershell
python src/ethics_runner/run_ethics_prompt_v9.py `
  --model_order 40 `
  --group gender `
  --rerun_manifest result/rerun_manifests/gender_v9_llama31_no_output.csv `
  --rerun_output_dir result/ETHICS/v9_pilot_rerun_staging
```

Jangan menyalin command ini sebelum `--rerun_manifest` dan `--rerun_output_dir` benar-benar ditambahkan ke runner.

## Checklist validasi

- [ ] Jumlah manifest no-output = 3.424.
- [ ] Jumlah manifest reasoning-only = 750.
- [ ] Tidak ada file valid yang masuk manifest.
- [ ] Tidak ada duplikasi `model + persona + question_id`.
- [ ] API key memiliki quota yang cukup.
- [ ] Hasil rerun ditulis ke staging.
- [ ] Final content tidak kosong.
- [ ] Reasoning-only tidak kembali berhenti karena `length`.
- [ ] Provider error di `raw_api_response.error` membuat `success=false`.
- [ ] Report lama dibackup dan tidak ditimpa sebelum verifikasi.
- [ ] CSV/report dibangun ulang per model.

## Panduan operasional lengkap dengan `--skip_existing`

Bagian ini adalah workflow yang dapat dipakai dengan runner saat ini. Prinsipnya:

1. JSON gagal dibuatkan manifest.
2. JSON lama dalam manifest disalin ke backup (tidak dipindahkan).
3. Runner dipanggil dengan `--rerun_manifest --skip_existing --json_only`.
4. Runner hanya mengiterasi target manifest; JSON lain tidak masuk loop.
5. JSON target diganti dengan hasil rerun, sementara salinan lama tetap ada di backup.

### A. Tentukan cakupan rerun

Pilihan yang direkomendasikan:

```text
all_missing = 4.174 = no-output 3.424 + reasoning-only 750
```

Jika hanya ingin kegagalan API/no-output:

```text
no_output = 3.424
```

### B. Buat manifest dari JSON sumber

Jalankan dari `D:\TugasAkhir`. Ubah `$mode` menjadi `no_output` jika 750 reasoning-only belum ingin diulang.

```powershell
$mode = 'all_missing' # all_missing | no_output
$reportRoot = 'D:\TugasAkhir\result\reports\v9_pilot\gender'
$dataRoot = 'D:\TugasAkhir\result\ETHICS\v9_pilot\Gender Persona'
$manifestRoot = 'D:\TugasAkhir\result\rerun_manifests\gender_v9'
$backupRoot = 'D:\TugasAkhir\result\rerun_backup\gender_v9'

New-Item -ItemType Directory -Force -Path $manifestRoot | Out-Null
$items = @()

foreach ($csv in [IO.Directory]::EnumerateFiles($reportRoot, 'results_*.csv', [IO.SearchOption]::AllDirectories)) {
    foreach ($row in (Import-Csv -LiteralPath $csv | Where-Object answer_status -eq 'missing')) {
        $source = Join-Path (Join-Path (Join-Path $dataRoot $row.model) $row.persona_folder) $row.file
        if (-not [IO.File]::Exists($source)) {
            throw "Source JSON tidak ditemukan: $source"
        }

        $d = [IO.File]::ReadAllText($source) | ConvertFrom-Json
        $content = [string]$d.prediction.raw
        $reasoning = ''
        $finishReason = ''

        if ($d.response.raw_api_response.choices) {
            $choice = $d.response.raw_api_response.choices[0]
            if ([string]::IsNullOrWhiteSpace($content) -and $choice.message) {
                $content = [string]$choice.message.content
            }
            if ($choice.message) {
                $reasoning = [string]$choice.message.reasoning
            }
            $finishReason = [string]$choice.finish_reason
        }

        if ([string]::IsNullOrWhiteSpace($content) -and [string]::IsNullOrWhiteSpace($reasoning)) {
            $reason = 'no_output'
        }
        elseif ([string]::IsNullOrWhiteSpace($content)) {
            $reason = 'reasoning_without_final'
        }
        else {
            continue
        }

        if ($mode -eq 'no_output' -and $reason -ne 'no_output') {
            continue
        }

        $sourceFull = [IO.Path]::GetFullPath($source)
        $dataRootFull = [IO.Path]::GetFullPath($dataRoot).TrimEnd('\')
        if (-not $sourceFull.StartsWith($dataRootFull + '\', [StringComparison]::OrdinalIgnoreCase)) {
            throw "Source keluar data root: $sourceFull"
        }
        $relative = $sourceFull.Substring($dataRootFull.Length).TrimStart('\')
        $backup = Join-Path $backupRoot $relative
        $items += [pscustomobject]@{
            model_order = switch ($row.model) {
                'openai_gpt-oss-20b' { 31 }
                'qwen_qwen3-next-80b-a3b-thinking' { 32 }
                'meta-llama_llama-3.3-70b-instruct' { 35 }
                'meta-llama_llama-3.1-8b-instruct' { 40 }
                'mistralai_ministral-8b-2512' { 41 }
                'deepseek_deepseek-r1-distill-qwen-32b' { 43 }
                default { throw "Model tidak dikenal: $($row.model)" }
            }
            model_slug = $row.model
            persona_folder = $row.persona_folder
            file = $row.file
            question_id = $row.question_id
            rerun_reason = $reason
            old_finish_reason = $finishReason
            source_path = $source
            backup_path = $backup
        }
    }
}

$items | Group-Object model_slug | ForEach-Object {
    $path = Join-Path $manifestRoot ("{0}_{1}.csv" -f $_.Name, $mode)
    $_.Group | Export-Csv -LiteralPath $path -NoTypeInformation -Encoding utf8
    Write-Host "$($_.Name): $($_.Count) -> $path"
}

Write-Host "TOTAL MANIFEST: $($items.Count)"
```

Expected total:

| Mode | Expected |
|---|---:|
| `no_output` | 3.424 |
| `all_missing` | 4.174 |

Expected `all_missing` per model:

| Model slug | Manifest rows |
|---|---:|
| `deepseek_deepseek-r1-distill-qwen-32b` | 119 |
| `meta-llama_llama-3.1-8b-instruct` | 573 |
| `meta-llama_llama-3.3-70b-instruct` | 1.033 |
| `mistralai_ministral-8b-2512` | 1.000 |
| `openai_gpt-oss-20b` | 528 |
| `qwen_qwen3-next-80b-a3b-thinking` | 921 |

Hentikan proses jika jumlah manifest berbeda.

### C. Backup satu model sebelum rerun

Jalankan per model, bukan keenam model sekaligus. Contoh Llama 3.1:

```powershell
$manifest = 'D:\TugasAkhir\result\rerun_manifests\gender_v9\meta-llama_llama-3.1-8b-instruct_all_missing.csv'
$dataRoot = [IO.Path]::GetFullPath('D:\TugasAkhir\result\ETHICS\v9_pilot\Gender Persona')
$backupRoot = [IO.Path]::GetFullPath('D:\TugasAkhir\result\rerun_backup\gender_v9')
$rows = Import-Csv -LiteralPath $manifest

foreach ($row in $rows) {
    $source = [IO.Path]::GetFullPath($row.source_path)
    $backup = [IO.Path]::GetFullPath($row.backup_path)
    if (-not $source.StartsWith($dataRoot, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Source keluar workspace data: $source"
    }
    if (-not $backup.StartsWith($backupRoot, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Backup keluar backup root: $backup"
    }
    if (-not [IO.File]::Exists($source)) {
        throw "Source sudah tidak ada: $source"
    }
    New-Item -ItemType Directory -Force -Path ([IO.Path]::GetDirectoryName($backup)) | Out-Null
    Copy-Item -LiteralPath $source -Destination $backup
}

Write-Host "BACKED UP: $($rows.Count)"
```

Ganti nama manifest untuk model lain. Jangan hapus folder backup. Source JSON sengaja tetap di tempatnya; `--rerun_manifest` akan mengganti hanya source yang tercantum.

### D. Jalankan rerun per model

Semua command dijalankan dari `D:\TugasAkhir`.

Sebelum command asli, tambahkan `--dry_run_targets` untuk memastikan jumlah target tanpa API call, contoh:

```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 40 --rerun_manifest result/rerun_manifests/gender_v9/meta-llama_llama-3.1-8b-instruct_all_missing.csv --skip_existing --json_only --dry_run_targets
```

Output harus menunjukkan `Main calls: 573`, bukan seluruh dataset. Setelah benar, jalankan command tanpa `--dry_run_targets`.

#### DeepSeek R1 Distill — 119 calls utama + warmup implicit

```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --rerun_manifest result/rerun_manifests/gender_v9/deepseek_deepseek-r1-distill-qwen-32b_all_missing.csv --skip_existing --json_only --max_tokens 8192 --max_retries 8 --retry_base_sec 8 --sleep_sec 1
```

#### Llama 3.1 8B — 573 calls utama + warmup implicit

```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 40 --rerun_manifest result/rerun_manifests/gender_v9/meta-llama_llama-3.1-8b-instruct_all_missing.csv --skip_existing --json_only --max_tokens 4096 --max_retries 8 --retry_base_sec 8 --sleep_sec 1
```

#### Llama 3.3 70B — 1.033 calls utama + warmup implicit

```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 35 --rerun_manifest result/rerun_manifests/gender_v9/meta-llama_llama-3.3-70b-instruct_all_missing.csv --skip_existing --json_only --max_tokens 4096 --max_retries 8 --retry_base_sec 8 --sleep_sec 1
```

#### Ministral 8B — 1.000 calls utama + warmup implicit

```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 41 --rerun_manifest result/rerun_manifests/gender_v9/mistralai_ministral-8b-2512_all_missing.csv --skip_existing --json_only --max_tokens 4096 --max_retries 8 --retry_base_sec 8 --sleep_sec 1
```

#### GPT OSS 20B — 528 calls utama + warmup implicit

```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 31 --rerun_manifest result/rerun_manifests/gender_v9/openai_gpt-oss-20b_all_missing.csv --skip_existing --json_only --max_tokens 8192 --max_retries 8 --retry_base_sec 8 --sleep_sec 1
```

#### Qwen3 Next 80B Thinking — 921 calls utama + warmup implicit

```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 32 --rerun_manifest result/rerun_manifests/gender_v9/qwen_qwen3-next-80b-a3b-thinking_all_missing.csv --skip_existing --json_only --max_tokens 8192 --max_retries 8 --retry_base_sec 8 --sleep_sec 1
```

Catatan warmup: hanya persona implicit yang benar-benar muncul di manifest model tersebut yang mendapat warmup. Jumlah API call aktual adalah manifest rows ditambah satu warmup per persona implicit target.

### E. Validasi satu model setelah rerun

Contoh Llama 3.1:

```powershell
$manifest = 'D:\TugasAkhir\result\rerun_manifests\gender_v9\meta-llama_llama-3.1-8b-instruct_all_missing.csv'
$rows = Import-Csv -LiteralPath $manifest
$ok = 0
$stillBad = @()

foreach ($row in $rows) {
    if (-not [IO.File]::Exists($row.source_path)) {
        $stillBad += [pscustomobject]@{ file=$row.file; problem='file_not_recreated' }
        continue
    }
    $d = [IO.File]::ReadAllText($row.source_path) | ConvertFrom-Json
    $content = [string]$d.prediction.raw
    if ([string]::IsNullOrWhiteSpace($content) -and $d.response.raw_api_response.choices) {
        $content = [string]$d.response.raw_api_response.choices[0].message.content
    }
    if ([string]::IsNullOrWhiteSpace($content)) {
        $stillBad += [pscustomobject]@{
            file = $row.file
            problem = 'final_content_still_empty'
            success = $d.response.success
            error = $d.response.error
            finish_reason = $d.response.raw_api_response.choices[0].finish_reason
        }
    }
    else {
        $ok++
    }
}

Write-Host "VALID FINAL CONTENT: $ok / $($rows.Count)"
$stillBad | Format-Table -AutoSize
```

Jangan lanjut ke model berikutnya sebelum hasil batch saat ini diperiksa.

### F. Jika rerun gagal dan perlu rollback

Backup dibuat dengan `Copy-Item`, jadi JSON lama tetap tersedia. Untuk mengembalikan versi lama secara eksplisit:

```powershell
$manifest = 'D:\TugasAkhir\result\rerun_manifests\gender_v9\meta-llama_llama-3.1-8b-instruct_all_missing.csv'
$rows = Import-Csv -LiteralPath $manifest

foreach ($row in $rows) {
    if ([IO.File]::Exists($row.backup_path)) {
        New-Item -ItemType Directory -Force -Path ([IO.Path]::GetDirectoryName($row.source_path)) | Out-Null
        Copy-Item -LiteralPath $row.backup_path -Destination $row.source_path -Force
    }
}
```

Command rollback di atas menimpa hasil rerun. Jalankan hanya setelah membandingkan hasil rerun dengan backup.

### G. Sesudah semua model selesai

1. Pastikan semua manifest sudah tervalidasi.
2. Pertahankan backup sampai analisis akhir selesai.
3. Bangun ulang report menggunakan ekstraktor baru di `src/ethics_runner` setelah implementasinya selesai.
4. Jangan menggunakan `--rebuild_csv_only` sebagai pengganti report gender final karena outputnya berbeda dari `result/reports/v9_pilot/gender`.
5. Audit ulang jumlah `missing`, `abstain`, committed rate, dan distribusi persona.

`--json_only` menonaktifkan dua penulisan CSV yang biasanya dilakukan runner: rebuild CSV per-model di akhir `run_model`, dan `combined_v9_pilot.csv` di akhir program.
