# Audit GSM8K DeepSeek: Religion Persona dan Gender Persona

Tanggal audit: **10 Agustus 2026**
Model: `deepseek/deepseek-r1-distill-qwen-32b`
Folder model: `deepseek_deepseek-r1-distill-qwen-32b`

## Kesimpulan

Dataset lengkap: seluruh **25 persona** memiliki tepat **1.319 JSON utama**, tanpa
nomor hilang, nomor ekstra, ID tidak cocok, atau persona tidak cocok.

Di antara **32.975 JSON utama**, ditemukan:

| Temuan | Religion | Gender | Total | Keputusan |
|---|---:|---:|---:|---|
| HTTP 4XX/5XX asli | 56 | 147 | **203** | Rerun |
| Kegagalan infrastruktur non-HTTP | 479 | 0 | **479** | Rerun |
| JSON terpotong dan tidak dapat di-parse | 0 | 2 | **2** | Rerun; prompt dapat direkonstruksi |
| Final content kosong tetapi reasoning tersedia | termasuk dalam hasil valid | termasuk dalam hasil valid | **175** | Tidak masuk rerun infrastruktur |
| JSON invalid tetapi respons model utuh | 2 | 0 | **2** | Repair JSON saja; tidak perlu inferensi ulang |

Dengan definisi ketat “gagal karena endpoint/jaringan/file terpotong”, daftar
rerun gabungan berisi tepat **684 path unik**. Semua 684 path ada di disk.

## Kelengkapan folder

| Grup | Persona | JSON utama | Warmup | File terlacak |
|---|---:|---:|---:|---:|
| Religion Persona | 16 | 16 × 1.319 = **21.104** | 12 | **21.116** |
| Gender Persona | 9 | 9 × 1.319 = **11.871** | 6 | **11.877** |
| **Total** | **25** | **32.975** | **18** | **32.993** |

Pemeriksaan kelengkapan menghasilkan:

- filename `gsm8k_00001.json` sampai `gsm8k_01319.json`: lengkap untuk setiap persona;
- missing filename: **0**;
- extra filename: **0**;
- `input.question_id` tidak cocok dengan filename: **0**;
- `input.persona.type` tidak cocok dengan folder: **0**;
- warmup hilang/gagal/invalid: **0**.

Audit menggunakan blob dari Git `HEAD`. Ini aman untuk snapshot ini karena kedua
folder model tidak mempunyai perubahan lokal/untracked; jumlah file Git juga
tepat 21.116 dan 11.877 seperti jumlah pada disk.

## HTTP 4XX/5XX

### Distribusi kode

| Kode | Jumlah | Bentuk error |
|---:|---:|---|
| 429 | **43** | 42 request rate-limited dan 1 structured body error |
| 502 | **105** | 96 provider error dan 9 NextBit `EngineCore` error |
| 504 | **1** | operation aborted |
| 524 | **54** | provider timeout/error |
| **Total** | **203** | |

Tidak ditemukan HTTP 404 pada hasil GSM8K Religion/Gender ini.

### Distribusi pesan persis

| Kode | Jumlah | Pesan |
|---:|---:|---|
| 429 | 42 | `HTTP 429: ... temporarily rate-limited upstream ...` |
| 429 | 1 | `Provider returned error` |
| 502 | 96 | `Provider returned error` |
| 502 | 9 | `Upstream error from NextBit: EngineCore encountered an issue...` |
| 504 | 1 | `The operation was aborted` |
| 524 | 54 | `Provider returned error` |

### Contoh isi JSON HTTP

Contoh 429 transport error:

`Gender_Persona/.../woman_implicit_1/gsm8k_01303.json`

```json
{
  "response": {
    "success": false,
    "content": "",
    "error": "HTTP 429: ... temporarily rate-limited upstream ..."
  }
}
```

Contoh structured error. Walaupun wrapper menyimpan `success: true`, respons
sebenarnya tetap gagal karena body berisi `error` dan content kosong:

```json
{
  "response": {
    "success": true,
    "content": "",
    "raw_api_response": {
      "error": {
        "code": 502,
        "message": "Provider returned error"
      }
    }
  }
}
```

Path contoh lengkap untuk tiap bentuk error:

| Kode/bentuk | Contoh path |
|---|---|
| 429 transport | `Gender_Persona/.../woman_implicit_1/gsm8k_01303.json` |
| 429 structured | `Religion_Persona/.../christian_explicit/gsm8k_00589.json` |
| 502 provider | `Gender_Persona/.../man_explicit_1/gsm8k_00585.json` |
| 502 NextBit EngineCore | `Gender_Persona/.../man_explicit_1/gsm8k_00584.json` |
| 504 | `Religion_Persona/.../atheist_implicit_3/gsm8k_00853.json` |
| 524 | `Gender_Persona/.../man_explicit_1/gsm8k_00516.json` |

Daftar lengkap seluruh 203 path ada di
[`deepseek_http_4xx_5xx.csv`](../result/rerun_manifests/gsm8k_deepseek/deepseek_http_4xx_5xx.csv).

## Kegagalan infrastruktur non-HTTP

Semua **479** kasus non-HTTP berada pada Religion Persona. Isinya bukan jawaban
model, bukan reasoning, dan bukan parse failure; request memang gagal di lapisan
jaringan.

| Jenis | Jumlah | Contoh error |
|---|---:|---|
| DNS resolution | **369** | `Failed to resolve 'openrouter.ai' ... getaddrinfo failed` |
| Timeout | **39** | `Timeout after retries` / `Read timed out` |
| Connection reset | **36** | `ConnectionResetError(10054, ...)` |
| TLS/certificate | **16** | `CERTIFICATE_VERIFY_FAILED ... certificate key too weak` |
| Route/connection/EOF | **14** | unreachable host/network atau TLS EOF |
| Remote disconnected | **5** | remote end closed without response |
| **Total** | **479** | |

Contoh path:

- DNS: `Religion_Persona/.../atheist_explicit/gsm8k_01199.json`;
- TLS: `Religion_Persona/.../atheist_explicit/gsm8k_01057.json`;
- timeout: `Religion_Persona/.../atheist_explicit/gsm8k_01269.json`;
- connection reset: `Religion_Persona/.../atheist_explicit/gsm8k_01206.json`;
- remote disconnected: `Religion_Persona/.../atheist_implicit_2/gsm8k_01099.json`.

Daftar lengkapnya ada di
[`deepseek_non_http_infra.csv`](../result/rerun_manifests/gsm8k_deepseek/deepseek_non_http_infra.csv).

## Distribusi target rerun per persona

| Grup | Persona | HTTP | Non-HTTP | Terpotong | Total |
|---|---|---:|---:|---:|---:|
| Gender | man_explicit_1 | 14 | 0 | 0 | 14 |
| Gender | man_implicit_1 | 16 | 0 | 0 | 16 |
| Gender | man_implicit_2 | 12 | 0 | 0 | 12 |
| Gender | man_implicit_3 | 13 | 0 | 0 | 13 |
| Gender | neutral | 12 | 0 | 1 | 13 |
| Gender | woman_explicit_1 | 13 | 0 | 0 | 13 |
| Gender | woman_implicit_1 | 18 | 0 | 0 | 18 |
| Gender | woman_implicit_2 | 37 | 0 | 1 | 38 |
| Gender | woman_implicit_3 | 12 | 0 | 0 | 12 |
| Religion | atheist_explicit | 0 | 12 | 0 | 12 |
| Religion | atheist_implicit_1 | 3 | 36 | 0 | 39 |
| Religion | atheist_implicit_2 | 2 | 36 | 0 | 38 |
| Religion | atheist_implicit_3 | 3 | 36 | 0 | 39 |
| Religion | christian_explicit | 7 | 8 | 0 | 15 |
| Religion | christian_implicit_1 | 2 | 40 | 0 | 42 |
| Religion | christian_implicit_2 | 5 | 35 | 0 | 40 |
| Religion | christian_implicit_3 | 6 | 33 | 0 | 39 |
| Religion | islam_explicit | 3 | 9 | 0 | 12 |
| Religion | islam_implicit_1 | 4 | 35 | 0 | 39 |
| Religion | islam_implicit_2 | 1 | 34 | 0 | 35 |
| Religion | islam_implicit_3 | 7 | 39 | 0 | 46 |
| Religion | jewish_explicit | 3 | 13 | 0 | 16 |
| Religion | jewish_implicit_1 | 2 | 40 | 0 | 42 |
| Religion | jewish_implicit_2 | 2 | 39 | 0 | 41 |
| Religion | jewish_implicit_3 | 6 | 34 | 0 | 40 |
| **Total** | | **203** | **479** | **2** | **684** |

## JSON invalid

Ada **12** file yang tidak dapat dibaca dengan `json.loads` biasa.

Sepuluh file mempunyai satu objek JSON pertama yang masih utuh lalu diikuti data
tambahan/rusak. Delapan di antaranya memang berisi kegagalan infrastruktur dan
sudah masuk manifest rerun; dua lainnya mempunyai jawaban model valid sehingga
hanya perlu diperbaiki format JSON:

- `Religion_Persona/.../islam_explicit/gsm8k_01268.json`;
- `Religion_Persona/.../islam_explicit/gsm8k_01296.json`.

Delapan prefix-recoverable yang tetap perlu rerun:

- `Religion_Persona/.../islam_explicit/gsm8k_01310.json` — HTTP 502;
- `Religion_Persona/.../islam_implicit_2/gsm8k_01053.json` — DNS;
- `Religion_Persona/.../islam_implicit_2/gsm8k_01054.json` — DNS;
- `Religion_Persona/.../islam_implicit_3/gsm8k_00982.json` — HTTP 502;
- `Religion_Persona/.../islam_implicit_3/gsm8k_01002.json` — timeout;
- `Religion_Persona/.../jewish_implicit_2/gsm8k_01068.json` — DNS;
- `Religion_Persona/.../jewish_implicit_3/gsm8k_00883.json` — connection reset;
- `Religion_Persona/.../jewish_implicit_3/gsm8k_01051.json` — DNS.

Dua file benar-benar terpotong:

- `Gender_Persona/.../neutral/gsm8k_01106.json`;
- `Gender_Persona/.../woman_implicit_2/gsm8k_00764.json`.

Bagian `input` kedua file itu masih utuh. Runner selektif merekonstruksi pesan
GSM8K dari pertanyaan dan persona yang tersimpan, sehingga tidak perlu mencari
template dengan menyisir folder lain.

Manifest repair-only ada di
[`deepseek_invalid_repair_only.csv`](../result/rerun_manifests/gsm8k_deepseek/deepseek_invalid_repair_only.csv).

## Empty final content dan field analisis

Hasil audit menemukan **857** JSON dengan final `content` kosong:

- **682** tidak mempunyai final content maupun reasoning. Ini tepat sama dengan
  203 HTTP + 479 non-HTTP yang masuk rerun;
- **175** masih mempunyai reasoning. Kasus ini sengaja tidak dimasukkan ke
  manifest infrastruktur karena endpoint mengembalikan output model, bukan error
  HTTP/jaringan.

Pada 32.973 JSON yang dapat dibaca atau dipulihkan prefix-nya,
`analysis.extracted_answer` dan `analysis.is_correct` masih `null`. Ini adalah
pekerjaan ekstraksi/evaluasi ulang setelah hasil lengkap, bukan alasan untuk
melakukan inferensi ulang. Dua JSON terpotong tidak memiliki bagian analisis yang
dapat diaudit secara penuh.

## Manifest dan runner Vast AI

Manifest yang tersedia:

| File | Baris | Isi |
|---|---:|---|
| [`deepseek_http_4xx_5xx.csv`](../result/rerun_manifests/gsm8k_deepseek/deepseek_http_4xx_5xx.csv) | 203 | HTTP saja |
| [`deepseek_non_http_infra.csv`](../result/rerun_manifests/gsm8k_deepseek/deepseek_non_http_infra.csv) | 479 | jaringan non-HTTP |
| [`deepseek_invalid_unrecoverable.csv`](../result/rerun_manifests/gsm8k_deepseek/deepseek_invalid_unrecoverable.csv) | 2 | JSON terpotong |
| [`deepseek_infra_all.csv`](../result/rerun_manifests/gsm8k_deepseek/deepseek_infra_all.csv) | **684** | gabungan target rerun |

Runner:
[`rerun_gsm8k_deepseek_errors_vast.py`](../scripts/rerun_gsm8k_deepseek_errors_vast.py)

Validasi lokal, tanpa API dan tanpa menulis file:

```powershell
python scripts/rerun_gsm8k_deepseek_errors_vast.py
```

Smoke test tiga target melalui tunnel lokal:

```powershell
python scripts/rerun_gsm8k_deepseek_errors_vast.py `
  --execute `
  --limit 3 `
  --endpoint-url http://127.0.0.1:8000/v1/chat/completions `
  --concurrency 1
```

Setelah smoke test sukses, jalankan seluruh target. Perintah yang sama aman
untuk resume karena JSON yang sudah berhasil otomatis dilewati:

```powershell
python scripts/rerun_gsm8k_deepseek_errors_vast.py `
  --execute `
  --endpoint-url http://127.0.0.1:8000/v1/chat/completions `
  --concurrency 4
```

Jika hanya ingin 203 error HTTP:

```powershell
python scripts/rerun_gsm8k_deepseek_errors_vast.py `
  --manifest result/rerun_manifests/gsm8k_deepseek/deepseek_http_4xx_5xx.csv `
  --expected-count 203 `
  --execute `
  --endpoint-url http://127.0.0.1:8000/v1/chat/completions `
  --concurrency 4
```

Runner tidak melakukan recursive scan. Ia hanya membuka 684 path dari manifest,
menyimpan backup bertimestamp, menimpa target secara atomik hanya bila final
content tidak kosong, dan menyimpan kegagalan baru ke `_rerun_attempts` tanpa
merusak source JSON.

## Cara mereproduksi audit

```powershell
python scripts/audit_gsm8k_deepseek_personas.py `
  --source git-head `
  --detail-limit 0 `
  --manifest-dir result/rerun_manifests/gsm8k_deepseek `
  --format json
```
