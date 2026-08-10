# Audit error Qwen3 Next 80B A3B Thinking non-regional

Tanggal audit: **10 Agustus 2026**
Model: `qwen/qwen3-next-80b-a3b-thinking`
Slug folder: `qwen_qwen3-next-80b-a3b-thinking`

## Ruang lingkup

Audit ini hanya membaca hasil utama non-Regional Cultural:

- ETHICS `v9 [THIS IS THE MAIN]`: Gender, Neutral, dan Religion Persona;
- GSM8K: Gender dan Religion Persona;
- Regional Cultural tidak diperiksa dan tidak masuk angka laporan;
- tidak ada API call atau rerun.

Sebanyak **70.824 JSON utama** dan **36 cache warm-up** diperiksa. Jumlah JSON
utama yang seharusnya ada adalah 70.825; satu output GSM8K netral belum
terbentuk.

## Kesimpulan

| Kategori | Jumlah |
|---|---:|
| JSON gagal karena infrastruktur | **8.286** |
| JSON invalid tanpa jawaban model | **3** |
| JSON utama belum terbentuk | **1** |
| **Total output yang perlu dibuat ulang** | **8.290** |
| Warm-up gagal | 0 |
| HTTP 404 | 0 |

## Jenis kegagalan

| Jenis | Jumlah |
|---|---:|
| HTTP 403 — OpenRouter `Key limit exceeded (total limit)` | **7.048** |
| TLS/SSL connection interrupted | 1.160 |
| DNS resolution failure | 56 |
| Remote connection reset/disconnected | 16 |
| Connection aborted by local host (`WinError 10053`) | 6 |
| JSON invalid tanpa output model | 3 |
| File belum ada | 1 |
| **Total** | **8.290** |

Total kegagalan jaringan non-HTTP adalah **1.238**. Rerun 7.048 target HTTP 403
dengan key yang sama dan limit yang belum diperbaiki hanya akan menghasilkan
403 lagi. Quota/key harus diselesaikan terlebih dahulu.

## Distribusi per dataset

| Dataset dan grup | Error teknis | Invalid | Hilang | Perlu dibuat ulang |
|---|---:|---:|---:|---:|
| ETHICS Gender Persona | 762 | 0 | 0 | 762 |
| ETHICS Neutral Persona | 7 | 0 | 0 | 7 |
| ETHICS Religion Persona | 2.180 | 0 | 0 | 2.180 |
| GSM8K Gender Persona | 1 | 2 | 1 | 4 |
| GSM8K Religion Persona | 5.336 | 1 | 0 | 5.337 |
| **Total** | **8.286** | **3** | **1** | **8.290** |

## ETHICS Gender Persona

| Persona | Jaringan | HTTP 403 | Total |
|---|---:|---:|---:|
| `man_explicit_1` | 9 | 0 | 9 |
| `man_implicit_1` | 6 | 235 | 241 |
| `man_implicit_2` | 10 | 0 | 10 |
| `man_implicit_3` | 6 | 167 | 173 |
| `woman_explicit_1` | 14 | 0 | 14 |
| `woman_implicit_1` | 10 | 135 | 145 |
| `woman_implicit_2` | 9 | 157 | 166 |
| `woman_implicit_3` | 4 | 0 | 4 |
| **Total** | **68** | **694** | **762** |

ETHICS Neutral Persona mempunyai tujuh kegagalan jaringan dan tidak mempunyai
HTTP 403.

## ETHICS Religion Persona

| Persona | Jaringan | HTTP 403 | Total |
|---|---:|---:|---:|
| `atheist_explicit` | 8 | 0 | 8 |
| `atheist_implicit_1` | 9 | 179 | 188 |
| `atheist_implicit_2` | 10 | 171 | 181 |
| `atheist_implicit_3` | 6 | 219 | 225 |
| `christian_explicit` | 7 | 0 | 7 |
| `christian_implicit_1` | 13 | 97 | 110 |
| `christian_implicit_2` | 7 | 163 | 170 |
| `christian_implicit_3` | 7 | 160 | 167 |
| `islam_explicit` | 7 | 0 | 7 |
| `islam_implicit_1` | 7 | 179 | 186 |
| `islam_implicit_2` | 8 | 210 | 218 |
| `islam_implicit_3` | 6 | 152 | 158 |
| `jewish_explicit` | 6 | 0 | 6 |
| `jewish_implicit_1` | 8 | 143 | 151 |
| `jewish_implicit_2` | 10 | 162 | 172 |
| `jewish_implicit_3` | 7 | 219 | 226 |
| **Total** | **126** | **2.054** | **2.180** |

## GSM8K Gender Persona

| Persona | Kondisi | Jumlah |
|---|---|---:|
| `man_implicit_2` | error jaringan | 1 |
| `woman_implicit_2` | JSON invalid tanpa jawaban | 2 |
| `neutral` | file belum terbentuk | 1 |
| **Total** | | **4** |

JSON invalid:

- [`woman_implicit_2/gsm8k_00325.json`](../result/GMS8K/Gender_Persona/qwen_qwen3-next-80b-a3b-thinking/woman_implicit_2/gsm8k_00325.json)
  terpotong pada `input.persona.warmup_response`, sebelum request dan respons
  model ditulis;
- [`woman_implicit_2/gsm8k_00718.json`](../result/GMS8K/Gender_Persona/qwen_qwen3-next-80b-a3b-thinking/woman_implicit_2/gsm8k_00718.json)
  berukuran 0 byte.

File yang belum ada:

```text
result/GMS8K/Gender_Persona/qwen_qwen3-next-80b-a3b-thinking/neutral/gsm8k_01319.json
```

## GSM8K Religion Persona

| Persona | Jaringan | HTTP 403 | Error teknis |
|---|---:|---:|---:|
| `atheist_explicit` | 57 | 23 | 80 |
| `atheist_implicit_1` | 66 | 353 | 419 |
| `atheist_implicit_2` | 65 | 357 | 422 |
| `atheist_implicit_3` | 60 | 353 | 413 |
| `christian_explicit` | 61 | 28 | 89 |
| `christian_implicit_1` | 63 | 355 | 418 |
| `christian_implicit_2` | 71 | 339 | 410 |
| `christian_implicit_3` | 67 | 344 | 411 |
| `islam_explicit` | 59 | 19 | 78 |
| `islam_implicit_1` | 66 | 348 | 414 |
| `islam_implicit_2` | 68 | 332 | 400 |
| `islam_implicit_3` | 72 | 341 | 413 |
| `jewish_explicit` | 59 | 21 | 80 |
| `jewish_implicit_1` | 64 | 373 | 437 |
| `jewish_implicit_2` | 74 | 355 | 429 |
| `jewish_implicit_3` | 64 | 359 | 423 |
| **Total** | **1.036** | **4.300** | **5.336** |

Selain 5.336 error teknis pada tabel, satu file berikut berukuran 0 byte dan
memerlukan model call baru:

[`christian_implicit_3/gsm8k_01282.json`](../result/GMS8K/Religion_Persona/qwen_qwen3-next-80b-a3b-thinking/christian_implicit_3/gsm8k_01282.json)

## Contoh bukti

- HTTP 403 key-limit:
  [`ethics_commonsense_0230.json`](../result/ETHICS/v9%20%5BTHIS%20IS%20THE%20MAIN%5D/Gender%20Persona/qwen_qwen3-next-80b-a3b-thinking/man_implicit_1/ethics_commonsense_0230.json)
- TLS/SSL terputus:
  [`ethics_commonsense_0107.json`](../result/ETHICS/v9%20%5BTHIS%20IS%20THE%20MAIN%5D/Gender%20Persona/qwen_qwen3-next-80b-a3b-thinking/man_explicit_1/ethics_commonsense_0107.json)
- DNS gagal:
  [`ethics_commonsense_0288.json`](../result/ETHICS/v9%20%5BTHIS%20IS%20THE%20MAIN%5D/Gender%20Persona/qwen_qwen3-next-80b-a3b-thinking/man_explicit_1/ethics_commonsense_0288.json)
- remote connection reset:
  [`gsm8k_00927.json`](../result/GMS8K/Religion_Persona/qwen_qwen3-next-80b-a3b-thinking/atheist_explicit/gsm8k_00927.json)
- koneksi dibatalkan host:
  [`gsm8k_01259.json`](../result/GMS8K/Religion_Persona/qwen_qwen3-next-80b-a3b-thinking/atheist_explicit/gsm8k_01259.json)

## Temuan terpisah: field `prediction`

Hitungan awal 28.425 tidak boleh dibaca sebagai 28.425 jawaban kosong. Schema
GSM8K memang menyimpan jawaban pada `response.content` dan **tidak memiliki
objek `prediction`**. Karena itu, 27.634 JSON GSM8K sukses yang tidak mempunyai
`prediction.raw` adalah normal dan bukan error.

Yang masih menjadi kandidat audit isi/reparse hanya **791 JSON ETHICS**:

| Grup | Kandidat audit |
|---|---:|
| ETHICS Gender Persona | 159 |
| ETHICS Neutral Persona | 33 |
| ETHICS Religion Persona | 599 |
| **Total** | **791** |

Kandidat ETHICS ini tidak dimasukkan ke 8.290 target infrastruktur. Isinya perlu
dibandingkan dengan final content/reasoning sebelum diputuskan apakah cukup
di-reparse secara lokal atau benar-benar perlu dipanggil ulang.

## Keputusan rerun

Set kerja non-regional Qwen3 Next 80B adalah **8.290 output**:

- 8.286 error teknis;
- tiga JSON invalid tanpa jawaban model;
- satu JSON belum terbentuk;
- tidak memasukkan 791 kandidat ETHICS yang masih harus diaudit untuk reparse;
- tidak menganggap 27.634 JSON GSM8K tanpa objek `prediction` sebagai error;
- tidak memasukkan Regional Cultural.
