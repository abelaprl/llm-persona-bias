# Audit error GPT-OSS-20B non-regional

Tanggal audit: **10 Agustus 2026**
Model: `openai/gpt-oss-20b`
Slug folder: `openai_gpt-oss-20b`

## Ruang lingkup

Audit ini hanya membaca hasil utama non-Regional Cultural:

- ETHICS `v9 [THIS IS THE MAIN]`: Gender, Neutral, dan Religion Persona;
- GSM8K: Gender dan Religion Persona;
- folder Regional Cultural tidak diperiksa dan tidak masuk angka laporan ini;
- tidak ada API call, rerun, atau perubahan terhadap JSON hasil.

Sebanyak **70.824 JSON utama** dan **36 cache warm-up** dibaca. Jumlah JSON
utama yang seharusnya tersedia adalah 70.825; satu output GSM8K netral belum
terbentuk.

## Kesimpulan

| Kategori | Jumlah |
|---|---:|
| JSON gagal karena infrastruktur | **111** |
| JSON utama belum terbentuk | **1** |
| Total output yang perlu dibuat ulang | **112** |
| JSON invalid | 0 |
| Warm-up gagal | 0 |
| HTTP 403 | 0 |
| HTTP 404 | 0 |

Angka 5.669 yang berasal dari audit gabungan seluruh skenario tidak dapat
digunakan sebagai angka non-regional. Setelah Regional Cultural dikeluarkan,
set ketat yang perlu dibuat ulang adalah **112 output**.

## Jenis kegagalan

| Jenis | Jumlah | Makna |
|---|---:|---|
| DNS gagal me-resolve `openrouter.ai` | 44 | Alamat OpenRouter tidak dapat ditemukan oleh resolver jaringan |
| Koneksi TLS/SSL terputus | 65 | Sesi HTTPS berhenti sebelum respons selesai |
| Koneksi dibatalkan host (`WinError 10053`) | 1 | Koneksi aktif dihentikan perangkat lunak atau jaringan lokal |
| HTTP 401 `User not found` | 1 | Autentikasi/key tidak diterima saat request |
| File belum ada | 1 | Output belum pernah berhasil dibuat |
| **Total** | **112** | |

Tiga kategori jaringan pertama merupakan kegagalan infrastruktur dan dapat
direrun. Target HTTP 401 sebaiknya baru dijalankan kembali setelah key dipastikan
valid.

## ETHICS

ETHICS mempunyai **84 kegagalan infrastruktur**:

| Grup | Jumlah |
|---|---:|
| Gender Persona | 0 |
| Neutral Persona | 4 |
| Religion Persona | 80 |
| **Total** | **84** |

Distribusi per persona:

| Persona | Error jaringan |
|---|---:|
| `neutral` | 4 |
| `atheist_explicit` | 6 |
| `atheist_implicit_1` | 4 |
| `atheist_implicit_2` | 6 |
| `atheist_implicit_3` | 5 |
| `christian_explicit` | 6 |
| `christian_implicit_1` | 6 |
| `christian_implicit_2` | 5 |
| `christian_implicit_3` | 6 |
| `islam_explicit` | 4 |
| `islam_implicit_1` | 6 |
| `islam_implicit_2` | 3 |
| `islam_implicit_3` | 6 |
| `jewish_explicit` | 4 |
| `jewish_implicit_1` | 4 |
| `jewish_implicit_2` | 4 |
| `jewish_implicit_3` | 5 |
| **Total** | **84** |

Contoh bukti:

- DNS:
  [`ethics_commonsense_0221.json`](../result/ETHICS/v9%20%5BTHIS%20IS%20THE%20MAIN%5D/Neutral%20Persona/openai_gpt-oss-20b/neutral/ethics_commonsense_0221.json)
- TLS/SSL:
  [`ethics_commonsense_0223.json`](../result/ETHICS/v9%20%5BTHIS%20IS%20THE%20MAIN%5D/Neutral%20Persona/openai_gpt-oss-20b/neutral/ethics_commonsense_0223.json)
- koneksi dibatalkan host:
  [`ethics_deontology_0022.json`](../result/ETHICS/v9%20%5BTHIS%20IS%20THE%20MAIN%5D/Neutral%20Persona/openai_gpt-oss-20b/neutral/ethics_deontology_0022.json)

## GSM8K

GSM8K mempunyai **27 JSON error** dan **1 output hilang**.

### Gender Persona

| Persona | Kondisi | Jumlah |
|---|---|---:|
| `man_implicit_2` | error jaringan | 1 |
| `woman_implicit_2` | HTTP 401 `User not found` | 1 |
| `neutral` | file belum terbentuk | 1 |

Target HTTP 401:

[`woman_implicit_2/gsm8k_00638.json`](../result/GMS8K/Gender_Persona/openai_gpt-oss-20b/woman_implicit_2/gsm8k_00638.json)

File yang belum ada:

```text
result/GMS8K/Gender_Persona/openai_gpt-oss-20b/neutral/gsm8k_01319.json
```

### Religion Persona

| Persona | Error jaringan |
|---|---:|
| `atheist_explicit` | 4 |
| `christian_explicit` | 10 |
| `islam_explicit` | 6 |
| `jewish_explicit` | 3 |
| `jewish_implicit_1` | 2 |
| **Total** | **25** |

## Temuan terpisah: field `prediction`

Hitungan awal 34.331 tidak boleh dibaca sebagai 34.331 jawaban kosong. Sebanyak
**32.947** di antaranya adalah JSON GSM8K sukses; schema GSM8K memang menyimpan
jawaban pada `response.content` dan tidak mempunyai objek `prediction`, sehingga
kasus tersebut normal dan bukan error.

Sisanya, **1.384 JSON ETHICS**, adalah kandidat audit isi/reparse dan tidak
dimasukkan ke 112 kegagalan infrastruktur. Final content dan reasoning perlu
diperiksa terlebih dahulu untuk menentukan mana yang cukup di-reparse secara
lokal dan mana yang benar-benar perlu dipanggil ulang.

## Keputusan rerun

Set rerun infrastruktur non-regional GPT-OSS-20B adalah:

- 111 path JSON yang sudah ada tetapi gagal;
- 1 path GSM8K netral yang belum terbentuk;
- total **112 output**;
- tidak memasukkan 1.384 kandidat ETHICS yang masih memerlukan audit/reparse;
- tidak menganggap 32.947 JSON GSM8K tanpa objek `prediction` sebagai error;
- tidak memasukkan Regional Cultural.
