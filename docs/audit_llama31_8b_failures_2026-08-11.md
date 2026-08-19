# Audit kegagalan Meta Llama 3.1 8B Instruct

Tanggal audit: **11 Agustus 2026**  
Model: `meta-llama/llama-3.1-8b-instruct`  
Model order: **40**  
Slug: `meta-llama_llama-3.1-8b-instruct`

## Ringkasan

Audit mencakup hasil utama ETHICS dan GSM8K, baik Regional Cultural maupun
nonregional. Tidak ada API call dan tidak ada JSON hasil utama yang diubah.

| Scope | ETHICS | GSM8K | Total gagal |
|---|---:|---:|---:|
| Nonregional | 2.279 | 4.166 | **6.445** |
| Regional Cultural | 4.499 | 503 | **5.002** |
| **Total** | **6.778** | **4.669** | **11.447** |

Jumlah tersebut adalah target JSON per persona, bukan jumlah pertanyaan unik.

## Distribusi penyebab

| Penyebab | Jumlah |
|---|---:|
| HTTP 403, key limit lama | 9.839 |
| TLS/SSL terputus | 1.481 |
| DNS gagal | 84 |
| Remote connection reset | 23 |
| Koneksi dibatalkan host | 16 |
| HTTP 401 | 1 |
| Infrastruktur non-HTTP lainnya | 1 |
| JSON GSM8K belum terbentuk | 2 |
| **Total** | **11.447** |

Tidak ditemukan warmup gagal: **60 warmup** yang tersedia pada seluruh scope
lolos pemeriksaan.

## Distribusi grup

| Dataset/grup | Jumlah gagal |
|---|---:|
| ETHICS Gender Persona | 0 |
| ETHICS Neutral Persona | 8 |
| ETHICS Religion Persona | 2.271 |
| ETHICS Regional Cultural Persona | 4.499 |
| GSM8K Gender Persona | 3 |
| GSM8K Religion Persona | 4.163 |
| GSM8K Regional Cultural Persona | 503 |

## Dua output GSM8K yang belum ada

Keduanya berada pada `Gender_Persona/.../neutral`:

- `gsm8k_00920.json`;
- `gsm8k_01319.json`.

Manifest menyimpan template persona dan sumber pertanyaan secara eksplisit.
Dry-run runner berhasil merekonstruksi payload keduanya tanpa menulis file.

## Catatan tentang manifest Gender lama

Manifest lama `result/rerun_manifests/gender_v9/`
`meta-llama_llama-3.1-8b-instruct_all_missing.csv` berisi 573 baris, tetapi
menunjuk `result/ETHICS/v9_pilot`, yang sudah tidak ada. Sampel target dari
manifest itu sekarang mempunyai respons sukses dan final content pada
`result/ETHICS/v9 [THIS IS THE MAIN]`. Karena itu manifest lama tidak digunakan.

## Metode audit

Inventaris memeriksa seluruh **116.151 JSON utama yang tersedia** dan semua ID
yang seharusnya ada. Ripgrep menyaring indikator kegagalan, error terstruktur,
dan final content kosong; **11.460 kandidat/support file** kemudian diparse
sebagai JSON untuk klasifikasi dan pemilihan sumber recovery. Dua file yang
tidak tersedia ditemukan lewat selisih inventaris ID.

Manifest keluaran:

- `result/rerun_manifests/llama31_8b_nonregional/model_all_failures.csv`;
- `result/rerun_manifests/llama31_8b_regional/model_all_failures.csv`.

Masing-masing tersedia pula dalam varian `model_ethics_failures.csv` dan
`model_gsm8k_failures.csv`.

## Keamanan rerun

Runner `scripts/rerun_llama31_8b_openrouter.py`:

- hanya membuka path yang tercantum dalam manifest;
- memisahkan Regional dan nonregional;
- membuat backup sebelum mengganti file yang ada;
- menulis percobaan gagal ke `_rerun_attempts`;
- hanya mengganti/membuat JSON jika OpenRouter memberi final content non-empty;
- melakukan resume dengan melewati target yang sudah sukses;
- menolak manifest dengan jumlah selain 6.445 atau 5.002.

