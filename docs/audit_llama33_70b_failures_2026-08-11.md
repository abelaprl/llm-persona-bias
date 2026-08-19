# Audit kegagalan Meta Llama 3.3 70B Instruct

Tanggal audit: **11 Agustus 2026**  
Model: `meta-llama/llama-3.3-70b-instruct`  
Model order: **35**  
Slug: `meta-llama_llama-3.3-70b-instruct`

## Ringkasan

Audit mencakup hasil utama ETHICS dan GSM8K, baik Regional Cultural maupun
nonregional. Tidak ada API call dan tidak ada JSON hasil utama yang diubah.

| Scope | ETHICS | GSM8K | Total gagal |
|---|---:|---:|---:|
| Nonregional | 2.249 | 5.409 | **7.658** |
| Regional Cultural | 32 | 562 | **594** |
| **Total** | **2.281** | **5.971** | **8.252** |

Jumlah tersebut adalah target JSON per persona, bukan jumlah pertanyaan unik.

## Distribusi penyebab

| Penyebab | Nonregional | Regional | Total |
|---|---:|---:|---:|
| HTTP 403, total limit key lama | 6.364 | 0 | **6.364** |
| TLS/SSL terputus | 1.117 | 497 | **1.614** |
| Final content kosong | 108 | 30 | **138** |
| DNS gagal | 41 | 56 | **97** |
| Remote connection reset | 19 | 3 | **22** |
| Koneksi dibatalkan host | 7 | 8 | **15** |
| HTTP 401 | 1 | 0 | **1** |
| JSON GSM8K belum terbentuk | 1 | 0 | **1** |
| **Total** | **7.658** | **594** | **8.252** |

Kegagalan parser jawaban saja tidak dimasukkan. Manifest ini khusus kegagalan
koleksi/infrastruktur: HTTP, jaringan, final content kosong, atau file hilang.

## Distribusi folder/persona

| Dataset/grup | Jumlah gagal |
|---|---:|
| ETHICS Gender Persona | 2 |
| ETHICS Neutral Persona | 15 |
| ETHICS Religion Persona | 2.232 |
| GSM8K Gender Persona | 12 |
| GSM8K Religion Persona | 5.397 |
| ETHICS Regional Cultural Persona | 32 |
| GSM8K Regional Cultural Persona | 562 |

Daftar lengkap seluruh persona, ID soal, penyebab, dan relative path terdapat
pada dua manifest CSV di bagian `Manifest keluaran`.

## Contoh isi kegagalan

| Penyebab | Contoh target |
|---|---|
| Final content kosong | `ETHICS/Gender Persona/man_implicit_2/ethics_commonsense_0367` |
| HTTP 401 | `GSM8K/Gender_Persona/woman_explicit_1/gsm8k_00965` |
| HTTP 403 | `ETHICS/Religion Persona/atheist_implicit_1/ethics_commonsense_0250` |
| DNS | `ETHICS/Neutral Persona/neutral/ethics_virtue_0219` |
| Host aborted | `ETHICS/Religion Persona/islam_implicit_3/ethics_virtue_0469` |
| Remote reset | `ETHICS/Religion Persona/christian_implicit_2/ethics_virtue_0365` |
| TLS/SSL | `ETHICS/Neutral Persona/neutral/ethics_commonsense_0115` |
| File hilang | `GSM8K/Gender_Persona/neutral/gsm8k_01319` |

## Satu output GSM8K yang belum ada

Target yang hilang adalah:

`Gender_Persona/meta-llama_llama-3.3-70b-instruct/neutral/gsm8k_01319.json`

Manifest menyimpan dua sumber recovery:

- template persona dari `neutral/gsm8k_01318.json`;
- pertanyaan yang sama dari `man_explicit_1/gsm8k_01319.json`.

Dry-run runner berhasil merekonstruksi payload target ini tanpa menulis file.

## Warmup

Seluruh **60 warmup yang tersedia** lolos pemeriksaan. Folder Neutral Persona
tidak mempunyai file warmup untuk model ini; kondisi tersebut bukan kegagalan
target utama dan tidak dimasukkan ke manifest.

## Metode audit

Inventaris memeriksa **116.152 JSON utama yang tersedia** dari total **116.153
target yang seharusnya ada**. Ripgrep menyaring indikator kegagalan, error
terstruktur, dan final content kosong; **8.701 kandidat** kemudian diparse untuk
klasifikasi. Selisih inventaris menemukan satu file GSM8K yang hilang.

Audit dilakukan per grup dan hasil parsial kemudian digabung. Penggabungan
menolak target duplikat dan sekarang mempunyai guard jumlah tetap: 7.658 untuk
nonregional dan 594 untuk regional.

Manifest keluaran:

- `result/rerun_manifests/llama33_70b_nonregional/model_all_failures.csv`;
- `result/rerun_manifests/llama33_70b_regional/model_all_failures.csv`.

Masing-masing tersedia pula dalam varian `model_ethics_failures.csv` dan
`model_gsm8k_failures.csv`.

## Keamanan rerun

Runner `scripts/rerun_llama33_70b_openrouter.py`:

- hanya membuka path yang tercantum dalam manifest;
- memisahkan Regional dan nonregional;
- membuat backup sebelum mengganti file yang ada;
- menulis percobaan gagal ke `_rerun_attempts`;
- hanya mengganti/membuat JSON jika OpenRouter memberi final content non-empty;
- melakukan resume dengan melewati target yang sudah sukses;
- merekonstruksi satu target yang hilang dari sumber manifest;
- menolak manifest dengan jumlah selain 7.658 atau 594.

