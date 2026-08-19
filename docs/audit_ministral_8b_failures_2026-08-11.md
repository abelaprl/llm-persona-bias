# Audit kegagalan Ministral 8B

Tanggal audit: **11 Agustus 2026**  
Model: `mistralai/ministral-8b-2512`  
Model order: **41**  
Slug: `mistralai_ministral-8b-2512`

## Ringkasan

Audit mencakup hasil utama ETHICS dan GSM8K, baik Regional Cultural maupun
nonregional. Tidak ada API call dan tidak ada JSON hasil utama yang diubah.

| Scope | ETHICS | GSM8K | Total gagal |
|---|---:|---:|---:|
| Nonregional | 2.308 | 4.028 | **6.336** |
| Regional Cultural | 3.847 | 278 | **4.125** |
| **Total** | **6.155** | **4.306** | **10.461** |

Jumlah tersebut adalah target JSON per persona, bukan jumlah pertanyaan unik.

## Distribusi penyebab

| Penyebab | Nonregional | Regional | Total |
|---|---:|---:|---:|
| HTTP 403, total limit key lama | 5.255 | 3.837 | **9.092** |
| TLS/SSL terputus | 1.025 | 220 | **1.245** |
| DNS gagal | 38 | 48 | **86** |
| Koneksi dibatalkan host | 8 | 15 | **23** |
| Remote connection reset | 7 | 1 | **8** |
| HTTP 504 | 1 | 1 | **2** |
| HTTP 502 | 0 | 2 | **2** |
| HTTP 500 | 0 | 1 | **1** |
| HTTP 401 | 1 | 0 | **1** |
| JSON GSM8K belum terbentuk | 1 | 0 | **1** |
| **Total** | **6.336** | **4.125** | **10.461** |

Tidak ditemukan final content kosong pada model ini. Kegagalan parser jawaban
saja tidak dimasukkan ke manifest rerun infrastruktur.

## Distribusi folder/persona

| Dataset/grup | Jumlah gagal |
|---|---:|
| ETHICS Gender Persona | 0 |
| ETHICS Neutral Persona | 11 |
| ETHICS Religion Persona | 2.297 |
| GSM8K Gender Persona | 2 |
| GSM8K Religion Persona | 4.026 |
| ETHICS Regional Cultural Persona | 3.847 |
| GSM8K Regional Cultural Persona | 278 |

Daftar lengkap persona, ID soal, penyebab, dan relative path terdapat pada dua
manifest CSV di bagian `Manifest keluaran`.

## Contoh isi kegagalan

| Penyebab | Contoh target |
|---|---|
| HTTP 401 | `GSM8K/Gender_Persona/woman_implicit_2/gsm8k_00620` |
| HTTP 403 | `ETHICS/Religion Persona/atheist_implicit_1/ethics_commonsense_0199` |
| HTTP 500 | `GSM8K/regional_cultural_persona/east_asian_implicit_3/gsm8k_00570` |
| HTTP 502 | `GSM8K/regional_cultural_persona/east_asian_implicit_2/gsm8k_00007` |
| HTTP 504 | `ETHICS/Religion Persona/islam_implicit_2/ethics_commonsense_0360` |
| DNS | `ETHICS/Neutral Persona/neutral/ethics_commonsense_0340` |
| Host aborted | `ETHICS/Religion Persona/atheist_implicit_2/ethics_deontology_0058` |
| Remote reset | `ETHICS/Religion Persona/atheist_explicit/ethics_commonsense_0306` |
| TLS/SSL | `ETHICS/Neutral Persona/neutral/ethics_commonsense_0142` |
| File hilang | `GSM8K/Gender_Persona/neutral/gsm8k_01319` |

## Satu output GSM8K yang belum ada

Target yang hilang adalah:

`Gender_Persona/mistralai_ministral-8b-2512/neutral/gsm8k_01319.json`

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
terstruktur, dan final content kosong; **10.463 kandidat** kemudian diparse untuk
klasifikasi. Selisih inventaris menemukan satu file GSM8K yang hilang.

Audit dilakukan per grup dan hasil parsial kemudian digabung. Penggabungan
menolak target duplikat dan mempunyai guard jumlah tetap: 6.336 untuk
nonregional dan 4.125 untuk regional.

Manifest keluaran:

- `result/rerun_manifests/ministral_8b_nonregional/model_all_failures.csv`;
- `result/rerun_manifests/ministral_8b_regional/model_all_failures.csv`.

Masing-masing tersedia pula dalam varian `model_ethics_failures.csv` dan
`model_gsm8k_failures.csv`.

## Keamanan rerun

Runner `scripts/rerun_ministral_8b_openrouter.py`:

- hanya membuka path yang tercantum dalam manifest;
- memisahkan Regional dan nonregional;
- membuat backup sebelum mengganti file yang ada;
- menulis percobaan gagal ke `_rerun_attempts`;
- hanya mengganti/membuat JSON jika OpenRouter memberi final content non-empty;
- melakukan resume dengan melewati target yang sudah sukses;
- merekonstruksi satu target yang hilang dari sumber manifest;
- menolak manifest dengan jumlah selain 6.336 atau 4.125.
