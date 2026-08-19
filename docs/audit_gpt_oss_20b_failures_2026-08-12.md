# Audit kegagalan GPT-OSS-20B

Tanggal audit: **12 Agustus 2026**  
Model: `openai/gpt-oss-20b`  
Model order: **31**  
Slug: `openai_gpt-oss-20b`

## Ringkasan rerun

Audit mencakup hasil utama ETHICS dan GSM8K, baik Regional Cultural maupun
nonregional. Tidak ada API call dan tidak ada JSON hasil utama yang diubah.

| Scope | ETHICS | GSM8K | Total rerun |
|---|---:|---:|---:|
| Nonregional | 86 | 32 | **118** |
| Regional Cultural | 4.995 | 563 | **5.558** |
| **Total** | **5.081** | **595** | **5.676** |

Jumlah tersebut adalah target JSON per persona, bukan pertanyaan unik.

## Reasoning-only dipisahkan

Respons yang mempunyai reasoning tetapi final content kosong tidak dimasukkan
ke manifest API rerun. Respons tersebut masih dapat diaudit atau direparse
secara lokal.

| Scope | ETHICS | GSM8K | Review/reparse-only |
|---|---:|---:|---:|
| Nonregional | 1.382 | 105 | **1.487** |
| Regional Cultural | 1.018 | 40 | **1.058** |
| **Total** | **2.400** | **145** | **2.545** |

Daftar lengkapnya disimpan pada `model_review_only.csv` di setiap direktori
manifest. Daftar ini tidak dibaca oleh runner OpenRouter.

## Distribusi penyebab rerun

| Penyebab | Nonregional | Regional | Total |
|---|---:|---:|---:|
| HTTP 403, total limit key lama | 0 | 4.992 | **4.992** |
| TLS/SSL terputus | 65 | 496 | **561** |
| DNS gagal | 44 | 52 | **96** |
| Koneksi dibatalkan host | 1 | 9 | **10** |
| Remote connection reset | 0 | 8 | **8** |
| Final content dan reasoning kosong | 6 | 0 | **6** |
| HTTP 401 | 1 | 0 | **1** |
| JSON GSM8K belum terbentuk | 1 | 0 | **1** |
| JSON ETHICS mempunyai trailing corruption | 0 | 1 | **1** |
| **Total** | **118** | **5.558** | **5.676** |

## Distribusi folder/persona

| Dataset/grup | Jumlah rerun |
|---|---:|
| ETHICS Gender Persona | 1 |
| ETHICS Neutral Persona | 4 |
| ETHICS Religion Persona | 81 |
| GSM8K Gender Persona | 6 |
| GSM8K Religion Persona | 26 |
| ETHICS Regional Cultural Persona | 4.995 |
| GSM8K Regional Cultural Persona | 563 |

Daftar lengkap persona, ID soal, penyebab, dan relative path terdapat pada dua
manifest rerun.

## Contoh isi kegagalan

| Penyebab | Contoh target |
|---|---|
| Final kosong | `ETHICS/Gender Persona/man_explicit_1/ethics_deontology_0266` |
| HTTP 401 | `GSM8K/Gender_Persona/woman_implicit_2/gsm8k_00638` |
| HTTP 403 | `ETHICS/Regional Cultural Persona/east_asian_explicit/ethics_justice_0138` |
| DNS | `ETHICS/Neutral Persona/neutral/ethics_commonsense_0221` |
| Host aborted | `ETHICS/Neutral Persona/neutral/ethics_deontology_0022` |
| Remote reset | `GSM8K/regional_cultural_persona/east_asian_explicit/gsm8k_00422` |
| TLS/SSL | `ETHICS/Neutral Persona/neutral/ethics_commonsense_0223` |
| File hilang | `GSM8K/Gender_Persona/neutral/gsm8k_01319` |
| Trailing corruption | `ETHICS/Regional Cultural Persona/north_american_implicit_3/ethics_virtue_0379` |

## Dua recovery khusus

Target GSM8K yang belum terbentuk:

`Gender_Persona/openai_gpt-oss-20b/neutral/gsm8k_01319.json`

Payload direkonstruksi dari template persona `neutral/gsm8k_01318.json` dan
pertanyaan yang sama pada `man_explicit_1/gsm8k_01319.json`.

Target ETHICS `ethics_virtue_0379.json` mempunyai satu objek JSON lengkap lalu
25 karakter duplikat di belakangnya. Runner membaca objek lengkap pertama dan
membuang trailing corruption hanya pada payload kerja. File utama tetap tidak
diubah sampai endpoint mengembalikan final content non-empty.

Kedua payload telah lolos hydration dan validasi request tanpa API call.

## Warmup

Seluruh **60 warmup yang tersedia** lolos pemeriksaan. Folder Neutral Persona
tidak mempunyai warmup untuk model ini dan kondisi tersebut bukan kegagalan.

## Metode audit

Inventaris memeriksa **116.152 JSON utama yang tersedia** dari total **116.153
target yang seharusnya ada**. Sebanyak **8.227 kandidat dan support file**
diparse setelah penyaringan indikator error/final kosong. Selisih inventaris
menemukan satu file GSM8K yang hilang.

Audit dilakukan per grup dan hasil parsial kemudian digabung. Penggabungan
menolak target duplikat dan mempunyai guard jumlah tetap: 118 untuk
nonregional dan 5.558 untuk regional.

Manifest keluaran:

- `result/rerun_manifests/gpt_oss_20b_nonregional/model_all_failures.csv`;
- `result/rerun_manifests/gpt_oss_20b_regional/model_all_failures.csv`;
- `result/rerun_manifests/gpt_oss_20b_nonregional/model_review_only.csv`;
- `result/rerun_manifests/gpt_oss_20b_regional/model_review_only.csv`.

## Keamanan rerun

Runner `scripts/rerun_gpt_oss_20b_openrouter.py`:

- hanya membuka path manifest rerun, bukan `model_review_only.csv`;
- memisahkan Regional dan nonregional;
- membuat backup sebelum mengganti file yang ada;
- menulis percobaan gagal ke `_rerun_attempts`;
- hanya mengganti/membuat JSON jika final content non-empty;
- melakukan resume dengan melewati target yang sudah sukses;
- mendukung dua recovery khusus yang telah divalidasi;
- menolak manifest dengan jumlah selain 118 atau 5.558.

