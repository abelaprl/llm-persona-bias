# Implementation Plan — GSM8K Runner: Regional Cultural Persona

> **Catatan 14 Agustus 2026:** status parsing terbaru dan perintah operasional
> berada di `docs/plan_parsing_gsm8k_regional_cultural_persona.md`. OLMo tidak
> termasuk enam model utama dan tidak masuk laporan atau agregasi.

> Status: **PLAN / REVIEW ONLY — belum ada kode yang ditulis.**
> Tujuan dokumen ini: review desain dan menangkap masalah *sebelum* development dimulai.
> Disusun mengikuti alur spec-driven: **SPEC → DESIGN → IMPL → CHECKLIST → TEST → INTEGRATION**.

---

## 0. Ringkasan Temuan Penting (baca ini dulu)

| # | Temuan | Severity | Dampak |
|---|--------|----------|--------|
| **T1** | `inputs/regional_cultural_persona.json` — **16 persona tersimpan di disk (4268 byte) & tervalidasi** (4 region × 1 explicit + 3 implicit) | ✅ SELESAI | Schema valid, nama unik, type-inference benar, encoding bersih. |
| **T2** | Struktur output | ✅ DIPUTUSKAN | **Samakan dengan religion runner**: 1 file per soal di `<model>/<persona>/gsm8k_<qid>.json`, warmup di `<model>/<persona>/warmup/`. → `save_result_json`, skema JSON, `--skip_existing` semua **dipakai ulang apa adanya**. |
| **T3** | GSM8K `gsm8k_test.json` punya **1319 baris**, hanya key `question` + `answer`, **tidak ada `idx`** | 🟡 INFO | Fallback `_qid` (1..N) di religion runner sudah menangani ini. Aman. |
| **T4** | Temperature | ✅ DIPUTUSKAN | **Pakai `temperature=0.2`** (sama seperti religion runner). `max_tokens=4096`. |
| **T5** | File persona religion mengandung artefak encoding (`â€"` = em-dash rusak). | ✅ DIPUTUSKAN | **Perbaiki artefak encoding** — loader baru harus membersihkan `â€"`/mojibake dan file disimpan UTF-8 bersih. |
| **T6** | Model #43 = `deepseek/deepseek-r1-distill-qwen-32b` (reasoning/thinking) | ✅ DIKONFIRMASI | **Pakai model order 43.** `clean_api_response` sudah handle field `reasoning`. |

> **Rekomendasi:** selesaikan T1 (isi file persona) dan putuskan T2 (struktur output) sebelum implementasi. Keduanya menentukan bentuk akhir runner.

---

## 1. Architecture Analysis — Cara Kerja Religion Runner

File referensi: `src/gsm8k_runner/run_gsm8k_religion_personas.py`

### 1.1 Main Execution Flow (`main()`)

```
parse_args()
  └─ set OUTPUT_ROOT global dari --output_root
load_model_config(model_order)      → ambil 1 model dari config by "order"
load_religion_personas(persona_file) → dict {persona_name: cfg}
load_gsm8k_samples(limit_samples)    → list soal (di-assign _qid)
filter personas (--personas atau semua)
pisahkan implicit vs explicit
[1/2] WARMUP — hanya untuk persona IMPLICIT:
   ensure_warmup() → load_warmup() → validasi → regen bila invalid
[2/2] RUN per persona:
   for persona in selected:
     load_warmup (kalau implicit)
     for sample in samples:
       (skip kalau --skip_existing & file ada)
       build_messages()
       call_openrouter_api()
       create_result_json()
       save_result_json()   ← 1 file per SOAL
```

### 1.2 Persona Loading Flow (`load_religion_personas`)

- Baca file JSON (harus **list** of objects).
- Per item ambil: `persona` (nama), `prompt`, `id`.
- Tipe ditentukan dari nama: jika string mengandung `"implicit"` → `implicit`, selain itu → `explicit`.
- Hasil: `dict[name] = {id, type, seed_prompt, explicit_prompt}`.
  - `implicit` → `seed_prompt = prompt`, `explicit_prompt = None`.
  - `explicit` → `explicit_prompt = prompt`, `seed_prompt = None`.
- Item tanpa nama/prompt/id valid akan **di-skip diam-diam**.

### 1.3 GSM8K Loading Flow (`load_gsm8k_samples`)

- Baca `data/gsm8k_test.json` (list).
- Potong `[:limit_samples]` bila `limit_samples > 0`.
- Assign `_qid = 1..N` bila belum ada.
- Field yang dipakai hilir: `question`, `answer`, `idx`/`_qid`.

### 1.4 Model Loading Flow (`load_model_config` + `model_slug`)

- Baca `config/model_keys.json`, cari objek dengan `order == model_order`.
- `model_slug = model_id.replace("/", "_")`.
- Untuk #43 → `model_id = "deepseek/deepseek-r1-distill-qwen-32b"` → slug `deepseek_deepseek-r1-distill-qwen-32b`.
- API call (`call_openrouter_api`): header `Authorization: Bearer <api_key_alias>`, retry eksponensial untuk 429/5xx/timeout, `temperature=0.2`, `max_tokens=4096`.

### 1.5 Warmup Mechanism (khusus persona implicit)

- `ensure_warmup`: kirim `seed_prompt` ke model sekali, simpan respons sebagai `warmup/persona_warmup.json`.
- Saat run, `build_messages` untuk implicit menyusun percakapan multi-turn:
  `system → user(warmup_q) → assistant(warmup_a) → user(soal)`.
- Untuk explicit: `system → user(explicit_prompt + "\n\n" + soal)` (tanpa warmup).

### 1.6 Result Writing Flow (`create_result_json` + `save_result_json`)

- `create_result_json` membangun objek dengan blok: `metadata`, `input` (termasuk persona + question), `request`, `analysis` (placeholder), `response`.
- `save_result_json` menulis ke:
  `OUTPUT_ROOT / <model_slug> / <persona_name> / gsm8k_<qid:05d>.json`
  → **satu file JSON per soal per persona.**

---

## 2. Reusability Analysis

### 2.1 Bisa dipakai TANPA modifikasi (copy apa adanya)

| Komponen | Alasan |
|---|---|
| `clean_bpe_artifacts` | Util murni string. |
| `clean_api_response` | Sudah handle `content` + `reasoning` (relevan untuk model #43). |
| `load_model_config` | Logika lookup by `order` identik. |
| `model_slug` | Identik. |
| `call_openrouter_api` | Logika HTTP + retry identik. |
| `ensure_warmup` / `load_warmup` / `is_valid_warmup_data` | Mekanik warmup + folder `<persona>/warmup/` tidak bergantung domain persona. |
| `build_messages` | Bergantung hanya pada `type` (implicit/explicit), bukan domain. |
| `load_gsm8k_samples` | Loader dataset identik. |
| `create_result_json` | Skema hasil identik. |
| `save_result_json` | ✅ Struktur output disamakan dengan religion runner → **tidak perlu diubah** (Q2). |

### 2.2 Perlu di-COPY lalu di-tweak ringan

| Komponen | Perubahan |
|---|---|
| `load_religion_personas` → `load_regional_personas` | Logika sama, tapi: (a) ganti nama; (b) **verifikasi penentuan tipe** — saat ini berbasis substring `"implicit"`. Aman bila penamaan regional mengikuti konvensi `*_implicit_*` / `*_explicit`. |
| `parse_args` | Ubah default: `--model_order=43`, `--persona_file=inputs/regional_cultural_persona.json`, `--output_root=result/GMS8K/regional_cultural_persona`. |
| Banner cetak (`print`) | Ganti label "RELIGION" → "REGIONAL CULTURAL". |

### 2.3 Perlu LOGIKA BARU (minimal)

| Komponen | Perubahan |
|---|---|
| `load_regional_personas` (copy dari `load_religion_personas`) | Tambah: warning untuk item dibuang (R3), validasi tipe eksplisit (R2), pembersihan artefak encoding pada `prompt` (R10). |
| `parse_args` / konstanta | Ganti default: `OUTPUT_ROOT`, `--model_order=43`, `--persona_file`, banner. |

> Karena Q2 = samakan dengan religion runner, **tidak ada lagi logika output baru**. `save_result_json`, skema JSON, warmup folder, dan `--skip_existing` semua dipakai ulang apa adanya.

---

## 3. Persona JSON Review

### 3.1 Status file target
`inputs/regional_cultural_persona.json` → **0 byte / kosong.** 🔴

Karena kosong, **hierarki, jumlah, dan identifier persona belum bisa ditentukan dari file.** Bagian di bawah adalah *skema yang diharapkan* berdasarkan pola religion runner + struktur output yang Anda contohkan (`western_explicit`, `western_implicit_1`, …).

### 3.2 Skema yang DIBUTUHKAN agar kompatibel dengan runner

File **harus** berupa JSON **list of objects**, tiap object minimal:

```json
[
  { "id": 1, "persona": "western_explicit",   "prompt": "..." },
  { "id": 2, "persona": "western_implicit_1", "type": "A - ...", "prompt": "..." },
  ...
]
```

Aturan kompatibilitas (diturunkan dari `load_religion_personas`):

1. **Top-level = list**, bukan dict. (Kalau dict → `RuntimeError`.)
2. Tiap item wajib punya `persona` (non-empty), `prompt` (non-empty), `id` (int > 0). Item tak lengkap **di-skip diam-diam** (silent) — risiko hilang tanpa peringatan.
3. **Tipe ditentukan dari substring nama**: harus mengandung `implicit` untuk persona implicit; selain itu dianggap explicit.
   - → Nama **harus** mengikuti konvensi `<region>_explicit` dan `<region>_implicit_<n>`.
4. Field `type` (mis. `"A - kitab"`) bersifat opsional/dokumentatif — **tidak dipakai** logika runner.

### 3.3 Concern kompatibilitas

- ⚠️ **Nama tanpa kata `implicit`** → otomatis diperlakukan explicit (tanpa warmup). Mis. `western_traditional` akan dianggap explicit walau dimaksud implicit.
- ⚠️ **Encoding**: hindari artefak em-dash rusak (`â€"`). Simpan UTF-8.
- ⚠️ **`id` duplikat / 0 / non-int** → item di-skip silently.
- ⚠️ Jumlah & daftar region belum diketahui (file kosong) → daftar output di §4 bersifat *template*, harus diperbarui setelah file diisi.

---

## 4. Output Mapping Plan

### 4.1 Struktur FINAL (Q2 = samakan dengan religion runner)

```
result/GMS8K/regional_cultural_persona/
└── deepseek_deepseek-r1-distill-qwen-32b/          ← <model_slug> #43
    ├── western_explicit/
    │   ├── gsm8k_00001.json                         ← 1 file per soal
    │   ├── gsm8k_00002.json
    │   └── ...
    ├── western_implicit_1/
    │   ├── warmup/
    │   │   └── persona_warmup.json                  ← warmup (Q3: di <persona>/warmup/)
    │   ├── gsm8k_00001.json
    │   └── ...
    ├── western_implicit_2/
    ├── western_implicit_3/
    ├── eastern_explicit/
    └── ...
```

Aturan path (sama persis religion runner):
- Explicit → `<model>/<persona>/gsm8k_<qid:05d>.json` (tanpa warmup).
- Implicit → `<model>/<persona>/warmup/persona_warmup.json` + `<model>/<persona>/gsm8k_<qid:05d>.json`.
- `<qid>` = `_qid` 1..N (GSM8K tanpa `idx`).

### 4.3 Daftar output FINAL (16 persona — sudah tervalidasi)

Folder `<model>` = `deepseek_deepseek-r1-distill-qwen-32b`. Tiap folder explicit berisi `gsm8k_<qid>.json`; tiap folder implicit berisi `warmup/persona_warmup.json` + `gsm8k_<qid>.json`.

| # | Persona | Tipe | Warmup? |
|---|---|---|---|
| 1 | `western_explicit` | explicit | — |
| 2 | `western_implicit_1` | implicit | ✅ |
| 3 | `western_implicit_2` | implicit | ✅ |
| 4 | `western_implicit_3` | implicit | ✅ |
| 5 | `east_asian_explicit` | explicit | — |
| 6 | `east_asian_implicit_1` | implicit | ✅ |
| 7 | `east_asian_implicit_2` | implicit | ✅ |
| 8 | `east_asian_implicit_3` | implicit | ✅ |
| 9 | `south_asian_explicit` | explicit | — |
| 10 | `south_asian_implicit_1` | implicit | ✅ |
| 11 | `south_asian_implicit_2` | implicit | ✅ |
| 12 | `south_asian_implicit_3` | implicit | ✅ |
| 13 | `african_explicit` | explicit | — |
| 14 | `african_implicit_1` | implicit | ✅ |
| 15 | `african_implicit_2` | implicit | ✅ |
| 16 | `african_implicit_3` | implicit | ✅ |

→ 4 explicit (tanpa warmup) + 12 implicit (dengan warmup). Total run = 16 persona × `limit_samples` soal.

### 4.2 Skema per file = identik religion runner

`create_result_json` dipakai apa adanya → blok `metadata` / `input` (question + persona) / `request` / `response` / `analysis`. Tidak ada skema agregat baru.

---

## 5. Risk Assessment

| Risiko | Detail | Mitigasi |
|---|---|---|
| **R1 — Persona file kosong** | 0 byte → runner gagal di `load_*_personas` (`No valid personas`). | Isi file dengan skema §3.2 sebelum run. |
| **R2 — Naming mismatch** | Tipe ditebak dari substring `implicit`. Nama menyimpang → salah klasifikasi explicit/implicit. | Patuhi konvensi `*_explicit` / `*_implicit_<n>`; tambahkan validasi eksplisit di loader baru. |
| **R3 — Schema mismatch** | Top-level dict, atau item tanpa `id/persona/prompt` → di-skip silently / error. | Loader baru sebaiknya **log warning** untuk item yang dibuang (tidak silent). |
| **R4 — Output collision** | Jika dua persona menghasilkan nama file sama, file saling menimpa. Nama dari `persona` → aman selama unik. | Pastikan `persona` unik; pertimbangkan guard "file exists". |
| **R5 — Model #43 selection** | `--model_order=43` harus benar-benar map ke `deepseek/deepseek-r1-distill-qwen-32b`. Sudah terverifikasi di config baris order 43. | Jadikan 43 sebagai default; cetak nama model di awal untuk konfirmasi. |
| **R6 — GSM8K compat** | Dataset hanya `question`+`answer`, tanpa `idx`. `_qid` fallback menangani. 1319 soal × N persona = banyak call. | Pakai `--limit_samples` kecil untuk smoke test; perhatikan rate limit & biaya. |
| **R7 — Temperature** | ✅ Diputuskan: `temperature=0.2`. | Tidak ada aksi; catat di metadata bahwa run tidak fully deterministik. |
| **R10 — Artefak encoding** | File persona bisa berisi mojibake (`â€"`). | ✅ Diputuskan diperbaiki: bersihkan di loader + simpan file UTF-8 bersih. |
| **R8 — Reasoning tokens** | Model thinking bisa boros token / lambat. `max_tokens=4096` mungkin terpotong. | Pantau `usage`; sesuaikan `max_tokens` bila perlu. |
| **R9 — Agregasi 1 file/persona** | ✅ TIDAK BERLAKU. Q2 = struktur per-soal (sama religion runner), penulisan tetap incremental per soal. | `--skip_existing` per-soal tetap berfungsi untuk resume. |

### Open Questions
- ✅ **Q1 (T1) — SELESAI:** File persona tersimpan di disk (4268 byte), 16 persona tervalidasi.

### Keputusan yang sudah dikunci
- ✅ **Q2 (T2):** Struktur output **disamakan dengan religion runner** — 1 file/soal `<model>/<persona>/gsm8k_<qid>.json`.
- ✅ **Q3:** Warmup persona implicit di **`<model>/<persona>/warmup/persona_warmup.json`** (sama religion runner).
- ✅ **Q4 (T4):** `temperature = 0.2`, `max_tokens = 4096`.
- ✅ **Model:** order **43** (`deepseek/deepseek-r1-distill-qwen-32b`).
- ✅ **Encoding:** perbaiki artefak mojibake (`â€"` dst.) di loader; simpan UTF-8 bersih.

---

## 6. Implementation Plan (langkah demi langkah — BELUM ditulis)

### Fase A — Prasyarat (blocker)
1. 🔴 **Save & isi `inputs/regional_cultural_persona.json`** — file masih 0 byte di disk. Save dari editor (Ctrl+S) lalu pastikan isinya sesuai skema §3.2 (list, `id`/`persona`/`prompt`, konvensi nama `*_explicit`/`*_implicit_<n>`, UTF-8 bersih).
2. Jawab Open Questions Q2 & Q3 (Q4/model/encoding sudah dikunci).

### Fase B — Buat runner baru
3. Salin `run_gsm8k_religion_personas.py` → `src/gsm8k_runner/run_gsm8k_regional_cultural_personas.py`.
4. Reuse tanpa ubah: util string, `load_model_config`, `model_slug`, `call_openrouter_api`, warmup helpers, `build_messages`, `load_gsm8k_samples` (§2.1).
5. Rename `load_religion_personas` → `load_regional_personas`; tambahkan: **warning** untuk item yang dibuang (atasi R3), validasi tipe eksplisit (atasi R2), dan **pembersihan artefak encoding** pada `prompt` (atasi R10 — bersihkan `â€"`/mojibake jadi karakter benar).
6. Ubah konstanta/`parse_args` default:
   - `OUTPUT_ROOT = result/GMS8K/regional_cultural_persona`
   - `--model_order=43`, `--persona_file=inputs/regional_cultural_persona.json`
   - `temperature=0.2`, `max_tokens=4096` (dikunci).

### Fase C — Output (tidak ada perubahan; Q2 = sama religion runner)
7. `save_result_json`, `create_result_json`, warmup folder, dan `--skip_existing` **dipakai ulang apa adanya**. Tidak ada writer/skema baru.

### Fase D — Test & Integrasi
10. **Smoke test**: `--limit_samples 1 --personas <satu_explicit>` → verifikasi 1 file persona terbentuk, JSON valid.
11. **Smoke test implicit**: pastikan warmup ter-generate & multi-turn tersusun benar.
12. Verifikasi mapping model #43 dari banner output.
13. Run subset kecil seluruh persona; cek tidak ada collision/overwrite.
14. Update CLAUDE.md "Key Scripts" + dokumentasi bila runner final.
15. (Hilir) sesuaikan rebuild CSV / accuracy bila skema agregat dipakai — skrip rebuild existing berasumsi 1 file/soal, jadi **kemungkinan butuh adapter** bila ingin masuk pipeline analisis.

---

## 7. Catatan untuk Pipeline Hilir

✅ Karena Q2 = struktur **identik religion runner** (1 JSON per soal), output regional **kompatibel** dengan `src/rebuild/*` dan `src/analysis/calc_accuracy.py` tanpa adapter — cukup arahkan ke `result/GMS8K/regional_cultural_persona/<model_slug>/`. Tidak ada trade-off pipeline yang tersisa.
