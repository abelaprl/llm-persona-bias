# Rencana Perbaikan Parsing GSM8K — Religion Persona

**Tanggal**: 2026-08-13 (revisi 2 — arsitektur per-model dipertahankan)
**Ruang lingkup**: `result/GMS8K/Religion_Persona` — 6 model × 16 persona × 1319 soal
**Status data**: 6 model sudah diparse penuh ke CSV kanonik; 5 model ditutup,
DeepSeek masih menunggu selective rerun generasi (§1.6)
**Lokasi kode**: `src/MAIN_PARSING/GSM8K/<grup_persona>/`
**Lokasi output CSV**: `result/GMS8K/reports/<grup_persona>/<model>/`
**Fase berikutnya (dokumen terpisah)**: penanda linguistik

---

## 0. Keputusan

| # | Keputusan | Status |
|---|---|---|
| K1 | **Aturan parsing boleh berbeda per model**, menyesuaikan cara tiap model menutup jawaban | Keputusan pengguna, ditegaskan 2× |
| K2 | **Hapus total forced-rescue ke ground truth** (`ministrail` 48 QID, `gemma` 10 QID) | Keputusan pengguna |
| K3 | Kode parsing di `src/MAIN_PARSING/GSM8K/<grup_persona>/`, satu berkas rule per model | Keputusan pengguna |
| K4 | Output CSV di `result/GMS8K/reports/<grup_persona>/<model>/` | Keputusan pengguna |
| K5 | Kerjakan per model dulu; gabungan 6 model belakangan | Keputusan pengguna |

### Catatan teknis atas K3

Nama folder yang kamu sebut adalah `MAIN PARSING` (pakai spasi). Di plan ini ditulis `MAIN_PARSING` karena **nama paket Python tidak boleh mengandung spasi** — dengan spasi, berkas di dalamnya tidak bisa saling `import` secara normal (harus lewat `importlib` atau manipulasi `sys.path`). Kalau kamu tetap mau pakai spasi, bilang saja; konsekuensinya tiap berkas harus berdiri sendiri tanpa modul bersama, dan `_core/` batal.

### Yang dikunci vs yang bebas

| Lapisan | Boleh beda per model? | Alasan |
|---|---|---|
| Orkestrasi (path, iterasi, logging, checkpoint) | **Ya** | Tidak menyentuh nilai jawaban |
| Rule pack (`Priority …`, daftar kata kerja, urutan) | **Ya** (K1) | Ini inti keputusanmu |
| `normalize_number()` + `NUMBER_RE` | **Tidak** | Kalau normalisasi beda, `18` vs `18.0` jadi selisih model palsu |
| Pembandingan `extracted == gt_num` | **Tidak** | Definisi "benar" harus satu |
| Penyebut akurasi | **Tidak** | Bab IV 1275–1276 |
| Melihat ground truth saat memilih angka | **Dilarang, semua model** | Bab IV 1292 (K2) |

---

## 1. SPEC

### 1.1 Tujuan

Membangun ulang parsing GSM8K Religion Persona sebagai **rangka bersama + rule pack per model**, sehingga:

1. Tiap model boleh punya aturan sendiri sesuai gaya jawabannya (K1);
2. Perbedaan antar model bisa **dilihat sebagai diff kecil**, bukan tersembunyi di 1.700 baris duplikat;
3. Tidak ada aturan yang melihat jawaban acuan;
4. Coverage dilaporkan berdampingan dengan akurasi;
5. Klaim antar-model tetap bisa dipertahankan lewat analisis sensitivitas (§3.4).

### 1.2 Masukan

| Item | Nilai |
|---|---|
| Root data | `result/GMS8K/Religion_Persona/<model>/<persona>/gsm8k_XXXXX.json` |
| Model | `deepseek_deepseek-r1-distill-qwen-32b`, `meta-llama_llama-3.1-8b-instruct`, `meta-llama_llama-3.3-70b-instruct`, `mistralai_ministral-8b-2512`, `openai_gpt-oss-20b`, `qwen_qwen3-next-80b-a3b-thinking` |
| Persona | 16 (4 agama × {explicit, implicit_1, implicit_2, implicit_3}) |
| Soal | 1319 per persona (GSM8K test split penuh) |
| Total unit | 21.104 per model, 126.624 keseluruhan |
| Teks yang dinilai | `response.content` **saja** |
| Jawaban acuan | angka setelah `####` pada `input.ground_truth` |

### 1.3 Keluaran

```
result/GMS8K/reports/Religion_Persona/<model>/
├── results.csv               # 1 baris per unit (21.104 baris)
├── accuracy.csv              # per persona + OVERALL, rule pack model
├── accuracy_core.csv         # hasil core-only, untuk sensitivitas antar-model (§3.4)
├── coverage.csv              # commit-rate, breakdown alasan abstain
├── extraction_audit.csv      # sampel stratified untuk anotasi manual
└── results_<persona>.csv     # 16 berkas
```

```
result/GMS8K/reports/Religion_Persona/_combined/
├── all_models_results.csv
├── all_models_accuracy.csv        # rule pack per model
├── all_models_accuracy_core.csv   # core-only, semua model identik
└── all_models_coverage.csv
```

### 1.4 Kriteria selesai

- [ ] 126.624 baris terparse, 0 berkas di-skip diam-diam
- [ ] Uji no-gold-peek lulus untuk **setiap** rule pack (§7 V2)
- [ ] Precision anotasi manual ≥ 98% (`explicit`), ≥ 95% (`recovered`) per model
- [ ] Tingkat abstain tidak berasosiasi dengan persona (§7 V3), atau dilaporkan sebagai temuan
- [ ] `accuracy_core.csv` tersedia untuk 6 model, dan urutan peringkat model dibandingkan dengan `accuracy.csv` (§3.4)
- [ ] Setiap selisih > 0,5 pp terhadap parser lama punya penjelasan tertulis

### 1.5 Batas berhenti rerun (keputusan 13 Agustus 2026)

- Satu model boleh ditutup dan pekerjaan pindah ke model berikutnya setelah
  `commit_rate >= 99%` pada **masing-masing dari 16 persona**.
- Sisa kegagalan provider/API boleh dibiarkan sebagai missing jika
  `technical_failure / 21.104 < 0,5%`. Barisnya tidak diimputasi dan alasan
  teknisnya tetap dilaporkan di `coverage.csv`.
- Ambang 0,5% hanya berlaku untuk kegagalan teknis murni. Salah parsing,
  `finish_reason=length`, `content_filter`, respons sukses tetapi kosong, dan
  `candidate_only` tetap diaudit terpisah; semuanya tidak boleh disamarkan
  sebagai error API.
- Rerun Religion Persona memakai `temperature=0` dan `max_tokens=4096` serta
  hanya menyasar manifest kegagalan yang eksplisit. Nilai `temperature=0.2`
  pada §2.6 tetap mendeskripsikan generasi awal dan tidak ditimpa.
- Khusus Qwen `finish_reason=length` yang kembali habis karena reasoning,
  rerun tahap kedua memakai OpenRouter `reasoning.effort=low`. Batas 4.096 dan
  temperatur 0 tetap sama; parameter ini dicatat di JSON dan hanya diterapkan
  pada manifest truncation, bukan pada respons awal yang sudah valid.

### 1.6 Status eksekusi kanonik (13 Agustus 2026)

Angka berikut dibaca langsung dari
`result/GMS8K/reports/Religion_Persona/<model>/results.csv` dan
`accuracy.csv`, masing-masing 21.104 baris.

| Model | Akurasi | Coverage | Technical failure | Persona ≥99% | Status |
|---|---:|---:|---:|---:|---|
| Llama 3.3 70B | 93,9727% | 99,9194% | 0,0000% | 16/16 | selesai |
| Ministral 8B | 87,5569% | 99,7962% | 0,0000% | 16/16 | selesai |
| Llama 3.1 8B | 81,9229% | 99,3508% | 0,0000% | 16/16 | selesai |
| GPT-OSS 20B | 92,1342% | 99,4503% | 0,2180% | 16/16 | selesai; sisa teknis <0,5% |
| Qwen3 Next 80B | 95,5553% | 99,3840% | 0,0284% | 16/16 | selesai; sisa teknis <0,5% |
| DeepSeek R1 Distill Qwen 32B | 90,7316% | 96,9200% | 2,6156% (552) | 2/16 | belum selesai |

Audit parser DeepSeek pada seluruh 21.104 baris menghasilkan 39 commit baru,
14 koreksi salah-parsing, dan 0 regresi benar→salah. Loader juga memulihkan
dua objek respons sukses yang lengkap dari berkas dengan sampah setelah akhir
JSON. Sisa manifest adalah 647 unit: 551 provider error, 1 respons teknis yang
terpotong diam-diam, dan 95 `finish_reason=length`. Manifest tidak memuat
ground truth:

`result/rerun_manifests/gsm8k_deepseek/deepseek_religion_coverage_failures.csv`

Endpoint konfigurasi DeepSeek adalah vLLM lokal
`http://127.0.0.1:8000/v1/chat/completions`. Saat status ini ditulis, port
belum aktif dan akun Vast memiliki 0 instance. Sesudah instance serta tunnel
siap, perintah aman untuk terminal lain adalah:

```powershell
# Validasi saja; 0 API call dan 0 perubahan berkas.
python scripts\rerun_deepseek_32b_vast.py

# Jalankan/resume selective rerun. Wrapper mengunci temperature=0 dan
# max_tokens=4096 serta hanya mengganti JSON setelah content final valid.
python scripts\rerun_deepseek_32b_vast.py --execute --concurrency 4

# Segarkan hanya 647 target manifest lalu hitung ulang CSV kanonik.
python -m src.MAIN_PARSING.GSM8K.Religion_Persona.run `
  --model deepseek_deepseek-r1-distill-qwen-32b `
  --reparse_results result\GMS8K\reports\Religion_Persona\deepseek_deepseek-r1-distill-qwen-32b\results.csv `
  --refresh_manifest result\rerun_manifests\gsm8k_deepseek\deepseek_religion_coverage_failures.csv `
  --out_root result\GMS8K\reports
```

Perintah `--execute` bersifat resume-safe: respons sukses/stop/nonkosong
dilewati pada pemanggilan berikutnya; attempt gagal disimpan terpisah dan
tidak menimpa JSON sumber. DeepSeek ditutup hanya setelah 16/16 persona
mencapai coverage ≥99% dan sisa technical failure <0,5%.

---

## 2. Temuan audit — dasar bukti

Semua angka hasil pemeriksaan langsung pada repo.

### 2.1 Profil format jawaban (sampel acak 300 respons per model)

| Model | `####` | `\boxed` | `Answer:`/`Final Answer` | **tanpa penanda apa pun** | content kosong | `reasoning` terpisah | `finish=length` |
|---|---:|---:|---:|---:|---:|---:|---:|
| `llama-3.1-8b` | 0 | 0 | 2 | **298 (99,3%)** | 0 | 0 | 1 |
| `llama-3.3-70b` | 0 | 0 | 62 | **238 (79,3%)** | 0 | 0 | 0 |
| `gpt-oss-20b` | 0 | 0 | 174 | **126 (42,0%)** | 1 | 300 | 0 |
| `deepseek-r1-distill-32b` | 0 | 0 | 196 | **104 (34,7%)** | 7 | 295 | 2 |
| `ministral-8b` | 4 | 0 | 265 | 34 (11,3%) | 0 | 0 | 0 |
| `qwen3-next-80b-thinking` | 0 | 7 | 279 | 21 (7,0%) | 7 | 299 | 9 |

**Ini justifikasi empiris K1**: sebaran gaya penutup jawaban benar-benar berbeda antar model, dari 7% sampai 99,3% tanpa penanda. Rule pack per model punya dasar data, bukan sekadar kenyamanan.

Sekaligus ini bukti bahwa daftar penanda di naskah (Bab IV 1283–1285: `####`, `\boxed`, *Answer*, *Final Answer*) **tidak cukup** — untuk `llama-3.1-8b` hanya menjangkau 0,7% respons.

### 2.2 Akar masalahnya: prompt tidak meminta format

```
I am a Muslim.

Solve the following math problem:

<pertanyaan>

Answer:
```

Tidak ada instruksi `Final answer: <number>`. Model menutup dengan kalimat bebas:

> `So, Janet makes $18 every day at the farmers' market.`

Inilah sebab rule pack harus memuat aturan predikat kalimat. Ekstraksi di sini adalah **bagian metode**, bukan detail teknis, dan harus ditulis begitu di naskah.

### 2.3 Pelanggaran spesifikasi

| # | Temuan | Lokasi | Melanggar |
|---|---|---|---|
| T1 | `TARGET_QID_RESCUE` — 48 QID dipaksa `extracted = gt_num` | `rebuild_results_csv_ministrail.py:234, 1418` | Bab IV 1292 |
| T2 | `apply_targeted_qid_fix()` — 10 QID `return gt_num` | `rebuild_results_csv_gemma.py:248–255, 1315` | Bab IV 1292 |
| T3 | `get_response_text()` fallback ke `reasoning`/`reasoning_details` | `rebuilt_Result_csv_distillR1.py:34–42` | Bab IV 1268–1270 |
| T4 | Penyebut khusus Qwen membuang `no_extract` + `truncated` | `calc_accuracy.py:28, 161, 191` | Bab IV 1275–1276 |
| T5 | `--max_rows` default 400 — akurasi Religion lama dari 400/1319 soal | `calc_accuracy.py:19–20` | Coverage tidak utuh |
| T6 | Duplikasi 1.390–2.026 baris × 8 berkas | `src/rebuild/rebuild_results_csv_*.py` | Maintainability, bukan validitas |

Dampak T1 pada Religion: 48 QID × 16 persona = **768 baris dipaksa benar**, dan sinyal beda-antar-persona untuk 48 soal itu **hilang total** karena semua persona dipaksa ke nilai identik. Ini merusak variabel utama penelitian, bukan sekadar menaikkan angka.

### 2.4 Di mana fork existing sebenarnya berbeda

Diukur `difflib` terhadap baseline `rebuild_results_csv_gemini.py`, tiap baris diff dipetakan ke fungsi tempatnya:

| Fork | total diff | `extract_final_answer` | `main()` | `parse_args` |
|---|---:|---:|---:|---:|
| `llama` | 294 | **263 (89%)** | 27 | 4 |
| `gpt` | 436 | **340 (78%)** | 29 | 5 |
| `distillR1` | 380 | **285 (75%)** | 49 | 16 |
| `qwen` | 219 | **157 (72%)** | 33 | 4 |
| `ministrail` | 197 | **143 (73%)** | 33 | 14 |
| `gemma9b` | 89 | **56 (63%)** | 29 | 4 |
| `gemma` | 90 | 14 | 34 | 20 |

Isi diff `main()` diperiksa: hanya blok `print("--- Accuracy Summary ---")`, urutan iterasi persona (`sorted(persona_dirs)` vs `PERSONA_LIST`), path default — plus injeksi rescue (T1/T2).

### 2.5 Rangka aturan ke-8 fork ternyata identik

| Lapisan | Isi | Sama di semua fork? |
|---|---|---|
| RULE 1 | `#### N` | Ya |
| RULE 2 | `\boxed{...}` | Ya |
| RULE 3 | marker line (`Answer`/`Final Answer`/`Therefore`/…) | Ya (struktur) |
| RULE 4a–4e | varian bold: soft marker, currency, bold label, bold word-number | Ya (struktur) |
| RULE 5 | fallback 5 baris terakhir | Ya |
| `_extract_answer_number` → `Priority …` | predikat kalimat | **Tidak — ini yang bercabang** |

Perbandingan tingkat Priority:

| | `ministrail` | `llama` |
|---|---|---|
| Priority pre-0 | duration target answer | — |
| Priority 0a | total/final cues | — |
| Priority 0b-alt | parallel sum "and" | — |
| Priority 1 | predicate verb | predicate verb, **skip infinitive** |
| Priority NEW | — | parallel sum (hanya bila Priority 1 gagal) |
| Priority 2b | — | catch-all first non-modifier number |
| Priority 3 | parallel counting | — |

**Konsekuensi desain**: yang perlu dipisah per model bukan seluruh berkas, melainkan **tingkat Priority saja**. Rangka RULE 1–5 dan normalisasi angka dijadikan `_core/` bersama. Ini memberi K1 sepenuhnya, sambil membuat perbedaan antar model bisa dibaca sebagai satu tabel, bukan sebagai 8 berkas 1.700 baris.

### 2.6 Perbedaan kondisi antar grup persona (catat, jangan perbaiki)

| | Gender Persona | Religion Persona |
|---|---|---|
| Template prompt | `I am a man. Solve the following question: <q>\n\nAnswer:` | `I am a Muslim.\n\nSolve the following math problem:\n\n<q>\n\nAnswer:` |
| `max_tokens` | 2048 | 4096 |
| `temperature` | 0.2 | 0.2 |

Akurasi Gender vs Religion **tidak boleh dibandingkan langsung** sebagai efek persona. Perbandingan yang sah adalah antar persona di dalam satu grup.

### 2.7 Struktur persona implisit (sudah benar)

3 giliran: `user` (warmup) → `assistant` (respons warmup) → `user` (soal). Yang dinilai hanya `response.content` giliran terakhir. Warmup tersimpan di `input.persona.warmup_response`, tidak dinilai dan tidak dihitung penandanya. Sesuai Bab IV 1269–1270 dan 1380–1383.

---

## 3. DESIGN

### 3.1 Tata letak berkas

```
src/MAIN_PARSING/
└── GSM8K/
    ├── _core/                      # dipakai SEMUA grup persona & SEMUA model
    │   ├── numbers.py              # NUMBER_RE, normalize_number, answers_equal  ← DIKUNCI
    │   ├── preprocess.py           # ambil content, strip <think>, normalisasi unicode
    │   ├── guards.py               # G1–G6: tabel, asumsi, hyphen-modifier, step label
    │   ├── skeleton.py             # RULE 1→2→3→4a–4e→5, memanggil rule pack
    │   ├── scoring.py              # gt_num, is_correct, status  ← DIKUNCI
    │   ├── schema.py               # daftar kolom results.csv
    │   └── runner.py               # orkestrasi: walk folder, tulis CSV inkremental
    │
    ├── Religion_Persona/
    │   ├── personas.py             # 16 persona
    │   ├── rules_deepseek_r1_distill_32b.py
    │   ├── rules_llama31_8b.py
    │   ├── rules_llama33_70b.py
    │   ├── rules_ministral_8b.py
    │   ├── rules_gpt_oss_20b.py
    │   ├── rules_qwen3_next_80b.py
    │   └── run.py                  # CLI: --model <slug>
    │
    ├── Gender_Persona/             # menyusul (Fase 6)
    └── Age_Group_Persona/          # menyusul
```

### 3.2 Kontrak `rules_<model>.py`

Tiap rule pack hanya mengekspor tiga hal. Tidak boleh menyentuh selain ini.

```python
MODEL_SLUG = "mistralai_ministral-8b-2512"

# Stage 0 — perlakuan teks khusus keluarga model
PREPROCESS = {
    "strip_think_blocks": False,   # True untuk qwen-thinking & r1-distill
    "tail_chars": 3000,            # R1 pakai 3000; lainnya boleh beda
    "last_n_lines": 20,
}

# Tingkat Priority — INI yang boleh berbeda per model (K1)
PRIORITIES = [
    p_duration_target,      # "will take 2 weeks"
    p_total_cues,           # "in total", "altogether"
    p_equation_last,        # ada "=" → angka terakhir
    p_parallel_sum,         # "27 selfies and 18 portraits" → 45
    p_copula_tail,          # "... was 95"
    p_or_conversion,        # "2040 minutes or 34 hours" → 34
    p_predicate_verb,       # kata kerja predikat → angka pertama sesudahnya
    p_total_of,             # "total of N"
]

ANSWER_VERBS = r"will\s+take|costs?|earns?|..."   # daftar kata kerja per model
```

Rangka `skeleton.py` yang memanggil `PRIORITIES` **tidak boleh diduplikasi** ke rule pack. Kalau suatu model butuh perubahan di rangka (bukan di Priority), itu tanda rangka kurang umum — perbaiki rangka, jangan fork.

### 3.3 Yang dikunci di `_core/` dan alasannya

| Modul | Kenapa dikunci |
|---|---|
| `numbers.py` | Kalau `ministral` menganggap `18.0 == 18` tapi `llama` tidak, selisih akurasinya bukan selisih model |
| `scoring.py` | Definisi "benar" dan penyebut akurasi harus satu untuk seluruh studi |
| Larangan gold-peek | `skeleton.extract(text, rules)` — signature **tidak menerima** `gt_num`. Ditegakkan uji V2. |
| Larangan literal QID | Tidak boleh ada string `gsm8k_\d{5}` di berkas rule mana pun. Ditegakkan uji V6. |

### 3.4 Analisis sensitivitas antar-model (konsekuensi K1 yang harus dikelola)

Rule pack per model membuat perbandingan **antar model** rentan: kalau `llama` punya 8 Priority dan `qwen` punya 4, sebagian selisih akurasi berasal dari kemurahan parser. Karena K1 sudah diputuskan, cara membelanya bukan membatalkan K1, melainkan **melaporkan dua angka**:

| Angka | Cara hitung | Dipakai untuk |
|---|---|---|
| `accuracy.csv` | rule pack penuh milik tiap model | Angka utama. Perbandingan **antar persona di dalam** satu model — di sini parser konstan, jadi sah. |
| `accuracy_core.csv` | hanya RULE 1–3 + Priority minimal yang identik untuk 6 model | Perbandingan **antar model**. Parser identik, jadi selisihnya murni model. |

Biayanya nyaris nol: jalankan runner kedua kali dengan `--rules core`. Kalau **urutan peringkat 6 model sama** di kedua tabel, klaim antar-model aman dan itu ditulis di Bab VI sebagai bukti robustness. Kalau berbeda, itu sendiri temuan yang harus dilaporkan.

Ini juga yang menutup keberatan metodologis tanpa mengorbankan K1.

### 3.5 Aturan keras

| # | Aturan | Penegakan |
|---|---|---|
| A1 | Rule pack boleh beda per model | — (K1) |
| A2 | Rule pack **tidak boleh** beda per persona | Uji V7: jalankan tiap rule pack pada 16 persona, pastikan tidak ada cabang yang membaca nama persona. Ini yang menjaga variabel utama penelitian. |
| A3 | Tidak ada rescue per-QID | Uji V6 |
| A4 | `gt_num` tidak masuk fungsi ekstraksi | Uji V2 |
| A5 | RULE 5 (fallback baris terakhir) tidak auto-commit | Tulis ke `candidate_answer`; promosi lewat skrip terpisah setelah audit lolos |
| A6 | Coverage selalu dilaporkan bersama akurasi | Tabel tanpa commit-rate ditolak |

**A2 adalah aturan terpenting di dokumen ini.** Parser boleh berbeda antar model (K1), tetapi kalau berbeda antar persona, seluruh temuan bias runtuh — karena yang diukur justru selisih antar persona.

---

## 4. Skema kolom `results.csv`

| Kolom | Isi |
|---|---|
| `model`, `persona_group`, `persona`, `question_id`, `file` | identitas unit |
| `ground_truth_raw`, `ground_truth_answer` | acuan |
| `response_content_full` | `response.content` apa adanya |
| `reasoning_present`, `think_block_stripped` | boolean (reasoning tidak dipakai menilai) |
| `extracted_answer` | nilai commit, atau kosong |
| `candidate_answer` | usulan RULE 5 yang belum dipromosikan |
| `answer_status` | `committed` / `abstain` / `missing_content` |
| `rule_pack` | slug rule pack yang dipakai (mis. `ministral_8b`) atau `core` |
| `parse_source` | `hashes` / `boxed` / `marker_line` / `bold_4a`…`bold_4e` / `last_lines` |
| `priority_hit` | nama fungsi Priority yang menentukan angka (mis. `p_predicate_verb`) |
| `extraction_confidence` | `explicit` / `recovered` / `candidate` / `none` |
| `abstain_reason` | `truncated` / `no_number` / `refusal` / `unclosed_think` |
| `is_correct` | boolean |
| `finish_reason`, `truncated_by_metadata`, `truncated_by_content` | |
| `prompt_tokens`, `completion_tokens`, `reasoning_tokens`, `total_tokens`, `latency_sec`, `retry_count` | |
| `text_tail` | 200 karakter terakhir content |

Kolom `priority_hit` adalah tambahan penting: ia membuat rule pack per model bisa diaudit — kamu bisa langsung tahu Priority mana yang paling sering dipakai model mana, dan mana yang tidak pernah kena (kandidat dihapus).

`accuracy.csv` wajib memuat: `committed`, `commit_rate`, `abstain`, `missing`. Akurasi utama memakai penyebut seluruh unit.

---

## 5. IMPLEMENTASI — berkas baru

| Berkas | Peran |
|---|---|
| `src/MAIN_PARSING/GSM8K/_core/*.py` | §3.1 |
| `src/MAIN_PARSING/GSM8K/Religion_Persona/rules_*.py` | 6 rule pack |
| `src/MAIN_PARSING/GSM8K/Religion_Persona/run.py` | CLI per model |
| `src/MAIN_PARSING/GSM8K/_core/calc_accuracy.py` | pengganti bersih: hapus `is_qwen_report`, `--max_rows` default = semua baris |
| `src/MAIN_PARSING/GSM8K/_core/coverage.py` | `coverage.csv` + uji asosiasi abstain × persona |
| `src/MAIN_PARSING/GSM8K/_core/build_audit_sample.py` | `extraction_audit.csv` stratified |
| `src/MAIN_PARSING/GSM8K/_core/promote_candidates.py` | promosi RULE 5 setelah audit |
| `src/test/test_gsm8k_parsing.py` | V2, V5, V6, V7 |

Skrip lama `src/rebuild/rebuild_results_csv_*.py` **tidak dihapus dan tidak diubah** — dipakai sebagai pembanding regresi (V4) dan jejak audit. Ditandai deprecated di docstring.

Perintah per model:

```powershell
.venv\Scripts\python.exe -m src.MAIN_PARSING.GSM8K.Religion_Persona.run `
  --model mistralai_ministral-8b-2512 `
  --input_root "result\GMS8K\Religion_Persona" `
  --out_root   "result\GMS8K\reports"
```

---

## 6. CHECKLIST

**Fase 0 — Ukur dulu**
- [ ] Untuk tiap model: berapa persen unit terekstrak **hanya** dengan RULE 1–3 (aturan naskah). Angka baseline untuk Bab VI.
- [ ] Distribusi `finish_reason` dan content kosong pada data penuh (bukan sampel) per model × persona.
- [ ] Verifikasi 21.104 berkas/model utuh dan `question_id` unik per persona.

**Fase 1 — Bangun `_core/`**
- [ ] `numbers.py`, `preprocess.py`, `guards.py`, `skeleton.py`, `scoring.py`, `runner.py`
- [ ] Rule pack `core` (RULE 1–3 + Priority minimal) sebagai baseline §3.4
- [ ] V2, V5, V6, V7 lulus

**Fase 2 — Rule pack pertama: `ministral-8b`**
Dipilih karena rule pack existing-nya paling matang (11,3% tanpa penanda) → paling cepat memvalidasi rangka.
- [ ] Port `Priority` dari `rebuild_results_csv_ministrail.py` ke `rules_ministral_8b.py`, **tanpa** `TARGET_QID_RESCUE` (K2)
- [ ] Parse penuh 21.104 unit
- [ ] Regresi terhadap skrip lama — selisih harus persis sebesar efek pencabutan rescue, tidak lebih
- [ ] `coverage.csv`, V3

**Fase 3 — `llama-3.1-8b`**
Dipilih kedua karena worst case (99,3% tanpa penanda) → paling menekan rangka.
- [ ] Port Priority dari `rebuild_results_csv_llama.py`
- [ ] Kalau rangka perlu diubah agar muat, **ubah rangka**, lalu jalankan ulang Fase 2 untuk memastikan tidak regresi
- [ ] Audit manual 150 sampel stratified per `parse_source` × `priority_hit`

**Fase 4 — 4 model sisanya**
- [ ] `llama-3.3-70b`, `gpt-oss-20b`, `qwen3-next-80b-thinking`, `deepseek-r1-distill-32b`
- [ ] `deepseek` dan `qwen`: `strip_think_blocks=True`; `get_response_text` **tidak** boleh fallback ke `reasoning` (T3)
- [ ] Audit 50 sampel per model, fokus `priority_hit` yang jarang muncul
- [ ] Priority yang tidak pernah kena di data → hapus dari rule pack, catat

**Fase 5 — Core run + gabungan**
- [ ] Jalankan `--rules core` untuk 6 model → `accuracy_core.csv`
- [ ] Bandingkan peringkat 6 model: rule pack vs core (§3.4)
- [ ] `_combined/all_models_*.csv`
- [ ] Tabel akurasi + commit-rate 6 model × 16 persona

**Fase 6 — Konsekuensi K2**
- [ ] Parse ulang Gender Persona + neutral dengan arsitektur yang sama
- [ ] Laporkan selisih akurasi `ministral-8b` (dan `gemma` bila dipakai) sebelum/sesudah pencabutan forced-rescue

---

## 7. TESTING & VALIDASI

| # | Uji | Isi | Lulus bila |
|---|---|---|---|
| **V1** | Anotasi manual | 300 respons stratified per `parse_source` × `priority_hit` × model, min 10/sel. Anotator tidak melihat label extractor maupun GT. Kuota rule-of-three: aturan eksplisit ≥ 50 hit, recovered ≥ 100 hit. | precision ≥ 98% (`explicit`), ≥ 95% (`recovered`) |
| **V2** | **No-gold-peek** | Parse 2×: JSON asli vs salinan dengan `input.ground_truth` dikosongkan | seluruh `extracted_answer` identik |
| **V3** | Differential measurement error | `answer_status` × `persona`, chi-square 16×3 per model, Bonferroni untuk 6 model | tidak signifikan; kalau signifikan → wajib dilaporkan sebagai keterbatasan model itu |
| **V4** | Regresi vs parser lama | Diff per unit, dikelompokkan: (a) rescue dicabut, (b) fallback reasoning dicabut, (c) aturan berubah | tiap kelompok punya penjelasan tertulis |
| **V5** | Unit test | Bibit dari `UNIT_TESTS` di `rebuilt_Result_csv_distillR1.py:355–367` + kasus Religion: salam penutup persona, `$18` tanpa penanda, `\boxed` ganda dipisah "or", angka ber-hubung, baris tabel, jam+menit | semua hijau |
| **V6** | Larangan literal QID | grep `gsm8k_\d{5}` di seluruh `rules_*.py` dan `_core/` | 0 hit |
| **V7** | **Rule pack persona-blind** | grep nama persona (`islam`, `christian`, `atheist`, `jewish`, `explicit`, `implicit`) di seluruh `rules_*.py`; plus jalankan rule pack pada 16 persona dan pastikan jalur kode yang dipakai tidak bergantung nama folder | 0 hit; A2 terjaga |

V2, V6, dan V7 adalah uji otomatis yang harus jalan tiap kali rule pack diubah. Ketiganya murah dan menjaga tiga klaim yang paling mudah diserang penguji.

---

## 8. INTEGRASI

```
Fase 0 (ukur)
   → Fase 1 (_core + rule pack "core" + V2/V5/V6/V7)
      → Fase 2 (ministral-8b — validasi rangka)
         → Fase 3 (llama-3.1-8b — tekan rangka; kalau rangka berubah, ulang Fase 2)
            → Fase 4 (4 model sisanya)
               → Fase 5 (core run + _combined + sensitivitas §3.4)
                  → Fase 6 (re-parse Gender + neutral)
                     → penanda linguistik (dokumen terpisah)
```

---

## 9. Yang harus diubah di naskah TA

| # | Lokasi | Perubahan |
|---|---|---|
| N1 | Bab IV 1283–1285 | Daftar penanda saat ini tidak cukup (bukti §2.1). Tambahkan tahap pemulihan berbasis kalimat kesimpulan dan laporkan persentase pemulihan per model. Kalau tidak, naskah tidak menggambarkan implementasi. |
| N2 | Bab IV, subbab baru | **Nyatakan terus terang bahwa aturan ekstraksi disusun per model**, dengan §2.1 sebagai justifikasi empiris, dan §3.4 sebagai pembelaannya. Ini jauh lebih kuat daripada menulis "satu aturan untuk semua" yang tidak sesuai kode. |
| N3 | Bab IV | Cantumkan bahwa `normalize_number`, definisi benar/salah, dan penyebut akurasi identik untuk semua model — yang berbeda hanya tingkat Priority. |
| N4 | Bab VI | Prompt tidak meminta format jawaban (§2.2) → keterbatasan. |
| N5 | Bab VI | Gender vs Religion beda template dan `max_tokens` (§2.6) → tidak dibandingkan langsung. |
| N6 | Bab IV/VI | Setiap tabel akurasi berdampingan dengan commit-rate. |
| N7 | Bab VI | Hasil §3.4: apakah peringkat model stabil antara rule pack dan core. |

---

## 10. Sambungan ke tahap penanda linguistik

`results.csv` sudah menyimpan yang akan dibutuhkan supaya parsing tidak perlu diulang:

- `response_content_full` utuh — Bab IV 1390 menuntut penanda diukur dari `content` yang dilihat pengguna;
- `reasoning_present`, `think_block_stripped` — Bab IV 1391–1395: kalau reasoning tidak bisa dipisah dengan aturan yang sama, respons dicatat **tidak tersedia** untuk penanda berbasis teks;
- respons warmup tetap di JSON dan tidak masuk `results.csv` — Bab IV 1380–1383;
- `parse_source` + `priority_hit` — memungkinkan mengecek apakah gaya penutup respons (yang memengaruhi penanda) berkorelasi dengan persona.

---

## 11. Risiko

| Risiko | Mitigasi |
|---|---|
| Rule pack per model menaikkan akurasi tidak merata → perbandingan antar model bias | §3.4: laporkan `accuracy_core.csv` berdampingan. Ini syarat, bukan opsi. |
| Rule pack diam-diam jadi persona-dependent saat di-tuning | V7 otomatis tiap perubahan. Ini risiko terbesar terhadap validitas temuan. |
| Tuning rule pack sambil melihat akurasi = overfitting ke label | Tuning hanya boleh melihat **teks respons**, bukan tabel akurasi. Kalau akurasi dipakai sebagai sinyal tuning, itu gold-peek tidak langsung. Catat urutan kerjanya di log. |
| Coverage `llama-3.1-8b` tetap rendah | Naikkan lewat RULE 5 **hanya** setelah audit. Kalau tetap rendah, itu temuan tentang model. |
| Re-parse Gender menurunkan angka yang sudah "beres" | Sudah diputuskan (K2). Sediakan tabel sebelum/sesudah. |
| 126.624 JSON lambat | Streaming per persona, CSV inkremental. Estimasi 10–20 menit/model. |
