# Handoff GPT-5.6 Sol — Perbaikan Parsing GSM8K Religion Persona

Tanggal audit: 13 Agustus 2026  
Workspace: D:\TugasAkhir  
Repo naskah: D:\ProposalTA-18222008

## Pesan yang dikirim ke GPT-5.6 Sol

Salin pesan ini ke terminal lain:

> Baca seluruh D:\TugasAkhir\docs\handoff_gpt56sol_perbaikan_parsing_gsm8k.md dan kerjakan instruksinya secara berurutan. Scope aktif hanya pilot meta-llama_llama-3.3-70b-instruct. Jangan menjalankan --all dan jangan memproses lima model lain. Mulai dari tindakan darurat, perbaiki parser dan output isolation, jalankan test, lalu re-parse model dan core Llama 3.3 70B secara lokal ke reports_parser_v2. Jangan memanggil API. Berhenti setelah pilot 21.104 unit tervalidasi dan serahkan hasil kepada pengguna.

## Scope aktif — hanya Llama 3.3 70B

Instruksi bagian ini mengalahkan instruksi perluasan enam model di bagian lain.

- Model: meta-llama_llama-3.3-70b-instruct
- Raw JSON: lengkap 21.104 unit
- Persona: 16
- Unit per persona: 1.319
- Jalankan rule pack model dan core hanya untuk model ini
- Jangan memakai opsi --all
- Jangan menjalankan combine enam model
- Jangan menyentuh output lima model lain
- Jangan rerun API, termasuk 202 row finish_reason=error
- Buat manifest teknis Llama 70B saja dan minta persetujuan sebelum langkah API
- Setelah pilot lolos, berhenti dan laporkan; perluasan enam model menunggu instruksi baru

## Tujuan

1. Menghentikan core parser yang menimpa report utama.
2. Memisahkan output rule pack model dan core.
3. Memperbaiki bug parsing yang sudah direproduksi.
4. Menambahkan regression test.
5. Menjamin satu row untuk setiap unit dan tidak ada silent skip.
6. Re-parse raw JSON Llama 3.3 70B secara lokal ke output versi baru.
7. Membuat manifest gangguan teknis Llama 3.3 70B tanpa menjalankan API.
8. Menghasilkan audit pilot model dan core untuk satu model.

## 1. Tindakan darurat

Pada pemeriksaan terakhir, proses Python PID 8040 yang mulai pada 09:30:49 menjalankan core parse. Proses tersebut sudah menimpa:

- DeepSeek results.csv dan coverage.csv dengan rule pack core;
- Llama 3.1 results.csv secara inkremental;
- sementara accuracy.csv dan file split lama masih berasal dari model run.

Akibatnya direktori report lama sudah tidak konsisten. PID dapat berubah atau proses mungkin sudah selesai saat dokumen ini dibaca. Jangan menghentikan proses Python lain tanpa verifikasi.

Jalankan:

~~~powershell
Set-Location -LiteralPath 'D:\TugasAkhir'

Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
Get-Process -Name python -ErrorAction SilentlyContinue |
    Select-Object Id, StartTime, CPU, WorkingSet, Path |
    Format-Table -AutoSize

$suspectPid = 8040
Get-Process -Id $suspectPid -ErrorAction SilentlyContinue |
    Select-Object Id, StartTime, CPU, Responding |
    Format-List

$report = 'D:\TugasAkhir\result\GMS8K\reports\Religion_Persona\meta-llama_llama-3.1-8b-instruct\results.csv'
Get-Item -LiteralPath $report |
    Select-Object Length, LastWriteTime |
    Format-List

$env:PYTHONIOENCODING = 'utf-8'
@'
import csv
from pathlib import Path

p = Path(r'D:\TugasAkhir\result\GMS8K\reports\Religion_Persona\meta-llama_llama-3.1-8b-instruct\results.csv')
with p.open(encoding='utf-8-sig', newline='') as f:
    rows = list(csv.DictReader(f))
print('rows:', len(rows))
print('rule packs:', sorted({r.get('rule_pack') for r in rows}))
print('last rows:', [(r.get('persona'), r.get('question_id')) for r in rows[-5:]])
'@ | python -
~~~

Jika PID 8040 masih aktif, file terus berubah, dan rule_pack berisi core, hentikan:

~~~powershell
Stop-Process -Id 8040
~~~

Verifikasi file tidak berubah lagi. Jangan jalankan mode semua-model maupun core lama sebelum output isolation diperbaiki.

## 2. Baca konteks dan lindungi perubahan pengguna

Sebelum edit:

1. Baca D:\TugasAkhir\AGENT.md sampai selesai.
2. Baca D:\ProposalTA-18222008\AGENT.md sampai selesai.
3. Baca D:\TugasAkhir\docs\plan_parsing_gsm8k_religion_persona.md sampai selesai.
4. Baca D:\ProposalTA-18222008\Bab IV - Perancangan.tex baris 1168–1296.
5. Khususnya pahami baris 1189–1190, 1223–1244, dan 1267–1296.
6. Periksa perubahan lokal dan jangan merusaknya.

~~~powershell
Set-Location -LiteralPath 'D:\TugasAkhir'
git status --short --untracked-files=no
git diff --name-only
~~~

Worktree sangat kotor dan perubahan yang ada milik pengguna. Dilarang menjalankan git reset, git clean, checkout paksa, atau menghapus report/raw JSON.

## 3. Batas metodologis

Konfigurasi utama TA harus tetap:

- temperature = 0;
- max_tokens = 4096;
- satu respons per kombinasi model, soal, dan persona;
- hanya response.content yang dinilai;
- reasoning terpisah tidak boleh diimputasi;
- ekstraksi tidak boleh melihat ground truth;
- tidak boleh ada rescue per-QID;
- rule pack boleh berbeda per model, tetapi tidak boleh berbeda per persona;
- coverage selalu dilaporkan bersama accuracy.

### Kebijakan rerun

Bedakan:

1. **Re-parse lokal**
   - Wajib setelah kode diperbaiki.
   - Tidak memanggil API.
   - Dijalankan terhadap semua raw JSON.

2. **Rerun API teknis**
   - Hanya untuk timeout, DNS, connection failure, HTTP/server error, SSE/JSON transport error, atau provider finish_reason=error.
   - Gunakan model, messages, warmup, temperature, dan max_tokens yang sama.
   - Jangan dieksekusi sebelum manifest disetujui pengguna.

3. **finish_reason=length**
   - Jangan rerun API untuk hasil utama.
   - Semua row length pada audit benar-benar mencapai completion_tokens=4096.
   - Ini hasil konfigurasi eksperimen, bukan gangguan teknis.
   - Jawaban eksplisit yang sudah lengkap sebelum truncation boleh dipertimbangkan dengan rule ketat.
   - Recovery lemah tidak boleh auto-commit.

Max token 8192 hanya boleh menjadi sensitivity analysis terpisah, bukan pengganti hasil utama 4096.

## 4. Baseline sebelum report ditimpa

| Model | Baris | Committed | Abstain | Missing | Accuracy |
|---|---:|---:|---:|---:|---:|
| Qwen 80B | 21.104 | 20.707 | 57 | 340 | 94,2096% |
| DeepSeek R1 32B | 21.094 | 20.221 | 239 | 634 | 89,3619% |
| Llama 3.1 8B | 21.104 | 20.965 | 139 | 0 | 81,1126% |
| Llama 3.3 70B | 21.104 | 20.852 | 252 | 0 | 90,8690% |
| Ministral 8B | 21.104 | 20.959 | 145 | 0 | 87,5900% |
| GPT-OSS 20B | 21.104 | 19.808 | 1.237 | 59 | 86,7703% |

Ini hanya baseline audit, bukan angka final.

DeepSeek kehilangan QID berikut karena JSON Extra data:

- islam_explicit: 1268, 1296, 1310;
- islam_implicit_2: 1053, 1054;
- islam_implicit_3: 982, 1002;
- jewish_implicit_2: 1068;
- jewish_implicit_3: 883, 1051.

Raw JSON di result\GMS8K\Religion_Persona adalah sumber utama. Jangan mempercayai report lama setelah pukul 09:30.

## 5. Perbaikan wajib

### P0 — Pisahkan output model dan core

File utama:

- src\MAIN_PARSING\GSM8K\Religion_Persona\run.py
- src\MAIN_PARSING\GSM8K\_core\runner.py

Masalah:

- suffix core hanya diterapkan pada accuracy_core.csv;
- parse_model selalu membuka results.csv dengan mode write;
- coverage selalu ditulis ke coverage.csv;
- core run menimpa model run.

Target per model:

~~~text
results.csv
accuracy.csv
coverage.csv
results_<persona>.csv
results_core.csv
accuracy_core.csv
coverage_core.csv
~~~

Implementasikan:

- parse_model menerima results filename;
- write_coverage menerima filename;
- suffix core dipakai konsisten;
- model split hanya berasal dari results.csv;
- hasil ditulis atomic ke temporary file lalu os.replace;
- file final tidak boleh terlihat setengah jadi.

### P0 — Jangan silent skip JSON invalid

File:

- src\MAIN_PARSING\GSM8K\_core\runner.py

Masalah sekarang: exception json.load dicetak sebagai SKIP lalu unit hilang dari CSV dan denominator.

Target:

- validasi 16 persona × 1.319 unit;
- invalid JSON mendapat row/status audit eksplisit, bukan menghilang;
- simpan input_json_valid, error_class, dan error_message aman;
- QID boleh diperoleh dari filename untuk pelaporan;
- jangan memakai GT untuk memilih jawaban;
- denominator tidak boleh berubah diam-diam;
- keputusan data unavailable harus konsisten dengan plan dan Bab IV.

### P0 — Perlakukan error dan truncation dengan benar

File:

- src\MAIN_PARSING\GSM8K\_core\preprocess.py
- src\MAIN_PARSING\GSM8K\_core\skeleton.py
- src\MAIN_PARSING\GSM8K\_core\runner.py

Audit non-stop committed:

| Model | Committed | Cocok GT |
|---|---:|---:|
| DeepSeek | 5 | 0 |
| Llama 3.1 | 112 | 3 |
| Llama 3.3 | 185 | 5 |
| Ministral | 28 | 7 |
| GPT-OSS | 11 | 2 |
| Qwen | 164 | 40 |

Kebijakan:

- error: jangan commit partial content; tandai technical failure;
- content_filter: jangan commit;
- length:
  - hanya explicit source tervalidasi seperti boxed atau strong Answer/Final Answer;
  - multiple/conflicting final answers menjadi abstain truncated;
  - Markdown heading hashes bukan jawaban;
  - bold_4a, bold_4d, currency_line, conclusion umum, final_sentence, dan last_lines tidak boleh auto-commit.

### P0 — Perbaiki hashes

File:

- src\MAIN_PARSING\GSM8K\_core\skeleton.py

Qwen memakai source hashes 20 kali dan seluruhnya salah. Model sering menulis heading:

~~~text
#### 1. Daily consumption
#### 2. Daily consumption
~~~

Regex lama juga menghasilkan:

~~~text
#### 18. Answer follows.  -> 1
~~~

Target:

- heading Markdown tidak dianggap marker jawaban;
- numeric boundary tidak boleh mengubah 18 menjadi 1;
- heading tidak boleh mengalahkan boxed/final answer yang muncul kemudian.

### P0 — Perbaiki resolver Answer clause

Contoh nyata:

~~~text
Answer: It will take John 2 weeks to save $80.
~~~

Hasil lama 80; jawaban eksplisit 2.

~~~text
Answer: 448 mL of medicine would be in 8 doses.
~~~

Hasil Qwen/DeepSeek lama 8; jawaban eksplisit 448.

Masalah berada pada urutan currency dan predicate di skeleton.py/priorities.py. Jangan mengganti semuanya menjadi first-number secara buta karena audit menunjukkan itu juga dapat menyebabkan regresi. Buat resolver answer clause dan audit tanpa GT.

### P1 — Perbaiki tanda negatif

File:

- src\MAIN_PARSING\GSM8K\_core\numbers.py

Reproduksi lama:

~~~text
find_numbers("- 7 packs")       -> [-7]
find_numbers("56 - 54 = 2")     -> [56, -54, 2]
~~~

Target:

- bullet tidak menjadi negatif;
- operator subtraction tidak menjadi unary minus;
- nilai negatif sah seperti -12 tetap didukung.

### P1 — Lindungi singkatan pada splitter

File:

- src\MAIN_PARSING\GSM8K\_core\skeleton.py

Teks:

~~~text
**Answer:** Mr. Ruther originally had 32 hectares.
~~~

Tidak boleh dipecah setelah Mr. Terapkan proteksi umum untuk Mr., Mrs., dan singkatan relevan tanpa rule persona/QID.

### P1 — Recovery bold/currency terlalu permisif

GT-match diagnostik bold_4d:

| DeepSeek | Llama 3.1 | Llama 3.3 | Ministral | GPT-OSS | Qwen |
|---:|---:|---:|---:|---:|---:|
| 26/142 | 0/15 | 0/47 | 77/438 | 29/128 | 4/38 |

Rule sering mengambil Step 1, 30% commission, bullet, atau nilai antara. Sampai audit manual lolos:

- jadikan bold_4d candidate-only atau nonaktif;
- batasi bold_4a dan currency_line;
- fallback last-lines tetap candidate sesuai A5;
- jangan promosi dengan melihat GT.

### P1 — Tambahkan provenance kegagalan

File:

- src\MAIN_PARSING\GSM8K\_core\schema.py
- src\MAIN_PARSING\GSM8K\_core\runner.py

Kolom minimum:

- response_success;
- response_error_present;
- error_class;
- input_json_valid;
- technical_failure;
- content_empty;
- reasoning_present;
- finish_reason;
- parser_version atau run_tag.

Jangan menyimpan secret.

Klasifikasi awal:

- DeepSeek missing: 492 response.success=False, 142 sukses tetapi content kosong;
- GPT-OSS missing: 58 sukses dengan reasoning terpisah dan content kosong, 1 gagal API;
- Qwen missing: 322 sukses dengan reasoning terpisah dan content kosong, 18 gagal API.

Reasoning tetap tidak boleh dipakai sebagai content.

## 6. Regression test wajib

Perluas src\test\test_gsm8k_parsing.py. Test sekarang lulus tetapi belum mencakup bug nyata. V7 dinamis sekitar baris 175 juga vacuous karena set hanya berisi satu pemanggilan.

Tambahkan test:

1. Hash 18 tidak menjadi 1.
2. Heading hashes tidak mengalahkan boxed/final answer.
3. Answer duration memilih 2, bukan 80.
4. Answer quantity memilih 448, bukan 8.
5. Bullet dan subtraction tidak menghasilkan negatif palsu.
6. Literal negatif tetap bekerja.
7. Mr./Mrs. tidak memutus answer clause.
8. finish_reason=error tidak committed.
9. content_filter tidak committed.
10. length + recovery lemah menjadi abstain truncated.
11. length + satu final answer eksplisit mengikuti kebijakan ketat.
12. Bold step label tidak auto-commit.
13. Invalid JSON tidak silent skip.
14. Core run tidak mengubah results.csv model.
15. Atomic output tidak terlihat parsial.
16. V2 no-gold-peek tetap lulus.
17. V6 no-QID-rescue tetap lulus.
18. V7 benar-benar menguji persona independence.

Jalankan:

~~~powershell
Set-Location -LiteralPath 'D:\TugasAkhir'
$env:PYTHONIOENCODING = 'utf-8'
python src\test\test_gsm8k_parsing.py
~~~

## 7. Output baru

Jangan tulis ke:

~~~text
D:\TugasAkhir\result\GMS8K\reports\Religion_Persona
~~~

Gunakan:

~~~text
D:\TugasAkhir\result\GMS8K\reports_parser_v2\Religion_Persona
~~~

### Smoke test

Jalankan setelah test lulus:

~~~powershell
Set-Location -LiteralPath 'D:\TugasAkhir'
$env:PYTHONIOENCODING = 'utf-8'
python -m src.MAIN_PARSING.GSM8K.Religion_Persona.run --model meta-llama_llama-3.3-70b-instruct --rules model --limit 20 --out_root 'result\GMS8K\reports_parser_v2_smoke'
python -m src.MAIN_PARSING.GSM8K.Religion_Persona.run --model meta-llama_llama-3.3-70b-instruct --rules core --limit 20 --out_root 'result\GMS8K\reports_parser_v2_smoke'

$smoke = 'D:\TugasAkhir\result\GMS8K\reports_parser_v2_smoke\Religion_Persona\meta-llama_llama-3.3-70b-instruct'
Get-ChildItem -LiteralPath $smoke -File |
    Select-Object Name, Length, LastWriteTime |
    Sort-Object Name |
    Format-Table -AutoSize
~~~

Pastikan results.csv dan results_core.csv keduanya ada dan tidak saling menimpa.

### Re-parse pilot Llama 3.3 70B

Jalankan hanya setelah smoke test dan seluruh test lulus:

~~~powershell
Set-Location -LiteralPath 'D:\TugasAkhir'
$env:PYTHONIOENCODING = 'utf-8'
python -m src.MAIN_PARSING.GSM8K.Religion_Persona.run --model meta-llama_llama-3.3-70b-instruct --rules model --out_root 'result\GMS8K\reports_parser_v2'
python -m src.MAIN_PARSING.GSM8K.Religion_Persona.run --model meta-llama_llama-3.3-70b-instruct --rules core --out_root 'result\GMS8K\reports_parser_v2'
~~~

Ini parsing lokal. Pastikan tidak ada kode jalur ini yang memanggil OpenRouter. Jangan jalankan combine dan jangan lanjut ke model lain.

## 8. Validasi v2

Acceptance criteria:

1. Hanya model meta-llama_llama-3.3-70b-instruct yang diproses.
2. Model merepresentasikan seluruh 21.104 unit secara eksplisit.
3. Ada 16 persona × 1.319 QID unik.
4. Tidak ada duplicate persona–QID.
5. Tidak ada exception JSON yang hanya dicetak lalu dilewati.
6. is_correct konsisten dengan extracted answer dan GT, tetapi fungsi ekstraksi tidak melihat GT.
7. accuracy.csv dan coverage.csv cocok dengan results.csv.
8. results_core.csv, accuracy_core.csv, coverage_core.csv tersedia untuk Llama 3.3 70B.
9. results.csv hanya memakai rule pack model.
10. results_core.csv hanya memakai rule pack core.
11. Tidak ada low-confidence recovery committed pada error/content_filter.
12. Kebijakan length dilaporkan per model × persona.
13. Combined tidak dibuat pada pilot ini.

Pemeriksaan ringkas:

~~~powershell
$env:PYTHONIOENCODING = 'utf-8'
@'
import csv
from collections import Counter
from pathlib import Path

d = Path(r'D:\TugasAkhir\result\GMS8K\reports_parser_v2\Religion_Persona\meta-llama_llama-3.3-70b-instruct')
for filename in ('results.csv', 'results_core.csv'):
    p = d / filename
    if not p.exists():
        print(filename, 'MISSING')
        continue
    with p.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    keys = [(r['persona'], r['question_id']) for r in rows]
    counts = Counter(r['persona'] for r in rows)
    print(
        filename,
        'rows=', len(rows),
        'unique=', len(set(keys)),
        'personas=', len(counts),
        'bad counts=', {k: v for k, v in counts.items() if v != 1319},
        'rules=', sorted({r['rule_pack'] for r in rows}),
    )
'@ | python -
~~~

## 9. Manifest rerun API teknis

Buat tanpa menjalankan API:

~~~text
result\GMS8K\audit\religion_persona_technical_rerun_manifest.csv
~~~

Kolom minimum:

- model;
- persona;
- question_id;
- raw_file;
- failure_class;
- response_success;
- finish_reason;
- provider_error_code;
- provider_error_message;
- retry_count_existing;
- eligible_for_technical_rerun;
- reason.

Scope manifest saat ini hanya:

- Llama 3.3: 202 finish_reason=error, mayoritas Internal Server Error 500;
- 23 finish_reason=length tidak eligible untuk rerun;
- jangan memindai atau memproses model lain pada fase pilot.

Untuk JSON Extra data:

- jangan overwrite;
- gunakan JSONDecoder.raw_decode hanya untuk audit/salvage;
- catat trailing content;
- validasi object pertama;
- klasifikasikan failure API, content kosong, atau response sukses;
- jangan otomatis memasukkan seluruh sepuluh file ke rerun.

Berhenti setelah manifest dan estimasi request selesai. Minta persetujuan pengguna sebelum request API.

## 10. Audit manual dan anti-overfitting

1. Buat extraction_audit.csv yang distratifikasi berdasarkan model, parse_source, priority_hit, confidence, finish_reason, dan persona.
2. Anotator melihat response text dan extracted span tanpa GT.
3. Jangan tune rule menggunakan aggregate accuracy.
4. Kasus bug yang sudah diketahui boleh menjadi regression test.
5. Fallback candidate hanya dipromosikan lewat prosedur audit plan.
6. Hitung ulang asosiasi status parsing × persona setelah v2.

Audit lama menunjukkan asosiasi signifikan untuk DeepSeek, Llama 3.1, Llama 3.3, dan GPT-OSS. Ministral tidak signifikan setelah Bonferroni; Qwen tidak signifikan. Jangan salin angka lama sebagai hasil final.

## 11. Deliverable

Serahkan:

1. Daftar file yang diubah.
2. Akar masalah dan perbaikan setiap bug.
3. Output seluruh regression test.
4. Bukti smoke model/core tidak overwrite.
5. Report v2 Llama 3.3 70B.
6. Tabel integritas 21.104 unit untuk model dan core.
7. Accuracy, accuracy_core, coverage, dan coverage_core Llama 3.3 70B.
8. Konfirmasi bahwa Combined tidak dijalankan.
9. extraction_audit.csv pilot atau generatornya.
10. Manifest 202 error teknis tanpa mengeksekusi API.
11. Perbedaan angka lama vs v2 beserta penyebab.
12. Catatan revisi yang dibutuhkan pada Bab IV/Bab VI.

## Larangan keras

- Jangan menaikkan max_tokens pada hasil utama.
- Jangan rerun row length untuk memperoleh jawaban lebih baik.
- Jangan fallback dari content ke reasoning.
- Jangan memakai GT untuk memilih kandidat.
- Jangan membuat rule persona atau QID.
- Jangan menulis ke report lama.
- Jangan menjalankan core sebelum output isolation benar.
- Jangan memanggil API tanpa persetujuan.
- Jangan menghapus raw JSON, CSV lama, atau perubahan pengguna.
- Jangan lanjut ke lima model lain setelah pilot Llama 3.3 70B selesai.
