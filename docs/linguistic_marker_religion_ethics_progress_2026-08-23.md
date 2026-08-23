# Progres Penanda Linguistik Religion Persona pada ETHICS — 23 Agustus 2026

Status: full extraction v1.0.0 selesai; hasil masih `candidate` dan belum
menjalankan inferensi statistik.

## FASE 1 — SPEC

Tujuan pekerjaan ini adalah menjalankan registry penanda linguistik agama yang
dibuat pada sesi sebelumnya terhadap seluruh data kanonik ETHICS Religion
Persona. Unit analisis tetap satu respons model untuk satu pertanyaan. Teks yang
diukur hanya penjelasan yang terlihat oleh pengguna; hidden reasoning, ground
truth, dan baris jawaban teknis tidak menjadi permukaan marker.

Masukan utama:

`result/ETHICS/reports/Religion_Persona/_combined/all_models_results_full.csv`

Kriteria selesai untuk tahap ini adalah seluruh respons diproses, baris
`abstain` tetap disertakan, `missing` tidak diubah menjadi nol, count/presence/
density tersedia, serta bundle audit dan manifest dapat dibuka ulang.

## FASE 2 — DESIGN PLAN

- Satu registry yang sama diterapkan kepada 6 model dan 17 skenario.
- Empat identitas agama tidak digabung: Islam, Christian, Jewish, dan atheist.
- Explicit, I1, I2, I3, dan Neutral tetap menjadi skenario terpisah.
- Registry memisahkan identity reference, devotional reference, scripture,
  practice, named concept, formula, direct address, perspective adoption,
  moral authority, secular-humanist framing, pluralism, irrelevance, blessing,
  dan respect acknowledgement. Tidak dibuat skor agama gabungan.
- Full extraction bersifat deskriptif. Uji statistik baru boleh dijalankan
  setelah marker lolos audit kontekstual.
- Output memakai folder baru dan tidak menimpa hasil lama.

## FASE 3 — IMPLEMENTATION

Kode utama:

- `src/MAIN_PARSING/LINGUISTIC_MARKERS/ethics_religion_registry.py`;
- `src/MAIN_PARSING/LINGUISTIC_MARKERS/religion_ethics_dataset_audit.py`;
- `src/MAIN_PARSING/LINGUISTIC_MARKERS/extractor.py`;
- `src/MAIN_PARSING/LINGUISTIC_MARKERS/tests/test_ethics_religion_registry.py`.

Versi registry:

`ethics-religion-lm-v1.0.0-candidate`

Folder hasil:

`result/ETHICS/reports/Religion_Persona/linguistic_markers_religion_candidate_20260823_v1_0_0/`

Runner kini memiliki dua pengaman tambahan. Pertama, DataFrame response-level
dilepaskan sebelum openpyxl membuat workbook agar kebutuhan memori tidak
bertumpuk. Kedua, dependency `openpyxl` diperiksa sebelum full scan. Opsi
`--finalize-existing` dapat melengkapi workbook dan manifest dari CSV write-once
yang sudah selesai tanpa mengulang ekstraksi dan tanpa menimpa artefak yang ada.

Open-anchor recall screen juga sudah diperbaiki pada kode agar memakai lexical
surface yang sama dengan marker. File screen dalam bundle v1.0.0 dibuat sebelum
perbaikan ini, sehingga beberapa unmatched candidate dapat berasal dari baris
teknis yang memang tidak diukur. Marker count/presence/density response-level
tidak terpengaruh oleh masalah tersebut.

## FASE 4 — CHECKLIST

- [x] Sumber kanonik dan SHA-256 diverifikasi.
- [x] Seluruh 154.428 kunci respons diproses.
- [x] Tidak ada duplikasi `model × persona × question_id`.
- [x] Semua 46.060 respons `abstain` tetap disertakan.
- [x] Sebanyak 2.381 respons `missing` menjadi `NA`, bukan nol.
- [x] Count, presence, dan density per 100 kata disimpan.
- [x] Workbook audit dan manifest selesai dibuat.
- [x] Tidak ada inferensi statistik pada registry yang belum diaudit.
- [ ] Precision sample 1.081 occurrence belum diberi label reviewer.
- [ ] Blind recall sample 306 respons belum diberi label reviewer.
- [ ] Dua reviewer manusia independen belum menyelesaikan audit.
- [ ] Surface-aligned open-anchor screen belum dibangun ulang sebagai versi
  output baru.

## FASE 5 — TESTING DAN HASIL FULL DATA

### Integritas sumber

SHA-256 input:

`c10a998ce65073e4adbd4a77e69cc1105540d0d600ceff9e8062d375c299533a`

Nilai tersebut identik dengan
`result/ETHICS/reports/Religion_Persona/_combined/all_models_results_full_manifest.json`.

| Ukuran | Nilai |
|---|---:|
| Respons unik | 154.428 |
| Model | 6 |
| Skenario | 17 |
| Pertanyaan unik | 1.514 |
| Duplikasi kunci | 0 |
| Respons dengan marker surface | 151.022 |
| Respons teknis saja | 1.025 |
| Abstain | 46.060 |
| Missing | 2.381 |
| Marker total | 50 |
| Marker khusus agama | 20 |

Workbook berhasil dibuka ulang dengan tujuh sheet. Jumlah baris pentingnya:
precision audit 1.081, blind recall 306, unmatched-anchor audit 100, marker
prevalence 49.600, dan data status 1.113 (di luar header).

### Pemeriksaan response-level

`response_level_markers.csv.gz` berisi tepat 154.428 baris, 6 model, 17
persona, dan 1.514 pertanyaan. Tidak ada kunci respons duplikat. Sebanyak
45.050 respons memuat setidaknya satu dari 20 marker agama kandidat.

Jumlah respons bertanda untuk beberapa marker utama:

| Marker | Respons bertanda |
|---|---:|
| `religion_moral_authority_framing` | 20.991 |
| `religion_scripture_reference` | 20.559 |
| `religion_generic_reference` | 18.458 |
| `religion_practice_community` | 16.236 |
| `religion_deity_invocation` | 13.521 |
| `religion_perspective_adoption` | 8.126 |
| `religion_formulaic_expression` | 797 |
| `religion_identity_irrelevance` | 499 |
| `religion_blessing_invocation` | 354 |
| `religion_persona_direct_address` | 301 |

Angka ini adalah jumlah marker terdeteksi, bukan efek kausal persona. Scenario
overlap dan perbedaan ketersediaan teks harus diperiksa sebelum interpretasi.

### Temuan deskriptif awal

Pada agregasi lintas model, perspective adoption paling tinggi pada
`christian_explicit` (43,43%), diikuti `islam_explicit` (19,75%),
`jewish_explicit` (8,15%), dan `atheist_explicit` (5,71%). Neutral hanya 0,03%.
Moral-authority framing juga tinggi pada `christian_explicit` (53,83%) dan
`islam_explicit` (50,92%), tetapi sangat heterogen antarmodel. Contohnya,
Ministral mencapai 91,7% pada Christian Explicit, sedangkan GPT-OSS hanya 4,1%.
Hasil ini belum boleh menjadi klaim Bab Hasil sebelum precision/recall audit.

### Kandidat recall dari full scan

Open-anchor screen menemukan bentuk sempit yang layak diaudit untuk v1.0.1:

- `insha'Allah` (12 respons);
- `halachically`, `halakhically`, dan `halakhah` (8 respons);
- `kavod habrit` (1 respons);
- `mitzvahs` dan variasi terkait (5 respons);
- `Quranic` (16 respons);
- konteks `commandment(s)` (132 respons).

Bentuk luas tidak boleh langsung ditambahkan. `belief/beliefs` memiliki 10.450
respons dan sering hanya merupakan disclaimer “I don't have personal beliefs”.
`fasting` mencampur puasa agama dengan puasa medis sebelum operasi. `creator`
mencampur sebutan Tuhan dengan pencipta karya, dan `lord` mencampur invokasi
agama dengan gelar seperti Lord Acton atau Lord Justice. `sacred` dan `holy`
juga memerlukan aturan konteks. Sesuai `AGENT.md`, kandidat tersebut harus
dibaca dan diaudit tanpa melihat ground truth sebelum promosi.

### Pengujian kode

- Test khusus ETHICS Religion: 16/16 lulus setelah perbaikan.
- Full suite final: 152/152 lulus di luar sandbox.
- Workbook `audit_manual_religion.xlsx` berhasil dibuka ulang dengan openpyxl.
- Manifest dan semua 17 file output tersedia.

Percobaan enam worker gagal sebelum menulis output karena `MemoryError` saat
enam proses Windows mengimpor pandas. Full run kemudian berhasil secara
sekuensial. Workbook awal gagal karena `.venv` tidak memiliki `openpyxl`; mode
recovery berhasil memakai `C:\Python314\python.exe`, yang memiliki dependency
tersebut.

## FASE 6 — INTEGRATION

Bundle v1.0.0 dapat dipakai sebagai dasar audit marker agama ETHICS dan untuk
merancang v1.0.1, tetapi belum sebagai sumber angka inferensial tesis. Analisis
berikutnya harus mempertahankan satu respons sebagai satu observasi, memisahkan
E/I1/I2/I3/Neutral, memakai pairing pada model–pertanyaan yang sama, melaporkan
effect size dan koreksi multiple comparison, serta membaca missing/technical-only
bersama prevalence marker.

Urutan berikutnya yang sah adalah membangun ulang surface-aligned anchor screen
ke folder baru, mengaudit kandidat sempit, memperbarui registry bila audit
lolos, menjalankan ulang full extraction ke versi baru, lalu menyelesaikan dua
review manusia sebelum inferensi statistik.
