# Audit Linguistic Marker ETHICS Gender Persona v1.5

Status: `ethics-gender-lm-v1.5.0-candidate`. Bundle ini adalah hasil yang
direkomendasikan untuk audit manual, tetapi belum boleh diperlakukan sebagai
hasil final inferensial sebelum precision/recall dinilai manusia.

## SPEC

Audit menggunakan tiga sumber:

- CSV canonical gabungan
  `result/benchmarks/ethics/reports/Gender_Persona/_combined/parser_refinement_20260820/all_models_results_full.csv`;
- bundle kandidat awal
  `result/benchmarks/ethics/reports/Gender_Persona/linguistic_markers_gender_candidate_20260821/`;
- referensi eksperimen lama
  `docs/ref/linguistic marker yang udah aku coba.zip`.

CSV canonical berisi 81.756 respons: 6 model × 9 persona × 1.514 pertanyaan.
Seluruh 54 kondisi memiliki tepat 1.514 respons, tidak ada duplicate response
key, dan SHA-256 input adalah
`9a442ffeda5b3308b21fd9c398ab2633e3ad835d41346f5fc1d4ed302b5ee5a4`.

Tujuan operasionalnya bukan memasukkan setiap kata corpus ke marker. Seluruh
respons harus dibaca, sedangkan marker hanya aktif jika konstruk linguistiknya
jelas. Bare `man`, `woman`, atau pronoun dari isi skenario tidak otomatis
dianggap persona uptake.

## DESIGN

Registry v1.5 memuat 41 marker yang dipakai identik pada semua model/persona:

- 15 marker `general_ethics` yang kompatibel dengan surface pipeline bersama;
- 15 marker `ethics_specific` untuk stance, connective, framing moral, affect,
  evaluasi moral positif/negatif, dan respons committed/conditional/refusal;
- 11 marker `gender_specific` untuk persona acknowledgment, direct address,
  first/second-person identity, conditional identity, perspective adoption,
  relevance denial, identity discourse/neutralization, gender-role framing,
  dan gendered normative evaluation.

Guardrail desain:

- hanya visible response yang dianalisis; hidden reasoning tidak menjadi fallback;
- respons abstain yang memiliki teks tetap dianalisis;
- missing dan technical-only menjadi NA, bukan zero;
- count, density per 100 kata, dan presence disimpan per marker;
- keluarga marker tidak dijumlahkan menjadi composite score;
- moral-polarity dan response-stance tidak mengubah `answer_status` parser;
- scenario-term overlap tersedia untuk audit;
- inferensi dinonaktifkan sampai audit manual disetujui.

## IMPLEMENTATION

Perubahan utama berada di:

- `src/MAIN_PARSING/LINGUISTIC_MARKERS/ethics_registry.py` untuk registry
  discourse ETHICS;
- `src/MAIN_PARSING/LINGUISTIC_MARKERS/gender_registry.py` untuk registry
  gabungan dan pola gender berbasis corpus;
- `src/MAIN_PARSING/LINGUISTIC_MARKERS/surface.py` untuk normalisasi quote
  tipografis pada measured surface;
- `src/MAIN_PARSING/LINGUISTIC_MARKERS/gender_dataset_audit.py` untuk full-run,
  polarity sequence, audit sampling, provenance, dan output bundle;
- `src/MAIN_PARSING/LINGUISTIC_MARKERS/tests/` untuk positive/negative context tests.

Extractor mengompakkan hasil per 2.500 respons agar full corpus tidak menumpuk
81.756 dictionary Python di memori. Output directory yang sudah berisi file
ditolak, sehingga bundle versi lama tidak tertimpa.

Referensi ZIP dipakai sebagai recall-oriented probe. Pola luas seperti bare
`must`, `right`, `result`, `but`, `may`, atau `could` tidak dipromosikan begitu
saja karena ambigu. Pola yang dipromosikan dipersempit dan diberi exclusion
rule berdasarkan konteks corpus.

## CHECKLIST

- [x] Seluruh 81.756 response key diproses dan identik dengan input.
- [x] Seluruh 54 model-persona condition lengkap dan seimbang.
- [x] Tidak ada duplicate key.
- [x] Terdapat 41 marker unik dan seluruhnya mempunyai minimal satu hit.
- [x] Semua kolom count/density/presence tersedia.
- [x] Density cocok dengan `100 × count / word_count`.
- [x] Missing/technical-only seluruhnya NA pada kolom marker.
- [x] Available text tidak memiliki NA pada kolom marker.
- [x] Seluruh 50 open gender forms sudah berada dalam closed lexicon.
- [x] Sampel precision dan blind recall terbentuk tanpa label otomatis.
- [x] Inferensi tetap nonaktif.
- [ ] Precision audit diselesaikan dua reviewer.
- [ ] False-negative/recall audit diselesaikan dua reviewer.
- [ ] Inter-rater agreement dihitung.
- [ ] Registry dipromosikan dari candidate menjadi validated.

## TESTING

Seluruh 101 unit/regression test lolos. Test mencakup visible/hidden surface,
missing dan technical-only, quote tipografis, positive/negative regex, ETHICS
discourse, gender context, exact key behavior, loader, statistics, regional,
religion, serta pipeline bersama.

Full bundle v1.5 juga lolos pemeriksaan berikut:

- 81.756 input rows = 81.756 response-level rows;
- 0 source duplicate dan 0 output duplicate;
- exact key-set equality;
- input SHA-256 cocok dengan manifest;
- 80.840 teks `available`, 544 `missing`, dan 372 `technical_only`;
- 18.521 respons berstatus abstain tetap tercakup;
- 41/41 marker memiliki hit dan domain presence hanya 0/1/NA;
- 3.755 occurrence disiapkan untuk precision audit;
- 162 respons (3 × 54 kondisi) disiapkan untuk blind recall audit;
- 930 perubahan polaritas awal-ke-akhir tercatat sebagai diagnostic sequence;
- 0 dari 50 open forms berada di luar closed lexicon.

Audit kualitatif v1.4 menemukan false positive pada `gender_role_framing`:
2.383 occurrence di 1.957 respons berupa `societal norms`, `societal
expectations`, `harmful stereotypes`, dan bentuk sejenis tanpa anchor gender.
V1.5 mewajibkan anchor `gender/gendered`; pemeriksaan full output menunjukkan
0 occurrence tanpa anchor tersebut. Akibat koreksi precision ini, candidate
coverage turun secara benar dari 55,63% pada v1.4 menjadi 53,62% pada v1.5.

## INTEGRATION

Bundle yang direkomendasikan:

`result/benchmarks/ethics/reports/Gender_Persona/linguistic_markers_gender_candidate_20260821_v1_5_0/`

Angka utama bundle:

- 8.502 broad gender candidate contexts ditemukan;
- 4.559 konteks (53,62%) ditangkap minimal satu registered gender marker;
- 3.943 konteks tidak dipromosikan otomatis;
- dari residual, 1.980 memiliki identity term yang juga ada di skenario dan
  1.315 memakai konstruksi third-person `the man/the woman/...`;
- residual eksplisit `user's gender` = 0 dan `you are a man/woman` = 0.

Coverage candidate bukan recall final dan bukan target 100%. Baseline v1.0
menangkap 2.549/8.502 (29,98%), v1.2 4.237/8.502 (49,84%), dan v1.5
4.559/8.502 (53,62%) setelah false-positive v1.4 dibuang. Residual teratas
adalah third-person scenario language yang memang tidak boleh otomatis dibaca
sebagai persona uptake.

File utama untuk tahap berikutnya:

- `marker_registry.csv`: definisi, pattern, version, exclusion rule, provenance;
- `response_level_markers.csv.gz`: satu observasi per respons;
- `marker_prevalence.csv`: descriptives per model/persona/subset/status;
- `gender_candidate_contexts.csv`: diagnostic broad contexts dan capture flag;
- `gender_unmatched_context_audit.csv`: sampel residual untuk false-negative review;
- `audit_marker_hits.csv`: occurrence-level precision audit;
- `audit_recall_responses.csv`: blind response-level recall audit;
- `manifest.json`: input hash, versi, konfigurasi, guardrail, dan daftar output.

