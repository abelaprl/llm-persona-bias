# Audit linguistic marker ETHICS Regional Cultural Persona — Codex R1

Tanggal audit: 2026-08-22
Status: **kandidat; belum divalidasi dua penilai manusia independen**

## Kesimpulan

Registry final pada audit ini adalah
`ethics-regional-cultural-lm-v1.0.16-candidate`. Pipeline membaca seluruh
154.428 baris respons dari gabungan terbaru, memakai surface respons yang sama
untuk semua model dan kondisi, dan menghasilkan 23 marker regional-cultural
ditambah 30 marker ETHICS umum. Seluruh open-anchor form sudah diperiksa:
tidak ada form yang masih berlabel `POTENTIAL_MISS` pada codebook kandidat ini.

Klaim tersebut berarti cakupan terhadap inventori anchor audit sudah lengkap,
bukan bahwa semua kemungkinan makna budaya di bahasa alami sudah terbukti
tertangkap. Registry tetap candidate dan inferensi tetap dinonaktifkan sampai
audit dua manusia, penyelesaian disagreement, dan inter-rater agreement selesai.

## Pemilihan input terbaru

Input kanonik yang dipakai:

`result/benchmarks/ethics/reports/Regional_Cultural_Persona/_combined/all_models_results_full.csv`

- Dibuat 2026-08-22 00:45:46 +07:00; lebih baru daripada
  `parser_refinement_20260821/` dan
  `parser_refinement_audited_aita_20260821/`.
- SHA-256:
  `09f96bec32feafe3426c39526c1d9a80453d09ede2edb06a25e90fbddacc89a5`.
- Manifest sumber menetapkan enam CSV regional kanonik ditambah enam CSV
  neutral kanonik, tanpa reparsing atau rescoring jawaban.
- Validasi manifest: header identik, setiap kondisi memiliki 1.514 soal,
  question ID identik, tidak ada key duplikat, dan identitas model/persona
  cocok dengan sumber.

Folder `persona_publication_analysis_v3_accuracy_paired_holm_six_models/`
lebih baru secara waktu, tetapi merupakan hasil analisis turunan, bukan dataset
respons kanonik, sehingga tidak dipakai sebagai input marker.

## Profil dataset

- Respons unik: 154.428
- Model: 6
- Kondisi/persona: 17 (neutral; explicit dan implicit 1–3 untuk empat region)
- Pertanyaan/skenario: 1.514/1.514
- Teks linguistik yang dapat dianalisis: 151.827
- Abstain yang tetap disertakan: 40.570
- Missing: 1.823; marker disimpan sebagai `NA`, bukan nol
- Marker total: 53; marker regional-cultural: 23
- Pengukuran: raw count, response presence, dan density per 100 kata
- Inferensi: tidak dijalankan

Empat identitas regional yang dibaca dari desain persona adalah North American,
East Asian, South Asian, dan Sub-Saharan African. Seluruh surface respons
dipindai, bukan hanya respons committed atau baris yang parser anggap benar.
Ground truth dan hidden reasoning tidak pernah digunakan sebagai fallback teks.

## Registry regional-cultural

| Marker | Respons dengan hit | Occurrence |
|---|---:|---:|
| `culture_region_north_american` | 2.049 | 3.376 |
| `culture_region_east_asian` | 4.369 | 10.364 |
| `culture_region_south_asian` | 3.836 | 10.193 |
| `culture_region_sub_saharan_african` | 3.238 | 4.670 |
| `culture_geocultural_broader_reference` | 9.878 | 17.644 |
| `culture_general_reference` | 32.977 | 97.859 |
| `culture_meta_framing` | 10.368 | 13.442 |
| `culture_relativist_framing` | 2.348 | 2.451 |
| `regional_persona_direct_address` | 209 | 211 |
| `regional_perspective_adoption` | 533 | 572 |
| `culture_collectivist_communal_orientation` | 3.936 | 6.178 |
| `culture_individual_autonomy_orientation` | 3.055 | 4.216 |
| `culture_family_obligation_orientation` | 3.166 | 4.425 |
| `culture_community_obligation_orientation` | 2.137 | 3.641 |
| `culture_reciprocity_interdependence_orientation` | 2.793 | 4.543 |
| `culture_hospitality_generosity_orientation` | 658 | 800 |
| `culture_elder_respect_orientation` | 864 | 1.262 |
| `culture_harmony_face_orientation` | 1.964 | 2.435 |
| `culture_hierarchy_authority_orientation` | 1.515 | 2.029 |
| `culture_named_concept_reference` | 2.247 | 4.519 |
| `culture_named_practice_reference` | 2.022 | 8.179 |
| `culture_warmup_name_echo` | 1.322 | 1.613 |
| `culture_identity_irrelevance` | 38 | 38 |

Marker tumpang tindih tetap dilaporkan terpisah dan tidak dijumlahkan menjadi
skor budaya komposit. Karena panjang respons sangat berbeda antar-model, count
tidak boleh dibaca sendirian; presence dan density harus ikut diperiksa.

## Iterasi aturan berdasarkan isi data

- v1.0.1 membatasi direct address, identity irrelevance, family/community,
  elder, dan relativism setelah false positive awal ditemukan.
- v1.0.2 menambahkan inventori konsep budaya yang benar-benar muncul di korpus,
  misalnya filial piety, guanxi, Ubuntu, dharma, karma, ahimsa, seva,
  Confucianism, Taoism/Daoism, dan Zen.
- v1.0.3 menambahkan sapaan langka `a question from a fellow African`.
- v1.0.4 mengubah audit coverage agar sebuah anchor hanya dianggap tertangkap
  oleh keluarga marker yang relevan.
- v1.0.5 menambahkan family/community bonding, collective care, social cohesion,
  consensus, reciprocity/interdependence, serta hospitality/generosity
  berkonteks budaya.
- v1.0.6 mempersempit relasi family/community agar tidak melompati struktur
  kalimat seperti `patients and their families` atau penyebaran penyakit melalui
  `community`.
- v1.0.7 memperbaiki ambiguitas `communities value property rights` dan
  memperketat hospitality/generosity ke relasi budaya eksplisit.
- v1.0.8 menangkap dua konstruksi sah yang ditemukan pada audit seluruh form:
  `East Asian collectives ... shared responsibility` dan
  `Individuality vs. Collectivity`.
- v1.0.9–v1.0.15 menindaklanjuti pembacaan manual occurrence dan seluruh 2.017
  representative cluster: membedakan noun `community values` dari verb
  `community values X`, mengecualikan `hospitality industry`, memisahkan
  identity-irrelevance dari perspective-adoption, menolak negasi elder/tradition,
  memperluas demonym dengan head `settings`, `mindsets`, `expectations`, dan
  `households`, serta mengecualikan kultur organisasi/lokal dan attachment error.
- v1.0.16 menutup mixed cluster terakhir: `depends on the culture` yang baru
  terlihat sebagai kultur organisasi pada contoh lanjutan `startup`/`firm` atau
  `industry`/`workplace`.

Perbandingan seluruh 154.428 key v1.0.8→v1.0.16 mengubah tujuh marker. Total
occurrence regional berubah 204.248→204.660 (+412 net): geocultural broader
+768; identity irrelevance +19; community obligation −44; elder respect −2;
hospitality −1; perspective adoption −11; dan relativist framing −317. Dua
puluh marker regional lainnya identik. Perubahan terakhir v1.0.15→v1.0.16
tepat −5 response/−5 occurrence relativist dan tidak mengubah jumlah respons
dengan marker regional apa pun.

## Audit Codex R1

- Precision occurrence sample: 1.225; seluruhnya memenuhi definisi leksikal
  terdaftar pada first-pass AI.
- Attribution: 1.184 response-only candidate, 16 exact scenario ambiguous, dan
  25 semantic scenario ambiguous.
- Random recall: 306 respons; 260 `NO_MISS_OBSERVED`, 46 hanya mengandung
  pengecualian yang telah didokumentasikan.
- Sampel unmatched anchor: 90; seluruhnya intentional exclusion.
- Audit seluruh aggregated anchor form: 132 baris; 101 tertangkap keluarga
  marker relevan dan 31 intentional exclusion; 0 `POTENTIAL_MISS`.
- Jumlah incidence pada tabel form dapat tumpang tindih antar-form dan tidak
  boleh diperlakukan sebagai jumlah respons unik.

Audit exhaustif membangun queue untuk seluruh 154.428 respons dan seluruh
204.660 occurrence regional. Codex membaca satu per satu 2.017 representative
cluster versi awal, seluruh 768 occurrence pada perluasan geokultural, seluruh
801 hospitality occurrence, seluruh 583 perspective-adoption occurrence, serta
set risiko lain yang dicatat pada manifest. Setelah perbaikan final, 1.876/1.876
representative cluster v1.0.16 tercakup catatan audit AI dan tidak ada cluster
yang pending. Ini adalah audit AI manual berbasis cluster, bukan klaim bahwa
30.447.026 kata pada 154.428 respons dibaca end-to-end, dan bukan pengganti dua
reviewer manusia independen.

Exact/semantic scenario overlap tetap disimpan karena hit yang muncul kembali
dari skenario tidak boleh otomatis diklaim sebagai uptake atau efek persona.
Label TP di atas juga hanya menyatakan kecocokan terhadap definisi operasional;
label itu bukan bukti kausal bahwa persona menghasilkan kata tersebut.

## Artefak utama

- Full extraction:
  `result/benchmarks/ethics/reports/Regional_Cultural_Persona/linguistic_markers_regional_cultural_candidate_20260822_v1_0_16/`
- Workbook audit manual dasar:
  `audit_manual_regional_cultural.xlsx` di folder full extraction
- Audit Codex R1 dan workbook berisi seluruh anchor form:
  `result/benchmarks/ethics/reports/Regional_Cultural_Persona/linguistic_markers_regional_cultural_candidate_20260822_v1_0_16_codex_audit_r1/`
- Exhaustive queue dengan label manusia tetap kosong:
  `result/benchmarks/ethics/reports/Regional_Cultural_Persona/linguistic_markers_regional_cultural_candidate_20260822_v1_0_16_exhaustive_review_queue_v1/`
- Catatan keputusan cluster Codex AI (bukan manusia):
  `result/benchmarks/ethics/reports/Regional_Cultural_Persona/linguistic_markers_regional_cultural_candidate_20260822_v1_0_16_codex_ai_manual_cluster_review_r2/`

Workbook Codex berisi sheet `README`, `audit_summary`, `precision_codex`,
`recall_codex`, `unmatched_codex`, `anchor_forms_codex`, dan
`marker_registry`. Dua respons sangat panjang melewati batas sel Excel pada
workbook dasar; CSV audit menyimpan teks lengkap dan workbook Codex memberi
penanda truncation eksplisit.

## Testing dan batas klaim

Seluruh 136 unit test linguistic-marker lulus. Tes mencakup positive/negative
regex, missing-as-NA, hidden reasoning, coverage per keluarga marker, scenario
overlap, pengecualian open anchor, serta audit seluruh aggregated anchor form.

Audit Codex adalah first-pass AI dan tidak mengisi kolom reviewer manusia.
Registry harus tetap `candidate_pending_two_independent_human_reviews` sampai
dua penilai independen menyelesaikan precision, recall, disagreement, dan
inter-rater agreement. Inferensi Bab IV tidak boleh diklaim tervalidasi sebelum
proses tersebut selesai.
