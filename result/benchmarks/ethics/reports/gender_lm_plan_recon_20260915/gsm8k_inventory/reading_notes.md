# Inventaris GSM8K untuk perencanaan LM ETHICS

Pemeriksaan 15 September 2026, terbatas pada pembacaan artefak dan respons. Tidak ada ekstraktor, notebook, label jawaban, atau hasil lama yang diubah. Pembacaan ini bukan anotasi manusia atau audit precision/recall.

## Sumber dan cakupan

- Sumber respons: `result/GMS8K/reports/Gender_Persona/publication_source_aligned_v1_20260817/_combined/all_models_results_full.csv`.
- Sumber marker: `result/linguistic_markers/gender_gsm8k_full_source_aligned_v1_2_1_20260817/response_records.csv`.
- Keduanya memuat 71.215 baris. Join satu-ke-satu berdasarkan model, persona, question_id cocok pada seluruh 71.215 baris; tidak ada baris yang tidak cocok.
- Target 71.226 = 6 model × 9 persona × 1.319 soal; source gap 11, missing 165, technical_only 802, available 70.248. Seluruh technical_only berasal dari GPT-OSS. Detail jumlah per model/persona berada di `inventory.json`.
- `visible_text` berbeda dari `explanation_text` pada 30.570 baris. Surface lexical dan pembukaan/gender lexicon harus disebutkan terpisah dalam kontrak pengukuran.

## Notebook dan visual yang benar-benar tersimpan

Notebook `notebooks/gender/linguistic marker/linguistic_marker_gender_gsm8k_direct_gender_v3.ipynb` memilih bundle sah terakhir menurut mtime, saat pemeriksaan yaitu `result/linguistic_markers/gender_gsm8k_bab4_visual_revision_v3_direct_gender_r2_20260817`.

Ada 26 sel, 13 sel kode dieksekusi, 0 output error tersimpan, 17 figur dalam index, dan 16 figur ditampilkan. Wordcloud nomor 11 supplementary tidak ditampilkan. Output provenance tersimpan menunjuk bundle r2, dan seluruh 16 PNG embedded identik byte-for-byte dengan PNG bundle. Hash tersedia di `notebook_embedded_png_verification.json`. Pemeriksaan ini tidak menjalankan ulang notebook.

Dua PNG dilihat langsung:

1. `images/13_gender_pronoun_direct_man_vs_woman_forest.png`: empat panel kondisi E/I1/I2/I3 dengan titik per model, garis nol, selisih prevalence Man−Woman dalam percentage point dan CI 95%. Kerangka ini mudah diadaptasi untuk ETHICS dengan pemisahan subset. Angka n_pairs dan jumlah pasangan diskordan perlu tetap tersedia di tabel pendamping. CI yang tidak melintasi nol bukan pengganti keputusan berdasarkan p yang dikoreksi.
2. `images/15_direct_gender_effect_consistency.png`: 16 outcome × 4 kondisi × 6 model. Efek, warna model, opacity berdasarkan Holm, ukuran berdasarkan nonzero, dan diamond median enam model membuat gambar padat. Untuk ETHICS, bagi menurut keluarga marker/subset dan sediakan angka effect, CI, dan dukungan pasangan; median model tetap deskriptif.

Bundle memuat 384 uji numeric primary, 56 dengan p_Holm < 0,05. Angka ini adalah inventaris kandidat, bukan temuan tervalidasi. Status manifest masih `candidate_pending_manual_precision_recall_audit`; precision occurrence 850 sampel/0 anotasi, recall response 108 sampel/0 anotasi.

## Registry dan lexicon

Registry GSM8K memuat 18 marker dalam 9 keluarga: organization 3; epistemic stance 4; logical connective 3; self-reference 2; mathematical presentation 2; engagement 1; textual elaboration 1; answer format 1; model identity 1. Peran: 13 primary, 2 supplementary, 2 format, 1 quality_control. Registry memberi sumber teoretis beserta operasionalisasi/adaptasi sendiri; tidak seluruh definisinya kutipan langsung dari sumber tersebut.

Gender lexicon memiliki 52 term: 4 masculine pronouns, 4 feminine pronouns, 22 masculine nouns, 22 feminine nouns. Metadata menyebut `gendered-visible-lexicon-v1`, tetapi tidak mencantumkan sitasi teoretis lexicon. Pronoun: he/him/his/himself dan she/her/hers/herself. Noun meliputi man/men/male/boy/father/dad/husband/son/brother/gentleman/uncle/guy serta pasangan feminin dan bentuk jamaknya. Penyebutan leksikal tidak membuktikan adopsi persona.

## Pembacaan 60 respons penuh

`gsm8k_full_response_samples.csv` menyimpan source key, respons sumber penuh, visible text penuh, explanation text penuh, dan protokol seleksi tanpa GT/correctness. Enam berkas `samples_*.txt` dibaca seluruhnya: 10 respons per model, mencakup sembilan persona dan satu respons tambahan yang lebih panjang. Total 6.618 kata menurut word_count tersimpan, rentang 49–312 kata. Seleksi purposive menurut persona dan kedekatan panjang 50/90/130/300 kata; bukan sampel acak untuk inferensi atau audit precision/recall.

Pengamatan konkret, bukan penilaian benar/salah jawaban:

- Llama 3.1 dan 3.3 banyak memakai pembukaan instruksional dan uraian aritmetika dalam sampel yang dibaca. Llama 3.1 man_implicit_1 `gsm8k_00078` menulis empat langkah bernomor biasa dan memiliki stored explicit-step count 4.
- DeepSeek man_explicit_1 `gsm8k_00195` memuat dua langkah seperti `1. **Calculate the total cost:**`, tetapi stored explicit-step count 0. Mistral man_explicit_1 `gsm8k_00238` memiliki tiga langkah bernomor tebal dan count 0. Pada sampel 60 respons, 24 memuat pola numbered-bold dan 23 di antaranya memiliki explicit-step count 0. Ini menunjukkan ketergantungan hit pada format Markdown yang harus diperiksa sebelum menyalin detector ke ETHICS.
- GPT-OSS neutral `gsm8k_00796` menggunakan tabel Markdown untuk tiga bulan payroll; registry tidak memiliki kategori tabel. GPT-OSS man_implicit_2 `gsm8k_00314` dimulai `**Answer: 120,000**`, tetapi opening tersimpan `document_heading`, bukan `answer_first`. Keputusan apakah format atau fungsi pembukaan yang diukur perlu dijelaskan.
- DeepSeek woman_explicit_1 `gsm8k_00902` memuat `Wait, this leads to a contradiction` dan GPT-OSS man_implicit_2 `gsm8k_00673` memuat `Wait – we mis-counted. Let's recount:`; keduanya stored self-correction 0. Mistral man_explicit_1 `gsm8k_00924` memuat `Let's re-evaluate` dan count 1. Perbedaan batas kalimat/punctuation layak masuk audit detector, bukan promosi otomatis aturan baru.
- Mistral man_implicit_3 `gsm8k_00489` berulang kali menyebut `girl Pomeranians`: noun bergender merujuk hewan. Qwen man_implicit_1 `gsm8k_00105` menyebut Mom sebagai pelaku dalam soal. Llama 3.1 man_implicit_2 `gsm8k_00045` menggunakan he/his untuk Charlie. Semua memperjelas kebutuhan membedakan konteks cerita dari adopsi identitas persona.
- Qwen woman_implicit_3 `gsm8k_00074` menggunakan heading dengan bold dan emoji; metadata format tidak boleh otomatis ditafsirkan sebagai kualitas penalaran.
- Llama 3.3 man_explicit_1 `gsm8k_00910` berakhir `Then,` pada visible text penuh. Baris tersimpan available dan response_truncated=False. Availability teks tidak membuktikan penjelasan lengkap; temuan ini tidak mengubah status atau label baris.

## Scan kandidat pola permukaan

`observed_surface_pattern_scan.csv` berisi hit literal per model/persona, berasal dari visible_text 71.215 respons. Semua hit hanya bukti cakupan pola, bukan validitas semantik atau jumlah kesalahan detector.

| Pola literal | Respons hit | Occurrence |
|---|---:|---:|
| Nomor di awal baris diikuti teks bold | 18.603 | 66.760 |
| Separator tabel Markdown | 822 | 825 |
| follow these steps | 3.178 | 3.178 |
| since | 17.367 | 25.321 |
| but | 2.769 | 27.943 |
| so | 32.688 | 84.411 |
| as a/an man/woman/male/female | 0 | 0 |

Ketiadaan satu frasa adopsi literal tidak berarti tidak ada bentuk adopsi lain. Bare since/but/so sengaja sempit atau dikecualikan dalam registry lama karena polifungsi; frekuensi di atas bukan izin memasukkannya tanpa konteks dan audit. Seluruh temuan dari pembacaan 60 respons tetap bersifat deskriptif.
