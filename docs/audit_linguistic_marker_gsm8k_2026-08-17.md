# Audit Awal Linguistic Marker untuk GSM8K

Tanggal audit: 17 Agustus 2026  
Status: audit metodologis dan keterbacaan; belum ada implementasi pipeline baru

## 1. Tujuan

Dokumen ini merangkum pembacaan kritis terhadap bahan eksplorasi linguistic marker yang tersimpan dalam:

`D:\TugasAkhir\docs\ref\linguistic marker yang udah aku coba.zip`

Audit dilakukan dengan mengacu pada rancangan penelitian dalam naskah tugas akhir di `D:\ProposalTA-18222008`, terutama ketentuan pada Bab IV mengenai:

- permukaan teks yang boleh dianalisis;
- satu respons sebagai satu observasi;
- perbandingan berpasangan berdasarkan pertanyaan yang sama;
- pemisahan kondisi neutral, explicit, implicit 1, implicit 2, dan implicit 3;
- penanganan missing value, abstention, dan kegagalan ekstraksi;
- pelaporan raw count dan density per 100 kata;
- effect size, confidence interval, dan koreksi pengujian berganda.

Ruang lingkup bahan yang diaudit terdiri atas:

- 8 notebook Jupyter;
- 14 laporan DOCX;
- 33 visual unik yang tertanam dalam notebook dan laporan.

## 2. Kesimpulan Utama

Bahan dalam ZIP sudah kaya sebagai eksplorasi awal, tetapi belum cukup aman untuk dipakai sebagai pipeline analisis final atau bukti utama pada Bab VI. Masalah terbesarnya bukan kekurangan jumlah grafik, melainkan ketidaksesuaian antara implementasi lama dan desain penelitian yang sudah ditetapkan dalam Bab IV.

Bab IV saat ini justru merupakan spesifikasi metodologis yang lebih kuat daripada notebook-notebook lama. Oleh karena itu, keputusan yang disarankan adalah membangun ulang pipeline linguistic marker berdasarkan kontrak Bab IV, bukan sekadar merapikan dan meneruskan notebook eksplorasi yang ada.

## 3. Kekuatan Bahan Eksplorasi Saat Ini

Beberapa bagian sudah memberikan fondasi yang berguna:

1. Taksonomi marker cukup luas, mencakup struktur pembuka, format matematika, stance epistemik, discourse connective, self-reference, identitas, dan metadata token reasoning.
2. Sudah ada gagasan menyimpan raw count sekaligus density yang dinormalisasi per 100 kata.
3. Beberapa laporan telah menyadari pentingnya membedakan extraction failure, nilai `none`, truncation, dan format jawaban akhir.
4. Eksplorasi berhasil menunjukkan kandidat fenomena penting: kondisi explicit dan implicit sering menghasilkan protokol penyajian jawaban yang sangat berbeda.
5. Struktur laporan per model, arsitektur, tipe reasoning, ukuran model, dan persona dapat menjadi bahan eksplorasi tambahan atau lampiran.
6. Kategori opener dan beberapa marker pemformatan berpotensi dipertahankan setelah definisinya diperbaiki dan divalidasi.

Kekuatan tersebut berarti bahan lama tidak perlu dibuang seluruhnya. Definisi marker, contoh respons, dan temuan eksploratif masih dapat dipakai sebagai sumber hipotesis, tetapi bukan sebagai hasil konfirmatori tanpa ekstraksi ulang.

## 4. Celah Metodologis Prioritas

### 4.1 Provenance dan reproducibility notebook

Pada `CoT_linguistic_marker_GSM8K_final.ipynb`, sel ekstraksi utama tidak tersimpan sebagai pernah dieksekusi, sedangkan sel-sel setelahnya memiliki output. Terdapat pula sel yang berisi sintaks Markdown tetapi disimpan sebagai code. Kondisi ini menunjukkan bahwa grafik yang tersimpan dapat berasal dari state kernel atau versi kode yang berbeda dari source cell saat ini.

Masalah lain yang ditemukan meliputi:

- path data lama yang masih menunjuk ke `E:\Tugas Akhir\...`;
- jumlah model yang berubah antara 12, 13, dan 14;
- cakupan model yang tidak sama dengan enam model pada rancangan TA;
- output notebook yang tidak selalu dapat ditelusuri ke input, extractor, dan versi pattern tertentu.

Dampaknya, angka dan visual lama tidak boleh langsung dipindahkan ke Bab VI sebelum direproduksi dari data sumber dengan pipeline yang versioned.

### 4.2 Permukaan teks yang dianalisis tidak konsisten

Sejumlah notebook memakai `response_content_full` dan bahkan menjadikan tag seperti `<think>` sebagai bagian dari opener. Beberapa analisis juga hanya memakai respons dengan `is_correct=True`.

Hal tersebut bertentangan dengan Bab IV, yang menetapkan bahwa:

- linguistic marker dihitung dari visible explanation saja;
- hidden reasoning atau isi `<think>` tidak boleh ikut dihitung;
- prompt, warm-up, dan salinan instruksi tidak termasuk permukaan analisis;
- penanda jawaban teknis seperti `Answer: 0` perlu dipisahkan dari penjelasan;
- respons benar, salah, dan abstain tetap menjadi bagian dari analisis teks selama teksnya tersedia.

Menganalisis hanya respons benar menimbulkan selection bias karena correctness sendiri dapat dipengaruhi oleh kondisi persona.

### 4.3 Desain berpasangan belum diterapkan

Beberapa notebook sudah mengekstrak `qid`, tetapi tidak menggunakannya dalam pengujian. Kondisi implicit 1, implicit 2, dan implicit 3 juga sering digabung menjadi satu kelompok implicit. Pada beberapa analisis, kondisi neutral tidak tersedia.

Uji yang dominan digunakan adalah Kruskal–Wallis dan Mann–Whitney untuk sampel independen. Padahal unit perbandingan yang ditetapkan dalam TA adalah respons terhadap pertanyaan yang sama pada kondisi berbeda.

Analisis final perlu mempertahankan tujuh perbandingan terencana:

1. neutral vs explicit;
2. neutral vs implicit 1;
3. neutral vs implicit 2;
4. neutral vs implicit 3;
5. explicit vs implicit 1;
6. explicit vs implicit 2;
7. explicit vs implicit 3.

Setiap perbandingan harus dilakukan di dalam kombinasi model, benchmark, dan identitas yang sama menggunakan pasangan `qid` yang tersedia.

### 4.4 Missing, abstention, dan extraction failure tercampur

Pada beberapa notebook, teks kosong diubah menjadi string kosong dan word count dipaksa minimal satu. Akibatnya, seluruh marker pada respons yang sebenarnya tidak tersedia berubah menjadi nol.

Dalam analisis ETHICS, `parse_source == "none"` juga digunakan untuk mewakili beberapa keadaan sekaligus, antara lain:

- model tidak memberikan jawaban yang dapat dinilai;
- parser gagal menemukan jawaban;
- respons terpotong;
- teks penjelasan tidak tersedia.

Keadaan tersebut mempunyai makna berbeda dan tidak boleh digabung. Minimal diperlukan status terpisah untuk:

- `text_available`;
- `abstained`;
- `answer_parse_failed`;
- `response_truncated`;
- `reasoning_unavailable`;
- `marker_not_applicable`.

Missing value harus disimpan sebagai `NA`, bukan nol.

### 4.5 Validitas konstruk marker belum memadai

Sejumlah regex atau lexicon terlalu luas dan rentan menangkap kata biasa dalam soal matematika. Contohnya:

- `about`, `around`, dan `could` dihitung sebagai hedge tanpa konteks;
- `always` dihitung sebagai booster;
- setiap kemunculan `you` dihitung sebagai engagement;
- `must`, `should`, `right`, `wrong`, `good`, dan `bad` dihitung sebagai moral language;
- kategori formal logic merupakan campuran LaTeX dan operasi aritmetika;
- sequential reasoning dapat terpicu hanya karena terdapat angka yang berdekatan dengan kata tertentu.

Terdapat pula skor komposit seperti `total_cot_density`, `hyland_interactional`, dan `liwc_reasoning_quality` yang mencampur density, indikator biner, dan marker yang tumpang tindih. Skor seperti ini belum memiliki dasar operasional yang cukup untuk disebut kualitas reasoning atau performa linguistik.

Rujukan Biber Dimension 5 juga belum cukup untuk menjadikan penggunaan LaTeX sebagai register formal. Format matematika dapat dilaporkan sebagai fitur penyajian, tetapi tidak boleh langsung disamakan dengan formalitas linguistik tanpa argumen dan validasi tambahan.

### 4.6 Belum ada validasi extractor

Belum ditemukan paket validasi yang lengkap berupa:

- registry marker yang versioned;
- positive unit tests;
- negative unit tests;
- contoh ambiguous match;
- audit false positive dan false negative;
- sampel anotasi manual;
- dua penilai dan ukuran kesepakatan antarrater;
- provenance lexicon dan perubahan pattern.

Tanpa validasi ini, penguji dapat berargumen bahwa perubahan yang dilaporkan merupakan perubahan perilaku regex, bukan perubahan linguistik model.

### 4.7 Eksplorasi ditulis seolah-olah konfirmatori

Pada notebook signifikansi gender, ketika tidak ditemukan hasil yang lolos koreksi Bonferroni, lima p-value terkecil dipilih untuk pengujian lanjutan. Praktik tersebut bersifat data-driven dan berisiko menghasilkan temuan semu.

Beberapa laporan juga menggunakan istilah seperti:

- "proxy terbaik";
- "gaya deliberatif";
- "kluster gaya reasoning";
- "stereotip budaya";
- perbedaan puluhan kali lipat;

berdasarkan mean deskriptif tanpa paired effect size, confidence interval, atau pengujian yang sesuai. Klaim seperti ini perlu diturunkan menjadi temuan eksploratif sampai tersedia bukti konfirmatori.

## 5. Inkonsistensi Antarlaporan

Beberapa nilai dan kesimpulan saling bertentangan sehingga perlu direproduksi dari sumber data:

1. Akurasi Llama 3.1 8B tercatat 73,4% pada satu laporan dan 83,59% pada laporan lain.
2. Laporan regional memperlihatkan word count explicit sekitar 120 dan implicit sekitar 85–102, tetapi ringkasannya membalik arah menjadi explicit sekitar 90 dan implicit sekitar 148.
3. Laporan religion juga membalik arah perubahan word count antara tabel dan narasi.
4. Nilai correct-versus-wrong pada laporan religion berbeda antara tabel dan uraian teks.
5. Laporan ETHICS v2 masih memuat tabel dan grafik lama berdasarkan 20 soal, kemudian menambahkan koreksi berdasarkan 200 soal di bagian akhir.
6. Klaim bahwa religious language tidak muncul pada semua respons sudah ditulis secara hard-coded dalam notebook.
7. Beberapa contoh yang disebut berasal langsung dari data mengandung artefak yang tidak dijelaskan, misalnya frasa `4933 muffin tins`.

Konsekuensinya, laporan DOCX lama sebaiknya diperlakukan sebagai catatan eksplorasi, bukan sumber angka final.

## 6. Penilaian Keterbacaan sebagai Pembaca TA

Dalam bentuk saat ini, pembaca sulit menjawab tiga pertanyaan dasar:

1. Marker mana yang merupakan outcome utama dan mana yang hanya eksplorasi tambahan?
2. Apakah perbedaan dihitung pada pertanyaan yang sama atau berasal dari komposisi sampel yang berbeda?
3. Apakah perubahannya konsisten dan bermakna, atau hanya didominasi oleh format prompt explicit dan implicit?

Dokumen lama lebih banyak menghasilkan profil gaya model atau persona, sedangkan pertanyaan penelitian TA lebih spesifik: apakah kondisi persona menggeser linguistic marker pada soal yang sama, seberapa besar pergeserannya, dan apakah pola tersebut konsisten pada enam model.

Laporan per persona ETHICS juga terlalu panjang dan repetitif. Pola tujuh bagian yang diulang untuk 41 persona menyulitkan pembaca menemukan temuan lintas persona. Detail per persona lebih sesuai ditempatkan dalam lampiran atau keluaran mesin, sedangkan Bab VI membutuhkan sintesis berdasarkan pertanyaan penelitian.

## 7. Penilaian Visual yang Sudah Ada

### 7.1 Visual yang bermasalah

- Grid pie chart opener sulit dibandingkan karena luas irisan antarpanel tidak memiliki baseline bersama dan label kategori kecil bertumpuk.
- Radar chart dengan banyak sumbu mencampur skala dan jenis metrik yang tidak sebanding.
- Master heatmap terlalu padat untuk halaman A4.
- Warna heatmap yang dinormalisasi per kolom tidak dapat dibandingkan antarmarker meskipun angka raw ditulis di dalam sel.
- Scatter plot dengan 16 persona terutama memisahkan protokol explicit dan implicit, tetapi diberi korelasi seolah-olah seluruh titik merupakan observasi independen.
- Bar chart akurasi menggunakan rentang sumbu sempit sehingga perbedaan kecil tampak lebih dramatis.
- Grafik delta tidak menampilkan confidence interval, jumlah pasangan, atau effect size.
- Bar chart semua model dan semua marker sekaligus terlalu padat dan mencampur metrik yang berbeda.

### 7.2 Prinsip visual yang disarankan

Satu visual sebaiknya menjawab satu klaim. Visual utama perlu menonjolkan perbedaan berpasangan dan ketidakpastian, bukan hanya mean absolut per persona.

Visual yang direkomendasikan adalah:

1. **Forest atau dot-whisker plot** untuk paired effect beserta 95% confidence interval.
2. **Effect-size heatmap** untuk melihat konsistensi arah efek lintas model; sel `NA` harus dibedakan dari nol.
3. **100% stacked bar atau small multiples** untuk distribusi kategori opener.
4. **Two-part plot** untuk marker sparse: prevalence pada panel pertama dan distribusi density kondisional pada respons nonzero pada panel kedua.
5. **Paired distribution plot** untuk dua sampai empat temuan utama, bukan untuk seluruh marker.

Radar chart, kumpulan pie chart, giant grouped bar, dan master heatmap seluruh marker tidak disarankan sebagai bukti utama. Jika tetap diperlukan untuk eksplorasi, visual tersebut dapat dipindahkan ke lampiran.

## 8. Rancangan Hierarki Tabel

### 8.1 Tabel quality control

Tabel pertama perlu menjelaskan kelengkapan data sebelum hasil marker ditampilkan.

Kolom minimal:

| Kolom | Makna |
|---|---|
| `model_id` | Model yang dianalisis |
| `benchmark` | GSM8K atau ETHICS |
| `identity_domain` | Gender, religion, atau regional |
| `condition_id` | Neutral, explicit, I1, I2, atau I3 |
| `n_expected` | Respons yang seharusnya tersedia |
| `n_response_available` | Respons yang berhasil diperoleh |
| `n_text_available` | Visible explanation tersedia |
| `n_abstained` | Abstention yang teridentifikasi |
| `n_parse_failed` | Jawaban gagal diparsing |
| `n_truncated` | Respons terpotong |
| `n_complete_pairs` | Pasangan `qid` lengkap untuk perbandingan |

### 8.2 Tabel registry marker

Kolom minimal:

| Kolom | Makna |
|---|---|
| `marker_id` | Identifier stabil |
| `marker_family` | Kelompok konstruk |
| `operational_definition` | Definisi yang benar-benar dihitung |
| `pattern_or_lexicon` | Pola atau daftar istilah |
| `exclusion_rule` | Konteks yang tidak boleh dihitung |
| `measurement_type` | Count, density, presence, atau category |
| `theoretical_source` | Provenance teori atau lexicon |
| `version` | Versi extractor |
| `validation_status` | Status unit test dan audit manual |

### 8.3 Tabel deskriptif

Untuk marker numerik, keluaran utama menggunakan median dan IQR. Density per 100 kata menjadi ukuran utama, sedangkan raw count dan presence menjadi ukuran pendukung.

Kolom minimal:

| Kolom | Makna |
|---|---|
| `model_id` | Model |
| `identity` | Identitas yang dianalisis |
| `condition_id` | N, E, I1, I2, atau I3 |
| `marker_id` | Marker |
| `n_available` | Jumlah respons yang dapat dianalisis |
| `median_density` | Median density per 100 kata |
| `iqr_density` | IQR density |
| `prevalence` | Proporsi respons dengan marker > 0 |
| `median_raw_count` | Median raw count |

### 8.4 Tabel inferensial utama

Kolom minimal:

| Kolom | Makna |
|---|---|
| `model_id` | Model |
| `identity` | Identitas |
| `marker_id` | Marker |
| `contrast_id` | Salah satu dari tujuh perbandingan terencana |
| `n_pairs` | Jumlah pasangan `qid` |
| `median_paired_delta` | Median perubahan di dalam pasangan |
| `ci_low` | Batas bawah bootstrap 95% CI |
| `ci_high` | Batas atas bootstrap 95% CI |
| `rank_biserial` | Paired rank-biserial effect size |
| `p_raw` | P-value sebelum koreksi |
| `p_holm` | P-value setelah koreksi Holm |
| `test_used` | Wilcoxon atau sign test sesuai kondisi data |

Untuk opener dan marker kategorikal, tabel menggunakan proporsi kategori dan prosedur paired categorical test yang ditetapkan dalam Bab IV.

## 9. Pemisahan Konstruk, Metadata, dan Quality Control

Tidak semua fitur lama sebaiknya disebut linguistic marker.

### 9.1 Kandidat outcome utama GSM8K

- struktur pembuka;
- mathematical presentation;
- explicit step organization;
- hedge dan booster yang telah divalidasi;
- assumption marker;
- self-correction;
- discourse connective;
- self-reference atau engagement yang didefinisikan secara sempit.

### 9.2 Outcome khusus atau supplementary

- religious identity mention;
- cultural identity mention;
- AI identity mention;
- marker yang sangat sparse;
- kandidat marker yang ditemukan secara post-hoc.

### 9.3 Metadata terpisah

- reasoning token count;
- completion token count;
- prompt token count;
- answer format;
- model architecture, reasoning type, dan size class.

### 9.4 Quality-control features

- keberadaan `boxed`;
- answer closure;
- `parse_source`;
- truncation;
- extraction status;
- text availability;
- abstention status.

Fitur quality control dapat membantu menjelaskan hasil, tetapi tidak boleh dicampurkan ke skor linguistic marker.

## 10. Serangan Reviewer yang Perlu Diantisipasi

Serangan metodologis yang paling mungkin adalah:

1. **"Penelitian hanya mengukur format prompt, bukan efek identitas."**  
   Pertahanan: gunakan neutral baseline, pertahankan I1–I3 secara terpisah, lakukan pairing berdasarkan `qid`, dan bedakan marker format dari marker linguistik.

2. **"Hidden chain-of-thought ikut dianalisis."**  
   Pertahanan: definisikan visible explanation secara deterministik, simpan provenance field, dan audit sampel pemisahan teks.

3. **"Regex hanya menangkap kata dalam soal matematika."**  
   Pertahanan: gunakan exclusion rules, negative tests, context-aware matching, audit manual, dan sensitivity analysis.

4. **"Pengujian mengabaikan desain berpasangan dan melakukan terlalu banyak tes."**  
   Pertahanan: gunakan tujuh planned contrasts, paired tests, effect size, bootstrap confidence interval, dan koreksi Holm per family.

5. **"Missing response dianggap tidak mengandung marker."**  
   Pertahanan: simpan missing sebagai `NA`, laporkan coverage, dan bedakan abstention dari kegagalan ekstraksi.

6. **"Temuan budaya dan agama merupakan interpretasi stereotip dari lexicon yang terlalu luas."**  
   Pertahanan: pisahkan identity mention yang eksplisit dari istilah umum, batasi klaim pada data, dan sertakan audit contoh kontekstual.

## 11. Rekomendasi Arsitektur Pekerjaan Berikutnya

Jika audit ini dilanjutkan menjadi implementasi, urutannya sebaiknya:

1. menulis specification untuk data contract dan permukaan teks;
2. menetapkan registry marker versi awal;
3. membuat extractor sebagai modul atau script, bukan hanya state notebook;
4. membuat positive dan negative unit tests;
5. menjalankan audit manual dua penilai;
6. menghasilkan satu long-form response-level dataset;
7. menghasilkan tabel QC sebelum analisis statistik;
8. menjalankan analisis paired sesuai Bab IV;
9. menghasilkan tabel dan visual publication-ready dari output statistik yang sama;
10. menulis narasi Bab VI setelah seluruh angka berhasil direproduksi.

Notebook tetap dapat digunakan untuk eksplorasi dan presentasi, tetapi logika ekstraksi dan statistik inti sebaiknya berada di modul yang dapat diuji dan dijalankan ulang.

## 12. Catatan Naskah TA

Terdapat satu masalah housekeeping yang tidak berkaitan langsung dengan linguistic marker: `D:\ProposalTA-18222008\5 Abstrak.tex` masih memuat abstrak penelitian Smart Scale IoT untuk jeruk, bukan abstrak penelitian LLM yang sedang dikerjakan.

## 13. Keputusan Audit

Pada tahap ini belum disarankan memakai angka, p-value, atau visual lama sebagai hasil final. Bahan lama dapat digunakan untuk:

- menyusun kandidat marker;
- menemukan kasus positif dan negatif untuk pengujian;
- membangun hipotesis eksploratif;
- menentukan visual yang perlu dihindari;
- merancang struktur laporan baru.

Tahap berikutnya yang disarankan adalah menyusun SPEC dan desain keluaran tabel/figur berdasarkan Bab IV sebelum menulis pipeline baru.
