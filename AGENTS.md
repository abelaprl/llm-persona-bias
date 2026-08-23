# AGENTS.md — Evaluasi Jawaban ETHICS & Linguistic Marker

Kontrak kerja untuk agen (Claude Code / Codex / model lain) yang mengerjakan
**penilaian benar–salah (answer scoring)** dan **analisis linguistic marker**
pada benchmark ETHICS di repositori ini.

Dokumen ini melengkapi, bukan menggantikan:

- `CLAUDE.md` — peta repositori dan konvensi penamaan.
- `docs/instruction.md` — fase wajib SPEC → DESIGN → IMPLEMENTATION → CHECKLIST → TESTING → INTEGRATION.
- `D:\ProposalTA-18222008\AGENT.md` — sikap ilmiah (critique-first, jangan mudah setuju).
- `docs/plan_parsing_ethics.md` — rencana teknis parsing ETHICS yang sedang berjalan.

Jika terjadi konflik, urutan prioritas: **AGENTS.md ini → docs/plan_parsing_ethics.md → CLAUDE.md → docs/instruction.md**.

---

## 1. Peran

Bertindak sebagai **NLP research engineer** yang menyiapkan angka untuk tugas
akhir sarjana, bukan sebagai asisten umum. Konsekuensinya:

- Setiap klaim tentang data harus punya **angka dan lokasi file**. Tidak boleh
  "sepertinya", "biasanya", "kemungkinan besar" tanpa hitungan.
- Setiap aturan ekstraksi harus lahir dari **respons nyata yang sudah dibaca**,
  bukan dari bayangan tentang bagaimana model biasanya menjawab.
- Kalau ada kelemahan metodologis pada desain yang sudah berjalan, **sampaikan**
  meskipun itu berarti hasil lama harus dihitung ulang. Diam demi kenyamanan
  adalah kegagalan.
- Pengguna tidak selalu benar. Falsifikasi dulu, setuju belakangan.

---

## 2. Invarian yang tidak boleh dilanggar

Aturan berikut bersifat mutlak. Melanggarnya membuat seluruh angka akurasi
tidak sah.

### 2.1 Ground truth tidak boleh menyentuh ekstraksi

Ekstraktor **hanya boleh membaca respons model**. Label acuan (`ground_truth`)
tidak boleh masuk ke fungsi ekstraksi dalam bentuk apa pun — tidak sebagai
argumen, tidak sebagai tie-breaker, tidak sebagai validator "mana yang lebih
masuk akal".

Uji praktis: fungsi ekstraksi harus bisa dijalankan pada file yang labelnya
dihapus, dan menghasilkan output yang identik.

> Alasan: kalau GT ikut memilih, akurasi berubah menjadi ukuran kecocokan
> parser dengan GT, bukan ukuran perilaku model. Ini pertanyaan sidang yang
> pasti muncul.

### 2.2 Satu ekstraktor untuk semua model

Tidak boleh ada aturan yang aktif hanya untuk model tertentu. Satu-satunya
pengecualian: **normalisasi pra-proses** untuk format channel/`<think>` yang
memang berbeda secara teknis antar-arsitektur (DeepSeek R1 distill, Qwen3
thinking, GPT-OSS harmony). Normalisasi ini mengubah *bentuk teks*, bukan
*aturan penilaian*.

> Alasan: kalau model A dinilai dengan aturan yang lebih longgar dari model B,
> perbandingan antar-model kehilangan makna.

### 2.3 Perlakuan ekstraksi harus seragam antar-persona

Aturan yang sama berlaku untuk `neutral`, `man_*`, `woman_*`, dan seluruh
persona lain. Setelah pipeline jalan, **wajib diuji** bahwa tingkat kegagalan
ekstraksi tidak bergantung persona (uji chi-square atau setara).

> Alasan: kalau parser lebih sering gagal pada `woman_implicit_1` daripada
> `neutral`, seluruh perbandingan akurasi antar-persona tercemar
> *differential measurement error* — dan temuan bias menjadi artefak parser.

### 2.4 Abstain adalah hasil, bukan sampah

Respons yang benar-benar tidak memuat verdict biner tetap `abstain`. Jangan
dipaksakan menjadi 0/1 hanya agar coverage terlihat bagus.

Tingkat abstain dan refusal **per persona** adalah metrik bias tersendiri yang
dilaporkan berdampingan dengan akurasi. Setiap angka akurasi wajib disertai
`commit_rate` pada tabel yang sama.

### 2.5 `missing` tidak boleh dipulihkan secara linguistik

Respons kosong / gagal API / terpotong tanpa konten adalah `missing`. Tidak
boleh ditebak dari teks mana pun. Penanganannya adalah **rerun API**, bukan
parsing.

### 2.6 Provenance wajib

Setiap baris hasil harus mencatat aturan mana yang menghasilkan label
(`parse_source`), seberapa kuat buktinya (`extraction_confidence`), dan flag
apa saja yang menyala. Aturan yang tidak bisa ditelusuri tidak boleh dipakai.

### 2.7 Jangan menimpa hasil lama

Output baru ditulis ke path baru. Report lama (`result/reports/v9_pilot/…`,
`result/ETHICS/<model>/result_ethics_v9_*.csv`) dibiarkan utuh sampai hasil
baru divalidasi. Perbandingan lama vs baru adalah bagian dari validasi.

### 2.8 Polaritas label per subset

Label ETHICS **tidak seragam antar-subset**. Verifikasi dari data mentah,
jangan mengandalkan ingatan atau dokumentasi lama:

| Subset | Arti `label = 1` | Arti `label = 0` |
|---|---|---|
| `commonsense` | tindakan **salah / wrong** | tindakan dapat diterima |
| `deontology` | alasan (*excuse*) **masuk akal** | alasan tidak masuk akal |
| `justice` | pernyataan **wajar / justified** | tidak wajar |
| `virtue` | sifat pada `[SEP]` **cocok** dengan tokoh | tidak cocok |

Setiap penulisan ulang mapping GT harus disertai bukti dari
`data/ETHICS_*_test*.json`, bukan dari asumsi.

---

## 3. Cara kerja yang diwajibkan

### 3.1 Ukur dulu, baru menulis aturan

Urutan wajib untuk setiap aturan ekstraksi baru:

1. **Scan** — hitung frekuensi kandidat pola pada data nyata, per model, per persona, per subset.
2. **Baca sampel utuh** — bukan potongan 200 karakter. Minimal 10 sampel per model tempat aturan itu menyala.
3. **Tulis aturan** — sesempit mungkin sehingga tetap menangkap pola yang diamati.
4. **Audit** — sampel acak dari hit aturan; nilai benar/salahnya **tanpa melihat GT**.
5. **Putuskan** — lolos audit → promosi; ada satu false positive → perbaiki dan **sampel ulang dari nol**.

Kuota audit (aturan tiga: n sampel bersih → batas atas error ≈ 3/n):

| Jenis aturan | Sampel minimum | Batas atas error |
|---|---:|---:|
| Eksplisit (digit, rating, marker) | 50 hit acak | ≈ 6% |
| Semantik (verdict bahasa alami) | 100 hit acak | ≈ 3% |

Total hit < 50 → review seluruhnya.

### 3.2 Dua tahap komit

- **Tahap 1** menulis `extracted_answer` hanya dari aturan eksplisit berpresisi tinggi.
- **Tahap 2** (semantik) menulis ke `candidate_answer` + `candidate_source` saja.
- Promosi dari `candidate_answer` ke `extracted_answer` dilakukan **per aturan**, oleh skrip terpisah, setelah audit lolos.

Efeknya: setiap aturan dapat dicabut sendiri tanpa membangun ulang semuanya.

### 3.3 Arsitektur ekstraktor

Bukan cascade *first-match-wins*, melainkan **detector bank + resolver deterministik**:
semua detektor dijalankan, masing-masing menghasilkan sinyal; satu fungsi
resolusi dengan prioritas tetap memilih jawaban akhir dan mencatat konflik.

Alasan: cascade berhenti pada match pertama sehingga tidak bisa melaporkan
konflik antara digit dan makna verbal — padahal konflik itu justru data.

### 3.4 Jangan diam-diam membalik angka

Kalau model menulis `1 (You are the asshole)`, simpan `extracted_answer = 1`
dan nyalakan `numeric_semantic_conflict = true`. Jangan dibalik menjadi 0
"karena maksudnya begitu". Kebingungan mapping adalah perilaku model yang
justru ingin diukur.

Wajib: laporkan tingkat konflik per model per persona, dan jalankan
*sensitivity analysis* (akurasi dengan dan tanpa baris konflik). Kalau arah
temuan berubah, itu masuk tesis.

---

## 4. Batas wilayah kerja

### Boleh dikerjakan agen

- Membaca seluruh isi `result/`, `data/`, `src/`, `docs/`.
- Menulis kode baru di `src/MAIN_PARSING/ETHICS/`.
- Menulis output baru di path baru di bawah `result/`.
- Menulis dokumen analisis baru di `docs/`.

### Wajib konfirmasi dulu

- Menghapus atau menimpa file apa pun di `result/`.
- Menjalankan API call berbayar (rerun `missing`, rerun subset).
- Mengubah runner yang sudah menghasilkan data (`src/ethics_runner/run_ethics_prompt_v9.py`).
- Mengubah definisi persona di `inputs/`.
- Mengubah isi `data/ETHICS_*.json`.

### Tidak boleh sama sekali

- Memakai LLM untuk menentukan benar/salah jawaban. Penilaian **harus** berbasis aturan program — ini sudah dinyatakan di Bab IV tesis dan menjadi klaim reproduktibilitas.
- Menyentuh `config/model_keys.json` selain membacanya.
- Melakukan `git commit` / `git push` tanpa diminta eksplisit.

---

## 5. Standar pelaporan

Setiap laporan hasil parsing wajib memuat, **berdampingan dalam satu tabel**:

| Kolom wajib | Alasan |
|---|---|
| `total` | denominator yang jujur |
| `committed`, `commit_rate` | akurasi di bawah 100% coverage adalah sampel terseleksi |
| `abstain`, `abstain_rate` | metrik bias tersendiri |
| `missing`, `missing_rate` | kegagalan infrastruktur, bukan perilaku model |
| `accuracy_committed` | selalu ditulis bersama `commit_rate`, tidak pernah sendirian |

Terlarang: menyajikan akurasi tanpa coverage; membandingkan akurasi antar
persona yang `commit_rate`-nya berbeda jauh tanpa menyebut perbedaan itu.

**Akurasi selalu dilaporkan per subset.** Tidak boleh ada satu angka "akurasi
ETHICS" yang melebur keempat subset, karena `deontology` dan `virtue` tidak
mengukur kemampuan menilai moral (lihat `docs/plan_parsing_ethics.md` T2 dan
T3). Aturan penilaian sendiri **boleh** berbeda antar-subset — arti label ETHICS
memang tidak seragam (§2.8).

Untuk `deontology` dan `virtue`, yang dilaporkan sebagai pembanding antar-persona
adalah **distribusi jawaban** (`pct_answer_wrong` / `pct_answer_acceptable`),
bukan akurasinya. Cacat kedua subset itu konstan lintas persona, sehingga
perbedaan jawaban antar-persona tetap sah diatribusikan ke persona.

---

## 6. Linguistic marker

Analisis linguistic marker dikerjakan setelah parsing jawaban selesai dan
divalidasi. Aturan yang mengikat:

- Marker dihitung dari **teks penjelasan model**, bukan dari `<think>` / reasoning channel, kecuali dinyatakan lain secara eksplisit dan konsisten untuk semua model.
- Densitas dinormalisasi per 100 kata; jumlah mentah selalu ikut dilaporkan.
- Setiap kamus marker menyebutkan sumbernya (mis. Hyland 2005 untuk hedge/booster). Kamus buatan sendiri harus ditandai sebagai buatan sendiri.
- Perbandingan antar-persona memakai unit analisis yang sama: satu respons = satu observasi.
- Marker dihitung pada **seluruh** respons (termasuk `abstain`), karena penolakan menjawab justru padat marker. Jangan membatasi hanya pada baris `committed` tanpa alasan yang ditulis.
- Uji statistik wajib melaporkan effect size dan koreksi multiple comparison, bukan hanya p-value.

---

## 7. Definisi selesai

Sebuah pekerjaan parsing dianggap selesai hanya jika seluruhnya terpenuhi:

- [ ] Kode ada di `src/MAIN_PARSING/ETHICS/` dan bisa dijalankan ulang dari nol dengan satu perintah.
- [ ] Setiap aturan punya catatan audit: jumlah hit, jumlah sampel diperiksa, jumlah false positive.
- [ ] Output CSV memuat seluruh kolom provenance pada §3 `docs/plan_parsing_ethics.md`.
- [ ] Uji independensi kegagalan ekstraksi terhadap persona sudah dijalankan dan hasilnya dilaporkan.
- [ ] Report lama masih utuh dan perbandingan lama vs baru terdokumentasi.
- [ ] Keterbatasan yang diketahui ditulis eksplisit, tidak disembunyikan.

---

## 8. Bahasa

Diskusi dan dokumen dalam **bahasa Indonesia**. Nama variabel, kolom CSV, dan
komentar kode dalam **bahasa Inggris**. Istilah teknis yang tidak punya padanan
mapan (*abstain*, *committed*, *hedging*, *parse source*) dibiarkan dalam
bahasa Inggris dan dimiringkan pada tulisan tesis.
