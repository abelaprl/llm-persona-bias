# Rencana Analisis Linguistic Marker — DeepSeek R1 Distill × ETHICS Benchmark

## Konteks

Model yang dianalisis adalah **DeepSeek R1 Distill (Qwen-32B)**, sebuah *thinking model* yang menjawab soal ETHICS benchmark secara naratif. Setiap soal diinjeksi satu dari 41 persona (gender, agama, regional, netral) sebelum pertanyaan moral diberikan.

**Hipotesis utama:** Jika model berperilaku bias, persona yang berbeda akan menghasilkan pola bahasa yang secara sistematis berbeda — bukan hanya pada jawaban akhir (0/1), tetapi pada *cara model bernalar dan mengekspresikan penilaian moralnya*.

Dataset yang tersedia:
- 8,020 respons (v9_pilot, 41 persona × 200 soal)
- Kolom: `raw` (teks penuh respons model), `persona`, `subset`, `ground_truth_label`, `prediction_label`, `is_correct`, `parse_source`

---

## 1. Certainty vs. Hedging Markers

**Pertanyaan:** Apakah model lebih ragu-ragu (hedging) untuk persona tertentu dibandingkan persona lain atau kondisi netral?

### Marker yang diamati

| Kategori | Contoh kata/frasa |
|---|---|
| **Certainty tinggi** | clearly, obviously, definitely, certainly, undeniably, absolutely, unquestionably |
| **Hedging lemah** | might, may, could, perhaps, possibly, arguably, seemingly |
| **Hedging kuat** | it depends, without more context, cannot determine, challenging to say |
| **Distancing** | one could argue, some might say, it could be argued, from some perspectives |

### Metrik
- Frekuensi per 100 kata untuk setiap kategori, dikelompokkan per persona
- Rasio certainty/hedging dibandingkan kondisi `neutral`
- Apakah pola hedging berkorelasi dengan tingginya jumlah `none` (gagal ekstrak)?

---

## 2. Moral Framing — Kerangka Etika yang Digunakan

**Pertanyaan:** Apakah persona agama mendorong model ke framing deontologis/religius, sementara persona netral cenderung ke framing konsekuensialis?

### Kategori framing

| Framing | Marker kunci |
|---|---|
| **Deontologis** | duty, obligation, must, should, principle, rule, right, wrong, moral law |
| **Konsekuensialis** | harm, benefit, outcome, consequence, impact, effect, result, utility |
| **Virtue ethics** | character, integrity, virtue, honesty, compassion, kindness, respect |
| **Religius** | God, sin, faith, scripture, divine, permissible, forbidden, Quran, Bible, Torah, haram, halal |
| **Kultural/kontekstual** | cultural norms, tradition, context-dependent, varies by culture, societal |

### Metrik
- Proporsi kalimat per framing category, dikelompokkan per `persona_type` (religion_explicit, religion_implicit, gender, regional, neutral)
- Chi-square test: apakah distribusi framing berbeda secara signifikan antar kelompok persona?
- Apakah explicit vs. implicit persona menghasilkan framing yang berbeda?

---

## 3. Epistemic Stance — Komitmen Model terhadap Jawabannya

**Pertanyaan:** Seberapa yakin model dalam mengambil posisi moral? Apakah model lebih sering "bersembunyi di balik konteks" untuk persona tertentu?

### Tiga stance yang diobservasi

**a. Committed verdict** — model jelas mengambil posisi
```
"This action is clearly wrong."
"I conclude that the scenario is acceptable."
"Therefore, this is not appropriate."
```

**b. Conditional verdict** — model memberikan jawaban tergantung kondisi
```
"If X, then it is right. If Y, then it is wrong."
"It depends on whether..."
"In some contexts acceptable, in others not."
```

**c. Refusal / deferral** — model menolak atau meminta klarifikasi
```
"I cannot determine without more context."
"This is a personal choice that varies by individual."
"I'm unable to provide a binary judgment."
```

### Metrik
- Proporsi setiap stance per persona
- Apakah `conditional verdict` lebih tinggi untuk persona tertentu → indikasi model menghindari penilaian langsung
- Korelasi stance dengan `parse_source`: `none` seharusnya didominasi conditional + refusal

---

## 4. Response Length dan Struktur

**Pertanyaan:** Apakah model menulis respons yang lebih panjang atau lebih terstruktur untuk persona tertentu?

### Yang diukur
- **Panjang respons** (word count, character count) per persona
- **Jumlah bullet point / numbered list** (hitung `\n-` atau `\n1.`)
- **Jumlah bold section header** (`**Section:**` pattern)
- **Jumlah "considerations"** yang dilisting (proxy: hitung paragraf bernomor)
- **Posisi verdict dalam teks** — early (kuartal pertama) vs. late (kuartal terakhir)

### Hipotesis
- Persona yang menghasilkan respons lebih panjang mungkin mendapat "perlakuan lebih serius"
- Posisi verdict di awal teks = model lebih confident; di akhir = lebih ragu

---

## 5. Persona Acknowledgment — Apakah Model Merujuk Balik ke Identitas Persona?

**Pertanyaan:** Apakah model secara eksplisit mengakui identitas persona dalam reasoning-nya, dan apakah ini mempengaruhi arah argumennya?

### Marker
- **Religius:** "as a Muslim", "from a Christian perspective", "according to Islamic principles", "the Bible says", "halal/haram", "Quran", "Torah"
- **Gender:** "as a woman", "as a man", "from a woman's perspective", "gender roles", "societal expectations for women/men"
- **Regional/kultural:** "in African culture", "in East Asian tradition", "Western values", "South Asian norms"

### Metrik
- Apakah explicit persona lebih sering di-acknowledge daripada implicit persona?
- Apakah model yang acknowledge persona lebih cenderung memberikan jawaban yang selaras dengan ekspektasi stereotip?
- Korelasi antara persona acknowledgment dengan akurasi (is_correct)

---

## 6. Discourse Connectives — Pola Argumentasi

**Pertanyaan:** Apakah model menggunakan lebih banyak counter-argument (however, but, on the other hand) untuk persona tertentu, mengindikasikan ambivalensi yang lebih tinggi?

### Marker

| Kategori | Contoh |
|---|---|
| **Additive** | furthermore, moreover, additionally, also, and |
| **Contrastive** | however, but, on the other hand, nevertheless, yet, although, despite |
| **Causal** | therefore, thus, hence, consequently, as a result, because |
| **Concessive** | while, although, even though, granted that, admittedly |
| **Conditional** | if, unless, provided that, assuming, in the event that |

### Metrik
- Rasio contrastive/causal per persona — tinggi berarti model banyak mempertimbangkan sisi berlawanan
- Tingginya conditional marker berkorelasi dengan `none` parse failures

---

## 7. Polarity Shift Detection

**Pertanyaan:** Apakah model memulai dengan posisi satu arah lalu berpindah di akhir? Persona mana yang paling sering mengalami polarity shift?

### Cara deteksi
1. Ekstrak sentiment/posisi kalimat pertama (menggunakan verdict markers di awal teks)
2. Ekstrak posisi kalimat terakhir (verdict akhir)
3. Cari mismatch: model mulai dengan "this is acceptable" tapi akhiri dengan "therefore wrong"

### Metrik
- Proporsi respons dengan polarity shift per persona
- Apakah polarity shift berkorelasi dengan false predictions (is_correct=False)?

---

## 8. Empathy dan Validating Language

**Pertanyaan:** Apakah model lebih sering memvalidasi atau bersimpati terhadap persona tertentu?

### Marker
```
"it is understandable that...", "this is a valid concern", 
"your feelings are valid", "I can see why...",
"it makes sense that...", "this is a difficult situation"
```

### Hipotesis
- Jika model lebih banyak menggunakan empathy language untuk persona tertentu, ini bisa mengindikasikan bias afektif — model lebih "berpihak" secara emosional pada kelompok tersebut.

---

## 9. Comparative Analysis: Explicit vs. Implicit Persona

Setiap dimensi di atas perlu dibandingkan antara:
- `explicit` persona (label identitas langsung: "I am a Muslim")
- `implicit` persona (narasi kontekstual tanpa menyebut identitas)
- `neutral` (tidak ada persona)

**Pertanyaan kunci:** Apakah implicit persona menghasilkan linguistic pattern yang berbeda dari explicit? Jika ya, ini mengindikasikan model merespons sinyal identitas implisit dalam teks — lebih berbahaya dari sudut pandang bias.

---

## 10. Proposed Output Metrics per Persona

Untuk setiap persona, hitung dan bandingkan:

| Metrik | Tipe |
|---|---|
| Mean word count | Numeric |
| Hedging ratio (hedge words / total words) | Ratio |
| Certainty ratio | Ratio |
| Religious framing score | Normalized count |
| Deontological vs. consequentialist ratio | Ratio |
| Contrastive connective density | Normalized count |
| Conditional clause density | Normalized count |
| Committed verdict rate | Proportion |
| Conditional verdict rate | Proportion |
| Refusal rate | Proportion |
| Persona acknowledgment rate | Proportion |
| Parse failure rate (`none`) | Proportion |
| Accuracy (is_correct) | Proportion |

---

## 11. Statistical Testing

- **ANOVA / Kruskal-Wallis** untuk metrik numerik antar grup persona
- **Chi-square** untuk proporsi (stance, framing type, parse_source)
- **Post-hoc pairwise comparison** antara setiap persona vs. `neutral` baseline
- **Effect size** (Cohen's d atau Cramér's V) untuk menentukan signifikansi praktis
- **Bonferroni correction** untuk multiple comparisons (41 persona × banyak metrik)

---

## 12. Prioritas Analisis

Urutan observasi yang disarankan:

1. **Parse failure rate per persona** — quick win, sudah ada di CSV (`parse_source == 'none'`)
2. **Response length per persona** — mudah dihitung dari kolom `explanation` atau `raw`
3. **Hedging vs. certainty ratio** — indikator paling langsung dari confidence model
4. **Moral framing** — menentukan apakah persona menggeser kerangka etika
5. **Verdict position** — awal vs. akhir teks
6. **Explicit vs. implicit comparison** — pertanyaan inti penelitian

---

## Catatan Implementasi

- Semua analisis dilakukan pada kolom `prediction.raw` dari file JSON per soal (bukan kolom `explanation` di CSV yang sudah di-truncate)
- Gunakan `parse_source != 'none'` untuk analisis akurasi; seluruh data (termasuk none) untuk analisis linguistic marker
- Bandingkan selalu terhadap baseline `neutral` persona
- Pisahkan analisis per ETHICS subset (`commonsense`, `deontology`, `justice`, `virtue`) karena tiap subset memiliki karakteristik soal yang berbeda
