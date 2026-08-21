# Audit refinement parser ETHICS Gender + Neutral — 2026-08-20

## Ruang lingkup

Input canonical adalah 81.756 respons Gender Persona + Neutral pada
`result/ETHICS/reports/Gender_Persona/_combined/all_models_results_full.csv`
dengan SHA-256
`93301f45395ae532193b76a166b88ef23c8e2d3dc206935ca5f5220a55b2d56e`.
Tidak ada API call, perubahan response text, atau penggunaan ground truth pada
jalur ekstraksi.

Refinement terdiri dari dua perubahan yang sengaja dipisahkan:

1. Marker eksplisit AITA setelah label ketat `Answer`, `Rating`, atau `Verdict`
   dapat menjadi `committed`.
2. Kandidat semantik setelah boilerplate `no_opinion_stop` atau
   `task_substitution` boleh muncul di `candidate_answer`, tetapi tetap
   `abstain`.

## Audit aturan AITA

Iterasi pertama menghasilkan 21 transisi `abstain -> committed`. Pembacaan
respons utuh menemukan 11 hit yang tidak aman: enam bentuk `likely YTA`, dua
rating `0.5`, satu jawaban bernuansa, satu frame `answer to the question`, dan
satu verdict kondisional dengan ekspansi akronim nonstandar. Aturan diperketat
dan audit diulang dari nol; hasil iterasi pertama tidak dipakai untuk promosi.

Aturan final hanya menerima label verdict yang bersebelahan langsung dengan
token AITA. Label yang hedged ditolak. Label final yang ketat boleh mengalahkan
glossary NTA/YTA sebelumnya, sedangkan glossary tanpa label final tetap
unresolved.

- Universe aturan final: 141 respons.
- Sampel blind deterministik: 50 respons, minimal 10 per model bila tersedia.
- Model dengan hanya empat hit diperiksa seluruhnya.
- Sampel benar: 50; salah: 0; unclear: 0.
- Baris yang benar-benar berubah: 12; semuanya dibaca utuh dan benar: 12/12.
- Seluruh recovery berada pada subset `commonsense` dan Gender Persona;
  Neutral tidak berubah.

Audit terisi dan riwayat aturan gagal tersedia di
`result/ETHICS/reports/_audit/gender_parser_refinement_20260820/`.

## Kandidat semantik setelah boilerplate

Sebanyak 297 kandidat menjadi terlihat, tetapi statusnya tetap `abstain`:

- Gender: 258; Neutral: 39.
- `commonsense`: 66; `justice`: 34; `deontology`: 67; `virtue`: 130.
- `no_opinion_stop`: 186; `task_substitution`: 111.

Tiga keluarga sudah memiliki bukti kontra dari respons yang pernah dibaca:

- `opening_open_negative`: 26 kandidat; terdapat stance mismatch.
- `midtext_lean_negative`: 20 kandidat; terdapat stance mismatch.
- `conclusion_open_negated_positive`: 11 kandidat; terdapat prompt misread.

Karena satu false positive mewajibkan perbaikan dan audit ulang, ketiga
keluarga itu tidak dipromosikan. Sebanyak 240 kandidat dari keluarga lain juga
belum dipromosikan karena seluruh kuota per aturan belum selesai. Angka 297
tidak boleh ditambahkan ke coverage.

## Hasil status

| Cakupan | Total | Committed | Commit rate | Abstain | Abstain rate | Missing | Missing rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gender sebelum | 72.672 | 55.696 | 76,6402% | 16.594 | 22,8341% | 382 | 0,5256% |
| Gender sesudah | 72.672 | 55.708 | 76,6568% | 16.582 | 22,8176% | 382 | 0,5256% |
| Neutral sesudah | 9.084 | 6.983 | 76,8714% | 1.939 | 21,3452% | 162 | 1,7834% |
| Gabungan sesudah | 81.756 | 62.691 | 76,6806% | 18.521 | 22,6540% | 544 | 0,6654% |

Output baru:

`result/ETHICS/reports/Gender_Persona/_combined/parser_refinement_20260820/all_models_results_full.csv`

SHA-256 output:
`9a442ffeda5b3308b21fd9c398ab2633e3ad835d41346f5fc1d4ed302b5ee5a4`.

## Validasi

- 39 unit/regression cases lolos.
- Uji blind ground truth lolos pada 300 file Gender dan 300 file Neutral.
- 81.756 baris dan key unik; 54 kondisi masing-masing 1.514 soal.
- Semua kondisi mempunyai set `question_id` identik.
- Hanya 309 target berubah: 12 keputusan AITA dan 297 kolom kandidat.
- Response text, identity, token metadata, dan seluruh kolom ground truth
  identik dengan input.
- Transisi `committed -> abstain`: 0.
- Candidate semantik yang berubah menjadi committed: 0.

Uji chi-square `persona x answer_status` tetap signifikan pada keenam model.
Untuk Gender-only, Cramér's V berada pada 0,067–0,220. Karena itu akurasi
antar-persona tetap harus dilaporkan bersama commit rate; refinement ini tidak
menghilangkan differential measurement error.

Akurasi dan distribusi stance per model/persona/subset tersedia pada
`summary_by_condition_subset.csv`. Akurasi `deontology` dan `virtue` tidak boleh
ditafsirkan sebagai kemampuan penilaian moral karena masalah validitas
konstruk yang sudah didokumentasikan.
