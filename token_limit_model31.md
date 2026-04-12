Hasil analisis model 31 (allenai_olmo-3-32b-think) - SEMUA 10,550 file:

Persona	Total	Correct	Wrong	NoExt	Empty	LenLim	Trunc	Acc
man_explicit_1	1319	1125	46	148	281	234	89	85.3%
man_implicit_1	1319	1078	51	190	247	201	138	81.7%
man_implicit_2	1319	1120	42	157	248	169	153	84.9%
man_implicit_3	1318	1123	48	147	219	169	165	85.2%
woman_explicit_1	1318	1112	54	152	293	237	62	84.4%
woman_implicit_1	1319	1042	54	223	259	213	136	79.0%
woman_implicit_2	1319	1070	52	197	238	185	201	81.1%
woman_implicit_3	1319	1126	43	150	221	180	149	85.4%
TOTAL	10550	8796	390	1364	2006	1588	1093	83.4%
Kesimpulan:

LenLim = 1588 (15%) → Model kehabisan token (max_tokens=2048) saat masih berpikir. Ini model "thinking" yang butuh token lebih banyak.

Empty = 2006 → Content kosong, kebanyakan karena hit token limit

Akurasi naik 73.4% → 83.4% setelah saya tambahkan ekstraksi dari reasoning field

Wrong = 390 (3.7%) → Model salah hitung, bukan masalah extraction

Pattern extraction terlewat = ~415 → Perlu perbaiki pattern lagi (misal **48 grams**)

Jadi EMPTY di model ini = model thinking yang kehabisan token, bukan error API.

Claude Opus 4.5 • 3x