# Struktur hasil benchmark

Dokumen ini menjelaskan struktur kanonik hasil benchmark setelah perapian pada
6 Oktober 2026. Struktur ini memisahkan data mentah, hasil parsing, laporan,
dan artefak audit tanpa menimpa hasil lama.

## Struktur kanonik

```text
result/benchmarks/
├── README.md
├── gsm8k/
│   ├── runs/
│   │   ├── main/
│   │   │   └── <persona_group>/<model>/<persona>/*.json
│   │   └── reruns/
│   │       └── <run_id>/
│   ├── reports/
│   │   └── legacy_exports/
│   │       └── runs_main/
│   └── README.md
└── ethics/
    ├── runs/
    │   ├── main/
    │   │   └── v9/<persona_group>/<model>/<persona>/*.json
    │   ├── pilot/
    │   │   └── v9/
    │   └── reruns/
    │       └── <run_id>/
    ├── parsed/
    │   └── legacy_parsing/
    │       └── v9_main/
    ├── reports/
    │   └── legacy_exports/
    │       └── pilot_v9/
    ├── audits/
    ├── logs/
    ├── archive/
    └── README.md
```

## Arti setiap area

- `runs/main/` adalah input mentah utama. File JSON di sini dipertahankan
  sebagai artefak respons model.
- `runs/pilot/` berisi percobaan pilot dan tidak digunakan sebagai denominator
  utama hasil final.
- `runs/reruns/` berisi percobaan ulang yang dipisahkan dari run utama.
- `parsed/` berisi hasil parsing atau cache parsing lama. Area ini bukan sumber
  respons mentah.
- `reports/` berisi tabel, ringkasan, dan keluaran analisis.
- `reports/legacy_exports/` menampung ekspor CSV/TXT lama yang dikeluarkan dari
  direktori run agar direktori run hanya berisi data respons.
- `audits/` berisi artefak pemeriksaan dan validasi.
- `logs/` berisi log eksekusi.
- `archive/` berisi artefak historis yang memang sudah dipisahkan dari alur
  utama.

## Perubahan yang sudah dilakukan

### GSM8K

Kelompok persona yang sebelumnya berada langsung di
`result/benchmarks/gsm8k/` dipindahkan ke:

```text
result/benchmarks/gsm8k/runs/main/
```

Kelompok yang dipertahankan adalah `Age_Group_Persona`, `Gender_Persona`,
`regional_cultural_persona`, dan `Religion_Persona`. Isi `reruns` dipindahkan
ke `runs/reruns/`.

Sebanyak 76 file CSV/TXT dari `runs/main/` dipindahkan ke
`reports/legacy_exports/runs_main/` dengan struktur relatif yang dipertahankan.

### ETHICS

Run utama dipindahkan dari `runs/v9_main/` ke:

```text
result/benchmarks/ethics/runs/main/v9/
```

Pilot dipindahkan dari `runs/pilot_test/` ke `runs/pilot/v9/`, sedangkan
percobaan ulang dipindahkan dari `runs/rerun_attempts/` ke `runs/reruns/`.

Di dalam run utama, respons dikelompokkan menurut persona group:

```text
runs/main/v9/
├── Gender Persona/
├── Neutral Persona/
├── Regional Cultural Persona/
└── Religion Persona/
```

Struktur model neutral yang sebelumnya berisi file JSON langsung juga
dinormalisasi agar konsisten dengan kelompok lain: file ditempatkan di bawah
subfolder persona `neutral`. Pada model DeepSeek, 1.514 file JSON dipindahkan
ke struktur tersebut.

Sebanyak 196 file CSV/TXT dari pilot dipindahkan ke
`reports/legacy_exports/pilot_v9/`. Setelah pemindahan, direktori raw run utama
GSM8K dan direktori pilot ETHICS tidak lagi memiliki file non-JSON.

## Aturan penggunaan

1. Respons mentah baru masuk ke `runs/main/`, `runs/pilot/`, atau `runs/reruns/`
   sesuai konteks eksekusinya.
2. Hasil parsing baru ditulis ke path baru di bawah `parsed/` atau `reports/`;
   jangan menimpa hasil lama.
3. Pilot tidak boleh digabungkan diam-diam ke denominator hasil utama.
4. Nama model, persona, dan run harus tetap dapat ditelusuri ke sumber
   responsnya.
5. File CSV/TXT hasil ekspor tidak ditempatkan kembali di direktori raw run.
6. Penghapusan atau penimpaan file di `result/` harus dikonfirmasi terlebih
   dahulu.

## Validasi setelah perapian

- Seluruh 662 file Python di `src/` berhasil diparse dengan AST; 0 file
  bermasalah.
- 14 pengujian statistik pada
  `src.core.linguistic_markers.tests.test_loaders_and_statistics.StatisticsTests`
  berhasil.
- Tiga CLI utama berhasil menampilkan bantuan (`run_parse.py`, runner ETHICS
  v9, dan runner GSM8K regional-cultural).
- Referensi path aktif pada konfigurasi dan runner telah diarahkan ke struktur
  baru.

Validasi loader lengkap belum diklaim lulus karena sebagian pengujian mencoba
menulis direktori temporary yang dibatasi oleh environment Windows saat
pengujian dijalankan.

## Catatan report historis

Sebagian direktori di bawah `result/benchmarks/*/reports/` masih memiliki nama
historis atau nama eksperimen lama yang ambigu. Direktori tersebut sengaja tidak
dipindahkan secara massal karena pemindahan tanpa klasifikasi dapat merusak
provenance. Hanya ekspor CSV/TXT yang jelas berasal dari raw run yang dipisahkan
ke `reports/legacy_exports/`.

README khusus benchmark tersedia di:

- `result/benchmarks/gsm8k/README.md`
- `result/benchmarks/ethics/README.md`

Perapian ini belum dibuatkan commit atau push Git.
