# Struktur hasil eksperimen

Folder ini menyimpan keluaran benchmark dan analisis. Keluaran baru memakai
`run_id` yang eksplisit dan tidak menimpa run lama.

```text
result/
├── benchmarks/
│   ├── ethics/
│   │   ├── runs/       # respons mentah dan eksperimen per run
│   │   ├── parsed/     # keluaran parsing
│   │   ├── reports/    # tabel dan visual ETHICS
│   │   ├── audits/     # audit dan catatan validasi
│   │   ├── logs/       # log eksekusi
│   │   └── archive/    # backup dan keluaran lama
│   ├── gsm8k/
│   ├── eli5/
│   ├── squad/
│   └── truthfulqa/
├── analyses/
│   └── linguistic_markers/
│       ├── runs/
│       ├── logs/
│       └── audits/
└── archive/
    └── reports_mixed_legacy/  # reports/ lama yang mencampur benchmark
```

Folder pilot ETHICS saat ini berada di
`benchmarks/ethics/runs/pilot_test/`. Folder `archive/` dipertahankan untuk
provenance dan tidak boleh dipakai sebagai lokasi keluaran baru.
