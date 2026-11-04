# Struktur keluaran GSM8K

Respons mentah disimpan di bawah `runs/` dan laporan analisis di bawah
`reports/`.

```text
gsm8k/
├── runs/
│   ├── main/
│   │   ├── Age_Group_Persona/<model>/<persona>/
│   │   ├── Gender_Persona/<model>/<persona>/
│   │   ├── regional_cultural_persona/<model>/<persona>/
│   │   └── Religion_Persona/<model>/<persona>/
│   └── reruns/<run_id>/
└── reports/
```

File JSON `gsm8k_*.json` adalah respons per soal. CSV/TXT ekspor historis
dipisahkan di `reports/legacy_exports/runs_main/` dan bukan input runner baru.
