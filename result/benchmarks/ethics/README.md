# Struktur keluaran ETHICS

```text
ethics/
├── runs/
│   ├── main/v9/
│   │   ├── Gender Persona/<model>/<persona>/
│   │   ├── Neutral Persona/<model>/neutral/
│   │   ├── Regional Cultural Persona/<model>/<persona>/
│   │   └── Religion Persona/<model>/<persona>/
│   ├── pilot/v9/
│   └── reruns/<run_id>/
├── parsed/legacy_parsing/v9_main/
├── reports/
├── audits/
├── logs/
└── archive/
```

`runs/main/v9` adalah input utama parser. `runs/pilot/v9` adalah keluaran
pilot dan prompt variant lama; jangan dipakai sebagai denominator utama tanpa
memilih run secara eksplisit. CSV/TXT historis dari pilot berada di
`reports/legacy_exports/pilot_v9/`.
