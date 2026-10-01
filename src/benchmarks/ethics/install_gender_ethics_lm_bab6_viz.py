"""Install and execute the Bab VI presentation in the existing ETHICS notebook.

Original presentation files are archived before any replacement. Frozen result
bundles and historical figure files are never overwritten or removed.
"""
from __future__ import annotations

from datetime import datetime
import html
import json
from pathlib import Path
import shutil

import nbformat
from nbclient import NotebookClient
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
ACTIVE = ROOT / "notebooks/gender/linguistic marker/ETHICS"
MAIN = ROOT / "result/ETHICS/reports/gender_lm_bab6_viz_20261002_r3"
SUPPORT = ROOT / "result/ETHICS/reports/gender_lm_reference_presentation_20260920_r3"
NOTEBOOK = "ETHICS_Gender_Linguistic_Markers_Analysis_v1_5_candidate.ipynb"


def install():
    from hashlib import sha256

    def digest(path):
        return sha256(path.read_bytes()).hexdigest()

    manifests = [json.loads((p / "manifest.json").read_text(encoding="utf-8"))
                 for p in (SUPPORT, MAIN)]
    hashes = {}
    for manifest in manifests:
        hashes.update(manifest["source_hashes"])
    for relative, expected in hashes.items():
        assert digest(ROOT / relative) == expected, relative

    main_index = pd.read_csv(MAIN / "figure_index.csv")
    old_index = pd.read_csv(SUPPORT / "figure_index.csv")
    retired = {"01_cakupan", "06_efek_langsung_umum", "07_efek_langsung_domain", "08_efek_langsung_biner"}
    combined = pd.concat([main_index, old_index.loc[~old_index.figure_id.isin(retired)]], ignore_index=True)
    order = ["00_cakupan", "02_panjang_penjelasan", *main_index.figure_id.tolist()[1:],
             "03_profil_umum_explicit", "04_profil_etika_explicit", "05_profil_gender_semua_skenario",
             "09_gender_terhadap_netral", "10_gender_implicit_vs_explicit", "11_keragaman_kategori", "12_konteks_rujukan"]
    combined = combined.set_index("figure_id").loc[order].reset_index()
    assert len(combined) == 13
    for row in combined.itertuples():
        bundle = MAIN if row.figure_id in set(main_index.figure_id) else SUPPORT
        for ext in ("png", "svg", "csv"):
            source = bundle / "figures" / f"{row.figure_id}.{ext}"
            assert digest(source) == getattr(row, f"{ext}_sha256"), source
            target = ACTIVE / "figures" / source.name
            if target.exists():
                assert digest(target) == digest(source), f"Figure collision: {target}"

    notebook = nbformat.read(ACTIVE / NOTEBOOK, as_version=4)
    assert "## 3. Perbandingan langsung Woman" in notebook.cells[6].source, "Unexpected notebook version"
    backup = ACTIVE / "_arsip" / ("sebelum_bab6_viz_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    backup.mkdir(parents=True, exist_ok=False)
    originals = [NOTEBOOK, "README.md", "figure_index.csv", "gallery.html", "manifest.json", "presentation_source.json"]
    backup_hashes = {}
    for name in originals:
        shutil.copy2(ACTIVE / name, backup / name)
        backup_hashes[name] = digest(ACTIVE / name)
        assert digest(backup / name) == backup_hashes[name]
    (backup / "backup_hashes.json").write_text(json.dumps(backup_hashes, indent=2), encoding="utf-8")

    notebook.cells[0].source += "\n\nVisual utama diperbarui: **Man − Woman**, konsisten dengan Gender GSM8K. Pembaruan hanya presentasi, bukan scoring akurasi atau penghitungan marker ulang."
    notebook.cells[1].source = notebook.cells[1].source.replace(
        "DOCUMENT = ACTIVE if Path.cwd() != BUNDLE else BUNDLE", "DOCUMENT = ACTIVE"
    ).replace("pd.read_csv(BUNDLE / 'figure_index.csv')", "pd.read_csv(ACTIVE / 'figure_index.csv')").replace(
        "(BUNDLE / 'manifest.json')", "(ACTIVE / 'manifest.json')"
    )
    notebook.cells[3].source = "show('00_cakupan')\nshow('02_panjang_penjelasan')"
    notebook.cells[6].source = """## 2. Hasil utama: perbandingan langsung Man − Woman

Empat panel E/I1/I2/I3 dan enam warna model mengikuti Gender GSM8K. **Positif = lebih tinggi pada Man; negatif = lebih tinggi pada Woman.** Seluruh 41 marker tercakup: 14 density bahasa umum, 12 density bahasa etika, 4 density wacana gender, dan 11 marker biner.

Density per 100 kata dibandingkan menggunakan matched rank-biserial. CI mean density pada sumber tidak ditempelkan pada rank-biserial. Marker biner memakai selisih prevalence dalam poin persentase dengan CI 95% marginal. Berlian median hanya muncul bila keenam model mempunyai estimasi tersedia; NA tidak dijadikan nol.

Warna pekat menunjukkan p Holm < 0,05; pudar bukan berarti efek nol. Holm mengikuti keluarga pengujian sumber, bukan koreksi global seluruh grafik. Ukuran titik rank menunjukkan jumlah pasangan dengan selisih bukan nol. `n_pairs` dan `n_changed` tersedia pada CSV. Kolom sumber tetap Woman−Man; kolom `display_*` mencatat pembalikan arah secara eksplisit. Tidak ada uji statistik baru."""
    notebook.cells[7].source = "\n".join(f"show('{name}')" for name in main_index.figure_id.tolist()[1:])
    notebook.cells[4].source = notebook.cells[4].source.replace("## 2. Profil marker", "## 3. Profil pendukung marker")
    notebook.cells[13].source = notebook.cells[13].source.replace(
        "    selected = select_contrasts(comparisons, marker_ids, comparison, measurement)\n    fig = effects(selected, marker_ids, 'Penjelajah kontras terpilih', comparison, measurement)",
        "    if comparison == 'direct':\n        from src.benchmarks.ethics.build_gender_ethics_lm_bab6_viz import select_direct, plot_direct_effects\n        selected = select_direct(comparisons, marker_ids, measurement)\n        fig = plot_direct_effects(selected, marker_ids, 'Penjelajah langsung Man − Woman', measurement)\n    else:\n        selected = select_contrasts(comparisons, marker_ids, comparison, measurement)\n        fig = effects(selected, marker_ids, 'Penjelajah kontras terpilih', comparison, measurement)"
    )
    notebook.cells[12].source += "\n\nTabel `chosen` mempertahankan arah `left − right` sumber. Untuk grafik langsung berarah Man − Woman, gunakan `lihat_efek(['stance_hedge'], comparison='direct')`."
    notebook.cells[16].source = """## Reproduksi

Menjalankan notebook memuat hasil dan memeriksa hash sumber, tanpa API maupun scoring ulang. Untuk membuat kembali lima visual utama, pilih direktori output yang belum ada:

```powershell
.\\.venv\\Scripts\\python.exe -X utf8 src/benchmarks/ethics/build_gender_ethics_lm_bab6_viz.py --output result/ETHICS/reports/gender_lm_bab6_viz_reproduction_01
```

Grafik pendukung dan tabel keragaman tetap berasal dari `gender_lm_reference_presentation_20260920_r3`. Gambar lama disimpan untuk penelusuran, tetapi tidak ditampilkan sebagai hasil utama."""
    notebook.cells = [notebook.cells[i] for i in [0, 1, 2, 3, 6, 7, 4, 5, *range(8, 17)]]
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None

    for row in main_index.itertuples():
        for ext in ("png", "svg", "csv"):
            filename = f"{row.figure_id}.{ext}"
            target = ACTIVE / "figures" / filename
            if not target.exists():
                shutil.copy2(MAIN / "figures" / filename, target)
    combined.to_csv(ACTIVE / "figure_index.csv", index=False)
    active_manifest = {**manifests[1], "figure_count": 13, "main_figure_count": 5,
                       "source_hashes": hashes, "supporting_bundle": SUPPORT.relative_to(ROOT).as_posix(),
                       "previous_presentation": backup.relative_to(ACTIVE).as_posix()}
    (ACTIVE / "manifest.json").write_text(json.dumps(active_manifest, indent=2), encoding="utf-8")
    provenance = json.loads((backup / "presentation_source.json").read_text(encoding="utf-8"))
    provenance.update(figure_bundle=MAIN.relative_to(ROOT).as_posix(),
                      supporting_bundle=SUPPORT.relative_to(ROOT).as_posix(),
                      style_version="gender-ethics-bab6-man-minus-woman-v3",
                      previous_active_artifacts=backup.relative_to(ACTIVE).as_posix())
    (ACTIVE / "presentation_source.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    sections = []
    for row in combined.itertuples():
        sections.append(f'<section><h2>{html.escape(row.figure_id)}</h2><img src="figures/{row.figure_id}.png"><p>{html.escape(row.caption)}</p>'
                        + " · ".join(f'<a href="figures/{row.figure_id}.{ext}">{ext.upper()}</a>' for ext in ("png", "svg", "csv")) + '</section>')
    (ACTIVE / "gallery.html").write_text('<!doctype html><html lang="id"><meta charset="utf-8"><title>Gender ETHICS — linguistic marker</title><style>body{font-family:Arial,sans-serif;margin:32px}img{max-width:100%}section{margin-bottom:48px}</style><h1>Gender ETHICS — linguistic marker</h1><p>Candidate; perbandingan langsung Man − Woman. Statistik sumber tidak berubah.</p>' + "\n".join(sections) + '</html>', encoding="utf-8")
    print(f"Backup verified: {backup}", flush=True)
    print("Executing the active notebook with 13 figures...", flush=True)
    try:
        NotebookClient(notebook, timeout=180, kernel_name="python3", resources={"metadata": {"path": str(ACTIVE)}}).execute()
        errors = [output for cell in notebook.cells if cell.cell_type == "code" for output in cell.outputs if output.output_type == "error"]
        images = sum("image/png" in output.get("data", {}) for cell in notebook.cells if cell.cell_type == "code" for output in cell.outputs)
        assert not errors and images == 13, (errors, images)
        for relative, expected in hashes.items():
            assert digest(ROOT / relative) == expected, relative
        nbformat.write(notebook, ACTIVE / NOTEBOOK)
    except Exception:
        for name in originals:
            shutil.copy2(backup / name, ACTIVE / name)
        raise
    print(f"Installed and executed: {ACTIVE / NOTEBOOK}; {images} embedded figures; frozen hashes unchanged", flush=True)


if __name__ == "__main__":
    install()
