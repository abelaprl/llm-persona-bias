import json
from pathlib import Path

root = Path(r"D:\TugasAkhir")
new_text = (
    "Dua metrik utama dihitung pada keempat subset: akurasi penuh "
    "(n_correct/total) dan proporsi wrong pada respons committed "
    "(n_answer_wrong/committed). Grafik dua metrik tersedia pada "
    "00_ethics_<domain>_two_metrics_all_subsets_heatmap."
)
for domain in ("gender", "religion", "regional"):
    folder = root / "notebooks" / domain / "ethics accuracy analysis"
    readme = folder / "README.md"
    if readme.exists():
        text = readme.read_text(encoding="utf-8")
        old_lines = [
            "Akurasi hanya per subset Commonsense/Justice, dibaca bersama commit_rate. Deontology/Virtue memakai distribusi wrong/acceptable.",
            "- Akurasi tetap per subset untuk Commonsense/Justice; Deontology/Virtue memakai stance.",
        ]
        for old in old_lines:
            text = text.replace(old, new_text)
        if "Dua metrik utama" not in text:
            text = new_text + "\n\n" + text
        readme.write_text(text, encoding="utf-8")

    # Update notebook documentation cells without touching the stored result images.
    notebooks = list(folder.glob("*.ipynb"))
    for nb_path in notebooks:
        try:
            nb = json.loads(nb_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        changed = False
        for cell in nb.get("cells", []):
            if cell.get("cell_type") != "markdown":
                continue
            source = "".join(cell.get("source", []))
            replacements = {
                "Akurasi hanya per subset Commonsense/Justice, dibaca bersama commit_rate. Deontology/Virtue memakai distribusi wrong/acceptable.": new_text,
                "- Akurasi tetap per subset untuk Commonsense/Justice; Deontology/Virtue memakai stance.": new_text,
            }
            updated = source
            for old, new in replacements.items():
                updated = updated.replace(old, new)
            if updated != source:
                cell["source"] = updated.splitlines(keepends=True)
                changed = True
        marker = "## Dua metrik utama pada seluruh subset"
        if not any(marker in "".join(c.get("source", [])) for c in nb.get("cells", [])):
            nb.setdefault("cells", []).append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    marker + "\n",
                    "\n",
                    "Setiap subset ETHICS menggunakan dua metrik yang sama:\n",
                    "- **Akurasi penuh** = `n_correct / total` seluruh respons.\n",
                    "- **Proporsi `wrong`** = `n_answer_wrong / committed` respons.\n",
                    "\n",
                    "Visual gabungan: `figures/00_ethics_" + domain + "_two_metrics_all_subsets_heatmap.png`.\n",
                    "Tabel: `tables/ethics_" + domain + "_two_metrics_all_subsets.csv`.\n",
                ],
            })
            changed = True
        if changed:
            nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")

    # Add the new artifact to manifests used as provenance indexes.
    for manifest_path in [
        folder / "bundle_manifest.json",
        folder / f"tables/ethics_{domain}_accuracy_parity_manifest_v1.json",
    ]:
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["primary_metrics_all_subsets"] = {
            "accuracy_full": "n_correct / total",
            "prop_wrong_committed": "n_answer_wrong / committed",
            "subsets": ["commonsense", "deontology", "justice", "virtue"],
        }
        relfig = f"figures/00_ethics_{domain}_two_metrics_all_subsets_heatmap.png"
        if isinstance(manifest.get("figure_files"), list) and relfig not in manifest["figure_files"]:
            manifest["figure_files"].append(relfig)
        if isinstance(manifest.get("active_figure_files"), list) and relfig not in manifest["active_figure_files"]:
            manifest["active_figure_files"].append(relfig)
        notes = manifest.get("notes")
        if isinstance(notes, list):
            notes = [n for n in notes if "accuracy_committed hanya" not in str(n) and "Deontology dan Virtue dilaporkan" not in str(n)]
            notes.append("Dua metrik utama untuk keempat subset: accuracy_full = n_correct/total dan prop_wrong_committed = n_answer_wrong/committed.")
            manifest["notes"] = notes
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
