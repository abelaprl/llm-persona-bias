"""Build a new B02 preview without modifying frozen or active artifacts."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np
import pandas as pd

from .gender_lm_presentation_v2_data import (
    GROUP_MEASUREMENTS, MARKER_GROUPS, PROTOTYPE_MARKERS, effect_view,
)
from .gender_lm_presentation_v2_plots import PRINT_STYLE, plot_direct_prototype
from .gender_lm_reference_data import (
    ACTIVE, COLORS, MODELS, MODEL_LABELS, NOTEBOOK, PERSONAS, SCENARIOS, SOURCE,
    digest, profiles, select_contrasts,
)

SOURCE_FILES = ("registry.json", "comparisons.csv", "marker_summary.csv", "response_markers.csv.gz")


def _snapshot() -> dict:
    paths = [SOURCE / name for name in SOURCE_FILES]
    paths += [ACTIVE / NOTEBOOK]
    paths += sorted(p for p in (ACTIVE / "figures").rglob("*") if p.is_file())
    return {p.resolve().as_posix(): digest(p) for p in paths}


def _validate_registry(registry: list) -> None:
    ids = [marker for group in MARKER_GROUPS.values() for marker in group]
    metadata = {row["marker_id"]: row for row in registry}
    if len(registry) != 41 or len(set(ids)) != 41 or set(ids) != set(metadata):
        raise ValueError("Marker grouping does not match frozen registry")
    for group_id, markers in MARKER_GROUPS.items():
        for marker in markers:
            wanted = "presence" if metadata[marker]["measurement"] == "binary" else "density"
            if GROUP_MEASUREMENTS[group_id] != wanted:
                raise ValueError("Mixed measurement in marker group")


def _geometry(fig) -> dict:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    visible = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text()]
    outside = []
    for artist in visible:
        box = artist.get_window_extent(renderer)
        if box.x0 < -.5 or box.y0 < -.5 or box.x1 > fig.bbox.width + .5 or box.y1 > fig.bbox.height + .5:
            outside.append(artist.get_text())
    smallest = min(t.get_fontsize() for t in visible)
    if outside or smallest < 9:
        raise ValueError(f"Invalid print geometry: outside={outside}, minimum_font={smallest}")
    return dict(width_cm=float(fig.get_figwidth() * 2.54), height_cm=float(fig.get_figheight() * 2.54),
                minimum_font_pt=float(smallest), text_outside_figure=outside)


def _readme(data, registry, report) -> str:
    pairs = report["n_pairs_by_model"]
    coverage = "\n".join(f"| {MODEL_LABELS[m]} | {pairs[m]['min']}–{pairs[m]['max']} |" for m in MODELS)
    def cell(value):
        return str(value).replace("|", "/").replace("\n", " ")
    catalog = "\n".join(
        f"| {r['marker_id']} | {cell(r['operational_definition'])} | {cell(r['theoretical_source'])} | "
        f"{cell(r['analysis_role'])} | {cell(r['custom_adaptation'])} |"
        for r in registry if r["marker_id"] in PROTOTYPE_MARKERS)
    return f"""# Prototipe B02 — Gender ETHICS

## SPEC dan DESIGN PLAN

Prototipe enam marker sikap epistemik, enam model, empat skenario E/I1/I2/I3 pada scope overall. Ini bukan penyelesaian visualisasi seluruh 41 marker. Tidak ada sampling respons, perhitungan ulang uji/CI, ekstraksi ulang, atau integrasi ke notebook/figures aktif.

## IMPLEMENTATION

![B02 — Man minus Woman](B02_direct_overall.png)

Gambar B02. Selisih rerata density Man−Woman (kemunculan per 100 kata). Titik berwarna menyatakan model; garis adalah CI 95% marginal untuk selisih rerata, bukan CI rank-biserial atau CI terkoreksi Holm. E adalah persona eksplisit; I1–I3 tiga narasi persona implisit. Kategori dapat overlap dan tetap berstatus candidate.

[SVG ukuran cetak](B02_direct_overall.svg) · [Tabel pembentuk](B02_direct_overall.csv) · [Profil dan jumlah mentah](marker_context.csv).

CSV pembentuk memuat {len(data)} baris (6 marker × 6 model × 4 skenario). Kolom sumber `effect`, `ci_low`, `ci_high`, `left`, `right`, `panel_gender`, `plot_value`, dan `matched_rank_biserial` tetap berorientasi sumber Woman−Man. Grafik **hanya** membaca `display_effect`, `display_ci_low`, `display_ci_high` yang dibalik menjadi Man−Woman. `display_sign=-1`; p dua sisi, Holm, pasangan, CI method, dan metadata sumber tidak diubah. Kolom `plot_*` adalah warisan adapter dan bukan dasar grafik ini.

`marker_context.csv` berisi 324 baris (6 marker × 6 model × 9 persona), dengan `value`/`prevalence_percent` dalam %, `density_per_100_words`, `raw_occurrences`, `total`, `n_available`, dan `n_unavailable`. Profil memakai seluruh penjelasan tersedia per persona, **bukan** populasi pairwise-complete yang digunakan untuk delta. Tidak boleh menafsirkan selisih profil sebagai delta berpasangan. Jumlah mentah tidak diturunkan dari delta.

Tabel 1. Rentang jumlah pasangan per model pada prototipe, sumber `comparisons.csv`, scope overall, direct B02 density.

| Model | n_pairs minimum–maksimum |
|---|---:|
{coverage}

Estimasi NA: {report['effect_na_count']}; estimasi nol: {report['effect_zero_count']}; estimasi tersedia tanpa CI lengkap: {report['ci_na_count']}; baris dengan flag `low_support` sumber: {report['low_support_count']}. Pasangan eligible dan komposisi empat subset tersedia pada CSV, termasuk `n_pairs_by_subset` dan `effective_subset_weights`. Model berpasangan sedikit tidak memiliki ketelitian setara dengan model berpasangan banyak. Status penjelasan tidak sama dengan status parser.

Tabel 2. Definisi operasional dan provenance enam marker dari registry frozen; pengelompokan B02 tidak membuat skor gabungan atau mengubah keluarga uji.

| Marker | Definisi sumber | Sumber teori | Peran analisis | Adaptasi/custom |
|---|---|---|---|---|
{catalog}

## CHECKLIST

- [x] 144 kontras dan 324 konteks cocok dengan tabel frozen.
- [x] Empat panel, enam warna model, 16 × 22 cm; font minimum 9 pt; PNG 300 dpi + SVG.
- [x] Semua kolom sumber dipertahankan, hash sumber dan artefak aktif sama sebelum/sesudah.
- [ ] Review keterbacaan oleh pengguna dan persetujuan perluasan 41 marker.
- [ ] Integrasi ke notebook/figures utama; belum dilakukan.

## TESTING

1. Jalankan `python -m unittest src.benchmarks.ethics.tests.test_gender_lm_presentation_v2 -v` dari akar repositori.
2. Untuk reproduksi, gunakan direktori baru: `python -m src.benchmarks.ethics.build_gender_lm_presentation_v2_preview --output result/benchmarks/ethics/reports/gender_lm_presentation_v2_preview_<tanggal>_<versi-baru>`. Path existing, termasuk direktori kosong, ditolak.
3. Baca `verification.json` dan `source_hashes.json`; periksa PNG pada lebar fisik 16 cm dan bandingkan titik/CI dengan kolom `display_*` dalam CSV.

## INTEGRATION dan keterbatasan

Sumber: `{SOURCE.resolve().as_posix()}`. Notebook aktif tetap `{(ACTIVE / NOTEBOOK).resolve().as_posix()}`; tidak ada pemasangan atau penghapusan. Task 4 rencana perluasan menunggu review prototipe.

Density memakai rerata selisih per respons pada pasangan soal yang sama. Uji Wilcoxon sumber tidak menguji estimand rerata yang sama dengan CI ini; Holm tetap disajikan pada tabel dan tidak menjadi pewarnaan signifikansi CI. CI sumber memakai subset-stratified paired percentile. Ringkasan overall mengikuti komposisi pasangan eligible, bukan rata-rata empat subset berbobot sama dan bukan angka akurasi ETHICS gabungan. Deontology/virtue tetap memiliki batasan tugas pada AGENTS.md. I1–I3 bukan replikasi acak populasi gender; perbedaan prompt tidak membuktikan sifat kelompok manusia. Marker tidak terdeteksi bukan bukti tidak adanya bias. Revisi presentasi tidak menyelesaikan validasi parser/kamus candidate.
"""


def build_preview(output: Path) -> dict:
    """Validate before creating output; existing targets are never overwritten."""
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output already exists: {output}")
    print("[1/5] Memeriksa hash sumber dan artefak aktif", flush=True)
    before = _snapshot()
    registry = json.loads((SOURCE / "registry.json").read_text(encoding="utf-8"))
    _validate_registry(registry)
    print("[2/5] Membaca dan memvalidasi tabel frozen", flush=True)
    comparisons = pd.read_csv(SOURCE / "comparisons.csv", float_precision="round_trip")
    selected = select_contrasts(comparisons, PROTOTYPE_MARKERS, "direct", "density")
    if len(selected) != 144 or not selected.scope.eq("overall").all():
        raise ValueError("Expected 144 overall direct B02 density rows")
    if not (selected.ci_level.eq(.95).all() and selected.ci_simultaneous.eq(False).all()
            and selected.ci_method.eq("subset_stratified_paired_percentile").all()):
        raise ValueError("Unexpected CI metadata for prototype caption")
    display = effect_view(selected, "direct")
    pd.testing.assert_frame_equal(display[selected.columns], selected, check_exact=True)
    summary = pd.read_csv(SOURCE / "marker_summary.csv", float_precision="round_trip")
    context = profiles(summary)
    context = context.loc[context.marker_id.isin(PROTOTYPE_MARKERS)].copy()
    keys = ["model_id", "persona", "marker_id"]
    expected = {(model, persona, marker) for model in MODELS for persona in PERSONAS for marker in PROTOTYPE_MARKERS}
    if (len(context) != 324 or context.duplicated(keys).any()
            or set(context[keys].itertuples(index=False, name=None)) != expected):
        raise ValueError("Expected 324 unique overall context rows")
    context["prevalence_percent"] = context.value
    print("[3/5] Merender empat panel pada ukuran 16 x 22 cm", flush=True)
    fig = plot_direct_prototype(display)
    try:
        geometry = _geometry(fig)
        report = dict(
            contrast_rows=len(display), context_rows=len(context), panels=len(fig.axes), models=len(MODELS),
            marker_count=len(PROTOTYPE_MARKERS), marker_ids=PROTOTYPE_MARKERS, scope="overall",
            direction="Man - Woman", measurement="density", dpi=300,
            effect_na_count=int(display.display_effect.isna().sum()),
            effect_zero_count=int(display.display_effect.eq(0).sum()),
            ci_na_count=int((display.display_effect.notna()
                             & display[["display_ci_low", "display_ci_high"]].isna().any(axis=1)).sum()),
            low_support_count=int(display.low_support.eq(True).sum()),
            n_pairs_by_model={model: {"min": int(rows.n_pairs.min()), "max": int(rows.n_pairs.max())}
                              for model, rows in display.groupby("model_id")},
            source_columns_equal=True, context_source_equal=True, geometry=geometry,
        )
        # All schema, key and geometry checks precede the first output write.
        output.mkdir(parents=True, exist_ok=False)
        print("[4/5] Menulis PNG, SVG, dan tabel pendamping ke path baru", flush=True)
        with plt.rc_context(PRINT_STYLE):
            fig.savefig(output / "B02_direct_overall.png", dpi=300)
            fig.savefig(output / "B02_direct_overall.svg")
        display.to_csv(output / "B02_direct_overall.csv", index=False)
        context.to_csv(output / "marker_context.csv", index=False)
    finally:
        plt.close(fig)
    print("[5/5] Mencocokkan ekspor dan hash sebelum/sesudah", flush=True)
    exported = pd.read_csv(output / "B02_direct_overall.csv", float_precision="round_trip")
    exported_context = pd.read_csv(output / "marker_context.csv", float_precision="round_trip")
    pd.testing.assert_frame_equal(exported, display.reset_index(drop=True), check_dtype=False, check_exact=True)
    pd.testing.assert_frame_equal(exported_context, context.reset_index(drop=True), check_dtype=False, check_exact=True)
    after = _snapshot()
    report["source_unchanged"] = all(after.get(p) == value for p, value in before.items()
                                     if p.startswith(SOURCE.resolve().as_posix() + "/"))
    report["active_unchanged"] = ({p: value for p, value in before.items() if p.startswith(ACTIVE.resolve().as_posix() + "/")}
                                  == {p: value for p, value in after.items() if p.startswith(ACTIVE.resolve().as_posix() + "/")})
    if before != after:
        raise RuntimeError("Source or active artifacts changed during build; preview is unverified")
    code_names = ["gender_lm_presentation_v2_data.py", "gender_lm_presentation_v2_plots.py",
                  "build_gender_lm_presentation_v2_preview.py", "gender_lm_reference_data.py",
                  "gender_lm_reference_plots.py", "restyle_gender_lm_gsm8k.py"]
    hashes = dict(before=before, after=after,
                  code={name: digest(Path(__file__).with_name(name)) for name in code_names})
    (output / "source_hashes.json").write_text(json.dumps(hashes, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "verification.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    (output / "README.md").write_text(_readme(display, registry, report), encoding="utf-8")
    print(f"Selesai: {len(display)} kontras, {len(context)} konteks; sumber/artefak aktif tidak berubah.", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_preview(args.output), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
