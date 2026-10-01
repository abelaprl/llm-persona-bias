"""Preview one figure without the NA gutter; do not modify active artifacts."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.benchmarks.ethics import build_gender_ethics_lm_print as plots


def main():
    output = ROOT / "result/ETHICS/reports/gender_lm_preview_without_na_20261002_r2"
    if output.exists():
        raise FileExistsError(output)
    fid = "02_bahasa_etika_man_minus_woman"
    csv = plots.ACTIVE / "figures" / (fid + ".csv")
    active_png = plots.ACTIVE / "figures" / (fid + ".png")
    original_hashes = {str(p): plots.direct.sha256(p) for p in (csv, active_png)}
    table = pd.read_csv(csv, float_precision="round_trip")
    snapshot = table.copy(deep=True)
    registry = pd.DataFrame(json.loads((plots.data.SOURCE / "registry.json").read_text(encoding="utf-8")))
    ids = [m for m in registry.marker_id if m in set(table.marker_id)]
    fig = plots.direct.plot_direct_effects(table, ids, plots.TITLES[fid], "density")
    unit = plots.configure_effect_axes(fig, table, ids, True)
    plots.decorate(fig, fig.axes[:], plots.TITLES[fid], unit + "\nMan − Woman", 20.6,
                   categorical=True, marker_ids=ids, median=True, significance=True)
    removed = 0
    for ax in fig.axes:
        retained = []
        for collection in list(ax.collections):
            xy = collection.get_offsets()
            if len(xy) and np.allclose(xy[:, 0], 1.135):
                collection.remove()
                removed += len(xy)
            else:
                retained.append((collection, xy.copy()))
        for patch in list(ax.patches):
            patch.remove()
        for label in list(ax.texts):
            if label.get_text() == "NA":
                label.remove()
        ax.set_xlim(-1.04, 1.04)
        for collection, positions in retained:
            np.testing.assert_array_equal(collection.get_offsets(), positions)
    absent = table.loc[table.display_effect.isna()]
    assert removed == len(absent) == 25
    assert absent.status.eq("no_change").all() and absent.n_changed.eq(0).all()
    pd.testing.assert_frame_equal(table, snapshot)
    output.mkdir(parents=True)
    fig.savefig(output / (fid + "_tanpa_na.png"), dpi=300)
    assert all(plots.direct.sha256(Path(p)) == h for p, h in original_hashes.items())
    metadata = dict(preview_only=True, active_artifacts_modified=False, source_hashes=original_hashes,
                    omitted_undefined_estimates=removed, finite_estimates=int(table.display_effect.notna().sum()),
                    caption="Sebanyak 25 perbandingan tanpa perubahan nilai penanda tidak ditampilkan karena rank-biserial tidak terdefinisi.")
    (output / "preview_manifest.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(output / (fid + "_tanpa_na.png"))


if __name__ == "__main__":
    main()
