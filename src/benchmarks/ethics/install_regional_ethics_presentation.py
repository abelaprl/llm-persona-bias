"""Install only verified Regional artifacts, with guarded rollback."""
import argparse
import json
from pathlib import Path
import shutil
import uuid
import nbformat
from .regional_ethics_presentation_data import ROOT, ACTIVE, sha256, write_json
from .regional_ethics_presentation_notebook import NOTEBOOK_NAME, make_notebook, validate_notebook
from .build_regional_ethics_presentation import verify_bundle
# Reuse only the generic journal; never call the Religion installer.
from .install_religion_ethics_print import InstallTransaction


def publish_files(source, active, archive, previous, names, fail_after=None):
    source, active, archive = map(lambda p: Path(p).resolve(), [source, active, archive])
    retired = sorted(set(previous) - set(names))
    for name in [*names, *retired]:
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('Unsafe install-relative path')
        target = (active / name).resolve()
        if not target.is_relative_to(active) or not (source / name).resolve().is_relative_to(source):
            raise ValueError('Install path escapes scope')
        if target.exists() and (name not in previous or sha256(target) != previous[name]):
            raise ValueError('Unknown or concurrently modified target: ' + str(target))
    transaction = InstallTransaction(archive, [source, active, archive])
    try:
        for name in retired:
            target = (active / name).resolve()
            if not target.exists() or sha256(target) != previous[name]:
                raise ValueError('Retired managed asset changed after preflight')
            backup = archive / 'previous' / name
            backup.parent.mkdir(parents=True, exist_ok=True)
            transaction.move(target, backup)
        for count, name in enumerate(names, 1):
            target = active / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                backup = archive / 'previous' / name
                backup.parent.mkdir(parents=True, exist_ok=True)
                if sha256(target) != previous[name]:
                    raise ValueError('Target changed after preflight')
                transaction.move(target, backup)
            transaction.copy_new(source / name, target)
            if fail_after == count:
                raise RuntimeError('Injected installation failure')
    except BaseException:
        transaction.rollback()
        raise
    return transaction


def install_verified(bundle, active_dir=ACTIVE):
    bundle, active = Path(bundle).resolve(), Path(active_dir).resolve()
    if active != ACTIVE.resolve() or bundle.parent != (ROOT / 'result/benchmarks/ethics/reports').resolve():
        raise ValueError('Installer target outside approved Regional scope')
    verification = verify_bundle(bundle)
    previous = {}
    old_manifest = active / 'installation.json'
    if old_manifest.is_file():
        old = json.loads(old_manifest.read_text(encoding='utf-8'))
        previous = old['managed_hashes']
        previous['installation.json'] = sha256(old_manifest)
    active.mkdir(parents=True, exist_ok=True)
    before = {str(p.relative_to(active)): sha256(p) for p in active.rglob('*') if p.is_file() and 'archive' not in p.relative_to(active).parts}
    archive = active / 'archive' / ('install_' + bundle.name + '_' + uuid.uuid4().hex[:8])
    archive.mkdir(parents=True, exist_ok=False)
    prepared = archive / 'prepared'
    prepared.mkdir()
    from .religion_ethics_gsm8k_style_notebook import execute_notebook
    notebook = execute_notebook(make_notebook(bundle, active))
    nbformat.write(notebook, prepared / NOTEBOOK_NAME)
    # Link validation uses the intended document directory, not the staging archive.
    validate_notebook(prepared / NOTEBOOK_NAME, bundle, document_dir=active)
    (prepared / 'figures').mkdir()
    for path in (bundle / 'figures').iterdir():
        if not path.is_file() or path.suffix not in ['.png', '.svg']:
            raise ValueError('Figures must be a flat PNG/SVG directory')
        shutil.copy2(path, prepared / 'figures' / path.name)
    relative_bundle = Path(__import__('os').path.relpath(bundle, active)).as_posix()
    (prepared / 'README.md').write_text(f'''# Regional/Cultural ETHICS — linguistic marker

Notebook utama: [{NOTEBOOK_NAME}]({NOTEBOOK_NAME}). Semua gambar aktif ada langsung di `figures/`, tanpa subfolder r1/r2.

Status **candidate**, belum audit dua reviewer manusia. Sumber frozen Regional v3 r5; policy explanation-v1. Label-only tidak menjadi nol marker berpenjelasan; abstain berteks tetap masuk. Tidak ada akurasi ETHICS gabungan.

- [Hasil verifikasi]({relative_bundle}/verification.json)
- [Katalog 53 marker]({relative_bundle}/tables/marker_catalog.csv)
- [Teks respons utuh dan sorotan]({relative_bundle}/evidence/response_cards.html)
- [Seluruh kontras baru]({relative_bundle}/tables/statistics/paired_contrasts.csv.gz)
- [Perbandingan lama–baru]({relative_bundle}/tables/statistics/old_new_comparison.csv.gz)

Kelengkapan 23 kategori regional bukan bukti seluruh cara budaya diungkapkan telah tertangkap. Nama wilayah merujuk persona eksperimen, bukan generalisasi penduduk. Overlap literal bukan hubungan kausal.

Notebook v2/atlas dan result lama tidak dihapus. Folder `archive/` menyimpan jurnal pemasangan dan pemulihan; berkas tak dikenal tidak ditimpa.
''', encoding='utf-8')
    names = [str(p.relative_to(prepared)) for p in prepared.rglob('*') if p.is_file()]
    report = dict(bundle=str(bundle), verification=verification, archive=str(archive),
                  managed_hashes={name: sha256(prepared / name) for name in names})
    write_json(prepared / 'installation.json', report)
    names += ['installation.json']
    current = {str(p.relative_to(active)): sha256(p) for p in active.rglob('*') if p.is_file() and 'archive' not in p.relative_to(active).parts}
    if current != before:
        raise ValueError('Active workspace changed during preparation')
    transaction = publish_files(prepared, active, archive, previous, names)
    try:
        for name, expected in report['managed_hashes'].items():
            if sha256(active / name) != expected:
                raise ValueError('Installed hash mismatch')
        validate_notebook(active / NOTEBOOK_NAME, bundle)
        verify_bundle(bundle)
    except BaseException:
        transaction.rollback()
        raise
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(install_verified(args.bundle), indent=2))
