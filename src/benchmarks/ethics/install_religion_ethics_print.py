"""Install a verified presentation with hash guards and recoverable archives."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys
import uuid

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import nbformat
import pandas as pd
from src.benchmarks.ethics.build_religion_ethics_gsm8k_style import sha256, write_json
from src.benchmarks.ethics.build_religion_ethics_expression import notebook_links
from src.benchmarks.ethics.build_religion_ethics_print import validate_main_cache, BASE, NOTEBOOK
from src.benchmarks.ethics.religion_ethics_gsm8k_style_notebook import rebase_notebook, execute_notebook


class InstallTransaction:
    """Journal explicit file operations; retain failed outputs for inspection."""

    def __init__(self, archive, allowed_roots):
        self.storage = archive.resolve() / '_transaction'
        self.storage.mkdir()
        self.roots = [path.resolve() for path in allowed_roots]
        self.operations = []

    def checked(self, path):
        path = path.resolve()
        if not any(path.is_relative_to(root) and path != root for root in self.roots):
            raise ValueError('Transaction path escapes approved directories: ' + str(path))
        return path

    def copy_new(self, source, target):
        source, target = self.checked(source), self.checked(target)
        if target.exists():
            raise ValueError('Copy would replace an existing asset: ' + str(target))
        # Atomic publication prevents a partial copy becoming an active image.
        staged = self.storage / ('copy_' + uuid.uuid4().hex)
        shutil.copy2(source, staged)
        digest = sha256(staged)
        staged.rename(target)
        self.operations.append(('new', target, None, digest))

    def move(self, source, target):
        source, target = self.checked(source), self.checked(target)
        if target.exists():
            raise ValueError('Move would replace an existing asset: ' + str(target))
        digest = sha256(source)
        source.rename(target)
        self.operations.append(('move', target, source, digest))

    def replace_bytes(self, target, content):
        target = self.checked(target)
        backup = None
        if target.exists():
            backup = self.storage / ('original_' + uuid.uuid4().hex)
            shutil.copy2(target, backup)
        staged = self.storage / ('staged_' + uuid.uuid4().hex)
        staged.write_bytes(content)
        digest = hashlib.sha256(content).hexdigest()
        staged.replace(target)
        self.operations.append(('replace' if backup else 'new', target, backup, digest))

    def rollback(self):
        conflicts = []
        for kind, target, original, digest in reversed(self.operations):
            if not target.exists() or sha256(target) != digest:
                conflicts.append(str(target))
                continue
            if kind == 'move':
                if original.exists():
                    conflicts.append(str(original))
                    continue
                target.rename(original)
            else:
                # Retain generated files, instead of deleting them during rollback.
                target.rename(self.storage / ('failed_' + uuid.uuid4().hex))
                if kind == 'replace':
                    original.replace(target)
        self.operations.clear()
        if conflicts:
            raise RuntimeError('Concurrent changes preserved; manual recovery required: ' + ', '.join(conflicts))


def preflight(bundle, notebook):
    manifest = json.loads((bundle / 'manifest.json').read_text(encoding='utf-8'))
    verification = json.loads((bundle / 'verification.json').read_text(encoding='utf-8'))
    if sha256(notebook) != manifest['previous_active_notebook_sha256']:
        raise ValueError('Active notebook changed after build; preserve user changes')
    current = {p.name:sha256(p) for p in (notebook.parent / 'figures').iterdir()
               if p.is_file() and p.suffix in ['.png','.svg']}
    if current != manifest['previous_figure_hashes']:
        raise ValueError('Active figures changed after build; installation stopped')
    if sha256(bundle / 'print.workspace.ipynb') != verification['notebook_sha256']:
        raise ValueError('Verified notebook has changed')
    for relative, expected in manifest['artifact_hashes'].items():
        path = (bundle / relative).resolve()
        if not path.is_relative_to(bundle.resolve()) or sha256(path) != expected:
            raise ValueError('Artifact changed or unsafe path: ' + relative)
    files = list((bundle / 'figures').iterdir())
    if len(files) != manifest['figures'] * 2 or any(p.suffix not in ['.png','.svg'] for p in files):
        raise ValueError('Unexpected new figure set')
    for path in files:
        if (notebook.parent / 'figures' / path.name).exists():
            raise ValueError('New filename collides with an existing local asset: ' + path.name)
    return manifest


def install(bundle):
    bundle = bundle.resolve()
    if bundle.parent != (ROOT / 'result/ETHICS/reports').resolve():
        raise ValueError('Bundle must be a direct child of the reports directory')
    manifest = preflight(bundle, NOTEBOOK)
    index = pd.read_csv(bundle / 'figure_index.csv')
    main = index[~index.section.isin(['expression_heatmaps','expression_cards','diversity'])]
    validate_main_cache(main, pd.read_csv(BASE / 'figure_index.csv'))
    for relative, digest in manifest['source_hashes'].items():
        if sha256(ROOT / relative) != digest:
            raise ValueError('Frozen source changed before installation: ' + relative)
    archive = (NOTEBOOK.parent / 'archive' / ('before_print_' + manifest['run_id'])).resolve()
    if not archive.is_relative_to(NOTEBOOK.parent.resolve()) or archive.exists():
        raise ValueError('Archive target is unsafe or already exists')
    original = nbformat.read(NOTEBOOK, as_version=4)
    archived = rebase_notebook(original, NOTEBOOK.parent, archive, ROOT)
    for cell in archived.cells:
        if cell.cell_type == 'code':
            cell.source = cell.source.replace("LOCAL_FIGURES = ACTIVE_DIR / 'figures'", "LOCAL_FIGURES = DOCUMENT_DIR / 'figures'")
            cell.source = cell.source.replace("SUPPLEMENT_LOCAL = ACTIVE_DIR / 'figures'", "SUPPLEMENT_LOCAL = DOCUMENT_DIR / 'figures'")
            for output in cell.get('outputs', []):
                for mime in ['text/html','text/markdown']:
                    if mime in output.get('data', {}):
                        output.data[mime] = output.data[mime].replace('../../figures/', 'figures/')
    moves = []
    for name, digest in manifest['previous_figure_hashes'].items():
        if Path(name).name != name:
            raise ValueError('Unsafe previous filename')
        source = (NOTEBOOK.parent / 'figures' / name).resolve()
        target = (archive / 'figures' / name).resolve()
        if source.parent != (NOTEBOOK.parent / 'figures').resolve() or target.parent != (archive / 'figures').resolve():
            raise ValueError('Figure move escapes the explicit source or archive')
        if sha256(source) != digest or target.exists():
            raise ValueError('Previous figure changed or archive collision')
        moves.append((source, target))
    older = NOTEBOOK.parent / 'archive/before_expression_20260919_r1' / NOTEBOOK.name
    old_hash = sha256(older)
    older_nb = nbformat.read(older, as_version=4)
    for cell in older_nb.cells:
        if cell.cell_type == 'code':
            cell.source = cell.source.replace("LOCAL_FIGURES = ACTIVE_DIR / 'figures'", "LOCAL_FIGURES = BUNDLE / 'figures'")
    # Execute and validate the verified fallback before changing workspace files.
    active = execute_notebook(nbformat.read(bundle / 'print.workspace.ipynb', as_version=4))
    notebook_links(active, NOTEBOOK.parent)
    notebook_links(older_nb, older.parent)
    if (bundle / 'installation_verification.json').exists():
        raise ValueError('Installation report already exists; preserve it')
    archive.mkdir()
    (archive / 'figures').mkdir()
    transaction = InstallTransaction(archive, [NOTEBOOK.parent, bundle])
    try:
        shutil.copy2(NOTEBOOK, archive / 'original.ipynb')
        shutil.copy2(older, archive / 'before_expression_original.ipynb')
        for source in (bundle / 'figures').iterdir():
            transaction.copy_new(source, NOTEBOOK.parent / 'figures' / source.name)
        # Resolve outputs to the newly installed local assets while old ones remain.
        active = execute_notebook(nbformat.read(bundle / 'print.workspace.ipynb', as_version=4))
        active_links = notebook_links(active, NOTEBOOK.parent)
        for source, target in moves:
            if sha256(source) != manifest['previous_figure_hashes'][source.name]:
                raise ValueError('Previous figure changed during installation: ' + source.name)
            transaction.move(source, target)
        transaction.replace_bytes(archive / NOTEBOOK.name, nbformat.writes(archived).encode('utf-8'))
        archive_links = notebook_links(archived, archive)
        if sha256(older) != old_hash or sha256(NOTEBOOK) != manifest['previous_active_notebook_sha256']:
            raise ValueError('Notebook changed during installation; user changes preserved')
        transaction.replace_bytes(older, nbformat.writes(older_nb).encode('utf-8'))
        transaction.replace_bytes(NOTEBOOK, nbformat.writes(active).encode('utf-8'))
        for source in (bundle / 'figures').iterdir():
            if sha256(source) != sha256(NOTEBOOK.parent / 'figures' / source.name):
                raise ValueError('Installed image differs from the verified source')
        result = dict(active_notebook=str(NOTEBOOK), notebook_sha256=sha256(NOTEBOOK),
                  notebook_links=active_links, archive=str(archive),
                  archive_links=archive_links, archived_image_files=len(manifest['previous_figure_hashes']),
                  png_count=len(list((NOTEBOOK.parent / 'figures').glob('*.png'))),
                  svg_count=len(list((NOTEBOOK.parent / 'figures').glob('*.svg'))),
                  main_pages_verified=len(main), main_panels_verified=int(main.panels.sum()),
                  previous_notebook_sha256=sha256(archive / 'original.ipynb'),
                  older_archive_navigation_repair=dict(path=str(older), before_sha256=old_hash, after_sha256=sha256(older)))
        write_json(archive / 'manifest.json', result)
        transaction.replace_bytes(bundle / 'installation_verification.json',
                                  json.dumps(result, ensure_ascii=False, indent=2).encode('utf-8'))
    except BaseException:
        transaction.rollback()
        failed = archive.with_name('failed_print_' + manifest['run_id'] + '_' + uuid.uuid4().hex[:8]).resolve()
        if failed.parent != archive.parent or failed.exists():
            raise RuntimeError('Rollback complete; diagnostic archive cannot safely be renamed')
        archive.rename(failed)
        print('Installation rolled back; diagnostics retained in ' + str(failed), flush=True)
        raise
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    install(parser.parse_args().bundle)
