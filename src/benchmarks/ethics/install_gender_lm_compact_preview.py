"""Install the approved preview location without replacing existing figures."""
import csv
import hashlib
import html
from html.parser import HTMLParser
import io
import json
import os
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'result/ETHICS/reports/gender_lm_compact_preview_20260920_r3'
TARGET = ROOT / 'notebooks/gender/linguistic marker/ETHICS'


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def install():
    index = list(csv.DictReader((SOURCE / 'figure_index.csv').open(encoding='utf-8')))
    assert len(index) == 14
    mapping = {}
    for row in index:
        for column in ['png_path', 'svg_path', 'source_csv']:
            source = (SOURCE / row[column]).resolve()
            mapping[source] = TARGET / 'figures' / f'ringkas_{row["figure_id"]}{source.suffix}'
    names = {'preview.html': 'gender_ethics_ringkas_preview.html',
             'Gender_ETHICS_ringkas_preview.ipynb': 'Gender_ETHICS_ringkas_preview.ipynb',
             'figure_index.csv': 'ringkas_figure_index.csv',
             'appendix_index.csv': 'ringkas_appendix_index.csv',
             'README.md': 'README_ringkas.md', 'verification.json': 'ringkas_verification.json'}
    mapping.update({(SOURCE / old).resolve(): TARGET / new for old, new in names.items()})
    manifest_path = TARGET / 'ringkas_install_manifest.json'
    collisions = [str(p) for p in [*mapping.values(), manifest_path] if p.exists()]
    if collisions:
        raise FileExistsError(f'Existing destinations must not be overwritten: {collisions}')
    source_manifest = json.loads((SOURCE / 'manifest.json').read_text(encoding='utf-8'))
    assert all(digest(SOURCE / name) == expected for name, expected in source_manifest['generated_files'].items())
    protected = {p: digest(p) for p in TARGET.rglob('*') if p.is_file()}

    def relocate(value):
        if value.startswith(('attachment:', 'https:', 'http:', '#')):
            return value
        path, separator, fragment = value.partition('#')
        original = (SOURCE / path).resolve()
        destination = mapping.get(original, original)
        return Path(os.path.relpath(destination, TARGET)).as_posix() + (separator + fragment if separator else '')

    def csv_bytes(rows):
        buffer = io.StringIO(newline='')
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        return buffer.getvalue().encode('utf-8')

    generated = {}
    page = (SOURCE / 'preview.html').read_text(encoding='utf-8')
    page = re.sub(r'(href|src)="([^"]+)"', lambda m: f'{m[1]}="{html.escape(relocate(html.unescape(m[2])), quote=True)}"', page)
    generated[mapping[(SOURCE / 'preview.html').resolve()]] = page.encode('utf-8')
    notebook = json.loads((SOURCE / 'Gender_ETHICS_ringkas_preview.ipynb').read_text(encoding='utf-8'))
    for cell in notebook['cells']:
        text = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        cell['source'] = re.sub(r'\]\(([^)]+)\)', lambda m: '](' + relocate(m[1]) + ')', text)
        cell['source'] = cell['source'].replace('tercantum di README.md', 'tercantum di README_ringkas.md')
    generated[mapping[(SOURCE / 'Gender_ETHICS_ringkas_preview.ipynb').resolve()]] = json.dumps(notebook, ensure_ascii=False, indent=1).encode('utf-8')
    for name in ['figure_index.csv', 'appendix_index.csv']:
        rows = list(csv.DictReader((SOURCE / name).open(encoding='utf-8')))
        for row in rows:
            for field in ['png_path', 'svg_path', 'source_csv']:
                if field in row:
                    row[field] = relocate(row[field])
        generated[mapping[(SOURCE / name).resolve()]] = csv_bytes(rows)
    readme = (SOURCE / 'README.md').read_text(encoding='utf-8')
    readme = readme.replace('verification.json', 'ringkas_verification.json')
    readme += ('\n## Lokasi pemasangan\n\n'
               '[Buka pratinjau](gender_ethics_ringkas_preview.html) atau '
               '[notebook ringkas](Gender_ETHICS_ringkas_preview.ipynb). '
               'PNG/SVG dan tabel per figur berada langsung di `figures/ringkas_01.*` sampai `figures/ringkas_14.*`. '
               'Statistik dan registry lengkap tetap tertaut ke bundle sumber. Tidak ada gambar lama yang dihapus.\n\n'
               'Pemasangan: `python -X utf8 -B -m src.benchmarks.ethics.install_gender_lm_compact_preview`. '
               'Installer menolak tujuan yang sudah ada agar tidak menimpa file.\n')
    generated[mapping[(SOURCE / 'README.md').resolve()]] = readme.encode('utf-8')

    class Links(HTMLParser):
        def __init__(self):
            super().__init__()
            self.links = []

        def handle_starttag(self, tag, attrs):
            self.links.extend(value for key, value in attrs if key in ('href', 'src'))

    parser = Links()
    parser.feed(page)
    destinations = {p.resolve() for p in mapping.values()}
    for link in parser.links:
        path = (TARGET / html.unescape(link).split('#')[0]).resolve()
        assert path in destinations or path.is_file(), str(path)
    # All destinations and links are validated before the first write.
    for source, target in mapping.items():
        if target in generated:
            target.write_bytes(generated[target])
        else:
            shutil.copyfile(source, target)
    assert all(digest(path) == expected for path, expected in protected.items())
    assert all((TARGET / html.unescape(link).split('#')[0]).is_file() for link in parser.links)
    for row in index:
        for field in ['png_path', 'svg_path', 'source_csv']:
            source = (SOURCE / row[field]).resolve()
            assert digest(source) == digest(mapping[source])
    manifest = dict(source=str(SOURCE), target=str(TARGET), figures=14, main_figure_directory='figures',
                    verified_html_links=len(parser.links), existing_files_unchanged=len(protected),
                    installed_files={str(path.relative_to(TARGET)): digest(path) for path in mapping.values()})
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest | {'installed_files': len(manifest['installed_files'])}, indent=2))


if __name__ == '__main__':
    install()
