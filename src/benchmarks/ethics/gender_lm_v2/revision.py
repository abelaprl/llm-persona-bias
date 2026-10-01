"""Recompute every surface; reuse unchanged evidence for a pure-verdict correction."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import gzip
import json
from pathlib import Path
import time

from . import VERSION
from . import run
from .sources import KEYS, load_source, reserve_output, sha256
from .surfaces import build_surfaces


def check_reuse(previous: dict, current: dict) -> None:
    if set(previous) != set(current) or any(previous[k] != current[k] for k in current if k != 'surfaces.py'):
        raise ValueError('Unchanged detector/reference instruments are required for evidence reuse')


def check_transition(before: str, after: str) -> None:
    if (before, after) != ('substantive_explanation', 'verdict_only'):
        raise ValueError('This revision only permits substantive_explanation -> verdict_only')


def revise(parent: Path, output: Path) -> dict:
    started = time.monotonic()
    previous = json.loads((parent / 'extraction_manifest.json').read_text(encoding='utf-8'))
    provenance = json.loads((parent / 'analysis_input_provenance.json').read_text(encoding='utf-8'))
    current = run.instrument_hashes()
    check_reuse(provenance['instrument_hashes'], current)
    registry_path = Path(__file__).with_name('registry.json')
    registry_hash = sha256(registry_path)
    if previous['registry_sha256'] != registry_hash or json.loads((parent / 'registry.json').read_text(encoding='utf-8')) != run.load_registry():
        raise ValueError('Parent registry differs')
    if run.evidence_path(parent).name not in provenance['artifacts']:
        raise ValueError('Selected parent evidence must be bound by the analysis provenance')
    for name, expected in provenance['artifacts'].items():
        if sha256(parent / name) != expected:
            raise ValueError('Parent artifact changed: ' + name)
    source = Path(previous['source'])
    source_hash = sha256(source)
    if source_hash != previous['source_sha256'] or source_hash != provenance['source_sha256']:
        raise ValueError('Original source changed')
    source_rows = load_source(source)
    if len(source_rows) != previous['rows']:
        raise ValueError('Revision requires the complete parent source')
    original = {tuple(row[k] for k in KEYS): row['response_content_full'] for row in source_rows.to_dict('records')}
    reserve_output(output)
    changed_ids, seen = set(), set()
    ledger, changed_spans = [], []
    statuses = Counter()
    total = 0
    with gzip.open(parent / 'response_markers.csv.gz', 'rt', encoding='utf-8', newline='') as mf, \
            gzip.open(parent / 'response_surfaces.csv.gz', 'rt', encoding='utf-8', newline='') as sf, \
            gzip.open(output / 'response_markers.csv.gz', 'xt', encoding='utf-8', newline='') as mo, \
            gzip.open(output / 'response_surfaces.csv.gz', 'xt', encoding='utf-8', newline='') as so:
        markers, surfaces = csv.DictReader(mf), csv.DictReader(sf)
        mw, sw = csv.DictWriter(mo, markers.fieldnames), csv.DictWriter(so, surfaces.fieldnames)
        mw.writeheader(); sw.writeheader()
        for record, texts in zip(markers, surfaces, strict=True):
            total += 1
            key = tuple(record[k] for k in KEYS)
            if record['response_id'] != texts['response_id'] or key != tuple(texts[k] for k in KEYS) or key in seen:
                raise ValueError('Parent row alignment/uniqueness failed')
            if key not in original or texts['raw_text'] != original[key]:
                raise ValueError('Parent raw response differs from original source')
            seen.add(key)
            surface = build_surfaces(texts['raw_text'])
            values = {name: getattr(surface, name) for name in run.SURFACE_COLUMNS if name != 'surface_flags'}
            values['surface_flags'] = json.dumps(surface.flags)
            changed = any(texts[name] != value for name, value in values.items())
            changed |= record['text_status'] != surface.status or int(record['word_count']) != surface.word_count
            if changed:
                check_transition(record['text_status'], surface.status)
                # Pure verdict has no spans, so scenario input cannot affect this rerun.
                replacement, evidence = run.extract_batch([{**{k: record[k] for k in KEYS},
                    'file': record['file'], 'response_content_full': texts['raw_text']}])[0]
                if evidence or replacement['response_id'] != record['response_id']:
                    raise ValueError('Pure verdict must retain its ID and produce no evidence')
                changed_ids.add(record['response_id'])
                ledger.append({**{k: record[k] for k in ['response_id', *KEYS]},
                    'old_status': record['text_status'], 'new_status': replacement['text_status'],
                    'old_word_count': record['word_count'], 'new_word_count': replacement['word_count'],
                    'raw_text': texts['raw_text']})
                record = {k: replacement[k] for k in markers.fieldnames}
                texts = {k: replacement[k] for k in surfaces.fieldnames}
                changed_spans.extend(evidence)
            mw.writerow(record); sw.writerow(texts)
            statuses[record['text_status']] += 1
            if total % 5000 == 0:
                print(f'Recomputed surfaces {total:,}; changed={len(changed_ids)}; elapsed={time.monotonic()-started:.0f}s', flush=True)
    if seen != set(original):
        raise ValueError('Revision did not cover every source response')
    spans_written = removed_spans = 0
    with gzip.open(run.evidence_path(parent), 'rt', encoding='utf-8', newline='') as src, \
            gzip.open(output / 'marker_spans.csv.gz', 'xt', encoding='utf-8', newline='') as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, reader.fieldnames); writer.writeheader()
        for row in reader:
            if row['response_id'] in changed_ids:
                removed_spans += 1
            else:
                writer.writerow(row); spans_written += 1
        writer.writerows(changed_spans)
    if run.instrument_hashes() != current or sha256(registry_path) != registry_hash or sha256(source) != source_hash:
        raise ValueError('Instrument, registry or source changed during revision')
    for name, expected in provenance['artifacts'].items():
        if sha256(parent / name) != expected:
            raise ValueError('Parent artifact changed during revision: ' + name)
    with (output / 'changed_responses.csv').open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['response_id', *KEYS, 'old_status', 'new_status', 'old_word_count', 'new_word_count', 'raw_text'])
        writer.writeheader(); writer.writerows(ledger)
    manifest = {'status': 'candidate', 'pipeline_version': VERSION, 'source': str(source.resolve()),
        'source_sha256': source_hash, 'source_hash_unchanged': True, 'rows': total, 'text_status': dict(statuses),
        'registry_sha256': registry_hash, 'markers': len(run.load_registry()),
        'instrument_hashes_at_extraction': current, 'limited_smoke_run': False,
        'extraction_strategy': 'all_surfaces_recomputed; unchanged_inputs_reuse_verified_parent_evidence',
        'parent_output': str(parent.resolve()), 'parent_artifact_hashes': provenance['artifacts'],
        'parent_historical_instrument_limitation': 'Parent snapshots were captured at analysis, not extraction; no retroactive extraction claim.',
        'changed_responses': len(changed_ids), 'unchanged_responses': total-len(changed_ids),
        'spans': spans_written + len(changed_spans), 'removed_spans': removed_spans,
        'elapsed_seconds': time.monotonic()-started, 'precision_recall_99_demonstrated': False}
    run.write_json_new(output / 'registry.json', run.load_registry())
    run.write_json_new(output / 'extraction_manifest.json', manifest)
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-output', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    revise(args.parent_output, args.output_dir)
