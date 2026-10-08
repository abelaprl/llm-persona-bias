"""Losslessly compact gzip CSV archives and restore their original CSV bytes.

Requires Python 3.14 with compression.zstd. Existing artifacts are never
overwritten. Gzip container metadata can differ after restoration; decompressed
CSV bytes must match the recorded SHA-256 exactly.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
from compression import zstd

BLOCK = 4 * 1024**2
OPTIONS = {
    zstd.CompressionParameter.compression_level: 9,
    zstd.CompressionParameter.window_log: 27,
    zstd.CompressionParameter.enable_long_distance_matching: 1,
    zstd.CompressionParameter.checksum_flag: 1,
}


def digest_stream(stream):
    digest = hashlib.sha256()
    total = 0
    while chunk := stream.read(BLOCK):
        digest.update(chunk)
        total += len(chunk)
    return digest.hexdigest(), total


def digest_file(path):
    with path.open('rb') as stream:
        return digest_stream(stream)[0]


def safe_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f'Path outside root: {relative}')
    return path


def compact(root, relative, expected_oid=None):
    source = safe_path(root, relative)
    if not str(source).endswith('.csv.gz'):
        raise ValueError('Expected .csv.gz input')
    target = source.with_suffix('.zst')
    if target.exists():
        raise FileExistsError(target)
    original_size = source.stat().st_size
    original_sha = digest_file(source)
    git_hash = hashlib.sha1(f'blob {original_size}\0'.encode())
    with source.open('rb') as stream:
        while chunk := stream.read(BLOCK):
            git_hash.update(chunk)
    if expected_oid and git_hash.hexdigest() != expected_oid:
        raise ValueError(f'Working file differs from committed blob: {relative}')
    raw_digest = hashlib.sha256()
    raw_size = 0
    temporary = target.with_name(target.name + '.compressing')
    with gzip.open(source, 'rb') as incoming, zstd.open(temporary, 'xb', options=OPTIONS) as outgoing:
        while chunk := incoming.read(BLOCK):
            outgoing.write(chunk)
            raw_digest.update(chunk)
            raw_size += len(chunk)
    with zstd.open(temporary, 'rb') as stream:
        restored_hash, restored_size = digest_stream(stream)
    if (restored_hash, restored_size) != (raw_digest.hexdigest(), raw_size):
        raise ValueError(f'Round-trip validation failed: {target}')
    if digest_file(source) != original_sha:
        raise ValueError(f'Source changed during compression: {source}')
    # Hard-link publication fails if a destination already exists on any platform.
    os.link(temporary, target)
    temporary.unlink()
    return {
        'original_path': source.relative_to(root).as_posix(),
        'compact_path': target.relative_to(root).as_posix(),
        'original_git_blob': git_hash.hexdigest(),
        'original_gzip_bytes': original_size,
        'original_gzip_sha256': original_sha,
        'compact_bytes': target.stat().st_size,
        'compact_sha256': digest_file(target),
        'csv_bytes': raw_size,
        'csv_sha256': restored_hash,
        'round_trip_verified': True,
    }


def verify(root, entry):
    target = safe_path(root, entry['compact_path'])
    if digest_file(target) != entry['compact_sha256']:
        raise ValueError(f'Archive hash mismatch: {target}')
    with zstd.open(target, 'rb') as stream:
        actual = digest_stream(stream)
    if actual != (entry['csv_sha256'], entry['csv_bytes']):
        raise ValueError(f'CSV content mismatch: {target}')


def restore(root, entry, output_root=None):
    verify(root, entry)
    destination = safe_path(output_root or root, entry['original_path'])
    if destination.exists():
        with gzip.open(destination, 'rb') as stream:
            actual = digest_stream(stream)
        if actual != (entry['csv_sha256'], entry['csv_bytes']):
            raise FileExistsError(f'Existing CSV differs; refusing overwrite: {destination}')
        return 'already_present'
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + '.restoring')
    digest = hashlib.sha256()
    total = 0
    with temporary.open('xb') as raw_file:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw_file, compresslevel=6, mtime=0) as outgoing:
            with zstd.open(safe_path(root, entry['compact_path']), 'rb') as incoming:
                while chunk := incoming.read(BLOCK):
                    outgoing.write(chunk)
                    digest.update(chunk)
                    total += len(chunk)
    if (digest.hexdigest(), total) != (entry['csv_sha256'], entry['csv_bytes']):
        raise ValueError(f'Restoration validation failed: {temporary}')
    # Exclusive destination creation also handles another process creating it.
    with destination.open('xb') as outgoing, temporary.open('rb') as incoming:
        while chunk := incoming.read(BLOCK):
            outgoing.write(chunk)
    temporary.unlink()
    return 'restored'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['compress', 'verify', 'restore'])
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--inventory', type=Path)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == 'compress':
        if not args.inventory:
            parser.error('--inventory is required for compress')
        if args.manifest.exists():
            raise FileExistsError(args.manifest)
        selected = [entry for entry in json.loads(args.inventory.read_text(encoding='utf-8'))['large_blobs'] if entry['path'].endswith('.csv.gz')]
        checkpoint = args.manifest.with_suffix('.progress.json')
        previous = json.loads(checkpoint.read_text(encoding='utf-8')) if args.resume and checkpoint.exists() else {}
        entries = previous.get('entries', [])
        failures = previous.get('failures', [])
        completed = {entry['original_path'] for entry in entries}
        failed = {entry['path'] for entry in failures}
        for number, item in enumerate(selected, 1):
            if item['path'] in completed or item['path'] in failed:
                continue
            print(f'[{number}/{len(selected)}] {item["path"]}', flush=True)
            try:
                entry = compact(root, item['path'], item['oid'])
            except (EOFError, gzip.BadGzipFile) as error:
                failures.append({**item, 'error': str(error), 'action': 'kept_original_gzip_unchanged'})
                print(f'  INVALID SOURCE: {error}; original retained', flush=True)
            else:
                entries.append(entry)
                print(f'  {entry["original_gzip_bytes"]:,} -> {entry["compact_bytes"]:,} bytes; exact CSV verified', flush=True)
            checkpoint.write_text(json.dumps({'entries': entries, 'failures': failures}, indent=2), encoding='utf-8')
        report = {'format': 'csv-zstandard-v1', 'compression': {'level': 9, 'window_log': 27, 'long_distance_matching': True}, 'entries': entries, 'failures': failures}
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        with args.manifest.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, indent=2)
        print(f'Done: {len(entries)} files; {len(failures)} invalid source archives retained', flush=True)
    else:
        entries = json.loads(args.manifest.read_text(encoding='utf-8'))['entries']
        for number, entry in enumerate(entries, 1):
            status = restore(root, entry, args.output_root) if args.action == 'restore' else verify(root, entry)
            print(f'[{number}/{len(entries)}] {entry["compact_path"]}: {status or "verified"}', flush=True)


if __name__ == '__main__':
    main()
