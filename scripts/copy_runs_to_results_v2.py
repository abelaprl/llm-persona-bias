#!/usr/bin/env python3
"""
Copy run outputs into `results/` (v2).

This variant avoids naming collisions and ensures the script is present in the
workspace where `ls` can see it.
"""
import argparse
import shutil
from pathlib import Path
import sys


def copy_files(src_dir: Path, dst_dir: Path, move: bool = False, dry_run: bool = False) -> int:
    if not src_dir.exists():
        return 0
    dst_dir.mkdir(parents=True, exist_ok=True)
    files = sorted([p for p in src_dir.iterdir() if p.is_file() and p.suffix == '.json'])
    count = 0
    for f in files:
        dest = dst_dir / f.name
        if dry_run:
            print(f"DRY-RUN: {f} -> {dest}")
            count += 1
            continue
        if move:
            shutil.move(str(f), str(dest))
        else:
            shutil.copy2(str(f), str(dest))
        count += 1
    return count


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task', required=True)
    p.add_argument('--order', required=True)
    p.add_argument('--move', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()

    repo_root = Path.cwd()
    logs_src = repo_root / 'logs_json' / args.task / str(args.order)
    chats_src = repo_root / 'chat_histories' / args.task / str(args.order)

    results_base = repo_root / 'results'
    logs_dst = results_base / 'raw_runs' / args.task / str(args.order)
    chats_dst = results_base / 'chat_histories' / args.task / str(args.order)

    print(f"Source logs: {logs_src}")
    print(f"Source chats: {chats_src}")
    print(f"Destination logs: {logs_dst}")
    print(f"Destination chats: {chats_dst}")

    logs_copied = copy_files(logs_src, logs_dst, move=args.move, dry_run=args.dry_run)
    chats_copied = copy_files(chats_src, chats_dst, move=args.move, dry_run=args.dry_run)

    print(f"Result: {logs_copied} log files, {chats_copied} chat files copied.")
    if logs_copied == 0 and chats_copied == 0:
        print("No files found to copy.")
        sys.exit(2)


if __name__ == '__main__':
    main()
