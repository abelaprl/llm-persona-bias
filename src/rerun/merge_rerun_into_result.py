import os
import shutil
import argparse
from pathlib import Path
import hashlib

def file_hash(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def merge_rerun_into_result(rerun_root, result_root):
    log = []
    changed_files = []
    missing_target = []
    total = 0
    overwritten = 0
    for persona in os.listdir(rerun_root):
        persona_dir = os.path.join(rerun_root, persona)
        if not os.path.isdir(persona_dir):
            continue
        for fname in os.listdir(persona_dir):
            if not fname.endswith('.json'):
                continue
            total += 1
            src = os.path.join(persona_dir, fname)
            tgt_dir = os.path.join(result_root, persona)
            tgt = os.path.join(tgt_dir, fname)
            if not os.path.exists(tgt_dir):
                os.makedirs(tgt_dir)
            existed = os.path.exists(tgt)
            old_hash = file_hash(tgt) if existed else None
            shutil.copy2(src, tgt)
            new_hash = file_hash(tgt)
            if existed:
                if old_hash != new_hash:
                    changed_files.append((persona, fname, old_hash, new_hash))
                    overwritten += 1
            else:
                missing_target.append((persona, fname))
                overwritten += 1
    print(f"Total rerun files found: {total}")
    print(f"Total successfully overwritten/copied: {overwritten}")
    print(f"Missing targets created: {len(missing_target)}")
    if missing_target:
        print("Examples of missing targets:")
        for ex in missing_target[:5]:
            print(f"  {ex}")
    if changed_files:
        print("Examples of changed files:")
        for ex in changed_files[:5]:
            print(f"  {ex[0]}/{ex[1]} (old_hash={ex[2]}, new_hash={ex[3]})")
    return changed_files, missing_target

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rerun_root', required=True)
    parser.add_argument('--result_root', required=True)
    args = parser.parse_args()
    merge_rerun_into_result(args.rerun_root, args.result_root)

if __name__ == '__main__':
    main()
