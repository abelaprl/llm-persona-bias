"""Print a verified push plan, or execute it only with --execute."""
import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAX_PUSH_BYTES = 2 * 1024**3


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT).decode().strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=ROOT / 'docs/push_batches_20261011.json')
    parser.add_argument('--remote', default='origin')
    parser.add_argument('--branch', default='ethics-accuracy')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding='utf-8'))
    commits = []
    parent = plan['base']
    for batch in plan['batches']:
        commit = batch['commit']
        if not 0 < batch['pack_bytes'] < MAX_PUSH_BYTES:
            raise RuntimeError(f'Unverified push size: {commit}')
        if git('rev-parse', commit + '^') != parent:
            raise RuntimeError(f'History no longer matches the plan: {commit}')
        commits.append(commit)
        parent = commit
    head = git('rev-parse', 'HEAD')
    if head != parent:
        if git('rev-parse', head + '^') != parent:
            raise RuntimeError('Unexpected commits after the verified plan; rebuild the plan')
        # The one final metadata commit is checked locally before pushing.
        process = subprocess.Popen(['git', '-c', 'pack.useSparse=false', 'pack-objects', '--stdout', '--revs', '--thin', '--window=0'], cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        process.stdin.write(f'{head}\n^{parent}\n'.encode())
        process.stdin.close()
        total = 0
        while chunk := process.stdout.read(4 * 1024**2):
            total += len(chunk)
        if process.wait() or total >= MAX_PUSH_BYTES:
            raise RuntimeError('Final metadata commit failed the pack size check')
        commits.append(head)
    start = 0
    if args.execute:
        remote_ref = git('ls-remote', args.remote, 'refs/heads/' + args.branch)
        if not remote_ref:
            raise RuntimeError('Remote branch is missing; inspect the remote before proceeding')
        remote_head = remote_ref.split()[0]
        if remote_head in commits:
            start = commits.index(remote_head) + 1
        elif remote_head != plan['base']:
            raise RuntimeError('Remote branch changed; refusing to force-push')
    for number, commit in enumerate(commits[start:], start + 1):
        command = ['git', '-c', 'pack.useSparse=false', '-c', 'pack.window=0', '-c', 'pack.compression=1',
                   '-c', 'pack.threads=2', 'push', args.remote,
                   f'{commit}:refs/heads/{args.branch}']
        print(f'[{number}/{len(commits)}] ' + ' '.join(command), flush=True)
        if args.execute:
            subprocess.run(command, cwd=ROOT, check=True)
    print('Done' if args.execute else 'Dry run only; use --execute to send these commits.', flush=True)


if __name__ == '__main__':
    main()
