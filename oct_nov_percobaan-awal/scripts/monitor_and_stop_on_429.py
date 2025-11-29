#!/usr/bin/env python3
"""
Monitor a background run log for rate-limit (429) messages. On detection:
- kill the running `run_batch_mmlu_women_implicit.py` process(es)
- copy partial outputs into `results/` using `copy_runs_to_results_v2.py`

Usage: set environment variables `MONITOR_LOG` and `TASK` and `ORDER`, or pass defaults.
"""
import os
import time
import subprocess
from pathlib import Path

MONITOR_LOG = os.environ.get('MONITOR_LOG', 'logs_nemotron_full.log')
TASK = os.environ.get('TASK', 'mmlu_women_implicit_strong')
ORDER = os.environ.get('ORDER', '14')
SLEEP_POLL = float(os.environ.get('MONITOR_POLL', '1.0'))


def tail_f(path: Path):
    # generator that yields new lines as they arrive
    with open(path, 'r', encoding='utf-8', errors='ignore') as fh:
        # seek to end
        fh.seek(0, 2)
        while True:
            line = fh.readline()
            if not line:
                time.sleep(SLEEP_POLL)
                continue
            yield line


def find_runner_pids():
    try:
        out = subprocess.check_output(['pgrep', '-f', 'run_batch_mmlu_women_implicit.py']).decode().strip()
        pids = [int(x) for x in out.split() if x.strip()]
        return pids
    except subprocess.CalledProcessError:
        return []


def kill_pids(pids):
    for pid in pids:
        try:
            subprocess.run(['kill', '-TERM', str(pid)])
        except Exception:
            pass


def copy_results(task, order):
    # call the copy script
    try:
        subprocess.run(['python3', 'scripts/copy_runs_to_results_v2.py', '--task', task, '--order', str(order)], check=False)
    except Exception as e:
        print('Copy script failed:', e)


def main():
    print('Monitoring', MONITOR_LOG)
    log_path = Path(MONITOR_LOG)
    if not log_path.exists():
        print('Log file not found, creating and waiting:', log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.touch()

    for line in tail_f(log_path):
        lower = line.lower()
        if '429' in lower or 'rate limit' in lower or 'rate-limit' in lower or 'rate_limit' in lower or 'quota' in lower:
            print('Detected rate-limit line:', line.strip())
            # find runner pids and kill
            pids = find_runner_pids()
            if pids:
                print('Killing runner pids:', pids)
                kill_pids(pids)
            else:
                print('No runner pids found')

            print('Copying partial outputs to results/')
            copy_results(TASK, ORDER)
            print('Monitor finished after copying results')
            return


if __name__ == '__main__':
    main()
