#!/usr/bin/env python3
"""
Periodic reporter: every N minutes write a short progress snapshot to a log file.
"""
import os
import time
from datetime import datetime
from pathlib import Path
import subprocess

POLL_MINUTES = int(os.environ.get('POLL_MINUTES', '5'))
TASK = os.environ.get('TASK', 'mmlu_women_implicit_strong')
ORDER = os.environ.get('ORDER', '14')
LOG_ROOT = Path.cwd()
REPORT_LOG = LOG_ROOT / 'logs_progress_reporter.log'
N_LAST = int(os.environ.get('N_LAST', '20'))


def tail_lines(path: Path, n: int):
    if not path.exists():
        return []
    try:
        out = subprocess.check_output(['tail', f'-n{n}', str(path)], stderr=subprocess.DEVNULL)
        return out.decode(errors='ignore').splitlines()
    except Exception:
        return []


def count_outputs():
    dirp = LOG_ROOT / 'logs_json' / TASK / str(ORDER)
    if not dirp.exists():
        return 0
    return sum(1 for p in dirp.iterdir() if p.is_file() and p.suffix == '.json')


def snapshot():
    ts = datetime.utcnow().isoformat() + 'Z'
    cnt = count_outputs()
    nemotron_log = LOG_ROOT / 'logs_nemotron_full.log'
    monitor_log = LOG_ROOT / 'logs_monitor_nemotron.log'
    lines = [f'[{ts}] outputs_count={cnt}']
    lines.append('--- last lines of nemotron log ---')
    lines.extend(tail_lines(nemotron_log, N_LAST) or ['(no nemotron log)'])
    lines.append('--- last lines of monitor log ---')
    lines.extend(tail_lines(monitor_log, N_LAST) or ['(no monitor log)'])
    return '\n'.join(lines) + '\n'


def main():
    print('Starting periodic reporter: every', POLL_MINUTES, 'minutes')
    while True:
        s = snapshot()
        with open(REPORT_LOG, 'a', encoding='utf-8') as f:
            f.write(s)
            f.flush()
        time.sleep(POLL_MINUTES * 60)


if __name__ == '__main__':
    main()
