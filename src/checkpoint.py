#!/usr/bin/env python3
"""Local checkpoint and optional remote sync; run on new elites and every 1500s.

No daemon is started by importing this module. Remote errors are reported,
not mistaken for a successful backup. Keep secrets outside this repository.
"""
import argparse
import fcntl
import datetime
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]

def run(args, timeout=30):
    try:
        p = subprocess.run(args, cwd=ROOT, text=True, capture_output=True,
                           timeout=timeout, env={**os.environ, 'GIT_TERMINAL_PROMPT': '0'})
        return {'command': args, 'returncode': p.returncode,
                'stdout': p.stdout.strip(), 'stderr': p.stderr.strip()}
    except subprocess.TimeoutExpired:
        return {'command': args, 'returncode': 124, 'stderr': 'Timeout'}

def _checkpoint_unlocked(reason='checkpoint', push=True):
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    # Gitignore cannot filter by file size: exclude raw dumps at every size,
    # and check all staged content for accidentally included large files.
    result = {'timestamp': stamp, 'reason': reason, 'steps': []}
    result['steps'].append(run(['git', 'add', '-A']))
    if result['steps'][-1]['returncode']:
        return result
    staged = run(['git', 'diff', '--cached', '--name-only', '-z'])
    for name in staged.get('stdout', '').split('\0'):
        path = ROOT / name
        if name and path.is_file() and path.stat().st_size > 10 * 1024 * 1024:
            result['error'] = f'Refusing staged file over 10 MiB: {name}'
            return result
    result['steps'].append(run(['git', 'commit', '-m', f'chore: checkpoint {stamp} - {reason}']))
    if push:
        remote = run(['git', 'remote', 'get-url', 'origin'])
        if remote['returncode']:
            result['remote_status'] = 'unavailable: no origin configured'
        else:
            pushed = run(['git', 'push', '-u', 'origin', 'main'], timeout=60)
            result['steps'].append(pushed)
            result['remote_status'] = 'pushed' if pushed['returncode'] == 0 else 'push_failed'
    else:
        result['remote_status'] = 'not_requested'
    return result

def checkpoint(reason='checkpoint', push=True):
    # Serialize independent research processes without placing a lock in Git's
    # own index.lock namespace. Kernel releases this lock if a process exits.
    with open(ROOT / '.git' / 'research-checkpoint.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        from progress_dashboard import update
        update()
        return _checkpoint_unlocked(reason, push)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reason', default='manual')
    parser.add_argument('--local-only', action='store_true')
    args = parser.parse_args()
    result = checkpoint(args.reason, not args.local_only)
    print(json.dumps(result, indent=2))
    raise SystemExit(1 if result.get('error') or result.get('remote_status') == 'push_failed' else 0)
