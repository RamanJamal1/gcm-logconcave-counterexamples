#!/usr/bin/env python3
"""Read-only readiness probes; never starts the overnight campaign.

Records only public network diagnostics and CLI status, never auth tokens.
Exit 0 means the checked gates pass; exit 1 means the campaign is not ready.
"""
import datetime
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]

def command(args, timeout=12):
    try:
        p = subprocess.run(args, cwd=ROOT, capture_output=True, text=True,
                           timeout=timeout, env={**os.environ, 'GIT_TERMINAL_PROMPT': '0'})
        return {'command': args, 'returncode': p.returncode,
                'stdout': p.stdout.strip(), 'stderr': p.stderr.strip()}
    except FileNotFoundError:
        return {'command': args, 'returncode': 127, 'stderr': 'not installed'}
    except subprocess.TimeoutExpired:
        return {'command': args, 'returncode': 124, 'stderr': 'timed out'}

def read_json(path):
    p = ROOT / path
    return json.loads(p.read_text()) if p.exists() else {}

def main():
    r = {'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'mode': 'preflight_only; overnight goal paused'}
    r['gh_installed'] = bool(shutil.which('gh'))
    r['gh_auth_status'] = command(['gh', 'auth', 'status'])
    r['github_identity'] = command(['gh', 'api', 'user', '--jq', '.login'])
    r['github_network'] = command(['curl', '-sS', '--max-time', '8', '-o', '/dev/null', '-w', '%{http_code}', 'https://api.github.com'])
    r['nvidia_smi'] = command(['timeout', '-k', '1s', '5s', 'nvidia-smi'], timeout=9)
    r['gpu_device_paths'] = [str(p) for p in Path('/dev').glob('nvidia*')]
    r['git_head'] = command(['git', 'rev-parse', 'HEAD'])
    r['git_origin'] = command(['git', 'remote', 'get-url', 'origin'])
    r['packages'] = {p: bool(importlib.util.find_spec(p)) for p in ('numpy', 'scipy', 'sympy', 'mpmath', 'torch', 'jax', 'flint', 'optax', 'evosax')}
    formula = read_json('certs/formula_tests.json')
    r['formula_validation_passed'] = formula.get('passed', False)
    r['formula_test_relative_errors'] = {m: item['relative_error'] for m, item in formula.get('runs', [{}])[-1].get('results', {}).items()}
    smoke = read_json('checkpoints/search_smoke_test.json')
    r['cpu_search_smoke'] = {k: smoke.get(k) for k in ('status', 'evaluations', 'gpu_evaluations', 'failures', 'elapsed_seconds')}
    r['blockers'] = []
    if r['github_identity']['returncode']:
        r['blockers'].append('CLI GitHub API identity cannot be verified; repository create/push and issue posting unverified.')
    if r['nvidia_smi']['returncode']:
        r['blockers'].append('GPU probe fails; requested RTX A6000 is not accessible from this execution environment.')
    if not r['formula_validation_passed']:
        r['blockers'].append('Numerical formula validation has not passed.')
    if smoke.get('status') != 'passed':
        r['blockers'].append('CPU search smoke test has not completed.')
    if not r['packages']['flint']:
        r['blockers'].append('python-flint/Arb unavailable; mpmath alone does not supply a rigorous certificate.')
    r['blockers'].extend([
        'Requested batched CUDA FP32/FP64 optimization pipeline is not implemented or validated; current search is CPU only.',
        'Rigorous interval certification with analytic tail bounds is not implemented or validated.',
        '25-minute continuous sync and new-elite hooks are not enabled; local checkpoint utility only.',
    ])
    r['ready_for_overnight'] = not r['blockers']
    (ROOT / 'certs/preflight.json').write_text(json.dumps(r, indent=2) + '\n')
    print(json.dumps(r, indent=2))
    return 0 if r['ready_for_overnight'] else 1

if __name__ == '__main__':
    raise SystemExit(main())
