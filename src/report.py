"""Generate an explicitly provisional briefing from actual campaign state.

Posting requires --post and uses the requested body-file GitHub command. A saved
issue receipt avoids duplicate posts. Never describes a numerical hit as proved.
"""
import argparse,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def report(state_path,post=False):
 s=json.loads(Path(state_path).read_text());certs=s.get('certificates',[])
 text=['# GCM research briefing','',f"Run status: {s['status']}",f"Device: {s.get('device','unknown')}",f"GPU exploration evaluations: {s['gpu_evaluations']}",f"FP64 refined candidates: {s['refinement_evaluations']}",'','**The minimal order is unresolved.** This file reports the recorded run only; it is not evidence of an eight-hour campaign.','', '| Order | Best dimensionless signed derivative | Status |','|---|---:|---|']
 for m in ['6','7','8']:
  e=s['best'].get(m);text.append(f"| {m} | {e['normalized'] if e else 'not evaluated'} | numerical, uncertified |")
 text+=['','Parameters are in checkpoints/elite_m*.json. Formula validation is in certs/formula_tests.json.','',f'Certification records: {json.dumps(certs)}','',f"Recorded errors: {json.dumps(s.get('errors',[]))}",'','A finite unsuccessful search proves no universal lower bound. Even a certified order-7 or order-8 violation would not determine minimality without lower-order theorems.']
 path=ROOT/'MORNING_BRIEFING.md';path.write_text('\n'.join(text)+'\n')
 if post:
  receipt=ROOT/'certs/github_issue.json'
  if receipt.exists():raise RuntimeError('Issue receipt exists; inspect before posting again')
  p=subprocess.run(['gh','issue','create','--title','Overnight Research Results: GCM Minimal Order m*','--body-file',str(path)],cwd=ROOT,text=True,capture_output=True,timeout=60)
  if p.returncode:raise RuntimeError(p.stderr)
  receipt.write_text(json.dumps({'url':p.stdout.strip()},indent=2)+'\n')
 return str(path)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('state');p.add_argument('--post',action='store_true');a=p.parse_args();print(report(a.state,a.post))
