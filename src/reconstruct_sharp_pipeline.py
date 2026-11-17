"""Promote sharp-bound candidates only after exact independent verification."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import json,hashlib,time
from pathlib import Path
from reconstruct_mckean_quotient import reconstruct
from verify_sos_standalone import verify
from checkpoint import checkpoint
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
 for m in [9,10,11]:
  source=ROOT/f'certs/sos_mckean_quotient_m{m}.json';reconstruct(m,source);path=ROOT/f'certs/exact_mckean_quotient_m{m}.json';r=verify(path);r.update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),certificate=str(path.relative_to(ROOT)))
  state_path=ROOT/'checkpoints/research_status.json';state=json.loads(state_path.read_text());state.setdefault('verified_sharp_certificates',{})[str(m)]=r
  if m not in state['sharp_gaussian_orders']:state['sharp_gaussian_orders'].append(m)
  state['sharp_gaussian_orders'].sort();state['updated']=time.time();state_path.write_text(json.dumps(state,indent=2)+'\n');print('INDEPENDENTLY VERIFIED SHARP BOUND',m,r,flush=True);checkpoint(f'exact independently verified sharp Gaussian bound m{m}')
