"""Promote nonnegative-order certificates only after independent exact checks.

This updates mathematical state only. Prose must be revised separately, so a
new sign certificate can never accidentally extend a sharp-bound claim.
"""
import argparse,json,hashlib,time
from pathlib import Path
from verify_sos_standalone import verify
ROOT=Path(__file__).resolve().parents[1]
def update(orders):
 path=ROOT/'checkpoints/research_status.json';state=json.loads(path.read_text())
 for m in orders:
  cert=ROOT/f'certs/exact_quotient_m{m}.json';result=verify(cert)
  assert result['exact_identity'] and result['positive_definite_grams'] and result['ibp_identities_verified']
  result['sha256']=hashlib.sha256(cert.read_bytes()).hexdigest();result['certificate']=str(cert.relative_to(ROOT));state.setdefault('verified_certificates',{})[str(m)]=result
  if m not in state['nonnegative_orders']:state['nonnegative_orders'].append(m)
 lower=6
 while lower in state['nonnegative_orders']:lower+=1
 upper=state['negative_witness_order'];assert lower<=upper,'Contradictory proof state requires investigation'
 changed=lower!=state['lower_bound'];state.update(lower_bound=lower,upper_bound=upper,updated=time.time(),exact_minimal_order_resolved=lower==upper)
 state['nonnegative_orders'].sort()
 if changed:state['report_revision_required']=True
 path.write_text(json.dumps(state,indent=2)+'\n');print(json.dumps({'lower_bound':lower,'upper_bound':upper,'verified_orders':orders}),flush=True);return state
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--orders',type=int,nargs='+',required=True);a=p.parse_args();update(a.orders)
