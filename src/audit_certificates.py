"""Audit proof provenance, exact parameter agreement, and certificate hashes.

Missing low-order verification records are filled by an independent exact
check. Previously verified identities are accepted only if their hashes match.
Numerical SDP candidates are excluded from this manifest.
"""
import json,hashlib,time
from pathlib import Path
from fractions import Fraction
from flint import arb,ctx
from checkpoint_io import write_json
from verify_sos_standalone import verify
from certify_min_curvature import certify as curvature_proof
ROOT=Path(__file__).resolve().parents[1]
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 ctx.dps=200;start=time.time();sp=ROOT/'checkpoints/research_status.json';state=json.loads(sp.read_text());records=[]
 for sharp,key in [(False,'verified_certificates'),(True,'verified_sharp_certificates')]:
  for m in (state['sharp_gaussian_orders'] if sharp else state['nonnegative_orders']):
   if not sharp and str(m) in state.get('propagated_signs',{}):continue
   if sharp:path=ROOT/(f'certs/exact_mckean_m{m}.json' if m<=8 else f'certs/exact_mckean_quotient_m{m}.json')
   else:path=ROOT/(f'certs/exact_sos_m{m}.json' if m<=8 else f'certs/exact_quotient_m{m}.json')
   stored=state.setdefault(key,{}).get(str(m))
   if stored is None:
    stored=verify(path);stored.update(certificate=str(path.relative_to(ROOT)),sha256=digest(path));state[key][str(m)]=stored
   assert stored['sha256']==digest(path),(key,m,'hash changed')
   assert stored['exact_identity'] and stored['positive_definite_grams'] and stored['ibp_identities_verified']
   records.append({'kind':'sharp_gaussian_bound' if sharp else 'universal_nonnegative_sign','order':m,'path':str(path.relative_to(ROOT)),'sha256':digest(path),'verification':stored['verifier']})
 for m,prop in state.get('propagated_signs',{}).items():
  assert str(prop['source_order']) in state['verified_certificates'];lemma=ROOT/prop['lemma'];assert digest(lemma)==prop['lemma_sha256'];records.append({'kind':'heat_flow_propagated_sign','order':int(m),**prop})
 a=ROOT/state['negative_witness_certificate'];b=ROOT/state['independent_negative_witness_certificate'];c=ROOT/state['curvature_certificate'];ra,rb,rc=[json.loads(p.read_text()) for p in (a,b,c)]
 assert ra['parameters']==rb['parameters']==rc['parameters'];assert ra['certified'] and rb['certified'] and rc['certified'];va,vb=[arb(r['signed_derivative_interval']) for r in (ra,rb)];assert va<0 and vb<0 and va.overlaps(vb)
 assert Fraction(rc['minimum_curvature_exact'])>Fraction(rc['simple_strict_lower_bound'])>0
 assert curvature_proof(rc['parameters']['amplitudes'],rc['parameters']['kappa'],rc['simple_strict_lower_bound'])==rc
 for p in (a,b,c):
  relative=str(p.relative_to(ROOT));assert state['witness_certificate_sha256'][relative]==digest(p);records.append({'kind':'negative_witness_or_global_curvature','order':state['negative_witness_order'],'path':relative,'sha256':digest(p)})
 family=ROOT/'certs/logcosh_family_coefficients.json';check=ROOT/'certs/independent_logcosh_family.json';r=json.loads(check.read_text());assert r['status']=='complete' and r['source_sha256']==digest(family);assert r['verified_orders']==list(range(2,18));records.append({'kind':'family_specific_positive_signs','orders':r['verified_orders'],'path':str(family.relative_to(ROOT)),'sha256':digest(family),'verification_record':str(check.relative_to(ROOT)),'verification_sha256':digest(check)})
 assert state['lower_bound']==max(state['nonnegative_orders'])+1<=state['upper_bound']==state['negative_witness_order']
 write_json(sp,state);out={'passed':True,'lower_bound':state['lower_bound'],'upper_bound':state['upper_bound'],'exact_minimal_order_resolved':state['exact_minimal_order_resolved'],'records':records,'seconds':time.time()-start,'note':'Analytic extension and downward propagation are mathematical arguments in the note; numerical search candidates are not included.'};write_json(ROOT/'certs/verified_manifest.json',out);print({'passed':True,'records':len(records),'bracket':[state['lower_bound'],state['upper_bound']],'seconds':time.time()-start},flush=True)
if __name__=='__main__':main()
