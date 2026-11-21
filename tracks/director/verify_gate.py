"""Gate for m*-relevant certificates: target_subtraction must be absent (or exactly '0'),
order must match the expected order, then run tracks/D_order14/verify_strict.py (python -O safe).
Usage: python3 -O verify_gate.py <cert.json> <expected_order> <out.json>"""
import json,sys,subprocess,os
cert,order,out=os.path.abspath(sys.argv[1]),int(sys.argv[2]),os.path.abspath(sys.argv[3])
d=json.load(open(cert))
if d.get('order')!=order: raise RuntimeError(f"order mismatch {d.get('order')} != {order}")
ts=d.get('target_subtraction','0')
if str(ts).strip() not in ('0',''): raise RuntimeError(f'nonzero target_subtraction not allowed: {str(ts)[:80]}')
vs=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','D_order14','verify_strict.py')
r=subprocess.run([sys.executable,'-O',vs,cert,out],capture_output=True,text=True)
if r.returncode!=0: raise RuntimeError('strict verifier failed: '+(r.stderr[-500:]))
res=json.load(open(out))
if not (res.get('exact_identity') and res.get('positive_definite_grams') and res.get('ibp_identities_verified') and res.get('python_optimize_flag')==1 and res.get('order')==order):
    raise RuntimeError('result flags not all confirmed')
print('GATE PASS',order,res['seconds'])
