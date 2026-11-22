"""Check that verify_sos_standalone.py rejects corrupted certificates.

Tampered copies are written to a temporary directory (never into certs/), and the
verifier is run on each with plain `python` and with `python -O`. Every corrupted
certificate must give a non-zero exit status; the untouched controls must pass.

Usage (from the repository root): python src/test_verifier_rejects.py
"""
import copy,json,subprocess,sys,tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
VERIFIER=ROOT/'src'/'verify_sos_standalone.py'

def load(name):return json.loads((ROOT/'certs'/name).read_text())

def cases():
 base=load('exact_sos_m6.json');sharp=load('exact_mckean_m6.json')
 c=copy.deepcopy(base);g=c['gram_matrices'][0];g[0][1]=g[1][0]=str(eval_fraction(g[0][1])+1/1000000)
 yield 'gram_entry_perturbed',c,False
 c=copy.deepcopy(base);c['ibp_multipliers'][0]=str(eval_fraction(c['ibp_multipliers'][0])+1)
 yield 'ibp_multiplier_changed',c,False
 c=copy.deepcopy(base);c['gram_matrices'][1][0][0]='-1'
 yield 'gram_not_positive_definite',c,False
 c=copy.deepcopy(base);g=c['gram_matrices'][0];g[0][1]=str(eval_fraction(g[0][1])+1)
 yield 'gram_not_symmetric',c,False
 c=copy.deepcopy(sharp);c['target_subtraction']='61*u2**6'
 yield 'wrong_sharp_subtraction',c,False
 c=copy.deepcopy(base);c['ibp_polynomials']=c['ibp_polynomials'][:-1];c['ibp_multipliers']=c['ibp_multipliers'][:-1]
 yield 'ibp_term_dropped',c,False
 yield 'control_sign_m6',base,True
 yield 'control_sharp_m6',sharp,True

def eval_fraction(s):
 from fractions import Fraction
 return Fraction(s)

def main():
 failures=0
 with tempfile.TemporaryDirectory() as tmp:
  for name,cert,should_pass in cases():
   path=Path(tmp)/f'{name}.json';path.write_text(json.dumps(cert))
   for flags in ([],['-O']):
    code=subprocess.run([sys.executable,*flags,str(VERIFIER),str(path)],capture_output=True,text=True).returncode
    ok=(code==0)==should_pass
    failures+=not ok
    print(f"{'PASS' if ok else 'FAIL'}  {name:28s} python {' '.join(flags) or '  '}  exit={code}  expected={'accept' if should_pass else 'reject'}",flush=True)
 print('all negative tests passed' if not failures else f'{failures} test(s) FAILED')
 return 1 if failures else 0

if __name__=='__main__':
 sys.exit(main())
