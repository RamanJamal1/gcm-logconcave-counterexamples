"""Finite-order signs for the Zou-Fan-Gao-Wang family (arXiv 2608.30275).

f_m ∝ exp(-kappa x^2/2)(1 + a cos x + b cos 2x), a = 2^(-m/4), b = 2^(-m), kappa = 8a.
The paper proves (-1)^m H^(m) < 0 only for "sufficiently large m". Here each
member f_M is evaluated at order M with the repo's rigorous Arb series certifier
(src/certify_fourier_series.py). For M not divisible by 4, a is replaced by a
rational within 1e-12 (rounded up, which keeps kappa = 8a admissible).
"""
import json,sys,time
from fractions import Fraction
from pathlib import Path
sys.path.insert(0,'src')
from certify_fourier_series import certify
import mpmath
rows=[]
for M in range(int(sys.argv[1]),int(sys.argv[2])+1):
    if M%4==0: a=Fraction(1,2**(M//4))
    else:
        a=Fraction(str(mpmath.nstr(mpmath.mpf(2)**(-mpmath.mpf(M)/4),30))).limit_denominator(10**12)
    b=Fraction(1,2**M);kappa=8*a
    t=time.time()
    try:
        r=certify(M,str(a),str(b),str(kappa),N=160,dps=120,mode_cutoff=16)
        row={'order':M,'a':str(a),'b':str(b),'kappa':str(kappa),'exact_member':M%4==0,'status':r['status'],'certified':r.get('certified'),
             'signed_derivative_interval':r['signed_derivative_interval'],'normalized_signed_interval':r['normalized_signed_interval'],'seconds':round(time.time()-t,2)}
    except Exception as e:
        row={'order':M,'a':str(a),'kappa':str(kappa),'error':repr(e)}
    rows.append(row);print(json.dumps(row),flush=True)
    Path(sys.argv[3]).write_text(json.dumps({'family':'Zou-Fan-Gao-Wang f_m (arXiv 2608.30275), member f_M evaluated at order M','method':'src/certify_fourier_series.certify, N=160, dps=120, mode_cutoff=16','rows':rows},indent=2)+'\n')
