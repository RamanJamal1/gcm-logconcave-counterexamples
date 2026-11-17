#!/usr/bin/env python3
"""Independent finite-difference checks, not interval certificates.

Test density p(x)=phi(x)*(1+a*sin(b*x)), a=1/10, b=1. Its exact
heat convolution is phi_s(x)*(1+c*sin(b*x/s)), s=1+t,
c=a*exp(-b*b*t/(2*s)). This identity independently follows by Gaussian
conditioning and provides the heat-kernel convolution without small-t
inner quadrature error. Outer entropy integration is high precision
Gauss-Hermite. Positive-time forward differences use exact rational weights.
"""
import argparse, json, math, time
from pathlib import Path
import mpmath as mp
import sympy as sp
from generated_integrands import INTEGRANDS
from derive_integrands import derive,u
ROOT=Path(__file__).resolve().parents[1]

def gh(n):
    start=time.time()
    x,w=mp.gauss_quadrature(n,'hermite')
    print(f'Gauss-Hermite n={n} generated in {time.time()-start:.1f}s',flush=True)
    return [(mp.sqrt(2)*x[k],w[k]/mp.sqrt(mp.pi)) for k in range(n)]

def log_jets(x,a,b):
    q=[1+a*mp.sin(b*x)]+[a*b**j*mp.sin(b*x+j*mp.pi/2) for j in range(1,9)]
    ell={}
    for n in range(1,9):
        ell[n]=(q[n]-sum(math.comb(n-1,k-1)*ell[k]*q[n-k] for k in range(1,n)))/q[0]
    ell[1]-=x;ell[2]-=1
    return ell

def entropy(t,rule,a,b):
    s=1+t;c=a*mp.exp(-b*b*t/(2*s));q=b/mp.sqrt(s)
    # Gaussian entropy term exact. Asymmetric perturbation contributes zero
    # to Gaussian normalization and second moment by oddness.
    return -mp.log(2*mp.pi*mp.e*s)/2+mp.fsum(w*(1+c*mp.sin(q*z))*mp.log1p(c*mp.sin(q*z)) for z,w in rule)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--nodes',type=int,nargs='+',default=[96,160]);ap.add_argument('--dps',type=int,default=100);args=ap.parse_args()
    started=time.time()
    mp.mp.dps=args.dps
    a=mp.mpf(1)/10;b=mp.mpf(1);h=mp.mpf('0.0001');N=24
    formulas=derive()
    # mpmath evaluation uses exact rational coefficients, avoiding generated
    # Python constants being silently promoted from binary inexact values.
    funcs={m:sp.lambdify([u[j] for j in range(2,m+1)],formulas[m],'mpmath') for m in [6,7,8]}
    # Backend-export smoke check at exact rational jets; coefficients are halves.
    sample={j:mp.mpf(j-4)/7 for j in range(2,9)}
    for m in [6,7,8]:
        exact=funcs[m](*[sample[j] for j in range(2,m+1)])
        assert abs(INTEGRANDS[m](sample)-exact)<mp.mpf('1e-90')
    weights={m:[mp.mpf(str(q.p))/int(q.q) for q in sp.finite_diff_weights(m,list(range(N)),0)[m][-1]] for m in [6,7,8]}
    runs=[]
    for n in args.nodes:
        rule=gh(n)
        direct={m:mp.mpf(0) for m in [6,7,8]}
        for z,w in rule:
            jets=log_jets(z,a,b)
            for m in direct:direct[m]+=w*(1+a*mp.sin(b*z))*funcs[m](*[jets[j] for j in range(2,m+1)])
        values=[entropy(k*h,rule,a,b) for k in range(N)]
        row={'nodes':n,'results':{}}
        for m in direct:
            fd=mp.fsum(w*v for w,v in zip(weights[m],values))/h**m
            err=abs(fd-direct[m])/abs(direct[m])
            row['results'][str(m)]={'integrand':mp.nstr(direct[m],70),'heat_convolution_forward_difference':mp.nstr(fd,70),'relative_error':mp.nstr(err,12),'passed':bool(err<mp.mpf('1e-11'))}
        runs.append(row)
        print(json.dumps(row,indent=2),flush=True)
    cross={}
    if len(runs)>1:
        for m in [6,7,8]:
            last=mp.mpf(runs[-1]['results'][str(m)]['integrand']);previous=mp.mpf(runs[-2]['results'][str(m)]['integrand'])
            cross[str(m)]=mp.nstr(abs(last-previous)/abs(last),12)
    result={'convention':'Boltzmann H=integral p log p, generator (1/2)dxx','density':'standard_normal(x)*(1+(1/10)*sin(x))','strict_convexity_global_lower_bound':'71/81','proof_of_convexity':'Vxx=1-qxx/q+(qx/q)^2 >=1-|qxx|/q >=8/9; conservative 71/81 also valid','digits':args.dps,'finite_difference':{'scheme':'24-point forward on nonnegative heat times','step':str(h),'exact_rational_weights':True},'convolution':'exact Gaussian conditioning formula, outer high-precision Gauss-Hermite integration','runs':runs,'cross_resolution_relative_errors':cross,'generated_backend_smoke_check':'passed','elapsed_seconds':time.time()-started,'rigorous_interval_certificate':False,'passed':all(r['passed'] for r in runs[-1]['results'].values()) and all(mp.mpf(v)<mp.mpf('1e-11') for v in cross.values())}
    (ROOT/'certs/formula_tests.json').write_text(json.dumps(result,indent=2)+'\n')
    assert result['passed'], 'Finite-difference/cross-resolution tolerance not met; see certs/formula_tests.json'
    print('PASS: all independent m=6,7,8 comparisons below 1e-11 relative error')

if __name__=='__main__':main()
