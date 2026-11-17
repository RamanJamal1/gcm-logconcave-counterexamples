"""Rigorous Arb integration and analytic Gaussian tails for Fourier families.

Only exact decimal/rational input parameters are certified. Other families are
explicitly unsupported, never silently labelled certified. H = integral p log p.
"""
import argparse,json,math
from fractions import Fraction
from pathlib import Path
from flint import arb,acb,ctx
from generated_integrands import INTEGRANDS
from derive_integrands import derive,u as symbols

def exact(v):
 f=Fraction(str(v));return arb(f.numerator)/f.denominator

def certify(candidate,order=6,dps=200,extent=16):
 ctx.dps=dps;ctx.threads=1
 c=candidate.get('parameters',candidate);family=c['family']
 if family not in ('fourier_multiplier','fourier_potential'):
  return {'status':'unsupported_family','certified':False,'family':family}
 a=list(map(exact,c['amplitudes']));w=list(map(exact,c['frequencies']));phase=list(map(exact,c['phases']))
 A={j:sum((abs(x)*abs(y)**j for x,y in zip(a,w)),arb(0)) for j in range(9)}
 if family=='fourier_multiplier':
  if not A[0]<1:return {'status':'positivity_not_proven','certified':False}
  curvature=1-A[2]/(1-A[0]);q={j:A[j]/(1-A[0]) for j in range(1,9)};bounds={1:q[1]}
  for n in range(2,9):bounds[n]=q[n]+sum(math.comb(n-1,k-1)*bounds[k]*q[n-k] for k in range(1,n))
  factor=1+A[0]
 else:curvature=1-A[2];bounds=A.copy();factor=A[0].exp()
 if not curvature>0:return {'status':'convexity_not_proven','certified':False,'curvature':str(curvature)}
 bounds[2]+=1
 import sympy as sp
 poly=sp.Poly(derive()[order],*[symbols[j] for j in range(2,order+1)])
 bound=arb(0)
 for powers,coeff in poly.terms():
  term=exact(abs(coeff))
  for j,power in enumerate(powers,2):term*=bounds[j]**power
  bound+=term
 L=arb(extent);tail_mass=2*factor*(-L*L/2).exp()/L
 def jets(x):
  ds={j:sum((aa*ww**j*(ww*x+pp+j*arb.pi()/2).cos() for aa,ww,pp in zip(a,w,phase)),acb(0)) for j in range(9)}
  if family=='fourier_multiplier':
   g=1+ds[0];q={j:ds[j]/g for j in range(1,9)};u={1:q[1]}
   for n in range(2,9):u[n]=q[n]-sum(math.comb(n-1,k-1)*u[k]*q[n-k] for k in range(1,n))
   density=(-x*x/2).exp()*g
  else:u={j:ds[j] for j in range(1,9)};density=(-x*x/2+ds[0]).exp()
  u[2]-=1
  return density,u
 # These functions are entire or meromorphic; poles automatically produce
 # nonfinite balls, satisfying acb.integral's analytic-domain contract.
 def f(x,analytic):
  p,u=jets(x);return (-1)**order*INTEGRANDS[order](u)*p
 tol=arb('1e-30')
 numerator=acb.integral(f,-L,L,abs_tol=tol,rel_tol=tol,eval_limit=200000).real
 numerator=numerator+arb(0,(bound*tail_mass).upper())
 if family=='fourier_multiplier':
  z=(2*arb.pi()).sqrt()*(1+sum((aa*(-ww*ww/2).exp()*pp.cos() for aa,ww,pp in zip(a,w,phase)),arb(0)))
 else:
  z=acb.integral(lambda x,_:jets(x)[0],-L,L,abs_tol=tol,rel_tol=tol,eval_limit=200000).real+arb(0,tail_mass.upper())
 value=numerator/z if z>0 else arb('nan')
 negative=bool(value.is_finite() and value<0)
 return {'status':'certified_negative' if negative else ('certified_nonnegative' if value.is_finite() and value>=0 else 'inconclusive'),'certified':negative,'order':order,'parameters':c,'digits':dps,'signed_derivative_interval':str(value),'convexity_lower_bound':str(curvature.lower()),'normalizer_interval':str(z),'absolute_numerator_tail_bound':str((bound*tail_mass).upper()),'extent':extent,'method':'Arb complex analytic integration with rigorous Gaussian tail majorant'}

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('candidate');p.add_argument('--order',type=int,choices=[6,7,8],default=6);p.add_argument('--output',required=True);a=p.parse_args()
 result=certify(json.loads(Path(a.candidate).read_text()),a.order);Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
