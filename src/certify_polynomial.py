"""Rigorous Arb certificate for V''=1+q² polynomial potentials."""
import argparse,json,math,time
from fractions import Fraction
from pathlib import Path
from flint import fmpq,fmpq_poly,arb,acb,ctx
from entropy_sparse import signed_polynomial

def rational(v):
 r=Fraction(str(v));return fmpq(r.numerator,r.denominator)
def ball(v):
 r=Fraction(str(v));return arb(r.numerator)/r.denominator
def horner(coeff,x):
 value=acb(0)
 for c in reversed(coeff):value=value*x+c
 return value

def certify(candidate,order=12,dps=200,extent=64):
 start=time.time();ctx.dps=dps;ctx.threads=1
 q=fmpq_poly([rational(v) for v in candidate['q_coefficients']]);tilt=rational(candidate['tilt']);square=q*q
 coefficients=[fmpq(0),tilt]+[c/((j+1)*(j+2)) for j,c in enumerate(square)];coefficients += [fmpq(0)]*max(0,3-len(coefficients));coefficients[2]+=fmpq(1,2);V=fmpq_poly(coefficients)
 jets={};deriv=V
 for j in range(1,order+1):deriv=deriv.derivative();jets[j]=-deriv
 F=fmpq_poly([])
 for powers,c in signed_polynomial(order).items():
  term=fmpq_poly([fmpq(c.numerator,c.denominator)])
  for j,power in enumerate(powers,1):
   if power:term*=jets[j]**power
  F+=term
 vc=[ball(c) for c in V];fc=[ball(c) for c in F];L=arb(extent)
 if not L>=4*abs(ball(tilt)):raise ValueError('extent too small for the Gaussian tail majorant')
 e=(-L*L/4).exp();tails=[arb.pi().sqrt()*(L/2).erfc(),2*e]
 for j in range(2,len(fc)):tails.append(2*L**(j-1)*e+2*(j-1)*tails[j-2])
 numerator_tail=2*sum((abs(c)*tails[j] for j,c in enumerate(fc)),arb(0));z_tail=2*tails[0]
 def density(x,analytic):return (-horner(vc,x)).exp()
 def integrand(x,analytic):return horner(fc,x)*density(x,analytic)
 tol=arb('1e-30');z=acb.integral(density,-L,L,abs_tol=tol,rel_tol=tol,eval_limit=200000).real+arb(0,z_tail.upper())
 numerator=acb.integral(integrand,-L,L,abs_tol=tol,rel_tol=tol,eval_limit=200000).real+arb(0,numerator_tail.upper())
 value=numerator/z
 return {'order':order,'family':'squared_polynomial_curvature','parameters':candidate,'curvature_lower_bound':'1','signed_derivative_interval':str(value),'normalizer_interval':str(z),'numerator_tail_bound':str(numerator_tail.upper()),'normalizer_tail_bound':str(z_tail.upper()),'potential_degree':len(vc)-1,'integrand_polynomial_degree':len(fc)-1,'digits':dps,'extent':extent,'certified':bool(z>0 and value.is_finite() and value<0),'status':'certified_negative' if z>0 and value.is_finite() and value<0 else 'nonnegative_or_inconclusive','seconds':time.time()-start,'method':'exact rational polynomial entropy integrand; Arb complex integration; Gaussian polynomial-moment tail recurrence'}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('candidate');p.add_argument('--order',type=int,default=12);p.add_argument('--output',required=True);a=p.parse_args();r=certify(json.loads(Path(a.candidate).read_text()),a.order);Path(a.output).write_text(json.dumps(r,indent=2)+'\n');print(r,flush=True)
