"""Finite-order Arb certificate for a Gaussian times two Fourier modes.

p=Z^-1 exp(-kappa*x²/2)(1+a*cos(x)+b*cos(2*x)). All parameters
are exact rationals; the infinite log series and every nonzero Gaussian-filtered
Fourier mode are enclosed analytically. No spatial quadrature is used.
"""
import argparse,json,math,time
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from flint import arb,ctx

def rat(v):
 f=Fraction(str(v));return arb(f.numerator)/f.denominator

@lru_cache(None)
def gamma_numerator(p,q,k=0):
 if (p-k)%2:return 0
 return sum(math.comb(q,j)*math.comb(p,(p+2*q-4*j-k)//2) for j in range(q+1) if 0<=p+2*q-4*j-k<=2*p)

def certify(m,a_string,b_string,kappa_string,N=96,dps=200,mode_cutoff=0):
 start=time.time();ctx.dps=dps;ctx.threads=1
 a,b,kappa=map(rat,[a_string,b_string,kappa_string]);r=abs(a)+abs(b)
 if not (r<1 and kappa>0):raise ValueError('inadmissible parameters')
 curvature=kappa-(abs(a)+4*abs(b))/(1-r)
 if not curvature>0:raise ValueError('convexity not proven')
 # Differentiation of exp(-L*t/(1+kappa*t)); the expression is a polynomial.
 coeff=[math.factorial(m)//math.factorial(m-j)*math.comb(m-1,j)*kappa**j for j in range(m)]
 T={L:sum((c*arb(L)**(m-j) for j,c in enumerate(coeff)),arb(0)) for L in range(-mode_cutoff**2,2*N+1)}
 ap=[a**p for p in range(N+1)];bp=[b**q for q in range(N+1)]
 periodic=arb(0)
 filters=[(-arb(k*k)/(2*kappa)).exp() for k in range(mode_cutoff+1)]
 for n in range(2,N+1):
  for p in range(n+1):
   q=n-p
   for k in range(p%2,mode_cutoff+1,2):
    g=gamma_numerator(p,q,k)
    if g:
     periodic+=arb((-1)**n*math.comb(n,p)*g*(1 if k==0 else 2))/(n*(n-1)*2**n)*ap[p]*bp[q]*T[(p+4*q-k*k)//2]*filters[k]
 # For each length n term, sum of absolute Fourier coefficients <= r^n.
 # Its exponent is B=(sum frequencies² - (sum frequencies)²)/2,
 # |B|<=2n². |T_m(B,kappa)| <= (2n²+m*kappa)^m.
 # Every nonzero mode is suppressed by exp(-1/(2*kappa)).
 filtering=(-arb((mode_cutoff+1)**2)/(2*kappa)).exp()*(2*N*N+m*kappa)**m*r*r/(1-r)
 # n^(2m) r^n has decreasing ratio, bounded by qtail for n>=N+1.
 qtail=r*(1+arb(1)/(N+1))**(2*m)
 if not qtail<1:raise ValueError('increase series cutoff')
 tail=(2+m*kappa)**m*r**(N+1)*arb(N+1)**(2*m)/(1-qtail)
 xi=1+a*(-1/(2*kappa)).exp()+b*(-2/kappa).exp()
 moment=math.factorial(m)*kappa**(m-1)*(a/2*(-1/(2*kappa)).exp()+2*b*(-2/kappa).exp())
 baseline=arb(math.factorial(m-1))/2*kappa**m
 value=baseline+(periodic+moment+arb(0,(filtering+tail).upper()))/xi
 variance=(1+a*(1-1/kappa)*(-1/(2*kappa)).exp()+b*(1-4/kappa)*(-2/kappa).exp())/(kappa*xi)
 normalized=value*variance**m
 result={'normalized_signed_midpoint':float(normalized.mid()),'normalized_signed_interval':str(normalized),'variance_interval':str(variance),'order':m,'family':'gaussian_two_fourier_modes','parameters':{'a':a_string,'b':b_string,'kappa':kappa_string},'density':'Z^-1 exp(-kappa*x^2/2)*(1+a*cos(x)+b*cos(2*x))','signed_derivative_interval':str(value),'convexity_lower_bound':str(curvature.lower()),'periodic_finite_sum':str(periodic),'nonzero_mode_absolute_bound':str(filtering.upper()),'series_tail_absolute_bound':str(tail.upper()),'mode_cutoff':mode_cutoff,'series_cutoff':N,'digits':dps,'certified':bool(value.is_finite() and value<0),'status':'certified_negative' if value.is_finite() and value<0 else 'inconclusive_or_nonnegative','seconds':time.time()-start,'method':'exact time-change polynomial, finite periodic Fourier sum, rigorous nonzero-mode and infinite-series bounds'}
 return result

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--orders',type=int,nargs='+',default=[40,44,48,52,56]);p.add_argument('--a',default='1/16384');p.add_argument('--b',default='1/72057594037927936');p.add_argument('--kappa',default='1/2048');p.add_argument('--cutoff',type=int,default=96);args=p.parse_args()
 for m in args.orders:
  r=certify(m,args.a,args.b,args.kappa,args.cutoff);Path(f'certs/fourier_series_m{m}.json').write_text(json.dumps(r,indent=2)+'\n');print(m,r['status'],r['signed_derivative_interval'],flush=True)
