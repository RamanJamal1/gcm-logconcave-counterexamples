"""Independent spatial Arb certificate via exact rational Laurent polynomials.

Unlike certify_fourier_series, this integrates the reduced entropy integrand
at time zero and uses a Gaussian spatial tail bound.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import math,json,time,argparse
from pathlib import Path
from checkpoint_io import write_json
from fractions import Fraction
from flint import arb,acb,fmpq,fmpq_poly,acb_poly,ctx
from entropy_sparse import signed_polynomial

def rational(s):
 f=Fraction(str(s));return fmpq(f.numerator,f.denominator)
def ball(s):
 f=Fraction(str(s));return arb(f.numerator)/f.denominator

def certify(m,amplitude_strings,k_s,L=120,dps=200):
 start=time.time();ctx.dps=dps;ctx.threads=1
 amplitudes=list(map(rational,amplitude_strings));k=rational(k_s);K=len(amplitudes);z=fmpq_poly([0,1]);coefficients=[fmpq(0)]*(2*K+1);coefficients[K]=1
 for j,a in enumerate(amplitudes,1):coefficients[K-j]=a/2;coefficients[K+j]=a/2
 G=fmpq_poly(coefficients);R={1:z*G.derivative()-K*G}
 for j in range(1,m):R[j+1]=z*(R[j].derivative()*G-j*R[j]*G.derivative())
 R[2]+=k*G**2;N=fmpq_poly([])
 terms=signed_polynomial(m)
 for e,c in terms.items():
  p=fmpq_poly([rational(c)*(-1)**m])
  for j,power in enumerate(e,1):
   if power:p*=R[j]**power
  N+=p
 aa=list(map(ball,amplitudes));kk=ball(k);center=2*K*m
 assert all(N[j]==N[2*center-j] for j in range(center+1)), 'Expected reciprocal numerator'
 cheb=[fmpq_poly([1]),z];C=fmpq_poly([N[center]])
 for j in range(1,center+1):
  if j>=2:cheb.append(2*z*cheb[-1]-cheb[-2])
  C+=2*N[center+j]*cheb[j]
 nc=[ball(c) for c in C];poly=acb_poly(nc)
 def integrand(x,analytic):
  c=x.cos();g=1+sum((a*(j*x).cos() for j,a in enumerate(aa,1)),acb(0))
  return poly(c)/g**(2*m-1)*(-kk*x*x/2).exp()
 A={j:sum((abs(a)*w**j for w,a in enumerate(aa,1)),arb(0)) for j in range(m+1)}
 if not A[0]<1:raise ValueError('Density positivity not proven')
 q={j:A[j]/(1-A[0]) for j in range(1,m+1)};bounds={1:q[1]}
 for j in range(2,m+1):bounds[j]=q[j]+sum(math.comb(j-1,l-1)*bounds[l]*q[j-l] for l in range(1,j))
 bounds[2]+=kk;bound=arb(0)
 for e,c in terms.items():
  term=abs(ball(c))
  for j,power in enumerate(e,1):
   if power:term*=bounds[j]**power
  bound+=term
 tail=2*(1+A[0])*bound*(-kk*L*L/2).exp()/(kk*L)
 normalizer=(2*arb.pi()/kk).sqrt()*(1+sum((a*(-w*w/(2*kk)).exp() for w,a in enumerate(aa,1)),arb(0)))
 total=arb(0);tol=arb('1e-24');pieces=[]
 print('exact rational numerator degree',len(nc)-1,'tail',str(tail),'setup_seconds',time.time()-start,flush=True)
 for index in range(4*L):
  left=arb(index)/4;right=arb(index+1)/4;part=acb.integral(integrand,left,right,abs_tol=tol,rel_tol=tol,eval_limit=100000).real
  if not part.is_finite():raise ArithmeticError(f'Nonfinite interval {left}:{right}')
  total+=2*part;pieces.append({'left':str(left),'right':str(right),'integral':str(part)});print(left,right,str(part),flush=True)
 value=(total+arb(0,tail.upper()))/normalizer;curvature=kk-A[2]/(1-A[0]);exact_curvature=None
 if K==3:
  try:
   from certify_min_curvature import certify as curvature_proof
   exact_curvature=curvature_proof(amplitude_strings,k_s);curvature=ball(exact_curvature['minimum_curvature_exact'])
  except AssertionError:pass
 return {'order':m,'parameters':{'amplitudes':amplitude_strings,'kappa':k_s},'certified':bool(value<0 and curvature>0),'signed_derivative_interval':str(value),'convexity_lower_bound':str(curvature),'exact_curvature_certificate':exact_curvature,'numerator_tail_bound':str(tail),'normalizer_interval':str(normalizer),'pieces':pieces,'digits':dps,'extent':L,'seconds':time.time()-start,'method':'independent exact rational spatial entropy polynomial, Arb complex analytic integration and Gaussian tail'}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--order',type=int,default=16);p.add_argument('--amplitudes',nargs='+',default=['1/15','1/40000','-1/18000000']);p.add_argument('--kappa',default='1/14');p.add_argument('--extent',type=int,default=120);p.add_argument('--output',default='certs/paired15_m16.json');a=p.parse_args();r=certify(a.order,a.amplitudes,a.kappa,a.extent);write_json(a.output,r);print(r['certified'],r['signed_derivative_interval'],flush=True)
