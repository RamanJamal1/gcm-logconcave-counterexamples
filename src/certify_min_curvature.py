"""Standard-library exact proof of the witness's global minimum curvature."""
from fractions import Fraction as Q
from pathlib import Path
import json

def add(a,b):return [(a[j] if j<len(a) else Q(0))+(b[j] if j<len(b) else Q(0)) for j in range(max(len(a),len(b)))]
def scale(a,c):return [x*c for x in a]
def mul(a,b):
 out=[Q(0)]*(len(a)+len(b)-1)
 for i,x in enumerate(a):
  for j,y in enumerate(b):out[i+j]+=x*y
 return out
def deriv(a):return [j*a[j] for j in range(1,len(a))]
def value(a,x):
 out=Q(0)
 for c in reversed(a):out=out*x+c
 return out
def certify(amplitudes=('1/16','1/45000','-1/20000000'),kappa='67/1000',lower_bound='0'):
 a,b,c=map(Q,amplitudes);k=Q(kappa)
 g=[1-b,a-3*c,2*b,4*c];gp=deriv(g);gpp=deriv(gp);one_minus_x2=[Q(1),Q(0),Q(-1)]
 H=add(mul(add(mul(one_minus_x2,gpp),scale(mul([Q(0),Q(1)],gp),-1)),g),scale(mul(one_minus_x2,mul(gp,gp)),-1))
 N=add(mul(deriv(H),g),scale(mul(H,gp),-2));constant=N[0];rest=sum(map(abs,N[1:]));assert abs(a)+abs(b)+abs(c)<1;assert constant+rest<0, 'Coefficient dominance does not prove monotonicity'
 minimum=k-value(H,Q(-1))/value(g,Q(-1))**2;assert minimum>Q(lower_bound)
 r={'certified':True,'parameters':{'amplitudes':list(map(str,[a,b,c])),'kappa':str(k)},'cosine_polynomial_g':list(map(str,g)),'log_curvature_numerator_H':list(map(str,H)),'derivative_numerator_N':list(map(str,N)),'constant_coefficient':str(constant),'absolute_sum_nonconstant_coefficients':str(rest),'N_uniform_upper_bound':str(constant+rest),'minimum_curvature_exact':str(minimum),'minimum_attained_when':'cos(x) = -1','simple_strict_lower_bound':str(Q(lower_bound)),'method':'exact Fraction polynomial arithmetic; N(c)<0 on [-1,1] by coefficient dominance; endpoint evaluation'}
 return r
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--amplitudes',nargs=3,default=['1/16','1/45000','-1/20000000']);p.add_argument('--kappa',default='67/1000');p.add_argument('--lower-bound',default='1/2500');p.add_argument('--output',default='certs/exact_min_curvature_m16.json');a=p.parse_args();r=certify(a.amplitudes,a.kappa,a.lower_bound);Path(a.output).write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
