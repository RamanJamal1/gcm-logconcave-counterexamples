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
if __name__=='__main__':
 a,b,c,k=Q(1,16),Q(1,45000),Q(-1,20000000),Q(67,1000)
 g=[1-b,a-3*c,2*b,4*c];gp=deriv(g);gpp=deriv(gp);one_minus_x2=[Q(1),Q(0),Q(-1)]
 H=add(mul(add(mul(one_minus_x2,gpp),scale(mul([Q(0),Q(1)],gp),-1)),g),scale(mul(one_minus_x2,mul(gp,gp)),-1))
 N=add(mul(deriv(H),g),scale(mul(H,gp),-2));constant=N[0];rest=sum(map(abs,N[1:]));assert abs(a)+abs(b)+abs(c)<1;assert constant<Q(-3,50);assert rest<Q(1,100)
 minimum=k-value(H,Q(-1))/value(g,Q(-1))**2;assert minimum>Q(1,2500)
 r={'certified':True,'parameters':{'amplitudes':list(map(str,[a,b,c])),'kappa':str(k)},'cosine_polynomial_g':list(map(str,g)),'log_curvature_numerator_H':list(map(str,H)),'derivative_numerator_N':list(map(str,N)),'constant_coefficient':str(constant),'absolute_sum_nonconstant_coefficients':str(rest),'N_uniform_upper_bound':'-1/20','minimum_curvature_exact':str(minimum),'minimum_attained_when':'cos(x) = -1','simple_strict_lower_bound':'1/2500','method':'exact Fraction polynomial arithmetic; N(c)<0 on [-1,1] by coefficient dominance; endpoint evaluation'}
 Path('certs/exact_min_curvature_m16.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
