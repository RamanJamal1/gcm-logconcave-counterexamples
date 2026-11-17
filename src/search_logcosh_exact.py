"""Exact signs for tilted sech-power densities, using beta-distribution moments.

p is proportional to exp(beta*x) cosh(x)^(-alpha), |beta|<alpha.
V''=alpha sech(x)^2>0. This family has exponential, not Gaussian, tails.
Finite exact checks do not establish a universal order bound.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import json,time
from pathlib import Path
from fractions import Fraction
from flint import fmpq_poly,fmpq
from entropy_sparse import polynomials
from checkpoint_io import write_json
ROOT=Path(__file__).resolve().parents[1]
def grouped(max_order=17):
 t=fmpq_poly([0,1]);D=fmpq_poly([1,0,-1]);u={1:-t}
 for j in range(2,max_order+1):u[j]=D*u[j-1].derivative()
 powers={};out={}
 for m,p in polynomials(max_order).items():
  groups={}
  for ex,c in p.items():
   degree=sum(ex);poly=fmpq_poly([fmpq(c.numerator,c.denominator)*(-1)**m])
   for i,n in enumerate(ex):
    if n:
     if (i+1,n) not in powers:powers[i+1,n]=u[i+1]**n
     poly*=powers[i+1,n]
   groups[degree]=groups.get(degree,fmpq_poly())+poly
  out[m]=groups
 return out

def value(groups,alpha,beta):
 a=fmpq(str(alpha));b=fmpq(str(beta));poly=sum((p*a**n for n,p in groups.items()),fmpq_poly());moments=[fmpq(1),b/a]
 for n in range(1,poly.degree()):moments.append((b*moments[-1]+n*moments[-2])/(a+n))
 return sum((c*moments[i] for i,c in enumerate(poly)),fmpq(0))

def main():
 start=time.time();g=grouped();rows=[]
 alphas=['1/100','1/30','1/10','1/3','1/2','1','2','3','5','10','30','100']
 for alpha in alphas:
  for ratio in ['0','1/4','1/2','3/4','9/10','99/100']:
   beta=str(Fraction(alpha)*Fraction(ratio));values={str(m):str(value(g[m],alpha,beta)) for m in range(6,18)}
   row={'alpha':alpha,'beta':beta,'exact_signed_derivatives':values,'negative_orders':[int(m) for m,v in values.items() if Fraction(v)<0]};rows.append(row)
   write_json(ROOT/'certs/logcosh_exact_scan.json',{'status':'running','family':'exp(beta*x) cosh(x)^(-alpha)','checks':rows,'seconds':time.time()-start})
   print(alpha,beta,row['negative_orders'],flush=True)
 write_json(ROOT/'certs/logcosh_exact_scan.json',{'status':'complete','family':'exp(beta*x) cosh(x)^(-alpha)','checks':rows,'seconds':time.time()-start,'method':'Exact polynomial entropy recurrence and exact beta moments; all finite parameters only.'})
if __name__=='__main__':main()
