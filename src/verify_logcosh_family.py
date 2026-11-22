"""Independent recurrence and beta-binomial verification of the sech family.

The score recurrence comes from the separate Fraction verifier. Expectations
use beta raw moments after t=2y-1, rather than the generator's moment recursion.
All polynomial and Bernstein checks are exact rational arithmetic.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import json,math,time,hashlib
from fractions import Fraction as F
from pathlib import Path
from flint import fmpq,fmpq_poly,fmpq_mpoly_ctx
from verify_sos_standalone import entropy_polynomial,require
from checkpoint_io import write_json
ROOT=Path(__file__).resolve().parents[1]
def main():
 start=time.time();path=ROOT/'certs/logcosh_family_coefficients.json';rows=json.loads(path.read_text())['records'];ctx=fmpq_mpoly_ctx.get(['a','s']);a,s=ctx.gens();t=fmpq_poly([0,1]);D=fmpq_poly([1,0,-1]);u={1:-t};powers={};checked=[]
 for j in range(2,18):u[j]=D*u[j-1].derivative()
 for row in rows:
  m=row['order'];groups={}
  for e,c in entropy_polynomial(m).items():
   degree=sum(e);p=fmpq_poly([fmpq(c.numerator,c.denominator)])
   for j,n in enumerate(e):
    if n:
     if (j+1,n) not in powers:powers[j+1,n]=u[j+1]**n
     p*=powers[j+1,n]
   groups[degree]=groups.get(degree,fmpq_poly())+p
  N=2*m;tail=[ctx.constant(1)]*(N+1)
  for j in range(N-1,-1,-1):tail[j]=tail[j+1]*(a+j)
  products=[ctx.constant(1)]
  for j in range(N):products.append(products[-1]*(a+s+2*j))
  moments=[sum((math.comb(k,j)*(-1)**(k-j)*products[j]*tail[j] for j in range(k+1)),ctx.constant(0)) for k in range(N+1)]
  num=sum((c*a**d*moments[k] for d,p in groups.items() for k,c in enumerate(p) if c),ctx.constant(0));transformed=ctx.constant(0)
  for (i,j),c in num.terms():
   require(j%2==0,'odd power of the beta-moment variable')
   transformed+=c*a**(i+j)*(1-s)**(j//2)
  expected=ctx.from_dict({tuple(e):fmpq(c) for e,c in row['polynomial_terms']})
  require(transformed==expected,'beta moment identity failed')
  reconstructed=ctx.constant(0);positive_endpoint=False
  for b in row['bernstein_by_alpha_power']:
   i,n=b['alpha_power'],b['degree'];coefficients=[F(v) for v in b['coefficients']];require(all(v>=0 for v in coefficients),'negative Bernstein coefficient')
   positive_endpoint|=coefficients[-1]>0
   reconstructed+=a**i*sum((fmpq(v.numerator,v.denominator)*math.comb(n,k)*s**k*(1-s)**(n-k) for k,v in enumerate(coefficients)),ctx.constant(0))
  require(reconstructed==transformed and positive_endpoint,'Bernstein identity or positive endpoint failed')
  checked.append(m);print('verified',m,time.time()-start,flush=True)
  write_json(ROOT/'certs/independent_logcosh_family.json',{'verified_orders':checked,'exact_beta_identity':True,'exact_bernstein_identity':True,'positive_on_entire_parameter_domain':True,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'seconds':time.time()-start,'status':'running'})
 r=json.loads((ROOT/'certs/independent_logcosh_family.json').read_text());r['status']='complete';write_json(ROOT/'certs/independent_logcosh_family.json',r)
if __name__=='__main__':main()
