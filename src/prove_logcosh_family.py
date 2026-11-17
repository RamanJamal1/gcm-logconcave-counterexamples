"""Attempt exact coefficientwise positivity for the whole tilted sech family.

For alpha=a>0 and beta^2=a^2(1-s), 0<s<=1, multiply S_m by
(a)_(2m). Exact positive polynomial coefficients suffice for positivity.
An unsuccessful coefficient test says nothing about the sign.
"""
import json,time,math
from pathlib import Path
from fractions import Fraction as F
from flint import fmpq_mpoly_ctx
from search_logcosh_exact import grouped,ROOT
from checkpoint_io import write_json

def main():
 start=time.time();groups=grouped();ctx=fmpq_mpoly_ctx.get(['a','b']);a,b=ctx.gens();N=[ctx.constant(1),b]
 for n in range(1,34):N.append(b*N[-1]+n*(a+n-1)*N[-2])
 records=[]
 for m in range(2,18):
  degree=2*m;tails=[ctx.constant(1)]*(degree+1)
  for k in range(degree-1,-1,-1):tails[k]=tails[k+1]*(a+k)
  poly=ctx.constant(0)
  for d,p in groups[m].items():
   for k,c in enumerate(p):
    if c:poly+=c*a**d*N[k]*tails[k]
  # b is used as the new variable s in the transformed polynomial.
  transformed=ctx.constant(0)
  for (i,j),c in poly.terms():
   assert j%2==0
   transformed+=c*a**(i+j)*(1-b)**(j//2)
  coeffs=list(transformed.coeffs());power_positive=bool(coeffs) and all(c>=0 for c in coeffs)
  by_a={}
  for (i,j),c in transformed.terms():by_a.setdefault(int(i),{})[int(j)]=F(str(c))
  bernstein=[]
  for i,co in sorted(by_a.items()):
   n=max(co);values=[sum((co.get(j,F(0))*F(math.comb(k,j),math.comb(n,j)) for j in range(k+1)),F(0)) for k in range(n+1)]
   bernstein.append({'alpha_power':i,'degree':n,'coefficients':list(map(str,values))})
  proved=all(F(v)>=0 for row in bernstein for v in row['coefficients']) and any(F(row['coefficients'][-1])>0 for row in bernstein)
  record={'order':m,'proved_positive_on_parameter_domain':proved,'power_basis_positive':power_positive,'bernstein_by_alpha_power':bernstein,'negative_coefficient_count':sum(c<0 for c in coeffs),'coefficient_count':len(coeffs),'domain':'alpha=a>0, beta^2=a^2(1-s), 0<s<=1','denominator':'product(a+j for j=0,...,2m-1)','polynomial_terms':[[list(map(int,e)),str(c)] for e,c in transformed.terms()] }
  records.append(record);print(m,proved,record['negative_coefficient_count'],len(coeffs),flush=True)
  write_json(ROOT/'certs/logcosh_family_coefficients.json',{'status':'running','records':records,'seconds':time.time()-start})
 write_json(ROOT/'certs/logcosh_family_coefficients.json',{'status':'complete','records':records,'seconds':time.time()-start,'note':'Nonnegative Bernstein coefficients in s on [0,1], and positive alpha powers, prove the whole family. A positive s=1 coefficient supplies strict positivity for 0<s<=1.'})
if __name__=='__main__':main()
