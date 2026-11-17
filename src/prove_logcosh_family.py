"""Attempt exact coefficientwise positivity for the whole tilted sech family.

For alpha=a>0 and beta^2=a^2(1-s), 0<s<=1, multiply S_m by
(a)_(2m). Exact positive polynomial coefficients suffice for positivity.
An unsuccessful coefficient test says nothing about the sign.
"""
import json,time,math
from pathlib import Path
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
  coeffs=list(transformed.coeffs());proved=bool(coeffs) and all(c>=0 for c in coeffs)
  record={'order':m,'proved_positive_on_parameter_domain':proved,'negative_coefficient_count':sum(c<0 for c in coeffs),'coefficient_count':len(coeffs),'domain':'alpha=a>0, beta^2=a^2(1-s), 0<s<=1','denominator':'product(a+j for j=0,...,2m-1)','positive_polynomial_terms':[[list(map(int,e)),str(c)] for e,c in transformed.terms()] if proved else None}
  records.append(record);print(m,proved,record['negative_coefficient_count'],len(coeffs),flush=True)
  write_json(ROOT/'certs/logcosh_family_coefficients.json',{'status':'running','records':records,'seconds':time.time()-start})
 write_json(ROOT/'certs/logcosh_family_coefficients.json',{'status':'complete','records':records,'seconds':time.time()-start,'note':'Coefficient positivity is a sufficient exact criterion. False entries are inconclusive.'})
if __name__=='__main__':main()
