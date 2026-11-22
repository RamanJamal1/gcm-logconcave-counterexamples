"""Independent exact polynomial verification with rigorous Arb matrix positivity.

The identity uses standard-library Fractions. Positive definiteness is checked
by outward-rounded interval Schur complements, starting from exact rationals.
This avoids enormous exact determinants at the larger orders.
"""
import argparse,json,time
from fractions import Fraction
from pathlib import Path
from flint import arb,ctx
import verify_sos_standalone as independent

def verify(path,dps=200):
 ctx.dps=dps;ctx.threads=1;records=[]
 def interval_positive(matrix):
  exact=[[Fraction(v) for v in row] for row in matrix];n=len(exact)
  independent.require(n>0 and all(len(row)==n for row in exact),'Gram matrix is not square')
  independent.require(all(exact[i][j]==exact[j][i] for i in range(n) for j in range(n)),'Gram matrix is not symmetric')
  a=[[arb(v.numerator)/v.denominator for v in row] for row in exact];pivots=[]
  for k in range(n):
   pivot=a[k][k];independent.require(pivot>0,lambda:('interval pivot not proved positive',k,str(pivot)));pivots.append(str(pivot))
   for i in range(k+1,n):
    ratio=a[i][k]/pivot
    for j in range(i,n):a[i][j]-=ratio*a[j][k];a[j][i]=a[i][j]
  records.append({'size':n,'positive_pivot_intervals':pivots});return pivots
 old=independent.positive_definite;independent.positive_definite=interval_positive
 try:result=independent.verify(path)
 finally:independent.positive_definite=old
 result['verifier']='independent Fraction polynomial identity and Arb interval Schur complements';result['digits']=dps;result['matrix_checks']=records
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('certificate');p.add_argument('--digits',type=int,default=200);a=p.parse_args();r=verify(a.certificate,a.digits);Path(f'certs/interval_sos_verification_m{r["order"]}.json').write_text(json.dumps(r,indent=2)+'\n');print({k:v for k,v in r.items() if k!='matrix_checks'},flush=True)
