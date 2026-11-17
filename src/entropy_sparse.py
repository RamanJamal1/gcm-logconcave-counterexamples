"""Exact sparse recurrence on monomial exponent tuples, avoiding symbolic expansion."""
from fractions import Fraction as F
from math import comb

def polynomials(max_order):
 length=2*max_order
 e=[0]*length;e[1]=2;out={2:{tuple(e):F(1,2)}}
 for order in range(3,max_order+1):
  nxt={}
  def add(e,c):
   e=tuple(e);nxt[e]=nxt.get(e,F(0))+c
  for e,c in out[order-1].items():
   active=[i for i in range(1,max_order) if e[i]]
   for i in active:
    for j in active:
     multiplicity=e[i]*(e[j]-(i==j))
     if multiplicity:
      a=list(e);a[i]-=1;a[j]-=1;a[i+1]+=1;a[j+1]+=1;add(a,-c*multiplicity/2)
    for k in range(1,i+1):
     a=list(e);a[i]-=1;a[k]+=1;a[i+1-k]+=1;add(a,c*e[i]*comb(i+1,k)/2)
  out[order]={e:c for e,c in nxt.items() if c}
 return out

def signed_polynomial(m):return {e:c*(-1)**m for e,c in polynomials(m)[m].items()}
if __name__=='__main__':
 import time,json
 from pathlib import Path
 start=time.time();p=polynomials(17)
 results={m:len(v) for m,v in p.items()}
 print(results,time.time()-start)
 Path('certs/sparse_recurrence_counts.json').write_text(json.dumps({'term_counts':results,'seconds':time.time()-start},indent=2)+'\n')
