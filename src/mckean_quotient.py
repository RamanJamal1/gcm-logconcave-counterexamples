"""Exact IBP quotient for the stronger Gaussian-curvature subtraction."""
import math
from functools import lru_cache
from pathlib import Path
import sympy as s
from entropy_sparse import signed_polynomial
from sos_probe import partitions
ROOT=Path(__file__).resolve().parents[1]
def setup(m):
 L=2*m
 def exp(parts):return tuple(parts.count(j) for j in range(1,L+1))
 @lru_cache(None)
 def reduce(e):
  if not e[0]:return {e:1}
  p=list(e);p[0]-=1;out={}
  for j,n in enumerate(p[:-1]):
   if n:
    f=p.copy();f[j]-=1;f[j+1]+=1
    for k,v in reduce(tuple(f)).items():out[k]=out.get(k,0)-n*v
  return {k:v for k,v in out.items() if v}
 bases=[[exp(p) for p in partitions(m,m) if max(p)>=3],[exp(p) for p in partitions(m-1,m-1) if max(p)>=3]]
 target={e:s.Rational(c.numerator,c.denominator) for e,c in signed_polynomial(m).items()}
 gaussian=[0]*L;gaussian[1]=m;gaussian=tuple(gaussian);target[gaussian]-=s.Rational(math.factorial(m-1),2)*(-1)**m;target={e:c for e,c in target.items() if c}
 keys=set(target);columns=[];layout=[]
 for block,base in enumerate(bases):
  for i,a in enumerate(base):
   for j in range(i,len(base)):
    b=base[j];e=[x+y for x,y in zip(a,b)];e[1]+=block
    d={k:v*(-1 if block else 1)*(1 if i==j else 2) for k,v in reduce(tuple(e)).items()}
    keys.update(d);columns.append(d);layout.append((block,i,j))
 keys=sorted(keys)
 # Keep every canonical equality. Exact reconstruction checks all rows;
 # there is no need for an expensive preliminary row-reduction here.
 rows=list(range(len(keys)))
 return bases,target,keys,columns,layout,rows
