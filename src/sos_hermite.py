"""Polynomial Hermite score basis to reduce monomial conditioning in SOS search."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,json,math,time
from functools import lru_cache
import numpy as np
import scipy.linalg as la
import sympy as s
import cvxpy as cp
from sos_probe import partitions
from sos_quotient import ROOT
from entropy_sparse import signed_polynomial

def setup(m):
 L=2*m
 def exp(parts):return tuple(parts.count(j) for j in range(1,L+1))
 def hermite(e):
  a=e[0];out={}
  for k in range(a//2+1):
   p=list(e);p[0]-=2*k;p[1]+=k;out[tuple(p)]=math.factorial(a)//(2**k*math.factorial(k)*math.factorial(a-2*k))
  return out
 @lru_cache(None)
 def reduce(e):
  if not e[0]:return {e:1}
  p=list(e);p[0]-=1;out={}
  for j,n in enumerate(p[:-1]):
   if n:
    f=p.copy();f[j]-=1;f[j+1]+=1
    for k,v in reduce(tuple(f)).items():out[k]=out.get(k,0)-n*v
  return {k:v for k,v in out.items() if v}
 bases=[[hermite(exp(p)) for p in partitions(m,m)],[hermite(exp(p)) for p in partitions(m-1,m-1)]]
 target={e:s.Rational(c.numerator,c.denominator) for e,c in signed_polynomial(m).items()};keys=set(target);cols=[];layout=[]
 for block,base in enumerate(bases):
  for i,a in enumerate(base):
   for j in range(i,len(base)):
    d={}
    for ea,ca in a.items():
     for eb,cb in base[j].items():
      e=[x+y for x,y in zip(ea,eb)];e[1]+=block
      for k,v in reduce(tuple(e)).items():d[k]=d.get(k,0)+ca*cb*v*(-1 if block else 1)*(1 if i==j else 2)
    d={k:v for k,v in d.items() if v};keys.update(d);cols.append(d);layout.append((block,i,j))
 keys=sorted(keys);return bases,target,keys,cols,layout,list(range(len(keys)))

def probe(m):
 start=time.time();bases,target,keys,cols,layout,rows=setup(m);norm=math.factorial(m-1);scales=[]
 for base in bases:
  ss=[]
  for poly in base:
   e=max(poly,key=lambda e:e[0]);moment=math.factorial(e[0])*math.prod(math.factorial(j)**n for j,n in enumerate(e) if j>0);ss.append(max(1,math.ceil(math.sqrt(moment))))
  scales.append(ss)
 G=np.array([[float(d.get(k,0))/(scales[bl][i]*scales[bl][j]) for d,(bl,i,j) in zip(cols,layout)] for k in keys]);b=np.array([float(target.get(k,0))/norm for k in keys]);rn=np.maximum(np.linalg.norm(G,axis=1),1e-20);G/=rn[:,None];b/=rn
 orth,tri=la.qr(G.T,mode='economic');condition=float(np.linalg.cond(tri));b=la.solve_triangular(tri.T,b,lower=True);G=orth.T
 print('Hermite setup',m,len(keys),len(cols),'condition',condition,'seconds',time.time()-start,flush=True)
 Qs=[cp.Variable((len(base),len(base)),symmetric=True) for base in bases];vec=cp.hstack([Qs[k][i,j] for k,i,j in layout]);margin=cp.Variable()
 prob=cp.Problem(cp.Maximize(margin),[G@vec==b]+[Q-margin*np.eye(len(base))>>0 for Q,base in zip(Qs,bases)])
 try:prob.solve(solver='CLARABEL',tol_gap_abs=1e-11,tol_feas=1e-11,tol_gap_rel=1e-11,max_iter=300)
 except Exception as e:return {'order':m,'status':'error','error':str(e),'proved':False,'seconds':time.time()-start}
 r={'order':m,'status':prob.status,'proved':False,'basis_type':'hermite_score','normalization':norm,'basis_scales':scales,'condition':condition,'seconds':time.time()-start}
 if all(Q.value is not None for Q in Qs):r.update(margin=float(margin.value),gram_matrices=[(Q.value*norm/np.outer(ss,ss)).tolist() for Q,ss in zip(Qs,scales)],min_scaled_eigenvalues=[float(np.linalg.eigvalsh(Q.value)[0]) for Q in Qs],residual=float(np.max(abs(G@vec.value-b))))
 return r
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--orders',type=int,nargs='+',default=[12]);a=p.parse_args()
 for m in a.orders:
  r=probe(m);(ROOT/f'certs/sos_hermite_m{m}.json').write_text(json.dumps(r,indent=2)+'\n');print({k:v for k,v in r.items() if k not in ['gram_matrices','basis_scales']},flush=True)
