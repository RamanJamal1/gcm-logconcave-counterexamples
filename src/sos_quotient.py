"""SDP in the exact integration-by-parts quotient eliminating the score u1.

This numerical locator does not certify anything by itself.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,json,math,time
from functools import lru_cache
from pathlib import Path
import numpy as np
import sympy as s
import cvxpy as cp
from derive_integrands import derive,u
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
 bases=[[exp(p) for p in partitions(m,m)],[exp(p) for p in partitions(m-1,m-1)]]
 target={e:s.Rational(c.numerator,c.denominator) for e,c in signed_polynomial(m).items()}
 keys=set(target);columns=[];layout=[]
 for block,base in enumerate(bases):
  for i,a in enumerate(base):
   for j in range(i,len(base)):
    b=base[j];e=[x+y for x,y in zip(a,b)];e[1]+=block
    d={k:v*(-1 if block else 1)*(1 if i==j else 2) for k,v in reduce(tuple(e)).items()}
    keys.update(d);columns.append(d);layout.append((block,i,j))
 keys=sorted(keys);G=s.polys.matrices.DomainMatrix.from_Matrix(s.Matrix([[d.get(k,0) for d in columns] for k in keys])).to_Matrix()
 # Choose independent equality rows exactly, checking target compatibility later.
 _,pivot=(G.T).rref();rows=list(pivot)
 return bases,target,keys,columns,layout,rows

def probe(m,solver):
 start=time.time();bases,target,keys,cols,layout,rows=setup(m);print('setup',m,len(keys),len(rows),len(cols),time.time()-start,flush=True)
 norm=math.factorial(m-1)
 scales=[]
 for base in bases:
  ss=[]
  for e in base:
   moment=math.prod(range(1,2*e[0],2))*math.prod(math.factorial(j)**n for j,n in enumerate(e) if j>0)
   ss.append(max(1,math.ceil(math.sqrt(moment))))
  scales.append(ss)
 G=np.array([[float(cols[c].get(keys[row],0))/(scales[layout[c][0]][layout[c][1]]*scales[layout[c][0]][layout[c][2]]) for c in range(len(cols))] for row in rows])
 b=np.array([float(target.get(keys[r],0))/norm for r in rows]);rownorm=np.maximum(np.linalg.norm(G,axis=1),1e-20);G/=rownorm[:,None];b/=rownorm
 Qs=[cp.Variable((len(b),len(b)),symmetric=True) for b in bases];vec=cp.hstack([Qs[k][i,j] for k,i,j in layout]);margin=cp.Variable()
 constraints=[G@vec==b]+[Q-margin*np.eye(len(base))>>0 for Q,base in zip(Qs,bases)]+[sum(cp.trace(Q) for Q in Qs)<=1e4]
 prob=cp.Problem(cp.Maximize(margin),constraints)
 try:
  if solver=='SDPA':prob.solve(solver=solver,epsilonStar=1e-25,epsilonDash=1e-25,mpfPrecision=256,maxIteration=300,numThreads=1,print='display')
  else:prob.solve(solver=solver,tol_gap_abs=1e-12,tol_feas=1e-12,tol_gap_rel=1e-12,max_iter=500)
 except Exception as e:return {'order':m,'status':'error','error':str(e),'proved':False,'seconds':time.time()-start}
 result={'order':m,'status':prob.status,'proved':False,'method':'exact IBP quotient; numerical SDP only','seconds':time.time()-start,'solver':solver,'normalization':norm,'basis_scales':scales}
 if all(Q.value is not None for Q in Qs):
  result.update(margin=float(margin.value),gram_matrices=[(Q.value*norm/np.outer(ss,ss)).tolist() for Q,ss in zip(Qs,scales)],min_scaled_eigenvalues=[float(np.linalg.eigvalsh(Q.value)[0]) for Q in Qs],residual=float(np.max(abs(G@vec.value-b))))
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--orders',type=int,nargs='+',default=[9]);p.add_argument('--solver',default='SDPA');a=p.parse_args()
 for m in a.orders:
  r=probe(m,a.solver);(ROOT/f'certs/sos_quotient_m{m}_{a.solver}.json').write_text(json.dumps(r,indent=2)+'\n');print({k:v for k,v in r.items() if k not in ['gram_matrices','basis_scales']},flush=True)
