"""Exploratory SDP for a log-concave integrated SOS certificate.

A solver status is NOT a theorem. Any feasible identity requires exact rational
reconstruction and a rigorous PSD proof before it can establish a sign.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import json,argparse,time
from pathlib import Path
import numpy as np
import sympy as s
import cvxpy as cp
from derive_integrands import derive,u
ROOT=Path(__file__).resolve().parents[1]

def partitions(weight,maxpart):
 if weight==0:yield ();return
 for k in range(min(weight,maxpart),1-1,-1):
  for rest in partitions(weight-k,k):yield (k,)+rest

def probe(m):
 start=time.time();symbols=[u[j] for j in range(1,m+1)]
 def mon(parts):return s.prod(u[j] for j in parts)
 def coeff(expr):return dict(s.Poly(s.expand(expr),*symbols).terms())
 target=coeff((-1)**m*derive(m)[m]);bases=[[mon(p) for p in partitions(m,m)],[mon(p) for p in partitions(m-1,m-1)]]
 ids=[]
 for parts in partitions(2*m-1,m-1):
  p=mon(parts);ids.append(coeff(u[1]*p+sum(s.diff(p,u[j])*u[j+1] for j in range(1,m))))
 allkeys=set(target)
 for a in ids:allkeys.update(a)
 blocks=[]
 for base,weight in zip(bases,[1,-u[2]]):
  block=[]
  for i in range(len(base)):
   for j in range(i,len(base)):
    c=coeff(weight*base[i]*base[j]*(1 if i==j else 2));allkeys.update(c);block.append((i,j,c))
  blocks.append(block)
 keys=sorted(allkeys);row={k:i for i,k in enumerate(keys)}
 D=np.zeros((len(keys),len(ids)))
 for j,d in enumerate(ids):
  for k,v in d.items():D[row[k],j]=float(v)
 lam=cp.Variable(len(ids));grams=[cp.Variable((len(b),len(b)),symmetric=True) for b in bases];rhs=[0]*len(keys)
 for Q,block in zip(grams,blocks):
  for i,j,d in block:
   for k,v in d.items():rhs[row[k]]=rhs[row[k]]+float(v)*Q[i,j]
 b=np.array([float(target.get(k,0)) for k in keys]);constraints=[Q>>0 for Q in grams]+[cp.hstack(rhs)==b+D@lam]
 problem=cp.Problem(cp.Minimize(sum(cp.trace(Q) for Q in grams)),constraints)
 try:problem.solve(solver='CLARABEL',tol_gap_abs=1e-9,tol_feas=1e-9,max_iter=300)
 except Exception as e:return {'order':m,'status':'solver_error','error':str(e),'proved':False}
 r={'order':m,'status':problem.status,'proved':False,'gram_sizes':[len(b) for b in bases],'ibp_identities':len(ids),'constraints':len(keys),'seconds':time.time()-start}
 if all(Q.value is not None for Q in grams):
  r['gram_matrices']=[Q.value.tolist() for Q in grams];r['ibp_multipliers']=lam.value.tolist();r['min_eigenvalues']=[float(np.linalg.eigvalsh(Q.value).min()) for Q in grams];r['equality_residual']=float(np.max(abs(np.array([float(v.value) if hasattr(v,'value') else v for v in rhs])-b-D@lam.value)))
 return r

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--orders',nargs='+',type=int,default=[5,6,7,8]);a=p.parse_args()
 for m in a.orders:
  r=probe(m);(ROOT/f'certs/sos_probe_m{m}.json').write_text(json.dumps(r,indent=2));print({k:v for k,v in r.items() if k not in ['gram_matrices','ibp_multipliers']},flush=True)
