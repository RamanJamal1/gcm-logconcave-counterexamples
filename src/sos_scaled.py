"""SDP with IBP multipliers eliminated numerically; exact verification is separate."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,json,time,math
from pathlib import Path
import numpy as np
import scipy.linalg as la
from scipy.sparse import coo_matrix
import cvxpy as cp
import sympy as s
from derive_integrands import derive,u
from sos_probe import partitions
ROOT=Path(__file__).resolve().parents[1]

def probe(m):
 start=time.time();symbols=[u[j] for j in range(1,m+1)];norm=math.factorial(m-1)
 def mon(parts):return s.prod(u[j] for j in parts)
 def coeff(expr):return dict(s.Poly(s.expand(expr),*symbols).terms())
 target=coeff((-1)**m*derive(m)[m]/norm)
 bases=[[mon(p) for p in partitions(m,m)],[mon(p) for p in partitions(m-1,m-1)]]
 ids=[]
 for parts in partitions(2*m-1,m-1):
  p=mon(parts);ids.append(coeff(u[1]*p+sum(s.diff(p,u[j])*u[j+1] for j in range(1,m))))
 keys=set(target)
 for d in ids:keys.update(d)
 columns=[];layout=[]
 for block,(base,weight) in enumerate(zip(bases,[1,-u[2]])):
  for i in range(len(base)):
   for j in range(i,len(base)):
    d=coeff(weight*base[i]*base[j]*(1 if i==j else 2));keys.update(d);columns.append(d);layout.append((block,i,j))
 keys=sorted(keys);rows={k:i for i,k in enumerate(keys)}
 D=np.zeros((len(keys),len(ids)))
 for j,d in enumerate(ids):
  for k,v in d.items():D[rows[k],j]=float(v)
 G=np.zeros((len(keys),len(columns)))
 for j,d in enumerate(columns):
  for k,v in d.items():G[rows[k],j]=float(v)
 b=np.array([float(target.get(k,0)) for k in keys])
 basis_scales=[]
 for base in bases:
  bs=[]
  for term in base:
   powers=s.Poly(term,*symbols).monoms()[0];f=1
   for j,power in enumerate(powers,1):
    f*=math.prod(range(1,2*power,2)) if j==1 else math.factorial(j-1)**power
   bs.append(math.sqrt(f))
  basis_scales.append(np.array(bs))
 column_scales=np.array([basis_scales[k][i]*basis_scales[k][j] for k,i,j in layout]);G=G/column_scales[None,:]
 # Numeric nullspace is used only to locate a candidate. Later exact rref and
 # polynomial comparison against all unprojected constraints are mandatory.
 null=la.null_space(D.T,rcond=1e-12);M=null.T@G;target_projected=null.T@b
 rownorm=np.maximum(np.linalg.norm(M,axis=1),1e-10);M=M/rownorm[:,None];target_projected/=rownorm
 grams=[cp.Variable((len(base),len(base)),symmetric=True) for base in bases]
 vector=cp.hstack([grams[k][i,j] for k,i,j in layout]);margin=cp.Variable()
 constraints=[M@vector==target_projected]+[Q-margin*np.eye(Q.shape[0])>>0 for Q in grams]+[sum(cp.trace(Q) for Q in grams)<=1e5]
 problem=cp.Problem(cp.Maximize(margin),constraints)
 try:problem.solve(solver='CLARABEL',tol_gap_abs=1e-12,tol_feas=1e-12,tol_gap_rel=1e-12,max_iter=500)
 except Exception as e:return {'order':m,'status':'solver_error','error':str(e),'proved':False,'seconds':time.time()-start}
 r={'order':m,'status':problem.status,'proved':False,'solver_normalization_factor':str(norm),'interior_margin':None if margin.value is None else float(margin.value),'gram_sizes':[len(base) for base in bases],'constraints_original':len(keys),'constraints_projected':len(M),'seconds':time.time()-start}
 if all(Q.value is not None for Q in grams):
  v=np.array([grams[k].value[i,j] for k,i,j in layout]);lam=la.lstsq(D,G@v-b)[0]
  r.update(gram_matrices=[(Q.value*norm/np.outer(sc,sc)).tolist() for Q,sc in zip(grams,basis_scales)],ibp_multipliers=(lam*norm).tolist(),min_eigenvalues=[float(la.eigvalsh(Q.value)[0]) for Q in grams],equality_residual=float(np.max(abs(G@v-b-D@lam))))
 return r

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--orders',nargs='+',type=int,default=[9,10]);a=p.parse_args()
 for m in a.orders:
  r=probe(m);(ROOT/f'certs/sos_scaled_m{m}.json').write_text(json.dumps(r,indent=2));print({k:v for k,v in r.items() if k not in ['gram_matrices','ibp_multipliers']},flush=True)
