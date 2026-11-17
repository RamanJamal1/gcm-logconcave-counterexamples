"""Numerical Gram congruence preconditioning; exact reconstruction is separate."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import numpy as np, scipy.linalg as la, cvxpy as cp
import json,math,time,argparse
from sos_quotient import setup,ROOT

def solve(m,floor,source):
 start=time.time();bases,target,keys,cols,layout,rows=setup(m);norm=math.factorial(m-1)
 data=json.loads((ROOT/source).read_text());scales=data['basis_scales'];dims=[len(x) for x in bases]
 transforms=[]
 for q,ss in zip(data['gram_matrices'],scales):
  q=np.asarray(q)*np.outer(ss,ss)/norm;v,U=la.eigh(q);transforms.append((U*np.sqrt(np.maximum(abs(v),floor)))@U.T)
 G=np.array([[float(d.get(k,0))/(scales[bl][i]*scales[bl][j]) for d,(bl,i,j) in zip(cols,layout)] for k in keys]);b=np.array([float(target.get(k,0))/norm for k in keys])
 new=np.zeros_like(G)
 for bl,n in enumerate(dims):
  indices=[a for a,(block,i,j) in enumerate(layout) if block==bl];pairs=np.array([(i,j) for block,i,j in layout if block==bl]).T;i,j=pairs
  mats=np.zeros((len(keys),n,n));mats[:,i,j]=G[:,indices]/np.where(i==j,1,2);mats[:,j,i]=mats[:,i,j]
  R=transforms[bl];mats=np.einsum('ia,kij,jb->kab',R,mats,R,optimize=True);new[:,indices]=mats[:,i,j]*np.where(i==j,1,2)
 rn=np.maximum(np.linalg.norm(new,axis=1),1e-30);new/=rn[:,None];b/=rn
 U,T=la.qr(new.T,mode='economic');b=la.solve_triangular(T.T,b,lower=True);new=U.T
 print('setup',m,'floor',floor,'condition',np.linalg.cond(T),'seconds',time.time()-start,flush=True)
 Qs=[cp.Variable((n,n),symmetric=True) for n in dims];vec=cp.hstack([Qs[bl][i,j] for bl,i,j in layout]);margin=cp.Variable()
 problem=cp.Problem(cp.Maximize(margin),[new@vec==b]+[Q-margin*np.eye(n)>>0 for Q,n in zip(Qs,dims)]+[sum(cp.trace(Q) for Q in Qs)<=100000])
 try:problem.solve(solver='CLARABEL',max_iter=400,tol_gap_abs=1e-10,tol_feas=1e-10,tol_gap_rel=1e-10)
 except Exception as e:return {'order':m,'status':'error','error':repr(e),'proved':False,'seconds':time.time()-start}
 result={'order':m,'status':problem.status,'proved':False,'seconds':time.time()-start,'floor':floor,'basis_scales':scales,'normalization':norm}
 if all(q.value is not None for q in Qs):
  matrices=[R@Q.value@R.T for R,Q in zip(transforms,Qs)]
  result.update(margin=float(margin.value),gram_matrices=[(q*norm/np.outer(ss,ss)).tolist() for q,ss in zip(matrices,scales)],min_scaled_eigenvalues=[float(la.eigvalsh(q)[0]) for q in matrices],residual=float(np.max(abs(new@vec.value-b))))
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--order',type=int,default=12);p.add_argument('--floor',type=float,default=.001);p.add_argument('--source',default='certs/sos_warm_projection_m12.json');a=p.parse_args();r=solve(a.order,a.floor,a.source);(ROOT/f'certs/sos_preconditioned_m{a.order}_{a.floor}.json').write_text(json.dumps(r,indent=2)+'\n');print({k:v for k,v in r.items() if k not in ['basis_scales','gram_matrices']},flush=True)
