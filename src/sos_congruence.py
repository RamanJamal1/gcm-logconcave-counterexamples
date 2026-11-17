"""Numerical SOS locator with Gram congruence preconditioning; never a proof."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,json,time,math
from pathlib import Path
import numpy as np,scipy.linalg as la,cvxpy as cp
from cvxpy.reductions.cvx_attr2constr import recover_value_for_leaf
ROOT=Path(__file__).resolve().parents[1]
def solve(m,seconds):
 start=time.time();d=np.load(ROOT/f'checkpoints/native_sos_m{m}.npz');A=d['A'];dims=list(map(int,d['dims']));scales=[d['scales0'],d['scales1']];norm=math.factorial(m-1);identity=np.concatenate([np.eye(n).ravel() for n in dims]);b=d['b']+1e-5*A@identity;offsets=np.cumsum([0]+[n*n for n in dims]);cho=la.cho_factor(A@A.T)
 source=ROOT/f'certs/sos_semismooth_warm_m{m}.json';warm=json.loads(source.read_text());qs=[np.array(q)*np.outer(ss,ss)/norm for q,ss in zip(warm['gram_matrices'],scales)];best=-float('inf');step=0
 while time.time()-start<seconds:
  floor=[1e-3,1e-5,1e-7,1e-9][step%4];Ws=[];blocks=[]
  for j,(q,n) in enumerate(zip(qs,dims)):
   w,U=la.eigh(q);W=U*np.sqrt(np.maximum(w,floor));Ws.append(W);blocks.append(np.matmul(np.matmul(W.T[None,:,:],A[:,offsets[j]:offsets[j+1]].reshape(len(b),n,n)),W[None,:,:]).reshape(len(b),-1))
  T=np.concatenate(blocks,axis=1);U,R=la.qr(T.T,mode='economic');rhs=la.solve_triangular(R.T,b,lower=True);T=U.T
  vs=[cp.Variable((n,n),symmetric=True) for n in dims];vec=cp.hstack([cp.vec(v,order='C') for v in vs]);problem=cp.Problem(cp.Minimize(sum(cp.trace(v) for v in vs)),[T@vec==rhs]+[v-1e-7*np.eye(n)>>0 for v,n in zip(vs,dims)])
  status='unknown'
  try:
   problem.solve(solver='CLARABEL',tol_feas=1e-10,tol_gap_abs=1e-10,tol_gap_rel=1e-10,max_iter=150,verbose=False);status=problem.status;rs=[v.value for v in vs]
  except Exception as e:
   status=repr(e);raw=problem._solver_cache['CLARABEL'].get_solution();data,_,_=problem.get_problem_data(cp.CLARABEL);mapping=data['param_prob'].var_id_to_col;rs=[recover_value_for_leaf(v,np.asarray(raw.x)[mapping[v.id]:mapping[v.id]+n*(n+1)//2]) for v,n in zip(vs,dims)]
  trial=[W@r@W.T for W,r in zip(Ws,rs)];xx=np.concatenate([q.ravel() for q in trial]);xx+=A.T@la.cho_solve(cho,b-A@xx);trial=[xx[offsets[j]:offsets[j+1]].reshape(n,n) for j,n in enumerate(dims)];eigs=[float(la.eigvalsh(q)[0]) for q in trial];mineig=min(eigs)
  print('step',step,'floor',floor,'minimum',eigs,'status',status,'seconds',time.time()-start,flush=True)
  if mineig>best:
   best=mineig;r={'order':m,'proved':False,'status':'numerical_congruence_candidate','solver_status':status,'normalization':norm,'basis_scales':[s.tolist() for s in scales],'gram_matrices':[(q*norm/np.outer(ss,ss)).tolist() for q,ss in zip(trial,scales)],'min_scaled_eigenvalues':eigs,'residual':float(np.max(abs(A@xx-b))),'seconds':time.time()-start,'iteration':step};(ROOT/f'certs/sos_congruence_m{m}.json').write_text(json.dumps(r,indent=2)+'\n')
  if mineig>1e-8:break
  if np.all(np.isfinite(xx)) and np.linalg.norm(xx)<1e12:qs=trial
  step+=1
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--order',type=int,default=12);p.add_argument('--seconds',type=float,default=600);a=p.parse_args();solve(a.order,a.seconds)
