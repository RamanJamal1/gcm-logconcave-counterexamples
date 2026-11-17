"""Solve a prepared exact-row-transformed SDP; outputs are numerical only."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,time,json,math
from pathlib import Path
import numpy as np,scipy.linalg as la,cvxpy as cp
ROOT=Path(__file__).resolve().parents[1]
def solve(m,sharp=False,delta=1e-5):
 start=time.time();tag='sharp_' if sharp else '';path=ROOT/f'checkpoints/native_sos_{tag}m{m}.npz'
 while not path.exists():time.sleep(5)
 d=np.load(path);A=d['A'];dims=list(map(int,d['dims']));scales=[d['scales0'].tolist(),d['scales1'].tolist()];identity=np.concatenate([np.eye(n).ravel() for n in dims]);b=d['b']+1e-5*(A@identity);norm=math.factorial(m-1)
 Qs=[cp.Variable((n,n),symmetric=True) for n in dims];vec=cp.hstack([cp.vec(Q,order='C') for Q in Qs]);problem=cp.Problem(cp.Minimize(sum(cp.trace(Q) for Q in Qs)),[A@vec==b]+[Q-delta*np.eye(n)>>0 for Q,n in zip(Qs,dims)])
 result={'order':m,'sharp':sharp,'proved':False,'normalization':norm,'basis_scales':scales,'margin_constraint':delta}
 try:
  problem.solve(solver='CLARABEL',tol_gap_abs=1e-10,tol_gap_rel=1e-10,tol_feas=1e-10,max_iter=250,static_regularization_constant=1e-12,dynamic_regularization_delta=1e-10,iterative_refinement_max_iter=30,verbose=True)
  result['status']=problem.status
  if all(Q.value is not None for Q in Qs):result.update(gram_matrices=[(Q.value*norm/np.outer(ss,ss)).tolist() for Q,ss in zip(Qs,scales)],min_scaled_eigenvalues=[float(la.eigvalsh(Q.value)[0]) for Q in Qs],residual=float(np.max(abs(A@vec.value-b))))
 except Exception as e:result.update(status='solver_error',error=repr(e))
 result['seconds']=time.time()-start;(ROOT/f'certs/sos_cached_clarabel_{tag}m{m}.json').write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items() if k not in ['gram_matrices','basis_scales']},flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--order',type=int,default=13);p.add_argument('--sharp',action='store_true');p.add_argument('--margin',type=float,default=1e-5);a=p.parse_args();solve(a.order,a.sharp,a.margin)
