"""Exact rational row elimination before floating SDP conditioning.

This locator never reports a proof; reconstruction and independent verification
are mandatory even if the numerical solver reports positive definiteness.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,json,time,math
import numpy as np,scipy.linalg as la,scipy.sparse as sparse,sympy as s,cvxpy as cp
from sos_quotient import setup,ROOT

def probe(m):
 start=time.time();bases,target,keys,cols,layout,rows=setup(m);norm=math.factorial(m-1);dims=[len(b) for b in bases]
 scales=[[max(1,math.ceil(math.sqrt(math.prod(range(1,2*e[0],2))*math.prod(math.factorial(j)**n for j,n in enumerate(e) if j>0)))) for e in base] for base in bases]
 A=s.Matrix([[d.get(k,0) for d in cols] for k in keys]);b=s.Matrix([target.get(k,0) for k in keys]);factors=np.array([1/(scales[bl][i]*scales[bl][j]) for bl,i,j in layout]);num=np.array(A,float)*factors;rn=np.maximum(np.linalg.norm(num,axis=1),1e-100)
 single={};keyindex={k:i for i,k in enumerate(keys)}
 for j,d in enumerate(cols):
  if len(d)==1:
   row=keyindex[next(iter(d))];strength=abs(num[row,j])
   if row not in single or strength>single[row][0]:single[row]=(strength,j)
 chosen=[single[row][1] for row in sorted(single)];chosen_set=set(chosen);rest=[j for j in range(len(cols)) if j not in chosen_set];remaining=[i for i in range(len(keys)) if i not in single]
 _,_,tail=la.qr((num/rn[:,None])[remaining][:,rest],mode='economic',pivoting=True);permutation=chosen+[rest[int(j)] for j in tail]
 print('begin exact RREF',m,len(keys),len(cols),'singleton pivots',len(chosen),time.time()-start,flush=True)
 R,pivots=A[:,permutation].row_join(b).rref();assert len(cols) not in pivots;rank=len(pivots)
 print('exact RREF complete',rank,time.time()-start,flush=True)
 G=np.array(R[:rank,:-1],float)*factors[permutation];bb=np.array(R[:rank,-1],float).ravel()/norm;rn=np.maximum(np.linalg.norm(G,axis=1),1e-100);G/=rn[:,None];bb/=rn
 U,T=la.qr(G.T,mode='economic');bb=la.solve_triangular(T.T,bb,lower=True);G=U.T;condition=float(np.linalg.cond(T));print('condition after exact row elimination',condition,flush=True)
 Qs=[cp.Variable((n,n),symmetric=True) for n in dims];full=cp.hstack([cp.vec(Q,order='F') for Q in Qs]);offsets=[0,dims[0]**2];indices=[offsets[layout[j][0]]+layout[j][1]+dims[layout[j][0]]*layout[j][2] for j in permutation]
 selector=sparse.csc_matrix((np.ones(len(indices)),(np.arange(len(indices)),indices)),shape=(len(indices),sum(n*n for n in dims)))
 delta=1e-6;problem=cp.Problem(cp.Minimize(sum(cp.trace(Q) for Q in Qs)),[G@selector@full==bb]+[Q-delta*np.eye(n)>>0 for Q,n in zip(Qs,dims)])
 try:problem.solve(solver='CLARABEL',tol_gap_abs=1e-9,tol_feas=1e-10,tol_gap_rel=1e-9,max_iter=300,verbose=True)
 except Exception as e:return {'order':m,'proved':False,'status':'solver_error','error':repr(e),'seconds':time.time()-start,'condition':condition}
 result={'order':m,'status':problem.status,'proved':False,'normalization':norm,'basis_scales':scales,'condition':condition,'seconds':time.time()-start}
 if all(Q.value is not None for Q in Qs):result.update(gram_matrices=[(Q.value*norm/np.outer(ss,ss)).tolist() for Q,ss in zip(Qs,scales)],min_scaled_eigenvalues=[float(la.eigvalsh(Q.value)[0]) for Q in Qs],residual=float(np.max(abs(G@selector@full.value-bb))))
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--order',type=int,default=12);a=p.parse_args();r=probe(a.order);(ROOT/f'certs/sos_exact_preconditioned_m{a.order}.json').write_text(json.dumps(r,indent=2)+'\n');print({k:v for k,v in r.items() if k not in ['gram_matrices','basis_scales']},flush=True)
