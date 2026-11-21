"""Max-margin SDP on cached native blocks via cvxpy: max t s.t. A y=b, Y_i >= t I. Numerical only."""
import os,sys,json,math,time,argparse
import numpy as np,cvxpy as cp,scipy.sparse as sp
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main(m,solver,cap,out,maxit):
 d=np.load(ROOT/f"checkpoints/native_sos_m{m}.npz");A=d["A"];b=d["b"];dims=list(map(int,d["dims"]));scales=[d["scales0"].tolist(),d["scales1"].tolist()];norm=math.factorial(m-1)
 # row scaling
 rs=1/np.maximum(np.linalg.norm(A,axis=1),1e-12);A=A*rs[:,None];b=b*rs
 Y=[cp.Variable((n,n),symmetric=True) for n in dims];t=cp.Variable()
 expr=A@cp.hstack([cp.vec(y,order="C") for y in Y])
 cons=[expr==b,t<=cap]+[y-t*np.eye(n)>>0 for y,n in zip(Y,dims)]
 pr=cp.Problem(cp.Maximize(t),cons);s=time.time()
 kw={"max_iter":maxit}
 pr.solve(solver=solver,verbose=True,**kw)
 print("status",pr.status,"t",t.value,time.time()-s,flush=True)
 if Y[0].value is None:return
 xx=np.concatenate([y.value.ravel() for y in Y]);A0=d["A"];res=float(abs(A0@xx-d["b"]).max())
 qs=[y.value+1e-5*np.eye(n) for y,n in zip(Y,dims)];import scipy.linalg as la
 eigs=[float(la.eigvalsh((q+q.T)/2)[0]) for q in qs]
 r={"order":m,"proved":False,"status":"numerical_seed_only","normalization":norm,"basis_scales":scales,"gram_matrices":[((q+q.T)/2*norm/np.outer(s_,s_)).tolist() for q,s_ in zip(qs,scales)],"min_scaled_eigenvalues":eigs,"residual":res,"t":float(t.value),"solver":solver}
 Path(out).write_text(json.dumps(r));print("eigs",eigs,"res",res)
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--order",type=int,default=14);p.add_argument("--solver",default="CLARABEL");p.add_argument("--cap",type=float,default=1.0);p.add_argument("--out",default="/tmp/cvx_m14.json");p.add_argument("--maxit",type=int,default=200);a=p.parse_args();main(a.order,a.solver,a.cap,a.out,a.maxit)
