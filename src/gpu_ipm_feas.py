"""GPU FP64 infeasible-start HKM interior-point feasibility solver on cached native blocks. Numerical only."""
import os,sys,json,math,time,argparse
import numpy as np,torch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
dev="cuda";dt=torch.float64
def run(m,iters,sigma,out,seconds,delta=1e-5):
 d=np.load(ROOT/f"checkpoints/native_sos_m{m}.npz");A=torch.tensor(d["A"],device=dev,dtype=dt);b=torch.tensor(d["b"],device=dev,dtype=dt);dims=list(map(int,d["dims"]));scales=[d["scales0"].tolist(),d["scales1"].tolist()];norm=math.factorial(m-1)
 one=torch.cat([torch.eye(n,device=dev,dtype=dt).reshape(-1) for n in dims]);b=b+(1e-5-delta)*(A@one);K=A.shape[0];off=np.cumsum([0]+[n*n for n in dims]);ntot=sum(dims)
 Ab=[A[:,off[k]:off[k+1]].reshape(K,n,n) for k,n in enumerate(dims)]
 C=[torch.eye(n,device=dev,dtype=dt) for n in dims]
 def AX(Xs):return sum((Ab[k].reshape(K,-1)@Xs[k].reshape(-1)) for k in range(len(dims)))
 def ATy(y):return [(y@Ab[k].reshape(K,-1)).reshape(dims[k],dims[k]) for k in range(len(dims))]
 rn=A.norm(dim=1);xi=max(10,math.sqrt(ntot),float((ntot*(b.abs()/(1+rn)).max())));eta=max(10,math.sqrt(ntot),float(rn.max()),float(ntot**0.5))
 X=[xi*torch.eye(n,device=dev,dtype=dt) for n in dims];S=[eta*torch.eye(n,device=dev,dtype=dt) for n in dims];y=torch.zeros(K,device=dev,dtype=dt)
 def maxstep(Mat,D):
  a=1e30
  for Mk,Dk in zip(Mat,D):
   L=torch.linalg.cholesky(Mk);Li=torch.linalg.solve_triangular(L,torch.eye(Mk.shape[0],device=dev,dtype=dt),upper=False);T=Li@Dk@Li.T;w=torch.linalg.eigvalsh((T+T.T)/2)[0]
   if w<0:a=min(a,float(-1/w))
  return a
 start=time.time();best=1e99
 for it in range(iters):
  Ax=AX(X);Rp=b-Ax;AtY=ATy(y);Rd=[C[k]-S[k]-AtY[k] for k in range(len(dims))]
  mu=float(sum((X[k]*S[k]).sum() for k in range(len(dims)))/ntot);pres=float(Rp.norm()/(1+b.norm()));dres=float(sum(r.norm()**2 for r in Rd)**0.5/(1+math.sqrt(ntot)))
  mins=[float(torch.linalg.eigvalsh(x)[0]) for x in X]
  print(it,"mu %.3e pres %.3e dres %.3e rp_abs %.3e minX %s bty %.4e ny %.3e t %.0f"%(mu,pres,dres,float(Rp.abs().max()),mins,float(b@y),float(y.norm()),time.time()-start),flush=True)
  if float(Rp.abs().max())<1e-9 and min(mins)>0:
   print("feasible found");break
  if time.time()-start>seconds:break
  Si=[torch.cholesky_inverse(torch.linalg.cholesky(s)) for s in S]
  M=torch.zeros(K,K,device=dev,dtype=dt)
  rhs=b-sigma*mu*AX(Si)+AX([X[k]@Rd[k]@Si[k] for k in range(len(dims))])
  for k in range(len(dims)):
   T=torch.matmul(torch.matmul(X[k][None],Ab[k]),Si[k][None]);M+=Ab[k].reshape(K,-1)@T.reshape(K,-1).T
  M=(M+M.T)/2
  try:
   L=torch.linalg.cholesky(M);dy=torch.cholesky_solve(rhs[:,None],L)[:,0]
  except Exception as e:
   print("chol fail, eig fallback");w,U=torch.linalg.eigh(M);dy=U@((U.T@rhs)/w.clamp_min(w.max()*1e-16))
  # one refinement
  dy=dy+torch.linalg.solve(M,rhs-M@dy) if False else dy
  Aty=ATy(dy);dS=[Rd[k]-Aty[k] for k in range(len(dims))];dX=[]
  for k in range(len(dims)):
   t=-X[k]+sigma*mu*Si[k]-X[k]@dS[k]@Si[k];dX.append((t+t.T)/2)
  ap=min(1.0,0.9*maxstep(X,dX));ad=min(1.0,0.9*maxstep(S,dS))
  X=[X[k]+ap*dX[k] for k in range(len(dims))];S=[S[k]+ad*dS[k] for k in range(len(dims))];y=y+ad*dy
  print("   step",ap,ad,flush=True)
 d0=np.load(ROOT/f"checkpoints/native_sos_m{m}.npz");xx=np.concatenate([x.cpu().numpy().ravel() for x in X]);res=float(abs(d0["A"]@xx-d0["b"]).max())
 qs=[(x.cpu().numpy()+1e-5*np.eye(n)) for x,n in zip(X,dims)];qs=[(q+q.T)/2 for q in qs];import scipy.linalg as la
 eigs=[float(la.eigvalsh(q)[0]) for q in qs]
 r={"order":m,"proved":False,"status":"numerical_GPU_IPM_seed","normalization":norm,"basis_scales":scales,"gram_matrices":[(q*norm/np.outer(s_,s_)).tolist() for q,s_ in zip(qs,scales)],"min_scaled_eigenvalues":eigs,"residual":res,"seconds":time.time()-start}
 Path(out).write_text(json.dumps(r));print("eigs",eigs,"res",res)
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--order",type=int,default=14);p.add_argument("--iters",type=int,default=120);p.add_argument("--sigma",type=float,default=0.2);p.add_argument("--out",default="/tmp/ipm_m14.json");p.add_argument("--delta",type=float,default=1e-5);p.add_argument("--seconds",type=float,default=1500);a=p.parse_args();run(a.order,a.iters,a.sigma,a.out,a.seconds,a.delta)
