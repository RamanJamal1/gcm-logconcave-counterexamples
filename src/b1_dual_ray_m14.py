"""Numerical Farkas-ray search for the order-14 SOS Gram system (min ||Ax-b|| over PSD x, FISTA). Numerical only."""
import sys,json,time,math,numpy as np,torch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];dev="cuda";dt=torch.float64
def load(m=14):
 d=np.load(ROOT/f"checkpoints/native_sos_m{m}.npz");A=torch.tensor(d["A"],device=dev,dtype=dt);b=torch.tensor(d["b"],device=dev,dtype=dt);dims=list(map(int,d["dims"]));return A,b,dims
def unpack(v,dims):
 out=[];o=0
 for n in dims:out.append(v[o:o+n*n].reshape(n,n));o+=n*n
 return out
def proj(v,dims,cap=None):
 out=[]
 for X in unpack(v,dims):
  X=(X+X.T)/2;w,U=torch.linalg.eigh(X);out.append(((U*w.clamp_min(0))@U.T).reshape(-1))
 return torch.cat(out)
def lam_max_blocks(S,dims):
 return [float(torch.linalg.eigvalsh((x+x.T)/2)[-1]) for x in unpack(S,dims)]
def fista(A,b,dims,iters,delta,tl=80):
 one=torch.cat([torch.eye(n,device=dev,dtype=dt).reshape(-1) for n in dims]);bd=b+(1e-5-delta)*(A@one)
 x=torch.zeros(A.shape[1],device=dev,dtype=dt);z=x.clone();t=1.0;t0=time.time();hist=[]
 for k in range(iters):
  r=A@z-bd;xn=proj(z-A.T@r,dims);fn=float((A@xn-bd).norm())
  if k>0 and fn>fprev:t=1.0;z=xn.clone();x=xn;fprev=fn;continue
  fprev=fn;tn=(1+math.sqrt(1+4*t*t))/2;z=xn+((t-1)/tn)*(xn-x);x=xn;t=tn
  if k%2000==0 or time.time()-t0>tl:
   rr=A@x-bd;hist.append((k,float(rr.norm())));print(k,"|r|",float(rr.norm()),"trX",float(sum(torch.trace(q) for q in unpack(x,dims))),flush=True)
  if time.time()-t0>tl:break
 return x,bd,hist
if __name__=="__main__":
 delta=float(sys.argv[1]);iters=int(sys.argv[2]);tl=float(sys.argv[3])
 A,b,dims=load();x,bd,h=fista(A,b,dims,iters,delta,tl)
 r=A@x-bd;y=-r/r.norm();S=-(A.T@y)  # S(y)=mat(A^T y)= -A^T r/|r|... sign: y=-r/|r| so A^T y=-A^T r/|r|
 S=A.T@y;lm=lam_max_blocks(S,dims);val=float(bd@y);ts=float(S@torch.cat([torch.eye(n,device=dev,dtype=dt).reshape(-1) for n in dims]))
 print("delta",delta,"|r|",float(r.norm()),"b.y",val,"lmax",lm,"trS",ts)
 np.save(f"/tmp/b1_y_{delta}_{sys.argv[3]}.npy",y.cpu().numpy())
