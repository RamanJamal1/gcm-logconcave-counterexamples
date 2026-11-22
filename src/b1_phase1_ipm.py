"""Phase-1 IPM: min t s.t. A x + t p = b_delta, X>=0, t>=0 ; dual y gives Farkas ray. Numerical only."""
import sys,json,time,math,numpy as np,torch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];dev="cuda";dt=torch.float64
def run(delta,iters,seconds,sigma=0.3,out=None):
 d=np.load(ROOT/"checkpoints/native_sos_m14.npz");A=torch.tensor(d["A"],device=dev,dtype=dt);b=torch.tensor(d["b"],device=dev,dtype=dt);dims=list(map(int,d["dims"]));K=A.shape[0]
 one=torch.cat([torch.eye(n,device=dev,dtype=dt).reshape(-1) for n in dims]);b=b+(1e-5-delta)*(A@one)
 p=b-A@one;ps=float(p.norm());print("|b|",float(b.norm()),"|p|",ps,flush=True)
 off=np.cumsum([0]+[n*n for n in dims]);Ab=[A[:,off[k]:off[k+1]].reshape(K,n,n) for k,n in enumerate(dims)]
 nb=len(dims);ntot=sum(dims)+1
 X=[torch.eye(n,device=dev,dtype=dt) for n in dims];Xt=torch.tensor(1.0,device=dev,dtype=dt)
 eta=10.0;S=[eta*torch.eye(n,device=dev,dtype=dt) for n in dims];St=torch.tensor(eta,device=dev,dtype=dt);y=torch.zeros(K,device=dev,dtype=dt)
 def AX(Xs,t):return sum(Ab[k].reshape(K,-1)@Xs[k].reshape(-1) for k in range(nb))+p*t
 def ATy(v):return [(v@Ab[k].reshape(K,-1)).reshape(dims[k],dims[k]) for k in range(nb)]
 def maxstep(Mat,D):
  a=1e30
  for Mk,Dk in zip(Mat,D):
   L=torch.linalg.cholesky(Mk);Li=torch.linalg.solve_triangular(L,torch.eye(Mk.shape[0],device=dev,dtype=dt),upper=False);T=Li@Dk@Li.T;w=torch.linalg.eigvalsh((T+T.T)/2)[0]
   if w<0:a=min(a,float(-1/w))
  return a
 t0=time.time();res=None
 for it in range(iters):
  Rp=b-AX(X,Xt);Aty=ATy(y);Rd=[-S[k]-Aty[k] for k in range(nb)];Rdt=1.0-St-float(p@y)
  mu=float((sum((X[k]*S[k]).sum() for k in range(nb))+Xt*St)/ntot);pres=float(Rp.norm()/(1+b.norm()));dres=float((sum(r.norm()**2 for r in Rd)+Rdt**2)**0.5)
  pobj=float(Xt);dobj=float(b@y)
  print(it,"mu %.2e pres %.2e dres %.2e t %.6e bty %.6e |y| %.3e el %.0f"%(mu,pres,dres,pobj,dobj,float(y.norm()),time.time()-t0),flush=True)
  if mu<1e-13 or time.time()-t0>seconds:break
  Si=[torch.cholesky_inverse(torch.linalg.cholesky(s)) for s in S];Sit=1/St
  M=torch.zeros(K,K,device=dev,dtype=dt)
  rhs=b-sigma*mu*(sum(Ab[k].reshape(K,-1)@Si[k].reshape(-1) for k in range(nb))+p*Sit)+sum(Ab[k].reshape(K,-1)@(X[k]@Rd[k]@Si[k]).reshape(-1) for k in range(nb))+p*(Xt*Rdt*Sit)-AX(X,Xt)
  for k in range(nb):
   T=torch.matmul(torch.matmul(X[k][None],Ab[k]),Si[k][None]);M+=Ab[k].reshape(K,-1)@T.reshape(K,-1).T
  M=(M+M.T)/2+torch.outer(p,p)*(Xt*Sit)
  # rhs derived: A dX + p dt = Rp, dX=-X+s mu Si - X dS Si, dS=Rd-A^T dy  => M dy = Rp +A(-X+s mu Si - X Rd Si)*(-1)... computed below directly
  rhs=Rp-(-AX(X,Xt)*0)  # placeholder overwritten
  base=[-X[k]+sigma*mu*Si[k]-X[k]@Rd[k]@Si[k] for k in range(nb)];bt=-Xt+sigma*mu*Sit-Xt*Rdt*Sit
  rhs=-(Rp-AX(base,bt)+p*0)*(-1)  # = A(base)+p bt - Rp  with sign fixed below
  rhs=AX(base,bt)-Rp
  # A(dX)+p dt = Rp, dX=base + X Aty(dy) Si ; => M dy = Rp - A(base)-p bt  ... with sign: dX = base + X(A^T dy)Si
  rhs=Rp-AX(base,bt)
  try:
   L=torch.linalg.cholesky(M);dy=torch.cholesky_solve(rhs[:,None],L)[:,0]
  except Exception:
   w,U=torch.linalg.eigh(M);dy=U@((U.T@rhs)/w.clamp_min(w.max()*1e-16))
  Ady=ATy(dy);dS=[Rd[k]-Ady[k] for k in range(nb)];dSt=Rdt-float(p@dy)
  dX=[];
  for k in range(nb):
   t=base[k]+X[k]@Ady[k]@Si[k];dX.append((t+t.T)/2)
  dXt=bt+Xt*float(p@dy)*Sit
  def ms(Mat,D,sc,ds):
   a=maxstep(Mat,D);
   if ds<0:a=min(a,float(-sc/ds))
   return a
  ap=min(1.0,0.9*ms(X,dX,Xt,dXt));ad=min(1.0,0.9*ms(S,dS,St,dSt))
  X=[X[k]+ap*dX[k] for k in range(nb)];Xt=Xt+ap*dXt;S=[S[k]+ad*dS[k] for k in range(nb)];St=St+ad*dSt;y=y+ad*dy
  print("   step %.3f %.3f"%(ap,ad),flush=True)
 np.save(f"/tmp/b1_phase1_y_{delta}.npy",y.cpu().numpy())
 Sy=ATy(y);lm=[float(torch.linalg.eigvalsh((s+s.T)/2)[-1]) for s in Sy];tr=float(sum(torch.trace(s) for s in Sy))
 print("FINAL delta",delta,"t",float(Xt),"b.y",float(b@y),"p.y",float(p@y),"lmaxS",lm,"trS",tr,"|y|",float(y.norm()))
if __name__=="__main__":run(float(sys.argv[1]),int(sys.argv[2]),float(sys.argv[3]))
