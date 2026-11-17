"""Semismooth Newton locator for a positive Gram representation.

Dual of minimum-Frobenius-norm PSD feasibility. All results remain numerical
until rational reconstruction and independent exact verification.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,json,time,math
from pathlib import Path
import numpy as np,scipy.linalg as la
ROOT=Path(__file__).resolve().parents[1]

def solve(m,seconds,delta):
 start=time.time();d=np.load(ROOT/f'checkpoints/native_sos_m{m}.npz');A=d['A'];dims=list(map(int,d['dims']));scales=[d['scales0'].tolist(),d['scales1'].tolist()];norm=math.factorial(m-1);rank=len(d['b']);identity=np.concatenate([np.eye(n).ravel() for n in dims]);b=d['b']+(1e-5-delta)*(A@identity);AA=A@A.T;cho=la.cho_factor(AA)
 offsets=np.cumsum([0]+[n*n for n in dims]);blocks=[A[:,offsets[j]:offsets[j+1]].reshape(rank,n,n) for j,n in enumerate(dims)]
 def project(y,hessian=False):
  v=A.T@y;xs=[];H=np.zeros((rank,rank)) if hessian else None
  for j,n in enumerate(dims):
   q=v[offsets[j]:offsets[j+1]].reshape(n,n);w,U=la.eigh(q,check_finite=False);positive=np.maximum(w,0);X=(U*positive)@U.T;xs.append(X.ravel())
   if hessian:
    difference=w[:,None]-w[None,:];numerator=positive[:,None]-positive[None,:];omega=np.divide(numerator,difference,out=np.zeros_like(difference),where=abs(difference)>1e-12*max(1,abs(w).max()));equal=abs(difference)<=1e-12*max(1,abs(w).max());omega[equal]=np.broadcast_to((w[:,None]+w[None,:]>0).astype(float),omega.shape)[equal];omega=np.maximum(omega,0)
    transformed=np.matmul(np.matmul(U.T[None,:,:],blocks[j]),U[None,:,:]);W=(transformed*np.sqrt(omega)[None,:,:]).reshape(rank,-1);H+=W@W.T
  x=np.concatenate(xs);return .5*float(x@x)-float(b@y),A@x-b,x,H
 y=la.cho_solve(cho,b);best=-float('inf');result={};iteration=0
 while time.time()-start<seconds and iteration<1000:
  value,g,x,H=project(y,True);residual=float(np.linalg.norm(g,np.inf));candidate=x+A.T@la.cho_solve(cho,-g);qs=[candidate[offsets[j]:offsets[j+1]].reshape(n,n)+delta*np.eye(n) for j,n in enumerate(dims)];eigs=[float(la.eigvalsh(q)[0]) for q in qs];mineig=min(eigs)
  if mineig>best:
   best=mineig;result={'order':m,'proved':False,'status':'numerical_semismooth_candidate','normalization':norm,'basis_scales':scales,'gram_matrices':[(q*norm/np.outer(ss,ss)).tolist() for q,ss in zip(qs,scales)],'min_scaled_eigenvalues':eigs,'residual_before_affine_projection':residual,'residual':float(np.max(abs(A@candidate-b))),'seconds':time.time()-start,'iteration':iteration,'delta':delta};(ROOT/f'certs/sos_semismooth_m{m}.json').write_text(json.dumps(result,indent=2)+'\n')
  if iteration%5==0:print(iteration,'affine_residual',residual,'min_eigenvalue',mineig,'dual_norm',np.linalg.norm(y),'seconds',time.time()-start,flush=True)
  if mineig>delta/4 and result.get('residual',1)<1e-7:break
  regularization=max(1e-10,min(.01,residual*.001));direction=la.solve(H+regularization*np.eye(rank),-g,assume_a='pos',check_finite=False);slope=float(g@direction);alpha=1.
  for _ in range(40):
   trial=y+alpha*direction;fnew,_,_,_=project(trial)
   if np.isfinite(fnew) and fnew<=value+1e-4*alpha*slope:break
   alpha*=.5
  y=trial;iteration+=1
  if not np.all(np.isfinite(y)) or np.linalg.norm(y)>1e16:break
 print({k:v for k,v in result.items() if k not in ['basis_scales','gram_matrices']},flush=True)
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--order',type=int,default=12);p.add_argument('--seconds',type=float,default=1800);p.add_argument('--margin',type=float,default=.001);a=p.parse_args();solve(a.order,a.seconds,a.margin)
