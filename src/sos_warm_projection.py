"""Douglas-Rachford feasibility search between the exact-identity affine space
and positive Gram matrices. Floating feasibility is only a reconstruction hint.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,json,math,time
import numpy as np
import scipy.linalg as la
from sos_quotient import setup,ROOT

def search(m,seconds,delta):
 start=time.time();bases,target,keys,cols,layout,rows=setup(m);norm=math.factorial(m-1);dims=[len(b) for b in bases]
 scales=[[max(1,math.ceil(math.sqrt(math.prod(range(1,2*e[0],2))*math.prod(math.factorial(j)**n for j,n in enumerate(e) if j>0)))) for e in base] for base in bases]
 weights=np.array([1 if i==j else math.sqrt(2) for bl,i,j in layout])
 G=np.array([[float(d.get(k,0))/(scales[bl][i]*scales[bl][j]*w) for d,(bl,i,j),w in zip(cols,layout,weights)] for k in keys]);b=np.array([float(target.get(k,0))/norm for k in keys]);rn=np.maximum(np.linalg.norm(G,axis=1),1e-20);G/=rn[:,None];b/=rn
 orth,tri=la.qr(G.T,mode='economic');b=la.solve_triangular(tri.T,b,lower=True);M=orth.T.copy();MT=M.T.copy()
 ix=[[j for j,(bl,i,k) in enumerate(layout) if bl==block] for block in range(2)]
 pairs=[np.array([(i,k) for bl,i,k in layout if bl==block]).T for block in range(2)]
 def unpack(x):
  out=[]
  for n,indices,(i,j) in zip(dims,ix,pairs):
   q=np.zeros((n,n));q[i,j]=x[indices]/weights[indices];q[j,i]=q[i,j];out.append(q)
  return out
 def pack(qs):return np.concatenate([q[i,j]*weights[indices] for q,indices,(i,j) in zip(qs,ix,pairs)])
 def affine(x):return x+MT@(b-M@x)
 def psd(x):
  out=[]
  for q in unpack(x):
   w,v=la.eigh(q,check_finite=False);out.append((v*np.maximum(w,delta))@v.T)
  return pack(out)
 x=MT@b
 previous_path=ROOT/f'certs/sos_uncapped_m{m}_CLARABEL.json'
 if previous_path.exists():
  warm=json.loads(previous_path.read_text());x=pack([np.array(q)*np.outer(ss,ss)/norm for q,ss in zip(warm['gram_matrices'],scales)])
 best=-float('inf');iteration=0;history=[];print('projection setup',m,time.time()-start,flush=True)
 while time.time()-start<seconds:
  y=psd(x);z=affine(2*y-x);f=1.5*(z-y);g=x+f;history.append((x.copy(),f.copy()))
  history=history[-7:]
  if len(history)>2 and iteration%5==0:
   dx=np.column_stack([history[j+1][0]-history[j][0] for j in range(len(history)-1)])
   df=np.column_stack([history[j+1][1]-history[j][1] for j in range(len(history)-1)])
   gamma=la.lstsq(df,f,cond=1e-10,check_finite=False)[0];proposal=g-(dx+df)@gamma
   py=psd(proposal);pf=1.5*(affine(2*py-proposal)-py)
   if np.linalg.norm(pf)<np.linalg.norm(f):g=proposal
   else:history=history[-2:]
  x=g;iteration+=1
  if iteration%200==0:
   candidate=affine(y);qs=unpack(candidate);eigs=[float(la.eigvalsh(q,check_finite=False)[0]) for q in qs];value=min(eigs);res=float(np.max(abs(M@candidate-b)))
   if value>best:
    best=value;r={'order':m,'status':'numerical_projection_candidate','proved':False,'normalization':norm,'basis_scales':scales,'gram_matrices':[(q*norm/np.outer(ss,ss)).tolist() for q,ss in zip(qs,scales)],'min_scaled_eigenvalues':eigs,'residual':res,'iterations':iteration,'seconds':time.time()-start}
    (ROOT/f'certs/sos_warm_projection_m{m}.json').write_text(json.dumps(r,indent=2)+'\n')
   print(iteration,'min_eigenvalue',value,'residual',res,'seconds',time.time()-start,flush=True)
   if value>delta/4 and res<1e-8:return r
 return r
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--order',type=int,default=12);p.add_argument('--seconds',type=float,default=1800);p.add_argument('--margin',type=float,default=1e-4);a=p.parse_args();search(a.order,a.seconds,a.margin)
