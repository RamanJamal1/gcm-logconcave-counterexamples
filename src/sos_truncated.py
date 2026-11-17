"""Explore smaller Gram bases by limiting powers of the first score u1."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,json,time,math
import numpy as np,scipy.linalg as la,scipy.sparse as sp,sdpap
from sos_quotient import setup as full_setup,ROOT

def setup(m,cap=1):
 bases,target,keys,cols,layout,rows=full_setup(m);selected=[[i for i,e in enumerate(base) if e[0]<=cap] for base in bases];maps=[{old:new for new,old in enumerate(ix)} for ix in selected];newcols=[];newlayout=[]
 for col,(block,i,j) in zip(cols,layout):
  if i in maps[block] and j in maps[block]:newcols.append(col);newlayout.append((block,maps[block][i],maps[block][j]))
 keys=sorted(set(target).union(*(set(c) for c in newcols)));return [[base[i] for i in ix] for base,ix in zip(bases,selected)],target,keys,newcols,newlayout,list(range(len(keys)))
def solve(m,cap):
 start=time.time();bases,target,keys,cols,layout,rows=setup(m,cap);norm=math.factorial(m-1);dims=[len(b) for b in bases];scales=[[max(1,math.ceil(math.sqrt(math.prod(range(1,2*e[0],2))*math.prod(math.factorial(j)**n for j,n in enumerate(e) if j>0)))) for e in base] for base in bases];offsets=[2,2+dims[0]**2];count=2+sum(n*n for n in dims);A=np.zeros((len(keys),count));b=np.array([float(target.get(k,0))/norm for k in keys]);index={k:i for i,k in enumerate(keys)}
 for col,(block,i,j) in zip(cols,layout):
  for key,value in col.items():
   r=index[key];v=float(value)/(scales[block][i]*scales[block][j]);n=dims[block];off=offsets[block]
   if i==j:A[r,off+i*n+j]+=v;A[r,0]+=v
   else:A[r,off+i*n+j]+=v/2;A[r,off+j*n+i]+=v/2
 zero=np.linalg.norm(A,axis=1)==0
 if np.any(zero & (b!=0)):
  missing=[list(keys[i]) for i in np.flatnonzero(zero & (b!=0))];r={'order':m,'score_power_cap':cap,'proved':False,'status':'restricted_basis_cannot_represent_target','missing_target_monomials':missing,'seconds':time.time()-start};(ROOT/f'certs/sos_truncated_m{m}_cap{cap}.json').write_text(json.dumps(r,indent=2)+'\n');print(r,flush=True);return
 rownorm=np.maximum(np.linalg.norm(A,axis=1),1e-100);A/=rownorm[:,None];b/=rownorm
 Q,R,piv=la.qr(A.T,mode='economic',pivoting=True);diag=abs(np.diag(R));rank=int(np.sum(diag>diag.max()*1e-12));ix=piv[:rank];originalA=A.copy();originalb=b.copy();A=A[ix];b=b[ix]
 # Add a generous trace cap, then maximize a common positive margin.
 trace=np.zeros(count);trace[0]=sum(dims);trace[1]=1
 for off,n in zip(offsets,dims):
  for j in range(n):trace[off+j*n+j]=1
 A=np.vstack([A,trace/1e5]);b=np.r_[b,1.];objective=np.zeros(count);objective[0]=-1
 print('truncated setup',m,cap,dims,'rows',len(keys),'rank',rank,'seconds',time.time()-start,flush=True)
 x,y,info,timing,extra=sdpap.solve(sp.csc_matrix(A),b,objective,sdpap.SymCone(l=2,s=tuple(dims)),sdpap.SymCone(f=len(b)),{'epsilonStar':1e-16,'epsilonDash':1e-16,'mpfPrecision':128,'maxIteration':100,'numThreads':1,'lambdaStar':1000.,'print':'display'})
 xx=np.asarray(x.toarray()).ravel();margin=xx[0];qs=[xx[off:off+n*n].reshape(n,n)+margin*np.eye(n) for off,n in zip(offsets,dims)];r={'order':m,'score_power_cap':cap,'proved':False,'status':str(info['phasevalue']),'normalization':norm,'basis_scales':scales,'gram_matrices':[(q*norm/np.outer(ss,ss)).tolist() for q,ss in zip(qs,scales)],'margin':float(margin),'min_scaled_eigenvalues':[float(la.eigvalsh(q)[0]) for q in qs],'full_residual':float(np.max(abs(originalA@xx-originalb))),'seconds':time.time()-start,'rank':rank,'all_rows':len(keys)};(ROOT/f'certs/sos_truncated_m{m}_cap{cap}.json').write_text(json.dumps(r,indent=2)+'\n');print({k:v for k,v in r.items() if k not in ['basis_scales','gram_matrices']},flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--orders',type=int,nargs='+',default=[12]);p.add_argument('--cap',type=int,default=1);a=p.parse_args()
 for m in a.orders:solve(m,a.cap)
