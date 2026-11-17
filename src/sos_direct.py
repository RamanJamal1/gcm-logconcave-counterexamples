"""Native conic SDP: avoid redundant free-variable splitting in the solver."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,json,math,time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import sdpap
from sos_quotient import setup,ROOT

def probe(m):
 start=time.time();bases,target,keys,cols,layout,rows=setup(m);norm=math.factorial(m-1);dims=[len(b) for b in bases]
 scales=[]
 for base in bases:
  scales.append([max(1,math.ceil(math.sqrt(math.prod(range(1,2*e[0],2))*math.prod(math.factorial(j)**n for j,n in enumerate(e) if j>0)))) for e in base])
 count=2+sum(n*n for n in dims);A=np.zeros((len(rows)+1,count));b=np.zeros(len(rows)+1);b[-1]=1e4
 offsets=[2,2+dims[0]**2]
 for col,(block,i,j) in zip(cols,layout):
  vals=np.array([float(col.get(keys[r],0))/(scales[block][i]*scales[block][j]) for r in rows])
  a=offsets[block]+i*dims[block]+j
  if i==j:
   A[:-1,a]=vals;A[:-1,0]+=vals;A[-1,a]=1
  else:A[:-1,a]=vals/2;A[:-1,offsets[block]+j*dims[block]+i]=vals/2
 b[:-1]=[float(target.get(keys[r],0))/norm for r in rows]
 A[-1,0]=sum(dims);A[-1,1]=1
 rn=np.maximum(np.linalg.norm(A,axis=1),1e-20);A/=rn[:,None];b/=rn
 c=np.zeros(count);c[0]=-1
 print('native conic setup',m,dims,A.shape,time.time()-start,flush=True)
 x,y,info,timing,extra=sdpap.solve(sp.csc_matrix(A),b,c,sdpap.SymCone(l=2,s=tuple(dims)),sdpap.SymCone(f=len(b)),{'epsilonStar':1e-25,'epsilonDash':1e-25,'mpfPrecision':256,'maxIteration':150,'numThreads':1,'print':'display'})
 xx=np.asarray(x.toarray()).ravel();margin=xx[0];grams=[];eigs=[]
 for offset,n,ss in zip(offsets,dims,scales):
  Q=xx[offset:offset+n*n].reshape(n,n)+margin*np.eye(n);eigs.append(float(np.linalg.eigvalsh(Q)[0]));grams.append((Q*norm/np.outer(ss,ss)).tolist())
 return {'order':m,'status':str(info['phasevalue']),'proved':False,'method':'native conic exact IBP quotient numerical locator','seconds':time.time()-start,'normalization':norm,'basis_scales':scales,'gram_matrices':grams,'margin':float(margin),'min_scaled_eigenvalues':eigs,'residual':float(np.max(abs(A@xx-b)))}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--orders',type=int,nargs='+',default=[9,10]);a=p.parse_args()
 for m in a.orders:
  r=probe(m);(ROOT/f'certs/sos_direct_m{m}.json').write_text(json.dumps(r,indent=2)+'\n');print({k:v for k,v in r.items() if k not in ['gram_matrices','basis_scales']},flush=True)
