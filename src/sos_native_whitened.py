"""Native minimum-trace SDP with row whitening and fixed positive Gram margin."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,json,math,time
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import sdpap
from sos_quotient import setup,ROOT

def probe(m,delta=1e-5):
 start=time.time();bases,target,keys,cols,layout,rows=setup(m);norm=math.factorial(m-1);dims=[len(b) for b in bases]
 scales=[[max(1,math.ceil(math.sqrt(math.prod(range(1,2*e[0],2))*math.prod(math.factorial(j)**n for j,n in enumerate(e) if j>0)))) for e in base] for base in bases]
 G=np.array([[float(col.get(k,0))/(scales[bl][i]*scales[bl][j]) for col,(bl,i,j) in zip(cols,layout)] for k in keys]);b=np.array([float(target.get(k,0))/norm for k in keys]);rn=np.maximum(np.linalg.norm(G,axis=1),1e-20);G/=rn[:,None];b/=rn
 orth,tri=la.qr(G.T,mode='economic');b=la.solve_triangular(tri.T,b,lower=True);G=orth.T
 count=sum(n*n for n in dims);A=np.zeros((len(keys),count));c=np.zeros(count);offsets=[0,dims[0]**2]
 for j,(block,i,k) in enumerate(layout):
  index=offsets[block]+i*dims[block]+k
  if i==k:A[:,index]=G[:,j];b-=delta*G[:,j];c[index]=1
  else:A[:,index]=G[:,j]/2;A[:,offsets[block]+k*dims[block]+i]=G[:,j]/2
 print('whitened native setup',m,dims,A.shape,time.time()-start,flush=True)
 x,y,info,timing,extra=sdpap.solve(sp.csc_matrix(A),b,c,sdpap.SymCone(s=tuple(dims)),sdpap.SymCone(f=len(b)),{'epsilonStar':1e-15,'epsilonDash':1e-15,'mpfPrecision':160,'maxIteration':60,'numThreads':1,'lambdaStar':1.0,'lowerBound':-1e100,'upperBound':1e100,'print':'display'})
 xx=np.asarray(x.toarray()).ravel();grams=[];eigs=[]
 for offset,n,ss in zip(offsets,dims,scales):
  Q=xx[offset:offset+n*n].reshape(n,n)+delta*np.eye(n);eigs.append(float(np.linalg.eigvalsh(Q)[0]));grams.append((Q*norm/np.outer(ss,ss)).tolist())
 return {'order':m,'status':str(info['phasevalue']),'proved':False,'method':'minimum-trace native conic with row whitening','seconds':time.time()-start,'normalization':norm,'basis_scales':scales,'gram_matrices':grams,'margin':delta,'min_scaled_eigenvalues':eigs,'residual':float(np.max(abs(A@xx-b)))}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--orders',type=int,nargs='+',default=[12]);a=p.parse_args()
 for m in a.orders:
  r=probe(m);(ROOT/f'certs/sos_native_whitened_m{m}.json').write_text(json.dumps(r,indent=2)+'\n');print({k:v for k,v in r.items() if k not in ['gram_matrices','basis_scales']},flush=True)
