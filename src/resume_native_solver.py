"""Reuse the numerically preconditioned conic matrix, avoiding exact RREF again."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='2'
import argparse,time,json,math
import numpy as np,scipy.sparse as sp,scipy.linalg as la,sdpap
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def solve(m,bits,threads):
 start=time.time();data=np.load(ROOT/f'checkpoints/native_sos_m{m}.npz');A=data['A'];b=data['b'];c=data['c'];dims=list(map(int,data['dims']));scales=[data['scales0'].tolist(),data['scales1'].tolist()];norm=math.factorial(m-1)
 print('cached native solve',m,bits,threads,A.shape,flush=True)
 x,y,info,timing,extra=sdpap.solve(sp.csc_matrix(A),b,c,sdpap.SymCone(s=tuple(dims)),sdpap.SymCone(f=len(b)),{'epsilonStar':1e-18,'epsilonDash':1e-18,'mpfPrecision':bits,'maxIteration':150,'numThreads':threads,'lambdaStar':1000.0,'upperBound':1e12,'lowerBound':-1e12,'print':'display'})
 xx=np.asarray(x.toarray()).ravel();grams=[];eigs=[];offset=0
 for n,ss in zip(dims,scales):
  Q=xx[offset:offset+n*n].reshape(n,n)+1e-5*np.eye(n);offset+=n*n;grams.append((Q*norm/np.outer(ss,ss)).tolist());eigs.append(float(la.eigvalsh(Q)[0]))
 r={'order':m,'status':str(info['phasevalue']),'proved':False,'normalization':norm,'basis_scales':scales,'seconds':time.time()-start,'gram_matrices':grams,'min_scaled_eigenvalues':eigs,'residual':float(np.max(abs(A@xx-b))),'solver_bits':bits,'solver_threads':threads};(ROOT/f'certs/sos_cached_native_m{m}_{bits}.json').write_text(json.dumps(r,indent=2)+'\n');print({k:v for k,v in r.items() if k not in ['gram_matrices','basis_scales']},flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--order',type=int,default=12);p.add_argument('--bits',type=int,default=128);p.add_argument('--threads',type=int,default=2);a=p.parse_args();solve(a.order,a.bits,a.threads)
