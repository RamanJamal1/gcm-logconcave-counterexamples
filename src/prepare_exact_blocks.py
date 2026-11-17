"""Prepare exact row transformations in small, resumable column blocks."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,json,time,math
from pathlib import Path
import numpy as np,scipy.linalg as la
from flint import fmpq
from sos_quotient import setup,ROOT
from affine_pivots import select,ExactPivotSolve

def prepare(m,blocksize=128,sharp=False):
 if sharp:
  from mckean_quotient import setup as selected_setup
 else:selected_setup=setup
 tag="sharp_" if sharp else ""
 start=time.time();bases,target,keys,cols,layout,rows=selected_setup(m);dims=[len(b) for b in bases];norm=math.factorial(m-1);scales=[[max(1,math.ceil(math.sqrt(math.prod(range(1,2*e[0],2))*math.prod(math.factorial(j)**n for j,n in enumerate(e) if j>0)))) for e in base] for base in bases]
 top,bottom,chosen,tail,A,factors=select(keys,cols,layout,scales);print('selected pivots',len(top),len(bottom),time.time()-start,flush=True);solver=ExactPivotSolve(keys,cols,top,bottom,chosen,tail);print('exact small inverse complete',time.time()-start,flush=True)
 cache=ROOT/f'logs/exact_blocks_{tag}m{m}';cache.mkdir(exist_ok=True);rank=len(keys);G=np.empty_like(A)
 for begin in range(0,len(cols),blocksize):
  end=min(len(cols),begin+blocksize);path=cache/f'block_{begin}_{end}.npy'
  if path.exists():block=np.load(path)
  else:
   upper,lower=solver.solve_columns([[fmpq(str(cols[j].get(keys[i],0))) for j in range(begin,end)] for i in top],[[fmpq(str(cols[j].get(keys[i],0))) for j in range(begin,end)] for i in bottom]);block=np.array([[float(upper[i,j]) for j in range(upper.ncols())] for i in range(upper.nrows())]+[[float(lower[i,j]) for j in range(lower.ncols())] for i in range(lower.nrows())]);np.save(path,block)
  G[:,begin:end]=block*factors[begin:end];print('block',begin,end,'seconds',time.time()-start,flush=True)
 bb=np.array([float(v)/norm for v in solver.solve_vector([fmpq(str(target.get(k,0))) for k in keys])]);rn=np.maximum(np.linalg.norm(G,axis=1),1e-100);G/=rn[:,None];bb/=rn;U,T=la.qr(G.T,mode='economic');bb=la.solve_triangular(T.T,bb,lower=True);G=U.T;condition=float(np.linalg.cond(T));print('whitened condition',condition,time.time()-start,flush=True)
 count=sum(n*n for n in dims);offsets=[0,dims[0]**2];native=np.zeros((rank,count));shift=np.zeros(rank)
 for column,(bl,i,j) in enumerate(layout):
  offset=offsets[bl];n=dims[bl]
  if i==j:native[:,offset+i*n+j]=G[:,column];shift+=G[:,column]
  else:native[:,offset+i*n+j]=G[:,column]/2;native[:,offset+j*n+i]=G[:,column]/2
 objective=np.zeros(count)
 for offset,n in zip(offsets,dims):
  for j in range(n):objective[offset+j*n+j]=1
 np.savez(ROOT/f'checkpoints/native_sos_{tag}m{m}.npz',A=native,b=bb-1e-5*shift,c=objective,scales0=scales[0],scales1=scales[1],dims=dims)
 (ROOT/f'certs/exact_block_preparation_{tag}m{m}.json').write_text(json.dumps({'order':m,'condition':condition,'seconds':time.time()-start,'rank':rank,'columns':len(cols),'proved':False,'method':'exact small Schur inverse and blocked rational row transformation; float SDP input only'},indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--order',type=int,default=13);p.add_argument('--blocksize',type=int,default=128);p.add_argument('--sharp',action='store_true');a=p.parse_args();prepare(a.order,a.blocksize,a.sharp)
