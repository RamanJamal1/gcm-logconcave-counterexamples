"""Whiten cached numerical SDP rows without replacing the original system."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,time,json
from pathlib import Path
import numpy as np,scipy.linalg as la
from checkpoint_io import write_json
ROOT=Path(__file__).resolve().parents[1]
def prepare(m):
 start=time.time();d=np.load(ROOT/f'checkpoints/native_sos_m{m}.npz');A=d['A'];b=d['b'];dims=list(map(int,d['dims']));offsets=np.cumsum([0]+[n*n for n in dims]);print('loaded',A.shape,time.time()-start,flush=True)
 Q,R=la.qr(A.T,mode='economic');del A
 b=la.solve_triangular(R.T,b,lower=True);np.savez(ROOT/f'checkpoints/whitened_transform_m{m}.npz',R=R);print('whitened',time.time()-start,flush=True);path=ROOT/f'logs/whitened_m{m}.dat-s';T=Q.T
 with path.open('w') as f:
  f.write(f'{len(b)}\n{len(dims)}\n'+ ' '.join(map(str,dims))+'\n'+' '.join(f'{-x:.17e}' for x in b)+'\n')
  for block,n in enumerate(dims,1):
   for i in range(n):f.write(f'0 {block} {i+1} {i+1} -1\n')
  for row in range(len(b)):
   for block,n in enumerate(dims):
    q=T[row,offsets[block]:offsets[block+1]].reshape(n,n)
    for i in range(n):
     for j in range(i,n):
      v=-(q[i,j]+q[j,i])/2
      if v:f.write(f'{row+1} {block+1} {i+1} {j+1} {v:.17e}\n')
 write_json(ROOT/f'certs/whitened_preparation_m{m}.json',{'order':m,'proved':False,'status':'numerical_row_transform_only','method':'QR of the original transposed coefficient system; original cache retained','dimensions':dims,'rows':len(b),'bytes':path.stat().st_size,'seconds':time.time()-start});print('exported',path.stat().st_size,time.time()-start,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--order',type=int,default=15);a=p.parse_args();prepare(a.order)
