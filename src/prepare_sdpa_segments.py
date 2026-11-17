"""Export a cached Gram SDP and a strictly positive numerical warm start.

SDPA dual Y is our Gram matrix: F0=-I, Fi=-Ai, c=-b.
All output here is numerical, never a mathematical certificate.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,json,math,time
from pathlib import Path
import numpy as np,scipy.linalg as la
ROOT=Path(__file__).resolve().parents[1]
def prepare(m):
 d=np.load(ROOT/f'checkpoints/native_sos_m{m}.npz');A=d['A'];b=d['b'];dims=list(map(int,d['dims']));scales=[d['scales0'],d['scales1']];offsets=np.cumsum([0]+[n*n for n in dims]);norm=math.factorial(m-1)
 path=ROOT/f'logs/native_m{m}.dat-s'
 with open(path,'w') as f:
  f.write(f'{len(b)}\n{len(dims)}\n'+ ' '.join(map(str,dims))+'\n'+' '.join(format(-x,'.17g') for x in b)+'\n')
  for block,n in enumerate(dims,1):
   for i in range(n):f.write(f'0 {block} {i+1} {i+1} -1\n')
  for row in range(len(b)):
   for block,n in enumerate(dims):
    q=A[row,offsets[block]:offsets[block+1]].reshape(n,n)
    for i in range(n):
     for j in range(i,n):
      if q[i,j]:f.write(f'{row+1} {block+1} {i+1} {j+1} {-q[i,j]:.17g}\n')
 warm=json.loads((ROOT/f'certs/sos_semismooth_warm_m{m}.json').read_text());qs=[np.array(q)*np.outer(ss,ss)/norm-1e-5*np.eye(n) for q,ss,n in zip(warm['gram_matrices'],scales,dims)];shift=max(1e-4,-min(la.eigvalsh(q)[0] for q in qs)+1e-4);qs=[q+shift*np.eye(n) for q,n in zip(qs,dims)]
 with open(ROOT/f'logs/native_m{m}_initial.ini-s','w') as f:
  f.write(' '+' '.join(['0']*len(b))+'\n')
  for k in (1,2):
   for block,n in enumerate(dims):
    q=np.eye(n) if k==1 else qs[block]
    for i in range(n):
     for j in range(i,n):
      if q[i,j]:f.write(f'{k} {block+1} {i+1} {j+1} {q[i,j]:.17g}\n')
 print('exported',path,'bytes',path.stat().st_size,'warm shift',shift,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--order',type=int,default=12);a=p.parse_args();prepare(a.order)
