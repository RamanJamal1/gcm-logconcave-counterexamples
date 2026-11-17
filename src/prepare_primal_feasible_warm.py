"""Prepare X=I, x=0 and a positive shifted numerical Gram Y for SDPA.

X=I is primal feasible for F0=-I. This is a numerical solver initialization,
not a certificate of the dual Gram equalities or entropy sign.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,json,math
from pathlib import Path
from fractions import Fraction
import numpy as np,scipy.linalg as la
from checkpoint_io import write_json
ROOT=Path(__file__).resolve().parents[1]
def prepare(m,source=None):
 d=np.load(ROOT/f'checkpoints/native_sos_m{m}.npz');A=d['A'];b=d['b'];dims=list(map(int,d['dims']));scales=[d['scales0'],d['scales1']];offsets=np.cumsum([0]+[n*n for n in dims]);norm=math.factorial(m-1)
 if source:
  r=json.loads(Path(source).read_text());qs=[np.array([[float(Fraction(str(v))) for v in row] for row in q])*np.outer(s,s)/norm-1e-5*np.eye(len(s)) for q,s in zip(r['gram_matrices'],scales)]
 else:
  x=np.concatenate([np.eye(n).ravel() for n in dims]);x+=A.T@la.solve(A@A.T,b-A@x,assume_a='pos');qs=[x[offsets[j]:offsets[j+1]].reshape(n,n) for j,n in enumerate(dims)]
 low=min(la.eigvalsh(q)[0] for q in qs);shift=max(.01,-low+.01);qs=[q+shift*np.eye(n) for q,n in zip(qs,dims)];path=ROOT/f'logs/primal_feasible_m{m}.ini-s'
 with path.open('w') as f:
  f.write(' '+' '.join(['0']*len(b))+'\n')
  for block,n in enumerate(dims):
   for i in range(n):f.write(f'1 {block+1} {i+1} {i+1} 1\n')
   for i in range(n):
    for j in range(i,n):f.write(f'2 {block+1} {i+1} {j+1} {qs[block][i,j]:.17e}\n')
 record={'order':m,'proved':False,'source':source,'shift':shift,'minimum_before_shift':float(low),'dual_residual':float(max(abs(A@np.concatenate([q.ravel() for q in qs])-b))),'primal_feasible_by_construction':True,'warm_file':str(path.relative_to(ROOT))};write_json(ROOT/f'certs/primal_warm_preparation_m{m}.json',record);print(record,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--order',type=int,required=True);p.add_argument('--source');a=p.parse_args();prepare(a.order,a.source)
