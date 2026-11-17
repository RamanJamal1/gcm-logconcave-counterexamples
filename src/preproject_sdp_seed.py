"""Distribute a numerical affine residual before exact rational repair.

This step is numerical only. The output must pass exact reconstruction and
independent verification before any positivity assertion.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import json,math,argparse
from pathlib import Path
from fractions import Fraction
import numpy as np,scipy.linalg as la
from checkpoint_io import write_json
ROOT=Path(__file__).resolve().parents[1]
def project(m,source,output):
 r=json.loads(Path(source).read_text());d=np.load(ROOT/f'checkpoints/native_sos_m{m}.npz');A=d['A'];b=d['b'];scales=r['basis_scales'];norm=math.factorial(m-1);dims=[len(s) for s in scales];offsets=np.cumsum([0]+[n*n for n in dims]);x=np.concatenate([(np.array([[float(Fraction(v)) for v in row] for row in q])*np.outer(s,s)/norm-1e-5*np.eye(len(s))).ravel() for q,s in zip(r['gram_matrices'],scales)])
 correction=A.T@la.solve(A@A.T,b-A@x,assume_a='pos');x+=correction;qs=[x[offsets[j]:offsets[j+1]].reshape(n,n)+1e-5*np.eye(n) for j,n in enumerate(dims)];r.update(gram_matrices=[(q*norm/np.outer(s,s)).tolist() for q,s in zip(qs,scales)],source=source,proved=False,status='numerically_affine_projected_seed',min_scaled_eigenvalues=[float(la.eigvalsh(q)[0]) for q in qs],residual=float(max(abs(A@x-b))));write_json(output,r);print({k:r[k] for k in ['order','min_scaled_eigenvalues','residual']})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('order',type=int);p.add_argument('source');p.add_argument('output');a=p.parse_args();project(a.order,a.source,a.output)
