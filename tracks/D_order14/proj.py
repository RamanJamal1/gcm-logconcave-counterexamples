"""Numerical affine projection of a Gram seed onto the exact constraint space (as src/preproject_sdp_seed.py),
using the exactly row-transformed system checkpoints/native_sos_m{m}.npz. Numerical only (proved=False).
Usage: proj.py m seed.json out.json"""
import sys,json,math,numpy as np,scipy.linalg as la,os
ROOT=os.path.expanduser('~/gcm-logconcave-counterexamples');m=int(sys.argv[1]);r=json.load(open(sys.argv[2]))
d=np.load(f'{ROOT}/checkpoints/native_sos_m{m}.npz');A=d['A'];b=d['b'];sc=r['basis_scales'];norm=math.factorial(m-1);dims=[len(s) for s in sc]
for k,s in enumerate(sc):
    if list(map(int,d[f'scales{k}']))!=list(map(int,s)):raise RuntimeError('basis scales differ from native npz')
x=np.concatenate([(np.array(q,dtype=float)*np.outer(s,s)/norm-1e-5*np.eye(len(s))).ravel() for q,s in zip(r['gram_matrices'],sc)])
before=float(np.max(np.abs(A@x-b)));x+=A.T@la.solve(A@A.T,b-A@x,assume_a='pos');after=float(np.max(np.abs(A@x-b)))
o=0;qs=[];eig=[]
for n,s in zip(dims,sc):
    Q=x[o:o+n*n].reshape(n,n);Q=(Q+Q.T)/2+1e-5*np.eye(n);o+=n*n;eig.append(float(la.eigvalsh(Q)[0]));qs.append((Q*norm/np.outer(s,s)).tolist())
r.update(gram_matrices=qs,min_scaled_eigenvalues=eig,residual=after,residual_before_projection=before,status='numerically_affine_projected_seed',proved=False,source=sys.argv[2])
json.dump(r,open(sys.argv[3],'w'));print(json.dumps({'before':before,'after':after,'min_eigs':eig}))
