"""SDPA-GMP solve of the order-m sparse SOS feasibility SDP (min trace, margin 1e-5 built into b).
Writes a numerical seed in the repo's Gram-JSON format (proved=False). Usage: gmp_solve.py m bits threads maxit"""
import os,sys,time,math,json
m,bits,threads,maxit=map(int,sys.argv[1:5])
os.environ['OMP_NUM_THREADS']=str(threads);os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np,scipy.sparse as sp,scipy.linalg as la,sdpap
ROOT=os.path.expanduser('~/gcm-logconcave-counterexamples');D=ROOT+'/tracks/D_order14/data'
A=sp.load_npz(f'{D}/A_m{m}.npz').tocsc();d=np.load(f'{D}/sdp_m{m}.npz');b=d['b'];c=d['c'];dims=[int(v) for v in d['dims']];scales=[d['scales0'].tolist(),d['scales1'].tolist()];norm=int(d['norm'])
tag=f'm{m}_{bits}b_it{maxit}';outp=f'{D}/sdpa_out_{tag}.txt'
t=time.time();print('start',A.shape,A.nnz,flush=True)
opt={'epsilonStar':1e-25,'epsilonDash':1e-25,'mpfPrecision':bits,'maxIteration':maxit,'numThreads':threads,'lambdaStar':1e4,'upperBound':1e15,'lowerBound':-1e15,'print':outp,'xPrint':'%+.45Fe','yPrint':'NOPRINT','sPrint':'%+.45Fe'}
x,y,info,timing,extra=sdpap.solve(A,b,c,sdpap.SymCone(s=tuple(dims)),sdpap.SymCone(f=len(b)),opt)
xx=np.asarray(x.toarray()).ravel();eigs=[];grams=[];o=0
for n,ss in zip(dims,scales):
    Q=xx[o:o+n*n].reshape(n,n);Q=(Q+Q.T)/2+1e-5*np.eye(n);o+=n*n;eigs.append(float(la.eigvalsh(Q)[0]));grams.append((Q*norm/np.outer(ss,ss)).tolist())
res=float(np.max(np.abs(A@xx-b)))
r={'order':m,'status':str(info.get('phasevalue')),'proved':False,'normalization':norm,'basis_scales':scales,'min_scaled_eigenvalues':eigs,'residual':res,'solver':'sdpap SDPA-GMP','bits':bits,'threads':threads,'maxIteration':maxit,'seconds':time.time()-t,'timing':{k:float(v) for k,v in timing.items()} if isinstance(timing,dict) else str(timing),'sdpa_output_file':outp,'gram_matrices':grams}
json.dump(r,open(f'{D}/seed_{tag}.json','w'))
print({k:v for k,v in r.items() if k!='gram_matrices' and k!='basis_scales'},flush=True)
