"""Congruence-preconditioned native double SDP locator; never itself a proof."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,json,time,math,subprocess
from pathlib import Path
import numpy as np,scipy.linalg as la
from checkpoint_io import write_json
ROOT=Path(__file__).resolve().parents[1]
def solve(m):
 start=time.time();d=np.load(ROOT/f'checkpoints/native_sos_m{m}.npz');A=d['A'];dims=list(map(int,d['dims']));scales=[d['scales0'],d['scales1']];norm=math.factorial(m-1);identity=np.concatenate([np.eye(n).ravel() for n in dims]);b=d['b']+1e-5*(A@identity);offsets=np.cumsum([0]+[n*n for n in dims]);warm=json.loads((ROOT/f'certs/sos_segment_m{m}.json').read_text());qs=[np.array(q)*np.outer(ss,ss)/norm for q,ss in zip(warm['gram_matrices'],scales)];history=[]
 env=os.environ.copy();env['LD_LIBRARY_PATH']=str(ROOT/'.venv/vendor/sdpa-double/usr/lib/x86_64-linux-gnu')+':'+str(ROOT/'.venv/vendor/sdpa-double/usr/lib/x86_64-linux-gnu/openblas-pthread');env['SDPA_LAMBDA']='1'
 for step,floor in enumerate([1e-5,1e-3,1e-1]):
  Ws=[];blocks=[]
  for j,(q,n) in enumerate(zip(qs,dims)):
   w,U=la.eigh(q);W=U*np.sqrt(np.maximum(w,floor));Ws.append(W);blocks.append(np.matmul(np.matmul(W.T[None,:,:],A[:,offsets[j]:offsets[j+1]].reshape(len(b),n,n)),W[None,:,:]).reshape(len(b),-1))
  T=np.concatenate(blocks,axis=1);U,R=la.qr(T.T,mode='economic');rhs=la.solve_triangular(R.T,b,lower=True);T=U.T;delta=1e-6;rhs-=delta*(T@identity);inp=ROOT/f'logs/congruence_native_m{m}_{step}.dat-s';out=ROOT/f'logs/congruence_native_m{m}_{step}.ini-s';log=ROOT/f'logs/congruence_native_m{m}_{step}.log'
  with inp.open('w') as f:
   f.write(f'{len(rhs)}\n2\n'+ ' '.join(map(str,dims))+'\n'+' '.join(f'{-v:.17e}' for v in rhs)+'\n')
   for bl,n in enumerate(dims):
    for i in range(n):f.write(f'0 {bl+1} {i+1} {i+1} -1\n')
    block=T[:,offsets[bl]:offsets[bl+1]].reshape(len(rhs),n,n)
    for row in range(len(rhs)):
     for i in range(n):
      for j in range(i,n):
       v=-block[row,i,j]
       if v:f.write(f'{row+1} {bl+1} {i+1} {j+1} {v:.17e}\n')
  print('prepared',step,time.time()-start,flush=True)
  with log.open('w') as f:r=subprocess.run([str(ROOT/'.venv/bin/sdpa_double'),str(inp),str(out),'100','64'],stdout=f,stderr=subprocess.STDOUT,env=env,timeout=240)
  if r.returncode:continue
  rs=[np.zeros((n,n)) for n in dims]
  for line in out.read_text().splitlines()[1:]:
   k,bl,i,j,v=line.split();k,bl,i,j=map(int,[k,bl,i,j])
   if k==2:rs[bl-1][i-1,j-1]=rs[bl-1][j-1,i-1]=float(v)
  trial=[W@(r+delta*np.eye(len(r)))@W.T for W,r in zip(Ws,rs)];xx=np.concatenate([q.ravel() for q in trial]);res=float(np.max(abs(A@xx-b)));xx+=A.T@la.solve(A@A.T,b-A@xx,assume_a='pos');trial=[xx[offsets[j]:offsets[j+1]].reshape(n,n) for j,n in enumerate(dims)];eigs=[float(la.eigvalsh(q)[0]) for q in trial];record={'order':m,'proved':False,'status':'numerical_congruence_seed','normalization':norm,'basis_scales':[s.tolist() for s in scales],'gram_matrices':[(q*norm/np.outer(ss,ss)).tolist() for q,ss in zip(trial,scales)],'min_scaled_eigenvalues':eigs,'residual_before_projection':res,'residual':float(np.max(abs(A@xx-b))),'seconds':time.time()-start,'floor':floor,'source_log':str(log.relative_to(ROOT))};write_json(ROOT/f'certs/sos_native_congruence_m{m}_{step}.json',record);print({k:v for k,v in record.items() if k not in ['basis_scales','gram_matrices']},flush=True)
  if min(eigs)>1e-8:return
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--order',type=int,default=12);a=p.parse_args();solve(a.order)
