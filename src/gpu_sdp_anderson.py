"""CUDA FP64 safeguarded Anderson-accelerated Douglas-Rachford locator for affine positive-semidefinite Grams.

This is a numerical feasibility search, not a proof. Every output still needs
exact rational reconstruction and an independent polynomial/positivity check.
"""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,time,json,math
from pathlib import Path
from fractions import Fraction
import numpy as np,torch
from checkpoint_io import write_json
ROOT=Path(__file__).resolve().parents[1]
def solve(m,seconds,source=None):
 torch.set_num_threads(1);start=time.time();d=np.load(ROOT/f'checkpoints/native_sos_m{m}.npz');dims=list(map(int,d['dims']));scales=[d['scales0'].tolist(),d['scales1'].tolist()];norm=math.factorial(m-1);offsets=np.cumsum([0]+[n*n for n in dims]);device='cuda';dtype=torch.float64;A=torch.tensor(d['A'],device=device,dtype=dtype);b=torch.tensor(d['b'],device=device,dtype=dtype);chol=torch.linalg.cholesky(A@A.T);delta=1e-5
 def affine(x):return x+A.T@torch.cholesky_solve((b-A@x)[:,None],chol)[:,0]
 def blocks(x):return [x[offsets[j]:offsets[j+1]].reshape(n,n) for j,n in enumerate(dims)]
 def positive(x):
  pieces=[]
  for q in blocks(x):
   w,U=torch.linalg.eigh((q+q.T)/2);pieces.append(((U*w.clamp_min(0))@U.T).reshape(-1))
  return torch.cat(pieces)
 if source:
  r=json.loads(Path(source).read_text());x=torch.tensor(np.concatenate([(np.array([[float(Fraction(str(v))) for v in row] for row in q])*np.outer(s,s)/norm-delta*np.eye(len(s))).ravel() for q,s in zip(r['gram_matrices'],scales)]),device=device,dtype=dtype)
 else:x=torch.cat([torch.eye(n,device=device,dtype=dtype).reshape(-1) for n in dims])
 best=-float('inf');iteration=0;history=[];state_path=ROOT/f'checkpoints/gpu_sdp_anderson_m{m}.json';candidate_path=ROOT/f'certs/sos_gpu_anderson_m{m}.json'
 while time.time()-start<seconds:
  p=positive(x);q=affine(2*p-x);f=1.6*(q-p);g=x+f;history.append((x.clone(),f.clone()));history=history[-7:]
  if len(history)>2 and iteration%5==0:
   dx=torch.stack([history[j+1][0]-history[j][0] for j in range(len(history)-1)],1);df=torch.stack([history[j+1][1]-history[j][1] for j in range(len(history)-1)],1);H=df.T@df;w,U=torch.linalg.eigh(H);inverse=torch.where(w>w[-1].clamp_min(1e-30)*1e-12,1/w.clamp_min(1e-30),torch.zeros_like(w));gamma=U@(inverse*(U.T@(df.T@f)));proposal=g-(dx+df)@gamma
   if torch.isfinite(proposal).all() and float(gamma.norm())<100:
    py=positive(proposal);pf=1.6*(affine(2*py-proposal)-py)
    if float(pf.norm())<float(f.norm()):g=proposal
    else:history=history[-2:]
   else:history=history[-2:]
  x=g
  if iteration%25==0:
   candidate=affine(p);qs=[q+delta*torch.eye(n,device=device,dtype=dtype) for q,n in zip(blocks(candidate),dims)];eigs=[float(torch.linalg.eigvalsh((q+q.T)/2)[0]) for q in qs];mineig=min(eigs);res=float((A@candidate-b).abs().max())
   if mineig>best:
    best=mineig;r={'order':m,'proved':False,'status':'numerical_GPU_affine_PSD_seed','normalization':norm,'basis_scales':scales,'gram_matrices':[(q.cpu().numpy()*norm/np.outer(s,s)).tolist() for q,s in zip(qs,scales)],'min_scaled_eigenvalues':eigs,'residual':res,'iteration':iteration,'seconds':time.time()-start,'source':source};write_json(candidate_path,r)
   state={'order':m,'status':'running','proved':False,'gpu_matrix_iterations':iteration+1,'best_min_scaled_eigenvalue':best,'current_min_scaled_eigenvalues':eigs,'residual':res,'seconds':time.time()-start,'candidate':str(candidate_path.relative_to(ROOT)),'device':torch.cuda.get_device_name()};write_json(state_path,state)
   if iteration%250==0:print(iteration,eigs,res,time.time()-start,flush=True)
   if mineig>1e-7 and res<1e-7:state['status']='ready_for_exact_reconstruction';break
  iteration+=1
 if state['status']=='running':state['status']='bounded_run_complete'
 write_json(state_path,state);print(state,flush=True);return state
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--order',type=int,default=15);p.add_argument('--seconds',type=float,default=600);p.add_argument('--source');a=p.parse_args();solve(a.order,a.seconds,a.source)
