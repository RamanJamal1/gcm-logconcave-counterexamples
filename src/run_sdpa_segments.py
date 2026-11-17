"""Resume high-precision numerical SDP work and retain every completed segment."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,json,time,subprocess,math,datetime,hashlib
from pathlib import Path
import numpy as np,scipy.linalg as la
ROOT=Path(__file__).resolve().parents[1]
def run(m,segments=30):
 os.chdir(ROOT);state_path=ROOT/f'checkpoints/sdpa_segments_m{m}.json';state=json.loads(state_path.read_text()) if state_path.exists() else {'order':m,'segment':0,'status':'running','proved':False,'history':[]};data=np.load(ROOT/f'checkpoints/native_sos_m{m}.npz');A=data['A'];b=data['b'];dims=list(map(int,data['dims']));scales=[data['scales0'],data['scales1']];norm=math.factorial(m-1)
 initial=state.get('initial_file')
 for _ in range(segments):
  n=state['segment']+1;output=f'logs/native_m{m}_segment_{n}.ini-s';log=f'logs/native_m{m}_segment_{n}.log';start=time.time();state['status']='running';state_path.write_text(json.dumps(state,indent=2)+'\n')
  with open(log,'w') as f:r=subprocess.run([str(ROOT/'.venv/bin/sdpa_segment'),f'logs/native_m{m}.dat-s',output,'2','128']+([initial] if initial else []),stdout=f,stderr=subprocess.STDOUT)
  if r.returncode:
   state.update(status='segment_failed',returncode=r.returncode);state_path.write_text(json.dumps(state,indent=2)+'\n');print(state,flush=True);return
  qs=[np.zeros((d,d)) for d in dims]
  with open(output) as f:
   next(f)
   for line in f:
    k,block,i,j,value=line.split();k,block,i,j=map(int,[k,block,i,j])
    if k==2:qs[block-1][i-1,j-1]=qs[block-1][j-1,i-1]=float(value)
  xx=np.concatenate([q.ravel() for q in qs]);residual=float(np.max(abs(A@xx-b)));qs=[q+1e-5*np.eye(d) for q,d in zip(qs,dims)];eigs=[float(la.eigvalsh(q)[0]) for q in qs]
  result={'order':m,'proved':False,'status':'numerical_segment_candidate','normalization':norm,'basis_scales':[s.tolist() for s in scales],'gram_matrices':[(q*norm/np.outer(ss,ss)).tolist() for q,ss in zip(qs,scales)],'min_scaled_eigenvalues':eigs,'residual':residual,'segment':n,'seconds':time.time()-start};candidate=ROOT/f'certs/sos_segment_m{m}.json';candidate.write_text(json.dumps(result,indent=2)+'\n')
  state.update(segment=n,initial_file=output,last_residual=residual,min_scaled_eigenvalues=eigs,status='segment_complete');state['history'].append({k:v for k,v in result.items() if k not in ['gram_matrices','basis_scales']});state_path.write_text(json.dumps(state,indent=2)+'\n');print('segment',n,'eigenvalues',eigs,'residual',residual,'seconds',result['seconds'],flush=True)
  from checkpoint import checkpoint
  checkpoint(f'm{m} high-precision SDP segment {n}, residual={residual:.3g}')
  if 'iterations=0 ' in Path(log).read_text():
   state['status']='stalled';state_path.write_text(json.dumps(state,indent=2)+'\n');return
  if min(eigs)>1e-8 and residual<1e-9:
   state['status']='ready_for_exact_reconstruction';state_path.write_text(json.dumps(state,indent=2)+'\n');return
  initial=output
  if datetime.datetime.now(datetime.timezone.utc).hour>=13:return
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--order',type=int,default=12);p.add_argument('--segments',type=int,default=30);a=p.parse_args();run(a.order,a.segments)
