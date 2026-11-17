"""GPU rationalization sweep, with every retained negative result checked by Arb.

The periodic GPU objective is a screening surrogate. Only the full Gaussian
Fourier-series enclosure can promote a candidate to a counterexample.
"""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,json,time,math,multiprocessing,hashlib
from pathlib import Path
from fractions import Fraction
from concurrent.futures import ProcessPoolExecutor
import torch
from periodic_cuda import evaluate
from gpu_search import atomic_json
from checkpoint import checkpoint
ROOT=Path(__file__).resolve().parents[1]

def certify_job(order,params):
 from certify_fourier_series import certify
 try:return certify(order,*params,N=128,dps=200,mode_cutoff=16)
 except Exception as e:return {'certified':False,'error':repr(e),'parameters':params,'order':order}

def main(seconds=3600,batch=65536,sync=False,state_path='checkpoints/campaign_C.json'):
 torch.set_num_threads(1);torch.manual_seed(170617);start=time.time();lastsync=start
 fractions=sorted({Fraction(p,q) for q in range(10,101) for p in range(1,11) if Fraction(1,100)<=Fraction(p,q)<=Fraction(1,8)})
 values64=torch.tensor([float(f) for f in fractions],device='cuda',dtype=torch.float64);values=values64.float();pool=ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn'));pending=[];seen=set()
 state={'status':'running','gpu_evaluations':0,'refinement_evaluations':0,'generation':0,'certifications':0,'best_certified':{},'errors':[],'method':'rational parameter GPU periodic screening; full Gaussian Arb verification','device':torch.cuda.get_device_name(0)}
 path=ROOT/state_path
 def save(reason=None):
  state['elapsed_seconds']=time.time()-start;atomic_json(path,state)
  if reason and sync:checkpoint(reason)
 def harvest(wait=False):
  for future,params,order in list(pending):
   if not wait and not future.done():continue
   r=future.result();pending.remove((future,params,order));state['certifications']+=1
   if r.get('certified'):
    from flint import arb
    score=float(arb(r['signed_derivative_interval']).mid());previous=state['best_certified'].get(str(order),{}).get('signed',float('inf'))
    if score<previous:
     name=f'certs/campaign_C_m{order}_best.json';atomic_json(ROOT/name,r);state['best_certified'][str(order)]={'signed':score,'parameters':r['parameters'],'certificate':name};save(f'certified rationalized m{order}={score:.12g}')
 def run_generation(order):
  idx=torch.randint(len(values),(batch,),device='cuda');aa=values[idx]
  denominator=500*torch.randint(20,1201,(batch,),device='cuda');bb=1/denominator
  # Explicitly try deleting the second Fourier mode in 1/16 of candidates.
  bb=torch.where(torch.arange(batch,device='cuda')%16==0,torch.zeros_like(bb),bb)
  bound=(aa+4*bb)/(1-aa-bb);offset=torch.randint(1,5,(batch,),device='cuda');ki=torch.searchsorted(values,bound)+offset-1;ki=ki.clamp(max=len(values)-1);kk=values[ki]
  valid=kk>bound;params=torch.stack([aa,bb,kk],dim=1);score=torch.full((batch,),float('inf'),device='cuda');micro=4096
  begin=0
  while begin<batch:
   end=min(batch,begin+micro)
   try:
    with torch.no_grad():score[begin:end]=evaluate(params[begin:end,:2],params[begin:end,2],order,256)
    begin=end
   except torch.cuda.OutOfMemoryError:
    if micro<=64:raise
    micro//=2;torch.cuda.empty_cache();state['errors'].append('FP32 OOM: reduced microbatch')
  score=torch.where(valid & torch.isfinite(score),score,torch.full_like(score,float('inf')));top=score.topk(min(512,batch),largest=False).indices;elite=torch.stack([values64[idx[top]],torch.where(bb[top]==0,torch.zeros_like(bb[top],dtype=torch.float64),1/denominator[top].double()),values64[ki[top]]],dim=1);state['gpu_evaluations']+=batch
  with torch.no_grad():
   coarse=evaluate(elite[:,:2],elite[:,2],order,1024);fine=evaluate(elite[:,:2],elite[:,2],order,2048)
  state['refinement_evaluations']+=len(elite);consistent=torch.isfinite(fine)&(abs(fine-coarse)<=1e-7*torch.maximum(torch.ones_like(fine),abs(fine)))
  for j in fine.argsort()[:16].tolist():
   if not consistent[j] or fine[j]>=0:continue
   original=int(top[j]);a=fractions[int(idx[original])];b=Fraction(0) if float(bb[original])==0 else Fraction(1,int(denominator[original]));k=fractions[int(ki[original])];strings=tuple(map(str,[a,b,k]));key=(order,strings)
   if key in seen:continue
   seen.add(key)
   # Queue is bounded so CPU memory remains independent of campaign duration.
   if len(pending)>=16:harvest(wait=True)
   pending.append((pool.submit(certify_job,order,strings),strings,order))
 while time.time()-start<seconds:
  order=16 if state['generation']%4==3 else 17
  run_generation(order);state['generation']+=1;harvest();save()
  print({k:v for k,v in state.items() if k not in ['errors']},flush=True)
  if time.time()-lastsync>=1500:save('campaign C periodic rationalization checkpoint');lastsync=time.time()
 harvest(wait=True);pool.shutdown();state['status']='bounded_run_complete';save('campaign C complete')
 return state
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seconds',type=float,default=3600);p.add_argument('--batch',type=int,default=65536);p.add_argument('--sync',action='store_true');p.add_argument('--state',default='checkpoints/campaign_C.json');a=p.parse_args();main(a.seconds,a.batch,a.sync,a.state)
