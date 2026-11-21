"""GPU rationalization sweep, with every retained negative result checked by Arb.

The periodic GPU objective is a screening surrogate. Only the full Gaussian
Fourier-series enclosure can promote a candidate to a counterexample.
"""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,json,time,math,multiprocessing,hashlib,heapq
from pathlib import Path
from fractions import Fraction
from concurrent.futures import ProcessPoolExecutor
import torch
from periodic_cuda import evaluate
from gpu_search import atomic_json
from checkpoint import checkpoint
ROOT=Path(__file__).resolve().parents[1]

def pool_workers():
 p=ROOT/'checkpoints/certification_workers.json'
 cap=max(1,(os.cpu_count() or 4)-2)
 return min(cap,max(1,int(json.loads(p.read_text()).get('workers',cap)))) if p.exists() else cap

def certify_job(order,params):
 from certify_multifourier_spatial import certify
 import contextlib
 try:
  with open(os.devnull,'w') as sink,contextlib.redirect_stdout(sink):return certify(order,list(params[:-1]),params[-1])
 except Exception as e:return {'certified':False,'error':repr(e),'parameters':params,'order':order}

def main(seconds=3600,batch=262144,sync=False,state_path='checkpoints/campaign_C.json',resume=True):
 torch.set_num_threads(1);torch.manual_seed(170617);start=time.time();lastsync=start
 research=json.loads((ROOT/'checkpoints/research_status.json').read_text());witness=json.loads((ROOT/research['negative_witness_certificate']).read_text());order=research['negative_witness_order'];reference=witness['parameters']['amplitudes'];K=len(reference)
 fractions=sorted({Fraction(p,q) for q in range(10,1001) for p in range(1,min(150,q//5)+1) if Fraction(1,100)<=Fraction(p,q)<=Fraction(1,5)})
 values64=torch.tensor([float(f) for f in fractions],device='cuda',dtype=torch.float64);values=values64.float();pool=ProcessPoolExecutor(max_workers=pool_workers(),mp_context=multiprocessing.get_context('spawn'),max_tasks_per_child=32);pending=[];seen=set();queue=[];QMAX=4096
 state={'status':'running','gpu_evaluations':0,'refinement_evaluations':0,'generation':0,'certifications':0,'best_certified':{},'errors':[],'method':'rational multi-harmonic GPU periodic screening; full spatial Arb verification','order':order,'device':torch.cuda.get_device_name(0)}
 path=ROOT/state_path
 if K==3:
  box_proof=json.loads((ROOT/'certs/curvature_rationalization_box.json').read_text());assert box_proof['certified']
 if resume and path.exists():
  state=json.loads(path.read_text())
  if state.get('rng_state'):torch.cuda.set_rng_state(torch.tensor(state['rng_state'],dtype=torch.uint8))
  seen={(order,tuple(v)) for v in state.get('seen_parameters',[])}
 prior_runtime=state.get('active_runtime_seconds',0)
 state['status']='running'
 def save(reason=None):
  state['configured_certification_worker_limit']=pool_workers();state['peak_cuda_memory_allocated_bytes']=max(state.get('peak_cuda_memory_allocated_bytes',0),torch.cuda.max_memory_allocated());state['elapsed_seconds']=time.time()-start;state['active_runtime_seconds']=prior_runtime+time.time()-start;state['rng_state']=torch.cuda.get_rng_state().tolist();state['seen_parameters']=[list(v) for _,v in sorted(seen)];atomic_json(path,state)
  if reason and sync:checkpoint(reason)
 def harvest(wait=False):
  for future,params,order in list(pending):
   if not wait and not future.done():continue
   r=future.result();pending.remove((future,params,order));state['certifications']+=1
   if r.get('certified'):
    from flint import arb
    score=float(arb(r['signed_derivative_interval']).mid());previous=state['best_certified'].get(str(order),{}).get('signed',float('inf'))
    if score<previous:
     name=f'certs/{Path(state_path).stem}_m{order}_best.json';atomic_json(ROOT/name,r);state['best_certified'][str(order)]={'signed':score,'parameters':r['parameters'],'certificate':name};save(f'certified rationalized m{order}={score:.12g}')
 def run_generation(order):
  a_ref=float(Fraction(reference[0]));lower=torch.searchsorted(values,torch.tensor(a_ref*.75,device='cuda')).item();upper=torch.searchsorted(values,torch.tensor(a_ref*1.25,device='cuda')).item()
  eligible=torch.tensor([j for j in range(lower,upper) if fractions[j].denominator<=100],device='cuda');idx=eligible[torch.randint(len(eligible),(batch,),device='cuda')];aa=values[idx];amplitudes=[aa];denominators=[]
  for j,ref in enumerate(reference[1:],1):
   rr=float(Fraction(ref));scale=10**max(0,math.floor(math.log10(1/abs(rr)))-2)
   denominator=(torch.exp(torch.randn(batch,device='cuda')*.3)/(abs(rr)*scale)).round().clamp(min=1)*scale
   value=math.copysign(1,rr)/denominator
   # Dropping each higher mode is explicitly included.
   value=torch.where(torch.arange(batch,device='cuda')%32==j,torch.zeros_like(value),value)
   amplitudes.append(value);denominators.append(denominator)
  amps=torch.stack(amplitudes,dim=1);frequencies=torch.arange(1,K+1,device='cuda');bound=(abs(amps)*frequencies**2).sum(1)/(1-abs(amps).sum(1))
  if K==3:
   # The interval box certificate proves the maximum occurs at cos(x)=-1.
   box=(amps[:,0]>=.04)&(amps[:,0]<=.09)&(amps[:,1]>=0)&(amps[:,1]<=.0001)&(amps[:,2]>=-.000001)&(amps[:,2]<=0)
   endpoint=(amps[:,0]-4*amps[:,1]+9*amps[:,2])/(1-amps[:,0]+amps[:,1]-amps[:,2])+1e-6
   bound=torch.where(box,endpoint,bound)
  offset=torch.randint(1,40,(batch,),device='cuda');ki=(torch.searchsorted(values,bound)+offset-1).clamp(max=len(values)-1);kk=values[ki]
  valid=kk>bound;params=torch.cat([amps,kk[:,None]],dim=1);score=torch.full((batch,),float('inf'),device='cuda');micro=65536
  begin=0
  while begin<batch:
   end=min(batch,begin+micro)
   try:
    with torch.no_grad():score[begin:end]=evaluate(params[begin:end,:K],params[begin:end,K],order,256)
    begin=end
   except torch.cuda.OutOfMemoryError:
    if micro<=64:raise
    micro//=2;torch.cuda.empty_cache();state['errors'].append('FP32 OOM: reduced microbatch')
  score=torch.where(valid & torch.isfinite(score),score,torch.full_like(score,float('inf')));top=score.topk(min(512,batch),largest=False).indices;elite=torch.stack([values64[idx[top]]]+[torch.where(amps[top,j+1]==0,torch.zeros(len(top),device='cuda',dtype=torch.float64),math.copysign(1,float(Fraction(reference[j+1])))/denominators[j][top].double()) for j in range(K-1)]+[values64[ki[top]]],dim=1);state['gpu_evaluations']+=batch
  with torch.no_grad():
   coarse=evaluate(elite[:,:K],elite[:,K],order,1024);fine=evaluate(elite[:,:K],elite[:,K],order,2048)
  state['refinement_evaluations']+=len(elite);consistent=torch.isfinite(fine)&(abs(fine-coarse)<=1e-7*torch.maximum(torch.ones_like(fine),abs(fine)))
  # Filter on the GPU, then one device-to-host transfer for the best 128 refined elites.
  sel=fine.argsort()[:128];sel=sel[consistent[sel]&(fine[sel]<0)];orig=top[sel]
  dens=torch.stack(denominators,1)[orig].tolist() if denominators else [[] for _ in range(len(orig))]
  signs=[1 if float(Fraction(r))>0 else -1 for r in reference[1:]]
  for fv,ii,kv,am,dn in zip(fine[sel].tolist(),idx[orig].tolist(),ki[orig].tolist(),amps[orig].tolist(),dens):
   a=fractions[ii];k=fractions[kv]
   rational_amplitudes=[a]+[Fraction(0) if am[l+1]==0 else Fraction(signs[l],int(dn[l])) for l in range(K-1)]
   strings=tuple(map(str,rational_amplitudes+[k]));key=(order,strings)
   if key in seen:continue
   seen.add(key);heapq.heappush(queue,(fv,strings))
   # Bounded best-first queue: CPU memory stays independent of campaign duration, GPU never waits.
   if len(queue)>QMAX:queue[:]=heapq.nsmallest(QMAX//2,queue)
 def feed(order):
  limit=2*pool_workers()
  while queue and len(pending)<limit:
   strings=heapq.heappop(queue)[1];pending.append((pool.submit(certify_job,order,strings),strings,order))
 lastsave=0
 while time.time()-start<seconds:
  run_generation(order);state['generation']+=1;harvest();feed(order)
  if time.time()-lastsave>=10:
   state['certification_queue']=len(queue);save();lastsave=time.time()
   print({k:v for k,v in state.items() if k not in ['errors','rng_state','seen_parameters']},flush=True)
  if time.time()-lastsync>=1500:save('campaign C periodic rationalization checkpoint');lastsync=time.time()
 harvest(wait=True);pool.shutdown();state['certification_queue']=len(queue);state['status']='bounded_run_complete';save('campaign C complete')
 return state
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seconds',type=float,default=3600);p.add_argument('--batch',type=int,default=262144);p.add_argument('--sync',action='store_true');p.add_argument('--state',default='checkpoints/campaign_C.json');a=p.parse_args();main(a.seconds,a.batch,a.sync,a.state)
