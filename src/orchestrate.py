"""Bounded, restartable CUDA campaign with local checkpoints and GitHub sync.

Use --seconds and --generations for a bounded readiness run. No import starts work.
"""
import os
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
import argparse,json,time,math,multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import torch
from cuda_pipeline import evaluate,refine,parameters,FAMILIES
from gpu_search import atomic_json
from checkpoint import checkpoint
ROOT=Path(__file__).resolve().parents[1]

def certification_job(c,order,path):
 from certify_arb import certify
 try:r=certify(c,order)
 except Exception as e:r={'status':'error','certified':False,'error':repr(e)}
 atomic_json(path,r);return r

def progress(state):
 lines=['# GCM research progress','',f"Status: {state['status']}",f"GPU evaluations: {state['gpu_evaluations']}",f"Completed generations: {state['generation']}",'','Numerical candidates are not interval certificates.']
 for m,e in state['best'].items():lines.append(f"- Order {m}: normalized signed derivative {e['normalized']:.12g}; certified: false")
 (ROOT/'PROGRESS.md').write_text('\n'.join(lines)+'\n')

def main(args):
 path=ROOT/args.state
 if path.exists():
  if not args.resume:raise RuntimeError('State exists: use --resume or a new --state path')
  state=json.loads(path.read_text())
  if state['seed']!=args.seed:raise ValueError('Resume seed mismatch')
 else:state={'schema':1,'seed':args.seed,'generation':0,'gpu_evaluations':0,'refinement_evaluations':0,'best':{},'certificates':[],'status':'running','errors':[]}
 torch.manual_seed(args.seed);rng=torch.Generator(device='cuda');rng.manual_seed(args.seed)
 if 'rng_state' in state:rng.set_state(torch.tensor(state['rng_state'],dtype=torch.uint8))
 pending=[];start=time.monotonic();last_sync=start;initial_generation=state['generation']
 state['status']='running';state['backend']='torch_cuda';state['device']=torch.cuda.get_device_name(0)
 def save(sync=False):
  state['rng_state']=rng.get_state().tolist();atomic_json(path,state);progress(state)
  if sync and args.sync:
   reason=' '.join(f'm{m}={v["normalized"]:.7g}' for m,v in state['best'].items())
   result=checkpoint(reason or 'campaign initialized');atomic_json(ROOT/'certs/latest_sync.json',result)
 pool=ProcessPoolExecutor(max_workers=min(args.workers,4),mp_context=multiprocessing.get_context('spawn'))
 try:
  while time.monotonic()-start<args.seconds and state['generation']-initial_generation<args.generations:
   g=state['generation'];family=FAMILIES[g%4];order=6+(g//4)%3;remaining=args.batch;elites=[];scores=[];micro=args.microbatch
   while remaining:
    n=min(micro,remaining)
    try:
     with torch.no_grad():
      raw=torch.randn(n,4,args.terms,device='cuda',generator=rng)
      values=evaluate(raw,family,args.grid)[0][:,order-6]
      values=torch.where(torch.isfinite(values),values,torch.full_like(values,float('inf')))
      elites.append(raw);scores.append(values)
      rr=torch.cat(elites);vv=torch.cat(scores);idx=torch.topk(vv,min(args.elites,len(vv)),largest=False).indices
      elites=[rr[idx]];scores=[vv[idx]]
     remaining-=n;state['gpu_evaluations']+=n
    except torch.cuda.OutOfMemoryError:
     if micro<=1:raise
     micro=max(1,micro//2);torch.cuda.empty_cache();state['errors'].append('exploration OOM: microbatch reduced')
   improved=False
   for raw in torch.split(elites[0],args.refine_chunk):
    try:
     optimized,result=refine(raw,family,order,args.refine_steps,args.dense_grid)
     state['refinement_evaluations']+=len(raw)
     # Independent dense resolution check after optimization.
     with torch.no_grad():fine=evaluate(optimized,family,2*args.dense_grid-1)
     for j in range(len(raw)):
      val=float(fine[0][j,order-6]);coarse=float(result[0][j,order-6]);curvature=float(fine[2][j])
      if not math.isfinite(val) or abs(val-coarse)>1e-7*max(1,abs(val)) or curvature<=0:continue
      if val<state['best'].get(str(order),{}).get('normalized',float('inf')):
       c=parameters(optimized[j],family);entry={'parameters':c,'normalized':val,'signed':float(fine[1][j,order-6]),'convexity_bound':curvature,'certified':False,'generation':g,'grid':2*args.dense_grid-1}
       state['best'][str(order)]=entry;atomic_json(ROOT/f'checkpoints/elite_m{order}.json',entry);improved=True
       if entry['signed']<-1e-8:
        target=ROOT/f'certs/m{order}_g{g}_{j}.json';pending.append((str(target),pool.submit(certification_job,c,order,target)))
       save(sync=True)
    except (RuntimeError,FloatingPointError) as e:
     state['errors'].append(str(e)[:300]);torch.cuda.empty_cache()
   state['generation']+=1
   for target,future in list(pending):
    if future.done():
     result=future.result();state['certificates'].append({'path':target,'status':result['status']});pending.remove((target,future));save(sync=result.get('certified',False))
   periodic=time.monotonic()-last_sync>=args.sync_seconds
   save(sync=periodic)
   if periodic:last_sync=time.monotonic()
   print(json.dumps({k:state[k] for k in ['generation','gpu_evaluations','refinement_evaluations','errors']}),flush=True)
  state['status']='bounded_run_complete'
 finally:
  pool.shutdown(wait=True)
  for target,future in pending:
   result=future.result();state['certificates'].append({'path':target,'status':result['status']})
  save(sync=True)
 print(json.dumps({'status':state['status'],'state':str(path),'best':state['best']},indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seconds',type=float,default=60);p.add_argument('--generations',type=int,default=12);p.add_argument('--batch',type=int,default=32768);p.add_argument('--microbatch',type=int,default=128);p.add_argument('--elites',type=int,default=512);p.add_argument('--refine-chunk',type=int,default=4);p.add_argument('--refine-steps',type=int,default=8);p.add_argument('--grid',type=int,default=513);p.add_argument('--dense-grid',type=int,default=2049);p.add_argument('--terms',type=int,default=3);p.add_argument('--workers',type=int,default=4);p.add_argument('--seed',type=int,default=260830275);p.add_argument('--state',default='checkpoints/campaign.json');p.add_argument('--resume',action='store_true');p.add_argument('--sync-seconds',type=float,default=1500);p.add_argument('--sync',action='store_true');main(p.parse_args())
