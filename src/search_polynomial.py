"""Supplementary mixed-precision search in a squared-polynomial curvature family."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,json,time,math,multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import torch
from polynomial_curvature import evaluate,refine,decode
from checkpoint import checkpoint
ROOT=Path(__file__).resolve().parents[1]
def certify_job(candidate,order,path):
 from certify_polynomial import certify
 try:r=certify(candidate,order)
 except Exception as e:r={'status':'error','certified':False,'error':repr(e)}
 Path(path).write_text(json.dumps(r,indent=2)+'\n');return r

def main(args):
 torch.set_num_threads(1);torch.manual_seed(4512);start=time.time();state={'status':'running','order':args.order,'gpu_evaluations':0,'refinement_evaluations':0,'best':None,'errors':[]};lastsync=start;pool=ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn'));pending=[]
 while time.time()-start<args.seconds:
  elite=None;scores=None
  for _ in range(args.batch//args.microbatch):
   with torch.no_grad():
    raw=torch.randn(args.microbatch,7,device='cuda');values=evaluate(raw,args.order)[0];values=torch.nan_to_num(values,nan=float('inf'),posinf=float('inf'),neginf=float('inf'))
    rr=raw if elite is None else torch.cat([elite,raw]);vv=values if scores is None else torch.cat([scores,values]);idx=vv.topk(min(args.elites,len(vv)),largest=False).indices;elite=rr[idx];scores=vv[idx]
   state['gpu_evaluations']+=len(raw)
  for chunk in elite.split(8):
   try:
    refined,result=refine(chunk,args.order);state['refinement_evaluations']+=len(chunk)
    with torch.no_grad():fine=evaluate(refined,args.order,4097)
    for j in range(len(chunk)):
     value=float(fine[0][j]);coarse=float(result[0][j])
     if not math.isfinite(value) or abs(value-coarse)>1e-7*max(1,abs(value)):continue
     if state['best'] is None or value<state['best']['normalized']:
      q,t=decode(refined[j:j+1]);entry={'normalized':value,'signed':float(fine[1][j]),'variance':float(fine[2][j]),'q_coefficients':q[0].cpu().tolist(),'tilt':float(t[0]),'curvature_lower_bound':1,'certified':False};state['best']=entry
      (ROOT/f'checkpoints/polynomial_best_m{args.order}.json').write_text(json.dumps(entry,indent=2)+'\n')
      if value<0:
       print('NEGATIVE SCREEN',entry,flush=True);target=ROOT/f'certs/polynomial_m{args.order}_{state["gpu_evaluations"]}_{j}.json';pending.append(pool.submit(certify_job,entry,args.order,str(target)))
      if args.sync:checkpoint(f'polynomial m{args.order} best {value:.8g}')
   except (RuntimeError,FloatingPointError) as e:state['errors'].append(str(e)[:300]);torch.cuda.empty_cache()
  for future in list(pending):
   if future.done():
    result=future.result();state.setdefault('certificates',[]).append(result);pending.remove(future)
    if result.get('certified') and args.sync:checkpoint('certified polynomial counterexample')
  state['elapsed_seconds']=time.time()-start;(ROOT/'checkpoints/polynomial_campaign.json').write_text(json.dumps(state,indent=2)+'\n');print({k:v for k,v in state.items() if k!='best'},flush=True)
  if time.time()-lastsync>1500 and args.sync:checkpoint('polynomial campaign periodic');lastsync=time.time()
 pool.shutdown(wait=True)
 for future in pending:state.setdefault('certificates',[]).append(future.result())
 state['status']='bounded_run_complete';(ROOT/'checkpoints/polynomial_campaign.json').write_text(json.dumps(state,indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--order',type=int,default=12);p.add_argument('--seconds',type=float,default=1800);p.add_argument('--batch',type=int,default=32768);p.add_argument('--microbatch',type=int,default=512);p.add_argument('--elites',type=int,default=512);p.add_argument('--sync',action='store_true');main(p.parse_args())
