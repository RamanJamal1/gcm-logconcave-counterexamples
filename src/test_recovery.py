"""Exercise recoverable CUDA OOM and process-pool certification paths."""
import argparse,json,tempfile
from pathlib import Path
from unittest.mock import patch
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import torch
import orchestrate

def main():
 target='checkpoints/recovery_test.json';p=orchestrate.ROOT/target
 if p.exists():p.unlink()
 args=argparse.Namespace(state=target,resume=False,seed=912,batch=4,microbatch=4,elites=1,terms=2,workers=4,seconds=60,generations=1,grid=257,refine_chunk=1,refine_steps=1,dense_grid=513,sync=False,sync_seconds=1500)
 original=orchestrate.evaluate;calls=0
 def fail_once(*a,**kw):
  nonlocal calls
  calls+=1
  if calls==1:raise torch.cuda.OutOfMemoryError('injected readiness failure')
  return original(*a,**kw)
 with patch.object(orchestrate,'evaluate',fail_once):orchestrate.main(args)
 state=json.loads(p.read_text());assert state['generation']==1 and state['gpu_evaluations']==4 and 'OOM' in state['errors'][0]
 c={'family':'fourier_multiplier','amplitudes':['0'],'frequencies':['1'],'phases':['0']}
 with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
  futures=[pool.submit(orchestrate.certification_job,c,m,orchestrate.ROOT/f'certs/worker_gaussian_{m}.json') for m in [6,7,8]]
  results=[f.result(timeout=90) for f in futures]
 assert all(r['status']=='certified_nonnegative' for r in results)
 (orchestrate.ROOT/'certs/recovery_tests.json').write_text(json.dumps({'oom_recovery_passed':True,'certification_process_pool_passed':True,'worker_limit':4,'tested_orders':[6,7,8]},indent=2))
if __name__=='__main__':main()
