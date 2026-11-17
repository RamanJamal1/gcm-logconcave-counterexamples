"""Run at most ten minutes of B, resuming saved RNG and measured active time."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='2';os.environ['TORCHINDUCTOR_COMPILE_THREADS']='1'
import json,time,datetime
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
 os.chdir(ROOT);state_path=ROOT/'checkpoints/campaign_B.json';state=json.loads(state_path.read_text());remaining=max(0,7200-state.get('active_runtime_seconds',0));seconds=min(600,remaining);status='bounded_run_complete' if remaining<=600 else 'chunk_complete'
 print('B chunk: remaining active seconds',remaining,'this segment',seconds,flush=True)
 if remaining:
  from orchestrate_compiled import main
  args=SimpleNamespace(orders=[7,8],seconds=seconds,generations=100000,batch=65536,microbatch=65536,elites=512,refine_chunk=16,refine_steps=8,grid=513,dense_grid=2049,terms=3,workers=4,seed=260830276,state='checkpoints/campaign_B.json',resume=True,sync_seconds=1500,sync=True,completion_status=status);main(args)
 else:state['status']='bounded_run_complete';state_path.write_text(json.dumps(state,indent=2)+'\n')
 if status=='bounded_run_complete':
  p=ROOT/'checkpoints/overnight_schedule.json';schedule=json.loads(p.read_text());schedule.update(status='campaign_B_complete',phase='C_pending',campaign_B_completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());p.write_text(json.dumps(schedule,indent=2)+'\n')
