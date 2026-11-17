"""Warm compiled kernels, then run the scheduled two-hour order-seven/eight sweep."""
import os
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['TORCHINDUCTOR_COMPILE_THREADS']='1'
import json,time,datetime,runpy,fcntl
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]

def main():
 lock=open(ROOT/'.git/campaign_b.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 os.chdir(ROOT)
 print('warming compiled campaign B kernels',flush=True)
 runpy.run_path(str(ROOT/'src/warm_compiled.py'),run_name='__main__')
 runpy.run_path(str(ROOT/'src/warm_refinement.py'),run_name='__main__')
 schedule=json.loads((ROOT/'checkpoints/overnight_schedule.json').read_text())
 earliest=datetime.datetime.fromisoformat(schedule['start_utc']).timestamp()+9000
 while time.time()<earliest:time.sleep(min(30,earliest-time.time()))
 while time.time()<earliest+180:
  a=json.loads((ROOT/'checkpoints/campaign_A.json').read_text())
  if a.get('status')=='bounded_run_complete':break
  time.sleep(10)
 from orchestrate_compiled import main as campaign
 state='checkpoints/campaign_B.json'
 args=SimpleNamespace(orders=[7,8],seconds=7200,generations=100000,batch=32768,microbatch=2048,elites=512,refine_chunk=16,refine_steps=8,grid=513,dense_grid=2049,terms=3,workers=4,seed=260830276,state=state,resume=(ROOT/state).exists(),sync_seconds=1500,sync=True)
 schedule.update(phase='B',status='campaign_B_running',campaign_B_started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());(ROOT/'checkpoints/overnight_schedule.json').write_text(json.dumps(schedule,indent=2)+'\n');print('starting campaign B',schedule['campaign_B_started_utc'],flush=True)
 campaign(args)
 schedule.update(status='campaign_B_complete',phase='C_pending');(ROOT/'checkpoints/overnight_schedule.json').write_text(json.dumps(schedule,indent=2)+'\n')
 print('campaign B complete',flush=True)
if __name__=='__main__':main()
